"""Operation Ship 75 strategy sweep: a per-cell Optuna study over the retrain grid.

For each ``(league, market)`` cell the sweep runs an Optuna study over the retrain axes
(normalization × dist-loss × blend-loss), training one deterministic ``meditate`` trial per grid
corner and scoring it through the *honest* production gate: the deterministic dump already carries
the pipeline's validation-fit joint calibration, and :func:`_score_normalization` runs the same
:func:`scorecard.gate_row` the production scorecard does — no test re-fit. The sweep is a fixed-HP
replica of the production HPO pipeline: same calibration, same gate, same dump decode; the *only*
differences are fixed hyperparameters in place of the Optuna search and the deterministic sandbox
write locations (so a trial never clobbers a real trained market). The objective minimizes the
negative ship slack, so the study's best trial is the most-shippable corner.

The retrain grid is enumerable and categorical today, so the sampler is
:class:`optuna.samplers.GridSampler` — exhaustive and deterministic, the right tool for a discrete
space (no TPE guessing over a dozen corners). The ``[kind, spec, stage]`` :data:`SEARCH_SPACE`
mirrors ``pipeline.py``'s ``hp_search_space`` (with a ``stage`` tag; every axis retrains today), so a
continuous retrain axis (a blend weight, say) is a one-line addition that flips the sampler to TPE.

Research scaffolding: the deterministic trials *rank* only — nothing ships off them. The top corner
is the real-HPO confirm candidate; a clean 5/5 on the official (full-HPO) scorecard is what ships.
"""

import importlib.resources as pkg_resources
import math
import pathlib
import subprocess
import time

import click
import optuna
import pandas as pd
import tabulate

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training import calibration
from sportstradamus.training.pipeline import LOSS_AUTO
from sportstradamus.training.scorecard import (
    apply_thresholds,
    gate_row,
    load_test_set,
    min_gate_slack,
)

# The normalization corners the SkewNormal gate can decode (and therefore score). The EB slug
# `centered_additive_eb_meanyr_k10` decodes off the dumped `GlobalMean` column (the gate re-adds
# the empirical-Bayes prior) — see docs/ship_gate.md.
_DECODABLE_SN_NORMS: tuple[str, ...] = (
    "ratio_meanyr",
    "centered_additive_mean10",
    "centered_additive_eb_meanyr_k10",
)

_SHIP_PRED_COL = "Blended_EV"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TEST_SETS_ROOT = pathlib.Path(str(pkg_resources.files(_data_pkg) / "test_sets"))
_DETERMINISTIC_MODEL_ROOT = _REPO_ROOT / "research" / "models" / "deterministic"
_DETERMINISTIC_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "deterministic"
# 30-minute ceiling; a deterministic meditate run is fast-HP, but SN cells with large datasets
# can take ~10 min, so 1800 s keeps CI from hanging without cutting off valid runs.
_MEDITATE_TRIAL_TIMEOUT_S = 1800
# Gates surfaced on the board / verdict (g6 is not scored here); a corner ships iff all pass.
_GATES: tuple[str, ...] = ("g1", "g2", "g3", "g4", "g5")

# Search axes in the ``[kind, spec, stage]`` shape — ``[kind, spec]`` mirrors pipeline's
# hp_search_space, ``stage`` tags the retrain grid the GridSampler enumerates. Every axis retrains:
# normalization and both loss axes change the trained model (the blend weight is fit *inside*
# meditate, so a ``crps`` blend needs a train). Calibration is deliberately NOT an axis: the
# pipeline auto-fits the joint ``(dispersion_cal, skew_cal)`` on validation per corner and the
# honest gate reads that fit off the dump — re-fitting calibration modes on the test dump would be
# the in-sample artifact the honest scorer exists to avoid. A continuous retrain axis would read
# ``["float", {"low": ...}, "retrain"]`` and require a ``suggest_float`` branch in
# :func:`_retrain_grid` plus a TPE sampler.
SEARCH_SPACE: dict[str, list] = {
    "normalization": ["categorical", list(_DECODABLE_SN_NORMS), "retrain"],
    "dist_training_loss": ["categorical", ["crps", "nll"], "retrain"],
    "blending_loss_fn": ["categorical", sorted(calibration.BLENDING_SLUGS), "retrain"],
}

# Default board: the covered-league withheld SkewNormal cells this lever can reach. NFL's
# passing-degenerate family and sharp-yardage cells are g1-blocked by any normalization
# (documented dead in docs/operation_ship_75.md), excluded on purpose.
DEFAULT_BOARD_CELLS: tuple[tuple[str, str], ...] = (
    ("WNBA", "AST"),
    ("WNBA", "PTS"),
    ("WNBA", "RA"),
    ("WNBA", "REB"),
    ("WNBA", "DREB"),
    ("WNBA", "fantasy points prizepicks"),
    ("NBA", "AST"),
    ("NBA", "DREB"),
    ("NBA", "FG3A"),
    ("NBA", "FGM"),
    ("NBA", "FGA"),
    ("NBA", "fantasy points prizepicks"),
    ("NFL", "carries"),
    ("NFL", "sacks taken"),
    ("NFL", "receptions"),
)

_BOARD_COLUMNS: list[str] = [
    "league",
    "market",
    "normalization",
    "dist_training_loss",
    "blending_loss_fn",
    "slack",
    "ships",
    "g1_pass",
    "g1_brier_diff_ci_hi",
    "g1_brier_skill",
    "g2_pass",
    "g2_star_z",
    "g3_pass",
    "g3_bench_z",
    "g4_pass",
    "g4_pit_ks",
    "g4_pit_ks_max",
    "g5_pass",
    "g5_ece_debiased",
    "central50_coverage",
    "dispersion_cal",
    "skew_cal",
    "n",
]

# Default output path for the living board — both CLI modes write here unless --out overrides.
STRATEGY_RESEARCH_BOARD: pathlib.Path = pathlib.Path(
    str(pkg_resources.files(_data_pkg) / "research" / "strategy_research_board.csv")
)


def _dump_paths(league: str, market: str, norm: str) -> tuple[pathlib.Path, pathlib.Path]:
    """CSV under ``test_sets/deterministic/<norm>/`` and pickle under ``research/models/deterministic/<norm>/``."""
    filename = market_file_slug(league, market)
    csv = _TEST_SETS_ROOT / "deterministic" / norm / f"{filename}.csv"
    mdl = _DETERMINISTIC_MODEL_ROOT / norm / f"{filename}.mdl"
    return csv, mdl


def _log_path(
    league: str,
    market: str,
    norm: str,
    dist_training_loss: str = LOSS_AUTO,
    blending_loss_fn: str = LOSS_AUTO,
) -> pathlib.Path:
    """Per-corner meditate log under ``research/logs/deterministic/<norm>/``.

    Keyed by the full corner (norm + both losses) so two loss corners of the same normalization
    don't overwrite each other's log — unlike the model/CSV dump, which meditate keys by
    normalization alone and each corner therefore retrains and scores before the next overwrites it.
    """
    filename = market_file_slug(league, market)
    return (
        _DETERMINISTIC_LOG_ROOT / norm / f"{filename}__{dist_training_loss}_{blending_loss_fn}.log"
    )


def _log_tail(path: pathlib.Path, n: int = 25) -> str:
    return "\n".join(path.read_text().splitlines()[-n:])


def _run_deterministic_meditate(
    league: str,
    market: str,
    norm: str,
    dist_training_loss: str = LOSS_AUTO,
    blending_loss_fn: str = LOSS_AUTO,
) -> None:
    """Train one deterministic ``(cell, normalization, dist-loss, blend-loss)`` trial via meditate.

    ``--deterministic`` pins RNGs and the fixed fast hyperparameters and dumps to the research
    sandbox (never production); ``--bypass-withholding`` lets a withheld cell train under the
    forced ``--target-normalization``. Non-``auto`` losses are forwarded as ``--dist-training-loss``
    / ``--blending-loss-fn`` so the sweep can vary them; ``auto`` leaves that arg off. The trained
    model is a *ranking* stand-in.

    meditate's full training log is captured to a per-corner file rather than streamed, so the
    sweep's own progress and verdict stay readable; on a failed trial the log's tail is surfaced.
    """
    cmd = [
        "poetry",
        "run",
        "meditate",
        "--league",
        league,
        "--market",
        market,
        "--deterministic",
        "--bypass-withholding",
        "--target-normalization",
        norm,
    ]
    if dist_training_loss != LOSS_AUTO:
        cmd += ["--dist-training-loss", dist_training_loss]
    if blending_loss_fn != LOSS_AUTO:
        cmd += ["--blending-loss-fn", blending_loss_fn]

    log_path = _log_path(league, market, norm, dist_training_loss, blending_loss_fn)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        try:
            subprocess.run(
                cmd,
                cwd=_REPO_ROOT,
                check=True,
                timeout=_MEDITATE_TRIAL_TIMEOUT_S,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            click.echo(f"  meditate failed — tail of {log_path}:\n{_log_tail(log_path)}", err=True)
            raise


def _score_normalization(league: str, market: str, norm: str) -> dict[str, object]:
    """Score a trained trial's dump; ``dispersion_cal`` / ``skew_cal`` are read from the pickle for context only."""
    csv_path, mdl_path = _dump_paths(league, market, norm)
    df = load_test_set(csv_path, _SHIP_PRED_COL)
    filedict = pd.read_pickle(mdl_path)
    row = apply_thresholds(
        gate_row(
            df, _SHIP_PRED_COL, league=league, market=market, strategy=norm, decode_strategy=norm
        )
    )
    return {
        "normalization": norm,
        "slack": min_gate_slack(row),
        "ships": bool(row.get("ship")),
        # All five gates (value + pass), not just g4 — so a normalization/loss choice's cost on
        # g1 (Brier non-inferiority) or g5 (ECE) is visible on the board, and g2/g3 + g1 brier
        # skill + central-50 coverage show how close the served EV sits to the outcome.
        "g1_pass": bool(row.get("g1_pass")),
        "g1_brier_diff_ci_hi": row.get("g1_brier_diff_ci_hi"),
        "g1_brier_skill": row.get("g1_brier_skill_score"),
        "g2_pass": bool(row.get("g2_pass")),
        "g2_star_z": row.get("g2_star_z"),
        "g3_pass": bool(row.get("g3_pass")),
        "g3_bench_z": row.get("g3_bench_z"),
        "g4_pass": bool(row.get("g4_pass")),
        "g4_pit_ks": row.get("g4_pit_ks"),
        "g4_pit_ks_max": row.get("g4_pit_ks_max"),
        "g5_pass": bool(row.get("g5_pass")),
        "g5_ece_debiased": row.get("g5_ece_debiased"),
        "central50_coverage": row.get("central50_coverage"),
        "dispersion_cal": filedict.get("dispersion_cal", 1.0),
        "skew_cal": filedict.get("skew_cal") or 0.0,
        "n": row.get("n_rows"),
    }


def _failed_gates(row: object) -> list[str]:
    """The scored gates a corner fails (empty when it ships). Accepts a dict or a DataFrame row."""
    return [g for g in _GATES if not row.get(f"{g}_pass", True)]


def _verdict(row: object) -> str:
    """``SHIP`` or ``KILL: g4 g5`` naming the failing gates — the human-scannable corner outcome."""
    if row.get("ships"):
        return "SHIP"
    failed = _failed_gates(row)
    return "KILL: " + " ".join(failed) if failed else "KILL"


def _run_and_score(
    league: str,
    market: str,
    *,
    normalization: str,
    dist_training_loss: str,
    blending_loss_fn: str,
) -> list[dict[str, object]]:
    """Train the ``(normalization × dist_training_loss × blending_loss_fn)`` retrain corner and score it.

    Calibration is not re-fit here: the dump already carries the pipeline's validation-fit joint
    calibration, and :func:`_score_normalization` reads it off the dump via the production
    :func:`scorecard.gate_row` — no test re-fit. The dump is keyed by normalization, so scoring
    right after this trial's train — before the next sequential trial overwrites it — keeps the
    loss/blend axes honest without a loss-keyed dump path (which is also why the sweep always
    retrains rather than reusing a dump). Returns a one-row list so the GridSampler objective and
    board assembly read uniformly.
    """
    corner = f"{normalization} · {dist_training_loss}/{blending_loss_fn}"
    click.echo(f"  training  {corner} …")
    start = time.monotonic()
    _run_deterministic_meditate(
        league,
        market,
        normalization,
        dist_training_loss=dist_training_loss,
        blending_loss_fn=blending_loss_fn,
    )
    row = _score_normalization(league, market, normalization)
    row["dist_training_loss"] = dist_training_loss
    row["blending_loss_fn"] = blending_loss_fn
    verdict = click.style(_verdict(row), fg="green" if row["ships"] else "red")
    elapsed = f"{time.monotonic() - start:.0f}s"
    click.echo(f"  {verdict}  {corner}  slack {float(row['slack']):+.3f}  ({elapsed})")
    return [row]


def _retrain_grid(space: dict[str, list]) -> dict[str, list]:
    """Loud on any non-categorical retrain axis — the only kind the GridSampler wires today."""
    grid: dict[str, list] = {}
    for name, (kind, spec, stage) in space.items():
        if stage != "retrain":
            continue
        if kind != "categorical":
            raise ValueError(
                f"model_strategy_sweep retrain axis {name!r} has unsupported kind {kind!r}; only "
                "'categorical' is wired — add the suggest_* branch + a TPE sampler when a "
                "continuous axis lands."
            )
        grid[name] = spec
    return grid


def search_cell(league: str, market: str, *, space: dict[str, list] = SEARCH_SPACE) -> pd.DataFrame:
    """Run the per-cell Optuna GridSampler study and return its board, ranked by ship slack.

    One honest row per retrain corner (normalization × dist-loss × blend-loss); the board carries
    each corner's slack / ship verdict / Gate-4 PIT-KS so the top row is the real-HPO confirm
    candidate. Sorted by ``slack`` descending.
    """
    grid = _retrain_grid(space)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(grid))

    def objective(trial: optuna.Trial) -> float:
        params = {name: trial.suggest_categorical(name, choices) for name, choices in grid.items()}
        rows = _run_and_score(
            league,
            market,
            normalization=params["normalization"],
            dist_training_loss=params["dist_training_loss"],
            blending_loss_fn=params["blending_loss_fn"],
        )
        trial.set_user_attr("rows", rows)
        return -max(float(row["slack"]) for row in rows)

    study.optimize(objective, n_trials=math.prod(len(choices) for choices in grid.values()))

    board = pd.DataFrame(
        [
            {"league": league, "market": market, **row}
            for trial in study.trials
            for row in trial.user_attrs["rows"]
        ]
    )
    return board.sort_values("slack", ascending=False, ignore_index=True)[_BOARD_COLUMNS]


def run_board(
    cells: tuple[tuple[str, str], ...] = DEFAULT_BOARD_CELLS, out: str | None = None
) -> pd.DataFrame:
    """Search every cell in ``cells``, printing each cell's verdict as it lands; write an
    incremental CSV after each so an interrupt keeps partial progress. Returns the concatenated
    board.
    """
    boards: list[pd.DataFrame] = []
    for league, market in cells:
        cell_board = search_cell(league, market)
        boards.append(cell_board)
        _print_cell_summary(cell_board)
        if out is not None:
            pd.concat(boards, ignore_index=True).to_csv(out, index=False)
    return pd.concat(boards, ignore_index=True)


def _upsert_cell(cell_board: pd.DataFrame, out: str) -> pd.DataFrame:
    """Merge one cell's rows into the board CSV at ``out`` — replacing any prior rows for that
    cell — so a single-cell run refreshes the living board instead of clobbering it. Returns the
    rows written for this cell (what the CLI summarizes).
    """
    path = pathlib.Path(out)
    if path.exists():
        prior = pd.read_csv(path)
        league, market = cell_board["league"].iloc[0], cell_board["market"].iloc[0]
        keep = prior[~((prior["league"] == league) & (prior["market"] == market))]
        pd.concat([keep, cell_board], ignore_index=True).to_csv(path, index=False)
    else:
        cell_board.to_csv(path, index=False)
    return cell_board


def _print_cell_summary(board: pd.DataFrame) -> None:
    """One cell's verdict: a colored best-corner headline over a narrow per-corner table.

    The board CSV keeps all ~24 columns; the screen shows only what a human scans (the winning
    corner, each corner's slack, and which gate blocks the losers).
    """
    best = board.iloc[0]
    headline = (
        f"\n{best['league']} {best['market']} — best: {best['normalization']} · "
        f"{best['dist_training_loss']}/{best['blending_loss_fn']} · slack "
        f"{float(best['slack']):+.3f} · "
    )
    click.echo(
        click.style(headline, bold=True)
        + click.style(_verdict(best), fg="green" if best["ships"] else "red", bold=True)
    )
    table = [
        [
            r["normalization"],
            r["dist_training_loss"],
            r["blending_loss_fn"],
            f"{float(r['slack']):+.3f}",
            "yes" if r["ships"] else "no",
            " ".join(_failed_gates(r)) or "-",
        ]
        for _, r in board.iterrows()
    ]
    click.echo(
        tabulate.tabulate(
            table,
            headers=["normalization", "dist", "blend", "slack", "ships", "failed gates"],
            tablefmt="github",
        )
    )


def _print_board_rollup(board: pd.DataFrame) -> None:
    """Board-wide tally: how many cells have a shipping corner, and each winner."""
    cells = list(board.groupby(["league", "market"], sort=False))
    shipping = [(lg, mkt, sub) for (lg, mkt), sub in cells if bool(sub["ships"].any())]
    click.echo(f"\n{len(cells)} cells swept · {len(shipping)} with a shipping corner")
    for lg, mkt, sub in shipping:
        best = sub.sort_values("slack", ascending=False).iloc[0]
        click.echo(
            click.style(
                f"  SHIP {lg} {mkt}: {best['normalization']} · "
                f"{best['dist_training_loss']}/{best['blending_loss_fn']} "
                f"(slack {float(best['slack']):+.3f})",
                fg="green",
            )
        )


@click.command(name="model-strategy-sweep")
@click.option("--league", default=None, help="League code, e.g. WNBA (single-cell mode).")
@click.option("--market", default=None, help="Market stem, e.g. AST (single-cell mode).")
@click.option(
    "--board/--no-board",
    default=False,
    help="Sweep the default covered-league board instead of one cell.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Board CSV path. Defaults to the package data dir "
    "(data/research/strategy_research_board.csv): --board overwrites it, a single cell upserts.",
)
def main(league: str | None, market: str | None, board: bool, out: str | None) -> None:
    """Operation Ship 75 strategy sweep — a per-cell GridSampler over the retrain grid
    (normalization × dist-loss × blend-loss), one honest val-fit→test gate row per corner.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = out or str(STRATEGY_RESEARCH_BOARD)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    if board:
        result = run_board(out=out)
        _print_board_rollup(result)
    else:
        if not (league and market):
            raise click.UsageError("pass --league and --market, or --board")
        cell_board = search_cell(league, market)
        result = _upsert_cell(cell_board, out)
        _print_cell_summary(cell_board)
    click.echo(f"\nboard: {out}")


if __name__ == "__main__":
    main()
