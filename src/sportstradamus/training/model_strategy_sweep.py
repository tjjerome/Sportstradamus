"""Operation Ship 75 strategy sweep: a per-cell Optuna study over a family's retrain grid.

For each ``(league, market)`` cell the sweep runs an Optuna study over the cell's distribution
family axes — SkewNormal sweeps ``normalization × dist-loss × blend-loss``; ZINB sweeps
``zinb-mode × count-dispersion-objective × blend-loss`` — training one deterministic ``meditate``
trial per grid corner and scoring it through the *honest* production gate: the deterministic dump
already carries the pipeline's validation-fit calibration, and :func:`_score_corner` runs the same
:func:`scorecard.gate_row` the production scorecard does — no test re-fit. The sweep is a fixed-HP
replica of the production HPO pipeline: same calibration, same gate, same dump decode; the *only*
differences are fixed hyperparameters in place of the Optuna search and the deterministic sandbox
write locations (so a trial never clobbers a real trained market). The objective minimizes the
negative ship slack, so the study's best trial is the most-shippable corner.

Families live in :data:`_FAMILIES`, a small registry keyed by the cell's ``dist``. Each
:class:`FamilySpec` names its grid axes, the ``stat_meta.json`` fields a winning corner persists,
and the shipped defaults for any non-persistable axis. Adding a family (or a future
distribution-family axis) is a registry entry, not an engine change. Every axis is categorical, so
the sampler is :class:`optuna.samplers.GridSampler` — exhaustive and deterministic, the right tool
for a discrete space.

Research scaffolding: the deterministic trials *rank* only — nothing ships off them. The confirm
loop (``--confirm``, :mod:`sportstradamus.training.model_strategy_confirm`) persists a winner and a
clean full-HPO 5/5 on the official scorecard is what actually ships.
"""

import importlib.resources as pkg_resources
import math
import pathlib
import subprocess
import time
from dataclasses import dataclass

import click
import optuna
import pandas as pd
import tabulate

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training import calibration
from sportstradamus.training.scorecard import (
    _DECODE_FALLBACK_STRATEGY,
    apply_thresholds,
    gate_row,
    load_test_set,
    min_gate_slack,
)
from sportstradamus.training.ship_config import (
    STAT_META_PATH,
    WITHHELD,
    load_stat_meta,
)

# The normalization corners the SkewNormal gate can decode (and therefore score). The EB slug
# `centered_additive_eb_meanyr_k10` decodes off the dumped `GlobalMean` column (the gate re-adds
# the empirical-Bayes prior) — see docs/ship_gate.md.
_DECODABLE_SN_NORMS: tuple[str, ...] = (
    "ratio_meanyr",
    "centered_additive_mean10",
    "centered_additive_eb_meanyr_k10",
)
_BLENDING: tuple[str, ...] = tuple(sorted(calibration.BLENDING_SLUGS))

# The SkewNormal family default training loss (the one that ships). A corner won under the other
# dist-loss can't be reproduced from stat_meta.json — dist-loss is a training-time knob that does
# not persist — so the actionable summary flags it and the confirm loop skips it.
_SN_DEFAULT_DIST_LOSS = "crps"

_SHIP_PRED_COL = "Blended_EV"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TEST_SETS_ROOT = pathlib.Path(str(pkg_resources.files(_data_pkg) / "test_sets"))
_DETERMINISTIC_MODEL_ROOT = _REPO_ROOT / "research" / "models" / "deterministic"
_DETERMINISTIC_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "deterministic"
# 30-minute ceiling; a deterministic meditate run is fast-HP, but cells with large datasets can
# take ~10 min, so 1800 s keeps CI from hanging without cutting off valid runs.
_MEDITATE_TRIAL_TIMEOUT_S = 1800
# The six offline ship gates (value + pass); a corner ships iff all pass. Mirrors
# scorecard._SHIP_GATES — apply_thresholds sets `ship` and min_gate_slack folds in all six.
_GATES: tuple[str, ...] = ("g1", "g2", "g3", "g4", "g5", "g6")

# Each swept axis' meditate CLI flag. A corner is realized by appending `--flag value` per axis;
# an axis a family doesn't sweep is simply absent (e.g. ZINB never forces --target-normalization,
# so meditate resolves it to the ratio_meanyr fallback — see _dump_subdir).
_AXIS_FLAG: dict[str, str] = {
    "normalization": "--target-normalization",
    "dist_training_loss": "--dist-training-loss",
    "blending_loss_fn": "--blending-loss-fn",
    "zinb_mode": "--zinb-mode",
    "count_dispersion_objective": "--count-dispersion-objective",
}


@dataclass(frozen=True)
class FamilySpec:
    """One distribution family's sweep definition.

    Attributes:
        axes: Grid axes ``{axis: choices}`` the GridSampler enumerates (all categorical today).
        persist: ``{axis: stat_meta.json field}`` a winning corner writes to ship the cell.
        defaults: ``{axis: shipped value}`` for each swept axis that does *not* persist — the value
            production actually ships. A corner is confirmable only when its non-persistable axes
            sit at these defaults (the confirm loop's reproducibility check).
    """

    axes: dict[str, tuple[str, ...]]
    persist: dict[str, str]
    defaults: dict[str, str]


_FAMILIES: dict[str, FamilySpec] = {
    "SkewNormal": FamilySpec(
        axes={
            "normalization": _DECODABLE_SN_NORMS,
            "dist_training_loss": ("crps", "nll"),
            "blending_loss_fn": _BLENDING,
        },
        persist={"normalization": "target_normalization", "blending_loss_fn": "blending"},
        defaults={"dist_training_loss": _SN_DEFAULT_DIST_LOSS},
    ),
    "ZINB": FamilySpec(
        axes={
            "zinb_mode": ("joint", "hurdle"),
            "count_dispersion_objective": ("crps", "pit_ks"),
            "blending_loss_fn": _BLENDING,
        },
        persist={
            "zinb_mode": "zinb_mode",
            "count_dispersion_objective": "count_dispersion_objective",
            "blending_loss_fn": "blending",
        },
        defaults={},
    ),
}

# One wide board schema across both families: a cell fills only its family's axis columns, the rest
# are blank. Kept as a fixed superset so the board CSV has a stable header regardless of which
# families were swept.
_AXIS_COLUMNS: list[str] = [
    "normalization",
    "dist_training_loss",
    "zinb_mode",
    "count_dispersion_objective",
    "blending_loss_fn",
]
_BOARD_COLUMNS: list[str] = [
    "league",
    "market",
    "family",
    *_AXIS_COLUMNS,
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
    "g6_pass",
    "central50_coverage",
    "dispersion_cal",
    "skew_cal",
    "n",
]

# Default output path for the living board — both CLI modes write here unless --out overrides.
STRATEGY_RESEARCH_BOARD: pathlib.Path = pathlib.Path(
    str(pkg_resources.files(_data_pkg) / "research" / "strategy_research_board.csv")
)


def _cell_family(league: str, market: str) -> str:
    """The registered sweep family for a cell, from its stat_meta ``dist``; loud on an unswept dist."""
    meta = load_stat_meta(pathlib.Path(str(STAT_META_PATH)))
    dist = meta.get(league, {}).get(market, {}).get("dist")
    if dist not in _FAMILIES:
        raise click.UsageError(
            f"{league} {market}: dist {dist!r} is not a swept family; known: {sorted(_FAMILIES)}"
        )
    return dist


def _dump_subdir(corner: dict[str, str]) -> str:
    """The deterministic dump subdir meditate keys a corner by: ``{target_normalization}{_hurdle}``.

    SkewNormal corners force a ``normalization`` slug; ZINB corners force none, so meditate resolves
    the count-branch target-normalization to the ``ratio_meanyr`` fallback (cli.meditate, via
    :func:`ship_config.resolve_flag_target_normalization`), and a ``hurdle`` mode appends ``_hurdle``
    (pipeline.py). This mirrors that formula exactly so scoring reads the file the trial wrote.
    """
    norm = corner.get("normalization", _DECODE_FALLBACK_STRATEGY)
    suffix = "_hurdle" if corner.get("zinb_mode") == "hurdle" else ""
    return f"{norm}{suffix}"


def _decode_strategy(corner: dict[str, str]) -> str:
    """The gate's decode strategy for a corner: the swept SN normalization, or ratio_meanyr for a count cell."""
    return corner.get("normalization", _DECODE_FALLBACK_STRATEGY)


def _dump_paths(
    league: str, market: str, corner: dict[str, str]
) -> tuple[pathlib.Path, pathlib.Path]:
    """CSV under ``test_sets/deterministic/<subdir>/`` and pickle under ``research/models/deterministic/<subdir>/``."""
    subdir = _dump_subdir(corner)
    filename = market_file_slug(league, market)
    csv = _TEST_SETS_ROOT / "deterministic" / subdir / f"{filename}.csv"
    mdl = _DETERMINISTIC_MODEL_ROOT / subdir / f"{filename}.mdl"
    return csv, mdl


def _corner_label(corner: dict[str, str]) -> str:
    """Human-scannable ``axis=value · axis=value`` for the trial's progress + log lines."""
    return " · ".join(f"{axis}={value}" for axis, value in corner.items())


def _log_path(league: str, market: str, corner: dict[str, str]) -> pathlib.Path:
    """Per-corner meditate log under ``research/logs/deterministic/<subdir>/``.

    Keyed by the full corner so two corners sharing a dump subdir (e.g. the loss axes of one
    normalization) don't overwrite each other's log — unlike the model/CSV dump, which meditate keys
    by subdir alone and each corner therefore retrains and scores before the next overwrites it.
    """
    filename = market_file_slug(league, market)
    tag = "_".join(f"{axis}={corner[axis]}" for axis in sorted(corner))
    return _DETERMINISTIC_LOG_ROOT / _dump_subdir(corner) / f"{filename}__{tag}.log"


def _log_tail(path: pathlib.Path, n: int = 25) -> str:
    return "\n".join(path.read_text().splitlines()[-n:])


def _run_deterministic_meditate(league: str, market: str, corner: dict[str, str]) -> None:
    """Train one deterministic ``(cell, corner)`` trial via meditate.

    ``--deterministic`` pins RNGs and the fixed fast hyperparameters and dumps to the research
    sandbox (never production); ``--bypass-withholding`` lets a withheld cell train. Each corner axis
    is forwarded as its ``--flag value`` (:data:`_AXIS_FLAG`) so the sweep varies it; an axis the
    family doesn't sweep is left off and meditate resolves its default. The trained model is a
    *ranking* stand-in.

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
    ]
    for axis, value in corner.items():
        cmd += [_AXIS_FLAG[axis], value]

    log_path = _log_path(league, market, corner)
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


def _score_corner(league: str, market: str, corner: dict[str, str]) -> dict[str, object]:
    """Score a trained corner's dump through the honest production gate.

    The dump's distribution is inferred from its columns (SkewNormal vs the count R/NB_P/Gate
    triple), so one gate path scores both families; ``decode_strategy`` is the swept SN
    normalization or the ratio_meanyr count fallback. ``dispersion_cal`` / ``skew_cal`` are read
    from the pickle for context only (a count cell dumps ``skew_cal`` absent → 0.0).
    """
    csv_path, mdl_path = _dump_paths(league, market, corner)
    df = load_test_set(csv_path, _SHIP_PRED_COL)
    filedict = pd.read_pickle(mdl_path)
    decode = _decode_strategy(corner)
    row = apply_thresholds(
        gate_row(
            df,
            _SHIP_PRED_COL,
            league=league,
            market=market,
            strategy=decode,
            decode_strategy=decode,
        )
    )
    return {
        **corner,
        "slack": min_gate_slack(row),
        "ships": bool(row.get("ship")),
        # All six gates (value + pass) so a corner's cost on any gate — g1 Brier non-inferiority,
        # g4 PIT-KS dispersion, g5 ECE, g6 anti-shrinkage — is visible on the board.
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
        "g6_pass": bool(row.get("g6_pass")),
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
    league: str, market: str, family: str, corner: dict[str, str]
) -> list[dict[str, object]]:
    """Train the ``corner`` retrain trial and score it; tag the row with its family.

    Calibration is not re-fit here: the dump already carries the pipeline's validation-fit
    calibration, and :func:`_score_corner` reads it off the dump via the production
    :func:`scorecard.gate_row` — no test re-fit. The dump is keyed by subdir, so scoring right after
    this trial's train — before the next sequential trial overwrites it — keeps the loss/mode axes
    honest without a per-corner dump path (which is also why the sweep always retrains rather than
    reusing a dump). Returns a one-row list so the GridSampler objective and board assembly read
    uniformly.
    """
    label = _corner_label(corner)
    click.echo(f"  training  {label} …")
    start = time.monotonic()
    _run_deterministic_meditate(league, market, corner)
    row = _score_corner(league, market, corner)
    row["family"] = family
    verdict = click.style(_verdict(row), fg="green" if row["ships"] else "red")
    elapsed = f"{time.monotonic() - start:.0f}s"
    click.echo(f"  {verdict}  {label}  slack {float(row['slack']):+.3f}  ({elapsed})")
    return [row]


def search_cell(league: str, market: str) -> pd.DataFrame:
    """Run the per-cell Optuna GridSampler study over the cell's family grid, ranked by ship slack.

    One honest row per retrain corner; the board carries each corner's slack / ship verdict / gate
    passes so the top row is the real-HPO confirm candidate. Sorted by ``slack`` descending.
    """
    family = _cell_family(league, market)
    grid = {axis: list(choices) for axis, choices in _FAMILIES[family].axes.items()}
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(grid))

    def objective(trial: optuna.Trial) -> float:
        corner = {axis: trial.suggest_categorical(axis, choices) for axis, choices in grid.items()}
        rows = _run_and_score(league, market, family, corner)
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
    ranked = board.sort_values("slack", ascending=False, ignore_index=True)
    return ranked.reindex(columns=_BOARD_COLUMNS)


def _board_cells(league: str | None = None) -> list[tuple[str, str]]:
    """Every withheld cell of a registered family in stat_meta.json, optionally one league.

    The board is self-maintaining: it follows ``shipped == "withheld"`` in stat_meta.json rather
    than a hardcoded list, so a cell shipped to devel drops out automatically.
    """
    meta = load_stat_meta(pathlib.Path(str(STAT_META_PATH)))
    return [
        (lg, mkt)
        for lg, markets in meta.items()
        if league is None or lg == league
        for mkt, cell in markets.items()
        if cell.get("dist") in _FAMILIES and cell.get("shipped") == WITHHELD
    ]


def _corner_count(cells: list[tuple[str, str]]) -> int:
    """Total deterministic trainings a board run will do — each cell's family grid size."""
    return sum(
        math.prod(len(c) for c in _FAMILIES[_cell_family(lg, mkt)].axes.values())
        for lg, mkt in cells
    )


def run_board(cells: list[tuple[str, str]], out: str | None = None) -> pd.DataFrame:
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


def _stat_meta_edit(row: object) -> str:
    """The exact stat_meta.json fields to persist a winning corner, resolved for its family.

    Reads the family's ``persist`` map so the operator doesn't have to translate board columns to
    field names: SkewNormal → ``target_normalization=…, blending=…``; ZINB → ``zinb_mode=…,
    count_dispersion_objective=…, blending=…``. A non-persistable axis (SN's dist-loss) is
    intentionally omitted — the shipped model uses the family default.
    """
    persist = _FAMILIES[row["family"]].persist
    return ", ".join(f"{field}={row[axis]}" for axis, field in persist.items())


def _repro_note(row: object) -> str:
    """Warn when the winning corner used a non-default axis it can't carry into stat_meta.

    Only SkewNormal's dist-loss is non-persistable today; a count corner's value is NaN (float), so
    the ``isinstance`` str guard keeps this quiet for families whose every axis persists.
    """
    dist_loss = row.get("dist_training_loss")
    if isinstance(dist_loss, str) and dist_loss != _SN_DEFAULT_DIST_LOSS:
        return (
            f"won under dist-loss={dist_loss}, which is not saved — "
            f"confirm under the default {_SN_DEFAULT_DIST_LOSS}"
        )
    return ""


def _print_cell_summary(board: pd.DataFrame) -> None:
    """One cell's verdict + the exact stat_meta.json edit to ship it, over a narrow per-corner table.

    The board CSV keeps all columns; the screen shows the verdict, what to change in stat_meta.json
    (for a shipping cell), and each corner's family axes / slack / blocking gate.
    """
    best = board.iloc[0]
    ships = bool(best["ships"])
    axes = list(_FAMILIES[best["family"]].axes)
    click.echo(
        click.style(f"\n{best['league']} {best['market']} — ", bold=True)
        + click.style(_verdict(best), fg="green" if ships else "red", bold=True)
        + click.style(f" (best slack {float(best['slack']):+.3f})", bold=True)
    )
    if ships:
        note = _repro_note(best)
        click.echo(
            f"  → stat_meta.json: {_stat_meta_edit(best)}" + (f"   ⚠ {note}" if note else "")
        )
    table = [
        [
            *(r[axis] for axis in axes),
            f"{float(r['slack']):+.3f}",
            "yes" if r["ships"] else "no",
            " ".join(_failed_gates(r)) or "-",
        ]
        for _, r in board.iterrows()
    ]
    click.echo(
        tabulate.tabulate(
            table, headers=[*axes, "slack", "ships", "failed gates"], tablefmt="github"
        )
    )


def _print_board_rollup(board: pd.DataFrame) -> None:
    """Board-wide tally + a scannable 'what to edit' table: one row per shipping cell naming the
    exact stat_meta.json fields to set. This is the takeaway — no cross-referencing the board CSV.
    """
    cells = list(board.groupby(["league", "market"], sort=False))
    shipping = [(lg, mkt, sub) for (lg, mkt), sub in cells if bool(sub["ships"].any())]
    click.echo(f"\n{len(cells)} cells swept · {len(shipping)} with a shipping corner")
    if not shipping:
        return
    click.echo("\nTo ship a cell, set these fields in its data/config/stat_meta.json entry:")
    rows = []
    for lg, mkt, sub in shipping:
        best = sub.sort_values("slack", ascending=False).iloc[0]
        rows.append(
            [
                f"{lg} {mkt}",
                best["family"],
                _stat_meta_edit(best),
                f"{float(best['slack']):+.3f}",
                _repro_note(best),
            ]
        )
    click.echo(
        tabulate.tabulate(
            rows,
            headers=["cell", "family", "set in stat_meta.json", "slack", "note"],
            tablefmt="github",
        )
    )


@click.command(name="model-strategy-sweep")
@click.option(
    "--league", default=None, help="League code, e.g. WNBA. Single-cell mode, or narrows --board."
)
@click.option("--market", default=None, help="Market stem, e.g. AST (single-cell mode).")
@click.option(
    "--board/--no-board",
    default=False,
    help="Sweep every withheld cell (both families) from stat_meta.json instead of one cell; --league narrows it.",
)
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="After ranking, persist each cell's best persistable corner to stat_meta.json and confirm "
    "it with a full-HPO retrain; failures auto-revert (stat_meta + pickle).",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the --confirm prompt (unattended). No effect without --confirm.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Board CSV path. Defaults to the package data dir "
    "(data/research/strategy_research_board.csv): --board overwrites it, a single cell upserts.",
)
def main(
    league: str | None, market: str | None, board: bool, confirm: bool, yes: bool, out: str | None
) -> None:
    """Operation Ship 75 strategy sweep — a per-cell GridSampler over the cell's family grid, one
    honest val-fit→test gate row per corner. ``--confirm`` then ships the winners end-to-end.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = out or str(STRATEGY_RESEARCH_BOARD)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    if board:
        cells = _board_cells(league)
        if not cells:
            raise click.UsageError(
                f"no withheld cells to sweep{f' in {league}' if league else ''}."
            )
        scope = f" ({league})" if league else ""
        click.echo(
            f"board{scope}: {len(cells)} cells · ~{_corner_count(cells)} deterministic trainings"
        )
        result = run_board(cells, out=out)
        _print_board_rollup(result)
    else:
        if not (league and market):
            raise click.UsageError("pass --league and --market, or --board")
        result = search_cell(league, market)
        _upsert_cell(result, out)
        _print_cell_summary(result)
    click.echo(f"\nboard: {out}")

    if confirm:
        from sportstradamus.training.model_strategy_confirm import run_confirm

        run_confirm(result, yes=yes)


if __name__ == "__main__":
    main()
