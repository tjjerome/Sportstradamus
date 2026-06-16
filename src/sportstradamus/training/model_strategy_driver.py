"""Operation Ship 75 search driver: an Optuna study over the retrain grid, per cell.

For each ``(league, market)`` cell the driver runs an Optuna study over the retrain axes
(normalization × dist-loss × blend-loss), training one deterministic ``meditate`` trial per grid
corner and scoring it through the *honest* production gate: the deterministic dump already carries
the pipeline's validation-fit joint calibration, and
:func:`model_strategy_search._score_normalization` runs the same :func:`scorecard.gate_row` the
production scorecard does — no test re-fit. The driver is a fixed-HP replica of the production HPO
pipeline: same calibration, same gate, same dump decode; the *only* differences are fixed
hyperparameters in place of the Optuna search and the deterministic sandbox write locations (so a
trial never clobbers a real trained market). The objective minimizes the negative ship slack, so
the study's best trial is the most-shippable corner.

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

import click
import optuna
import pandas as pd

from sportstradamus import data as _data_pkg
from sportstradamus.training import calibration
from sportstradamus.training.model_strategy_search import (
    _DECODABLE_SN_NORMS,
    _run_deterministic_meditate,
    _score_normalization,
)

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


def _run_and_score(
    league: str,
    market: str,
    *,
    normalization: str,
    dist_training_loss: str,
    blending_loss_fn: str,
) -> list[dict[str, object]]:
    """Score the ``(normalization × dist_training_loss × blending_loss_fn)`` retrain corner.

    Calibration is not re-fit here: the dump already carries the pipeline's validation-fit joint
    calibration, and :func:`model_strategy_search._score_normalization` reads it off the dump via
    the production :func:`scorecard.gate_row` — no test re-fit. The dump is keyed by normalization,
    so scoring right after this trial's train — before the next sequential trial overwrites it —
    keeps the loss/blend axes honest without a loss-keyed dump path. Returns a one-row list so the
    GridSampler objective and board assembly read uniformly.
    """
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
    return [row]


def _retrain_grid(space: dict[str, list]) -> dict[str, list]:
    """Loud on any non-categorical retrain axis — the only kind the GridSampler wires today."""
    grid: dict[str, list] = {}
    for name, (kind, spec, stage) in space.items():
        if stage != "retrain":
            continue
        if kind != "categorical":
            raise ValueError(
                f"model_strategy_driver retrain axis {name!r} has unsupported kind {kind!r}; only "
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
    """Search every cell in ``cells``; write an incremental CSV after each so an interrupt keeps
    partial progress. Returns the concatenated board.
    """
    boards: list[pd.DataFrame] = []
    for league, market in cells:
        boards.append(search_cell(league, market))
        if out is not None:
            pd.concat(boards, ignore_index=True).to_csv(out, index=False)
    return pd.concat(boards, ignore_index=True)


def _upsert_cell(cell_board: pd.DataFrame, out: str) -> pd.DataFrame:
    """Merge one cell's rows into the board CSV at ``out`` — replacing any prior rows for that
    cell — so a single-cell run refreshes the living board instead of clobbering it. Returns the
    rows written for this cell (what the CLI echoes).
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


@click.command(name="model-strategy-driver")
@click.option("--league", default=None, help="League code, e.g. WNBA (single-cell mode).")
@click.option("--market", default=None, help="Market stem, e.g. AST (single-cell mode).")
@click.option(
    "--board/--no-board",
    default=False,
    help="Search the default covered-league board instead of one cell.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Board CSV path. Defaults to the package data dir "
    "(data/research/strategy_research_board.csv): --board overwrites it, a single cell upserts.",
)
def main(league: str | None, market: str | None, board: bool, out: str | None) -> None:
    """Operation Ship 75 strategy research board — a per-cell GridSampler over the retrain grid
    (normalization × dist-loss × blend-loss), one honest val-fit→test gate row per corner.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = out or str(STRATEGY_RESEARCH_BOARD)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    if board:
        result = run_board(out=out)
    else:
        if not (league and market):
            raise click.UsageError("pass --league and --market, or --board")
        result = _upsert_cell(search_cell(league, market), out)
    click.echo(result.to_string(index=False))


if __name__ == "__main__":
    main()
