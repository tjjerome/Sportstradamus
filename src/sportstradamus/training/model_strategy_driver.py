"""Operation Ship 75 search driver: an Optuna study over the search grid, per cell.

For each ``(league, market)`` cell the driver runs an Optuna study over the *retrain* axes —
training one deterministic ``meditate`` trial per grid corner — then fans each trained corner out
across the *free post-hoc* axes (the four calibration modes, swept off the dump with no retrain).
Every board row is scored through the honest val-fit→test gate (the per-corner primitives live in
:mod:`sportstradamus.training.model_strategy_search`). The objective minimizes the negative best
ship slack over a corner's post-hoc fan-out, so the study's best trial is the most-shippable corner.

The retrain grid is enumerable and categorical today, so the sampler is
:class:`optuna.samplers.GridSampler` — exhaustive and deterministic, the right tool for a discrete
space (no TPE guessing over six corners). The ``[kind, spec, stage]`` :data:`SEARCH_SPACE` mirrors
``pipeline.py``'s ``hp_search_space`` (with a ``stage`` tag splitting retrain from post-hoc), so a
continuous retrain axis (a blend weight, say) is a one-line addition that flips the sampler to TPE.

Research scaffolding: the deterministic trials *rank* only — nothing ships off them. The top corner
is the real-HPO confirm candidate; a clean 5/5 on the official (full-HPO) scorecard is what ships.
"""

import math
import pathlib

import click
import optuna
import pandas as pd

from sportstradamus.training import calibration
from sportstradamus.training.model_strategy_search import (
    _DECODABLE_SN_NORMS,
    _run_deterministic_meditate,
    score_calibration_modes,
)
from sportstradamus.training.scorecard import CALIBRATION_MODES

# Search axes in the ``[kind, spec, stage]`` shape — ``[kind, spec]`` mirrors pipeline's
# hp_search_space, ``stage`` splits the retrain grid (GridSampler exhaustive) from the free
# post-hoc sweep (scored off each trained dump, no retrain). A continuous retrain axis would
# read ``["float", {"low": ...}, "retrain"]`` and require a ``suggest_float`` branch in
# :func:`_retrain_grid` plus a TPE sampler.
SEARCH_SPACE: dict[str, list] = {
    "normalization": ["categorical", list(_DECODABLE_SN_NORMS), "retrain"],
    "dist_training_loss": ["categorical", ["crps", "nll"], "retrain"],
    "blending_loss_fn": ["categorical", sorted(calibration.BLENDING_SLUGS), "posthoc"],
    "calibration": ["categorical", list(CALIBRATION_MODES), "posthoc"],
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
    "calibration",
    "slack",
    "ships",
    "dispersion_cal",
    "skew_cal",
    "g4_pit_ks",
    "g4_pit_ks_max",
    "n",
]


def _run_and_score(
    league: str,
    market: str,
    *,
    normalization: str,
    dist_training_loss: str,
    blending_choices: tuple[str, ...],
) -> list[dict[str, object]]:
    """Train one deterministic retrain corner, then sweep the free post-hoc axes off its dump.

    The retrain axes (``normalization``, ``dist_training_loss``) train one ``meditate`` trial; the
    calibration axis is then swept for free (four rows per blend objective, mean held fixed) and the
    blending axis loops its objectives. Only the ``nll`` blend objective ships today — a non-nll
    objective fails loud *before* any meditate runs, until ``calibration.fit_blend_weight`` grows a
    crps post-hoc refit (docs/operation_ship_75.md §8 crps-blending gate). The dump is keyed by
    normalization only, so scoring right after this trial's train — before the next sequential trial
    overwrites it — keeps the loss axis honest without a loss-keyed dump path.
    """
    unbuilt = [b for b in blending_choices if b != "nll"]
    if unbuilt:
        raise NotImplementedError(
            f"blending_loss_fn {unbuilt} need a post-hoc calibration.fit_blend_weight refit; only "
            "'nll' blending ships today (docs/operation_ship_75.md §8 crps-blending gate)."
        )
    _run_deterministic_meditate(
        league, market, normalization, dist_training_loss=dist_training_loss
    )
    rows: list[dict[str, object]] = []
    for blending in blending_choices:
        for row in score_calibration_modes(league, market, normalization):
            row["dist_training_loss"] = dist_training_loss
            row["blending_loss_fn"] = blending
            rows.append(row)
    return rows


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

    One trial per retrain corner, each fanned out over the free post-hoc sweep; the board carries
    every (corner × calibration-mode) row's slack / ship verdict / Gate-4 PIT-KS so the top row is
    the real-HPO confirm candidate. Sorted by ``slack`` descending.
    """
    grid = _retrain_grid(space)
    blending_choices = tuple(space["blending_loss_fn"][1])
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(grid))

    def objective(trial: optuna.Trial) -> float:
        params = {name: trial.suggest_categorical(name, choices) for name, choices in grid.items()}
        rows = _run_and_score(
            league,
            market,
            normalization=params["normalization"],
            dist_training_loss=params["dist_training_loss"],
            blending_choices=blending_choices,
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
    help="Write the board CSV here (incremental in --board mode).",
)
def main(league: str | None, market: str | None, board: bool, out: str | None) -> None:
    """Operation Ship 75 search driver — GridSampler over the retrain grid × free post-hoc sweep."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if board:
        result = run_board(out=out)
    else:
        if not (league and market):
            raise click.UsageError("pass --league and --market, or --board")
        result = search_cell(league, market)
        if out is not None:
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out, index=False)
    click.echo(result.to_string(index=False))


if __name__ == "__main__":
    main()
