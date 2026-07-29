"""How a cell's recipe space is proposed and when the search stops.

The sweep owns training, scoring, and the board; this module owns the sampler. It turns a cell's
applicable specs into one conditional Optuna study — which family, then that family's own axes —
budgets it, decides which axis values the cell can actually support, and says when to stop early.

Nothing here trains or scores anything, and nothing here imports the sweep.
"""

from __future__ import annotations

import collections
import hashlib
import math
import pathlib
import warnings
from collections.abc import Callable

import optuna

from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.baselines import resolve_denom_col
from sportstradamus.training.model_strategy.registry import (
    CellContext,
    StrategySpec,
    get_strategy,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_STUDY_ROOT = _REPO_ROOT / "research" / "optuna"

# Per-cell trial budget. An integer cell with mean <= the count ceiling declares ~312 corners, so 48
# is ~15% of the grid; the enqueued incumbent and seed corners carry the recipes a sampler at that
# coverage cannot be relied on to find.
MAX_TRIALS_PER_CELL: int = 48
# Trials a family must receive — capped by its own corner count — before the patience stop may fire,
# so an early plateau on one family cannot end the study before the others are sampled at all.
MIN_FAMILY_TRIALS: int = 4
# Trials without a new best slack before the study stops early. Each trial is one atomic meditate,
# so there is no cheap intermediate to prune; this is early stopping, not trial pruning.
_EARLY_STOP_PATIENCE: int = 12
# A fixed sampler seed makes a resumed study reproduce the suggestion sequence its journal recorded.
_TPE_SEED: int = 20260727


def cell_axis_choices(context: CellContext, spec: StrategySpec) -> dict[str, list[str]]:
    """The spec's declared axes with values this cell cannot support pruned out.

    Pruning happens here rather than through :class:`registry.Applicability` for two reasons: a
    callable applicability hook is not JSON-serializable and would break ``canonical_signature``, and
    applicability gates a whole spec — dropping SkewNormal because one of four normalization values
    is unavailable is the wrong granularity. Pruning only narrows what the sampler may propose; every
    proposed corner is still a member of ``strategy_controls(spec)``.
    """
    choices = {axis: list(values) for axis, values in spec.axes.items()}
    normalization = choices.get("normalization", [])
    if "ratio_projvol" in normalization:
        try:
            resolve_denom_col(
                "ratio_projvol",
                context.league,
                context.market,
                context.data_columns or frozenset(),
                zero_inflated=False,
            )
        except ValueError:
            normalization.remove("ratio_projvol")
    return choices


def suggest_corner(
    trial: optuna.Trial, context: CellContext, families: tuple[str, ...]
) -> tuple[StrategySpec, dict[str, str]]:
    """Suggest one family, then that family's own axes, as a conditional search space.

    Axis parameters are namespaced ``family__axis`` because the same axis name carries different
    meaning across families — ``normalization`` exists on the two continuous families alone, and a
    shared parameter name would make TPE model the union of their value sets.
    """
    spec = get_strategy(trial.suggest_categorical("family", list(families)))
    corner = dict(spec.fixed_controls)
    for axis, choices in cell_axis_choices(context, spec).items():
        corner[axis] = trial.suggest_categorical(f"{spec.slug}__{axis}", choices)
    return spec, corner


def enqueue_params(
    context: CellContext, spec: StrategySpec, corner: dict[str, str]
) -> dict[str, str] | None:
    """``corner`` as trial parameters, or ``None`` if this cell prunes one of its values."""
    choices = cell_axis_choices(context, spec)
    if any(corner.get(axis) not in values for axis, values in choices.items()):
        return None
    return {"family": spec.slug, **{f"{spec.slug}__{axis}": corner[axis] for axis in choices}}


def _grid_size(choices: dict[str, list[str]]) -> int:
    """Corners an axis map enumerates; an axis-free spec contributes its single fixed corner."""
    return math.prod((len(values) for values in choices.values()), start=1)


def reachable_corners(context: CellContext, families: tuple[str, ...]) -> int:
    """Corners the cell can actually reach — the declared grid less every pruned axis value."""
    return sum(_grid_size(cell_axis_choices(context, get_strategy(slug))) for slug in families)


# Set on each trial by the sweep's objective: False when the corner came from the evaluated-corner
# cache rather than a retrain. TPE re-proposes a seen corner often enough to matter (roughly a
# quarter of trials on an observed run), and such a trial is neither coverage nor progress.
TRAINED_TRIAL_ATTR = "trained"


def early_stop_callback(
    context: CellContext, families: tuple[str, ...]
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """An ``optimize`` callback that stops the study once it has both covered and plateaued.

    The coverage floor is per family and capped by that family's own grid, so a family with fewer
    corners than :data:`MIN_FAMILY_TRIALS` cannot hold the study open forever. Cache-served trials
    are skipped outright: counting them would let duplicate churn age out the patience window and
    end the search while real corners were still improving it.
    """
    floors = {
        slug: min(MIN_FAMILY_TRIALS, _grid_size(cell_axis_choices(context, get_strategy(slug))))
        for slug in families
    }
    seen: collections.Counter = collections.Counter()
    state = {"best": math.inf, "stale": 0}

    def stop_when_stale(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if not trial.user_attrs.get(TRAINED_TRIAL_ATTR, True):
            return
        seen[trial.params.get("family")] += 1
        if trial.value is not None and trial.value < state["best"]:
            state.update(best=trial.value, stale=0)
        else:
            state["stale"] += 1
        covered = all(seen[family] >= floor for family, floor in floors.items())
        if covered and state["stale"] >= _EARLY_STOP_PATIENCE:
            study.stop()

    return stop_when_stale


def cell_study(league: str, market: str, families: tuple[str, ...]) -> optuna.Study:
    """One resumable conditional-TPE study per cell and family pool, journalled under ``research/``.

    ``multivariate=True, group=True`` is the only Optuna 3.5 configuration that models a conditional
    space — with either off the sampler falls back to independent per-parameter sampling and cannot
    learn that a normalization value only means something under a continuous family. The journal
    lives under gitignored ``research/``, so resume state is machine-local by design.

    The pool is part of the study key because it *is* the ``family`` arm set, and Optuna fixes a
    categorical's choices at its first suggestion: resuming a journal written under a different pool
    raises ``CategoricalDistribution does not support dynamic value space`` and kills the run. That is
    not hypothetical — dropping Mixture invalidated every journal on disk, and a ``--dist-class``
    run would collide with a full-pool journal the same way. Keying on the pool starts a fresh study
    instead, and lets narrowed runs keep their own resumable state alongside the full one.

    All three of those APIs are marked experimental, and each announces itself once per cell. The
    warnings carry nothing actionable — the choices are deliberate, and the ``optuna >=3.5,<3.6``
    pin is what stops "the interface can change in the future" from reaching us — so they are
    silenced here rather than left to bury a real failure in a 30-cell run's output.
    """
    _STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    pool = hashlib.sha256("|".join(families).encode()).hexdigest()[:8]
    path = _STUDY_ROOT / f"{market_file_slug(league, market)}.{pool}.log"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return optuna.create_study(
            study_name=f"{league}__{market}__{pool}",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=_TPE_SEED),
            storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(path))),
            load_if_exists=True,
        )
