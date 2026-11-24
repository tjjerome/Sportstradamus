"""Refit-carry: serve a cell's booster(s) refit through its newest matrix row.

A cell's served booster fits the oldest 70 % of its matrix rows by date: the validation half
of the rest fits every calibration layer and the test half reads the six ship gates. A cell
whose stat_meta sets ``refit: "all"`` adds one step once that run is done. Its served
booster(s) refit on every matrix row at the selected hyperparameters and round count, with no
new early stopping and no new search, and every calibration layer carries over unchanged.

Only the pickle serves the refit. The gates, the ``model_stats`` gate columns, the test-set
dump and the SHAP pass keep describing the 70 % sibling: a refit's prediction on a held-out
row is in-sample, so it must never reach a dump, a calibration fit or a gate. The pickle
records the refit in four keys: ``refit_through`` (the newest game date the refit trained on),
``refit_rows``, and ``refit_rounds`` and ``sibling_booster_sha256``, both keyed by booster
(``"model"``, plus ``"mean_anchor"`` on a plain-anchored cell). The knob sits outside the
strategy identity, as ``book_pull`` does, so a refit pickle signs the corner its sibling signs.
"""

import hashlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbmlss.model import LightGBMLSS

from sportstradamus.helpers.config import ALL_ROWS_REFIT
from sportstradamus.helpers.config import refit as cell_refit
from sportstradamus.training import baselines
from sportstradamus.training.plain_baseline import _PARAMS, _SEED
from sportstradamus.training.structural_strategies import AFFINE_EXPERT_EXPERIMENTS


def resolve_refit(
    league: str,
    market: str,
    override: str | None,
    *,
    use_hurdle: bool,
    structural_strategy: str,
    holdout_blind: bool,
) -> bool:
    """Whether this run refits the cell's served booster(s) on every matrix row.

    Args:
        league: League slug.
        market: Market name.
        override: ``train_market``'s ``refit`` keyword; ``None`` reads the cell's stat_meta knob.
        use_hurdle: The cell trains the two-booster ``HurdleZINB``.
        structural_strategy: The run's structural calibration method.
        holdout_blind: The run is a holdout-blind search.

    Raises:
        ValueError: The knob is on for a run whose refit could not serve as its sibling does.
    """
    if (override or cell_refit(league, market)) != ALL_ROWS_REFIT:
        return False
    refusals = {
        "a hurdle cell's two boosters have no refit path": use_hurdle,
        "an affine-expert strategy's per-position experts are not refit": (
            structural_strategy in AFFINE_EXPERT_EXPERIMENTS
        ),
        "a holdout-blind run's held-out rows must never reach a fit": holdout_blind,
    }
    reasons = [reason for reason, refused in refusals.items() if refused]
    if reasons:
        raise ValueError(f"{league} {market}: refit 'all' refused: {'; '.join(reasons)}")
    return True


def lss_refit_rows(splits: dict, dist_info: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """The matrix rows the distributional model refits on, and its training target.

    A count family fits raw outcomes. A continuous family applies the sibling's own rule,
    :func:`~sportstradamus.training.baselines.continuous_fit_target`, so a zero-gated cell
    refits on its positive rows only, at the sibling's ``global_mean`` and ``denom_col``: the
    pickle's ``offset_meta`` then decodes the refit too. Rows keep the matrix's order, which
    LightGBM's row order follows.

    Args:
        splits: ``train_market``'s splits; reads ``X`` and ``y``, every matrix row.
        dist_info: :func:`~sportstradamus.training.pipeline._step_select_distribution`'s output.
    """
    labels = np.ravel(splits["y"].to_numpy())
    strategy = dist_info["target_normalization"]
    if strategy is None:
        return splits["X"], labels
    return baselines.continuous_fit_target(
        splits["X"],
        labels,
        strategy,
        hist_gate=dist_info["hist_gate"],
        global_mean=dist_info["global_mean"],
        denom_col=dist_info["denom_col"],
    )


def finish_refit(
    splits: dict,
    dates: pd.Series,
    *,
    sibling: LightGBMLSS,
    refit_model: LightGBMLSS,
    anchor: lgb.Booster | None,
    determinism: dict[str, bool | int],
) -> tuple[lgb.Booster | None, dict]:
    """Refit the plain mean anchor on every row, and describe the refit for the pickle.

    The anchor refits as ``training/plain_baseline.py`` fits it (its parameters and seed, on
    the same feature columns and categorical casts) on every matrix row's outcome, for the
    sibling anchor's early-stopped round count with no new early stopping. Each
    ``sibling_booster_sha256`` entry hashes the sibling booster's ``model_to_string()``.

    Args:
        splits: ``train_market``'s splits; reads ``X`` and ``y``, every matrix row.
        dates: Every matrix row's game date, indexed like the matrix.
        sibling: The distributional model fit on the oldest 70 % of rows.
        refit_model: The same recipe refit on :func:`lss_refit_rows`.
        anchor: The sibling's plain booster on a plain-anchored cell, else ``None``.
        determinism: LightGBM's bit-reproducibility params on a ``--deterministic`` run, else
            empty.

    Returns:
        The refit anchor (``None`` on an unanchored cell), and the pickle's ``refit_through``,
        ``refit_rows``, ``refit_rounds`` and ``sibling_booster_sha256``.
    """
    boosters = {"model": (sibling.booster, refit_model.booster)}
    refit_anchor = None
    if anchor is not None:
        refit_anchor = (
            lgb.LGBMRegressor(
                **_PARAMS,
                **determinism,
                n_estimators=anchor.best_iteration,
                random_state=_SEED,
                verbose=-1,
            )
            .fit(splits["X"], splits["y"]["Result"])
            .booster_
        )
        boosters["mean_anchor"] = (anchor, refit_anchor)
    record = {
        "refit_through": dates.loc[splits["X"].index].max().date().isoformat(),
        "refit_rows": len(splits["X"]),
        "refit_rounds": {
            key: refit_booster.current_iteration() for key, (_, refit_booster) in boosters.items()
        },
        "sibling_booster_sha256": {
            key: hashlib.sha256(sibling_booster.model_to_string().encode()).hexdigest()
            for key, (sibling_booster, _) in boosters.items()
        },
    }
    return refit_anchor, record
