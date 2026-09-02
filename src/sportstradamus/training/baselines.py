"""Target/baseline strategies for the SkewNormal training branch.

Single source of truth for the forward target transform applied during
training and the inverse decode applied at prediction time. The training
pipeline (``training/pipeline.py``) and the live prediction path
(``prediction/model_prob.py``) both call into this module, so the two cannot
drift between train and predict.

Each strategy bundles:

* ``forward(y, X, global_mean, denom_col)`` — transforms raw labels into the
  target the GBDT learns.
* ``decode_loc(loc, X, global_mean, denom_col)`` — converts the model's raw
  ``loc`` output back into an absolute EV contribution.
* ``decode_scale(scale, X, denom_col)`` — converts the model's raw ``scale``
  output into an absolute sigma.
* ``start_mode_flag`` — ``"normalized"`` / ``"offset"`` / ``"raw"``; threaded
  to ``set_model_start_values`` so its SkewNormal seeding matches the target
  space.
* ``offset_meta(global_mean, denom_col)`` — dict persisted to the model
  pickle, allowing the prediction-side mirror to recompute the baseline
  using the same ``global_mean`` snapshot the model was trained against.

Registered strategies (more queued as A/B follow-ups in
``docs/gbdt_mean_regression_plan.md``):

* ``ratio_meanyr`` — current production. ``y / MeanYr_clipped`` ratio
  target; decode multiplies ``loc`` and ``scale`` by ``MeanYr``.
* ``centered_additive_eb_meanyr_k10`` — Phase-A. Trains on
  ``y − EB(MeanYr, GamesPlayed, K=10)`` and adds the EB prior back to
  ``loc`` only at decode; ``scale`` left in absolute residual units. Targets
  the multiplicative-amplification artifact (corr(pred_loc, MeanYr) ≈ −0.6
  path-wide) identified in ``docs/OVERCONFIDENCE_INVESTIGATION.md``.
  Path-wide A/B (PR #46) showed FGA SHIP, every other SkewNormal market
  KILL — the long-horizon baseline only fits markets where production is
  structural (e.g. shot volume).
* ``centered_additive_mean10`` — same shape, swap the baseline to a
  trailing-10-game per-player mean. More responsive to current form; the
  hypothesis is that markets where production tracks recent role/rotation
  (PTS, REB, AST, PR, …) benefit from a faster-moving prior than MeanYr.
  Falls back to ``MeanYr`` / ``MeanYr_nonzero`` for sparse-history players.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportstradamus.helpers.distributions import (
    NONZERO_DENOM_GATE,
    DecodedParams,
    decode_predictive_mean,
)

# Empirical-Bayes shrinkage strength: pseudo-count of "global-mean games"
# mixed into each player's trailing per-player mean. Validated on the NBA
# FGA holdout (top-volume-quintile bias -2.0 -> -0.48 at K=10) in the
# overconfidence investigation; retune only after the baseline-source A/Bs
# settle.
EB_SHRINKAGE_K: float = 10.0

# Floor applied to ``MeanYr`` (or ``MeanYr_nonzero``) before division. Mirrors
# the legacy clip at pipeline.py:466; centralized here so train and predict
# never disagree.
_MEANYR_FLOOR: float = 0.5

# Floor applied to the ratio target post-divide, mirroring pipeline.py:468.
_RATIO_TARGET_FLOOR: float = 0.01

# Fallback denominator for ``ratio_projvol`` when a volume projection is missing:
# degrades cleanly to ``ratio_meanyr`` semantics and is present in every
# train, serve, and gate frame.
_PROJVOL_FALLBACK_COL: str = "MeanYr"

# ``ratio_projvol`` denominator routing: scoring market -> the volume market
# whose projected per-game total (the ``Player proj {vol} mean`` feature) is the
# opportunity denominator. NBA/WNBA carry a single volume stat (MIN), so any
# counting market routes to it via the per-league default below; NFL is
# role-specific. MLB/NHL fold in by adding rows here once their cells activate
# (volume_stats: MLB pitches thrown, NHL timeOnIce / shotsAgainst — MLB
# plateAppearances is now a structural projection, not a trained cell) — the
# resolver and the ``Player proj {vol} mean`` column builder are league-agnostic,
# so a fold-in is a data edit, not a logic change.
_PROJVOL_VOLUME_MARKET: dict[str, dict[str, str]] = {
    "NFL": {
        "completions": "attempts",
        "passing yards": "attempts",
        "passing tds": "attempts",
        "rushing yards": "carries",
        "receiving yards": "targets",
        "receptions": "targets",
    },
}
_PROJVOL_DEFAULT_VOLUME_MARKET: dict[str, str] = {"NBA": "MIN", "WNBA": "MIN"}


def compute_eb_prior(
    player_mean: np.ndarray | pd.Series,
    games_played: np.ndarray | pd.Series,
    global_mean: float,
    k: float = EB_SHRINKAGE_K,
) -> np.ndarray:
    """Empirical-Bayes shrink a per-player trailing mean toward the global mean.

    ``prior = (games * player_mean + k * global_mean) / (games + k)``

    Low-sample players shrink toward the league mean; high-sample players
    keep their season-to-date trailing mean. ``k`` is a pseudo-count of
    "global-mean games" mixed in.

    Args:
        player_mean: Per-player trailing mean for the target market.
        games_played: Count of prior games observed per player. Negative
            values are clipped to zero.
        global_mean: Marginal training mean for the market.
        k: Shrinkage strength. Defaults to :data:`EB_SHRINKAGE_K`.

    Returns:
        Per-row EB prior, same length as ``player_mean``.
    """
    pm = np.asarray(player_mean, dtype=float)
    g = np.clip(np.asarray(games_played, dtype=float), 0.0, None)
    return (g * pm + k * float(global_mean)) / (g + k)


@dataclass(frozen=True)
class TargetNormalization:
    """Bundle of forward/inverse transforms for one target/baseline choice."""

    slug: str
    start_mode_flag: str
    _forward: Callable[[np.ndarray, pd.DataFrame, float, str], np.ndarray]
    _decode_loc: Callable[[np.ndarray, pd.DataFrame, float, str], np.ndarray]
    _decode_scale: Callable[[np.ndarray, pd.DataFrame, str], np.ndarray]
    _encode_loc: Callable[[np.ndarray, pd.DataFrame, float, str], np.ndarray]
    _encode_scale: Callable[[np.ndarray, pd.DataFrame, str], np.ndarray]
    _offset_meta: Callable[[float, str], dict | None]

    def forward(
        self,
        y: np.ndarray,
        X: pd.DataFrame,
        global_mean: float,
        denom_col: str = "MeanYr",
    ) -> np.ndarray:
        """Transform raw labels into the target the GBDT learns."""
        return self._forward(y, X, global_mean, denom_col)

    def decode_loc(
        self,
        loc: np.ndarray,
        X: pd.DataFrame,
        global_mean: float,
        denom_col: str = "MeanYr",
    ) -> np.ndarray:
        """Convert raw ``loc`` into the absolute EV contribution."""
        return self._decode_loc(loc, X, global_mean, denom_col)

    def decode_scale(
        self,
        scale: np.ndarray,
        X: pd.DataFrame,
        denom_col: str = "MeanYr",
    ) -> np.ndarray:
        """Convert raw ``scale`` into an absolute sigma."""
        return self._decode_scale(scale, X, denom_col)

    def encode_loc(
        self,
        loc: np.ndarray,
        X: pd.DataFrame,
        global_mean: float,
        denom_col: str = "MeanYr",
    ) -> np.ndarray:
        """Map an absolute EV ``loc`` back into the model's normalized space.

        Exact inverse of :meth:`decode_loc`. Used by the Gate-4 served-predictive
        dump to store blended params so the scorecard's decode recovers them.
        """
        return self._encode_loc(loc, X, global_mean, denom_col)

    def encode_scale(
        self,
        scale: np.ndarray,
        X: pd.DataFrame,
        denom_col: str = "MeanYr",
    ) -> np.ndarray:
        """Map an absolute sigma back into the model's normalized space.

        Exact inverse of :meth:`decode_scale`.
        """
        return self._encode_scale(scale, X, denom_col)

    def offset_meta(
        self,
        global_mean: float,
        denom_col: str = "MeanYr",
    ) -> dict | None:
        """Pickle-persisted metadata for the prediction-side baseline mirror."""
        return self._offset_meta(global_mean, denom_col)


def _ratio_forward(
    y: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    meanyr = X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy()
    return np.clip(np.asarray(y, dtype=float) / meanyr, _RATIO_TARGET_FLOOR, None)


def _ratio_decode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    meanyr = X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy()
    return np.asarray(loc, dtype=float) * meanyr


def _ratio_decode_scale(scale: np.ndarray, X: pd.DataFrame, denom_col: str) -> np.ndarray:
    meanyr = X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy()
    return np.asarray(scale, dtype=float) * meanyr


def _ratio_encode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    meanyr = X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy()
    return np.asarray(loc, dtype=float) / meanyr


def _ratio_encode_scale(scale: np.ndarray, X: pd.DataFrame, denom_col: str) -> np.ndarray:
    meanyr = X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy()
    return np.asarray(scale, dtype=float) / meanyr


def _ratio_offset_meta(global_mean: float, denom_col: str) -> dict:
    return {"method": "ratio", "denom_col": denom_col}


def _centered_eb_forward(
    y: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    eb = compute_eb_prior(
        X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy(),
        X["GamesPlayed"].clip(lower=0).to_numpy(),
        global_mean,
        EB_SHRINKAGE_K,
    )
    return np.asarray(y, dtype=float) - eb


def _centered_eb_decode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    eb = compute_eb_prior(
        X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy(),
        X["GamesPlayed"].clip(lower=0).to_numpy(),
        global_mean,
        EB_SHRINKAGE_K,
    )
    return eb + np.asarray(loc, dtype=float)


def _centered_eb_decode_scale(scale: np.ndarray, X: pd.DataFrame, denom_col: str) -> np.ndarray:
    return np.asarray(scale, dtype=float)


def _centered_eb_encode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    eb = compute_eb_prior(
        X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy(),
        X["GamesPlayed"].clip(lower=0).to_numpy(),
        global_mean,
        EB_SHRINKAGE_K,
    )
    return np.asarray(loc, dtype=float) - eb


def _centered_eb_offset_meta(global_mean: float, denom_col: str) -> dict:
    return {
        "method": "eb_additive",
        "k": EB_SHRINKAGE_K,
        "global_mean": float(global_mean),
        "prior_col": denom_col,
        "games_col": "GamesPlayed",
        "denom_col": denom_col,
    }


def _mean10_baseline(X: pd.DataFrame, denom_col: str) -> np.ndarray:
    """Trailing-10-game per-player mean, with denom_col fallback for sparse rows.

    Mean10 captures recent form; it falls back to ``denom_col`` (typically
    ``MeanYr`` / ``MeanYr_nonzero``) when missing so brand-new players with
    no rolling-10 history don't blow up the residual. Floor mirrors the
    ratio strategy's ``MeanYr`` floor.
    """
    return X["Mean10"].fillna(X[denom_col]).clip(lower=_MEANYR_FLOOR).to_numpy()


def _centered_mean10_forward(
    y: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    return np.asarray(y, dtype=float) - _mean10_baseline(X, denom_col)


def _centered_mean10_decode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    return _mean10_baseline(X, denom_col) + np.asarray(loc, dtype=float)


def _centered_mean10_decode_scale(scale: np.ndarray, X: pd.DataFrame, denom_col: str) -> np.ndarray:
    return np.asarray(scale, dtype=float)


def _centered_mean10_encode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    return np.asarray(loc, dtype=float) - _mean10_baseline(X, denom_col)


def _centered_mean10_offset_meta(global_mean: float, denom_col: str) -> dict:
    return {
        "method": "mean10_additive",
        "global_mean": float(global_mean),
        "prior_col": "Mean10",
        "prior_fallback_col": denom_col,
        "denom_col": denom_col,
    }


def _projvol_denominator(X: pd.DataFrame, proj_col: str) -> np.ndarray:
    """Projected-volume denominator with a per-row season-mean fallback.

    Missing projections arrive as ``0`` (the ``fillna(0)`` sentinel in
    ``stats/base.py``) or ``NaN`` (absent player join); both fall back to
    :data:`_PROJVOL_FALLBACK_COL` so the rate target is always finite and
    the decode stays exact.
    """
    proj = X[proj_col].replace(0, np.nan)
    return proj.fillna(X[_PROJVOL_FALLBACK_COL]).clip(lower=_MEANYR_FLOOR).to_numpy()


def _ratio_projvol_forward(
    y: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    denom = _projvol_denominator(X, denom_col)
    return np.clip(np.asarray(y, dtype=float) / denom, _RATIO_TARGET_FLOOR, None)


def _ratio_projvol_decode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    return np.asarray(loc, dtype=float) * _projvol_denominator(X, denom_col)


def _ratio_projvol_decode_scale(scale: np.ndarray, X: pd.DataFrame, denom_col: str) -> np.ndarray:
    return np.asarray(scale, dtype=float) * _projvol_denominator(X, denom_col)


def _ratio_projvol_encode_loc(
    loc: np.ndarray, X: pd.DataFrame, global_mean: float, denom_col: str
) -> np.ndarray:
    return np.asarray(loc, dtype=float) / _projvol_denominator(X, denom_col)


def _ratio_projvol_encode_scale(scale: np.ndarray, X: pd.DataFrame, denom_col: str) -> np.ndarray:
    return np.asarray(scale, dtype=float) / _projvol_denominator(X, denom_col)


def _ratio_projvol_offset_meta(global_mean: float, denom_col: str) -> dict:
    return {
        "method": "projvol_ratio",
        "denom_col": denom_col,
        "fallback_col": _PROJVOL_FALLBACK_COL,
    }


_TARGET_NORMALIZATIONS: dict[str, TargetNormalization] = {
    "ratio_meanyr": TargetNormalization(
        slug="ratio_meanyr",
        start_mode_flag="normalized",
        _forward=_ratio_forward,
        _decode_loc=_ratio_decode_loc,
        _decode_scale=_ratio_decode_scale,
        _encode_loc=_ratio_encode_loc,
        _encode_scale=_ratio_encode_scale,
        _offset_meta=_ratio_offset_meta,
    ),
    "centered_additive_eb_meanyr_k10": TargetNormalization(
        slug="centered_additive_eb_meanyr_k10",
        start_mode_flag="offset",
        _forward=_centered_eb_forward,
        _decode_loc=_centered_eb_decode_loc,
        _decode_scale=_centered_eb_decode_scale,
        _encode_loc=_centered_eb_encode_loc,
        _encode_scale=_centered_eb_decode_scale,
        _offset_meta=_centered_eb_offset_meta,
    ),
    "centered_additive_mean10": TargetNormalization(
        slug="centered_additive_mean10",
        start_mode_flag="offset",
        _forward=_centered_mean10_forward,
        _decode_loc=_centered_mean10_decode_loc,
        _decode_scale=_centered_mean10_decode_scale,
        _encode_loc=_centered_mean10_encode_loc,
        _encode_scale=_centered_mean10_decode_scale,
        _offset_meta=_centered_mean10_offset_meta,
    ),
    "ratio_projvol": TargetNormalization(
        slug="ratio_projvol",
        start_mode_flag="normalized",
        _forward=_ratio_projvol_forward,
        _decode_loc=_ratio_projvol_decode_loc,
        _decode_scale=_ratio_projvol_decode_scale,
        _encode_loc=_ratio_projvol_encode_loc,
        _encode_scale=_ratio_projvol_encode_scale,
        _offset_meta=_ratio_projvol_offset_meta,
    ),
}


# Public list of slugs accepted by the ``meditate --target-normalization`` flag.
TARGET_NORMALIZATION_SLUGS: tuple[str, ...] = tuple(_TARGET_NORMALIZATIONS.keys())


def get_target_normalization(slug: str) -> TargetNormalization:
    """Resolve a target-strategy slug to a :class:`TargetNormalization`.

    Args:
        slug: One of :data:`TARGET_NORMALIZATION_SLUGS`.

    Returns:
        The registered strategy.

    Raises:
        ValueError: If ``slug`` is not a registered strategy.
    """
    if slug not in _TARGET_NORMALIZATIONS:
        raise ValueError(f"Unknown target strategy {slug!r}; valid: {TARGET_NORMALIZATION_SLUGS}")
    return _TARGET_NORMALIZATIONS[slug]


def resolve_denom_col(
    slug: str,
    league: str,
    market: str,
    columns,
    *,
    zero_inflated: bool,
) -> str:
    """Resolve the denominator column a SkewNormal cell normalizes against.

    ``ratio_projvol`` divides by a market-specific projected-volume feature
    (``Player proj {vol} mean``); every other strategy uses the season mean
    (``MeanYr_nonzero`` for zero-inflated cells, else ``MeanYr``). The chosen
    column flows through ``pipeline._step_select_distribution`` to the model
    pickle's ``offset_meta``, the served-predictive ``DenomCol`` dump, and the
    Gate-4 scorecard decode, so train, serve, and gate share one denominator.

    Args:
        slug: Target-normalization slug.
        league: League slug (e.g. ``"NFL"``).
        market: Market slug; hyphens are normalized to spaces before lookup.
        columns: Available feature columns (anything supporting ``in``).
        zero_inflated: Whether the cell drops zero rows / serves against the
            non-zero season mean.

    Returns:
        The denominator column name.

    Raises:
        ValueError: If ``ratio_projvol`` has no volume mapping for the cell, or
            the projected-volume column is absent (a volume market's own matrix
            or an un-regenerated cache) — the cell is not eligible.
    """
    if slug == "ratio_projvol":
        market_key = market.replace("-", " ")
        vol = _PROJVOL_VOLUME_MARKET.get(league, {}).get(
            market_key
        ) or _PROJVOL_DEFAULT_VOLUME_MARKET.get(league)
        if vol is None:
            raise ValueError(f"ratio_projvol: no projected-volume mapping for {league}/{market}")
        col = f"Player proj {vol} mean"
        if col not in columns:
            raise ValueError(
                f"ratio_projvol: projected-volume column {col!r} absent for {league}/{market} "
                "— cell not eligible (volume market or un-regenerated training cache)"
            )
        return col
    if zero_inflated and "MeanYr_nonzero" in columns:
        return "MeanYr_nonzero"
    return "MeanYr"


def serve_denom_col(offset_meta: dict | None, hist_gate: float, columns) -> str:
    """SkewNormal decode denominator, preferring the value training persisted.

    The serve-side companion to :func:`resolve_denom_col`, which makes the same
    choice at train time from the cell's zero-rate. The denom (``MeanYr`` vs
    ``MeanYr_nonzero``) is a property of the trained model, so reading it back off
    the pickle keeps serve in lockstep even when the runtime ``stat_zi``
    (gitignored, per-box) is stale or absent. Falls back to the legacy
    ``hist_gate`` recompute for pre-persist pickles; the ``prior_*`` keys cover
    already-shipped centered pickles without a retrain (mean10 stores the denom as
    ``prior_fallback_col``, EB as ``prior_col``).
    """
    om = offset_meta or {}
    persisted = om.get("denom_col") or om.get("prior_fallback_col")
    if persisted is None and om.get("method") == "eb_additive":
        persisted = om.get("prior_col")
    if persisted is not None:
        return persisted
    return (
        "MeanYr_nonzero"
        if (hist_gate > NONZERO_DENOM_GATE and "MeanYr_nonzero" in columns)
        else "MeanYr"
    )


def serve_offset_mode(dist: str, target_normalization: str) -> bool:
    """Whether serve seeds the SkewNormal with the centered-residual offset.

    Mirrors training's ``offset_mode = strategy.start_mode_flag == "offset"``
    (``pipeline.train_market``) so a model trained against any offset strategy seeds
    the same way at predict time. Derived from the registry — not a hardcoded
    ``offset_meta["method"]`` string — so a new offset strategy needs no serve-side
    edit. Only the SkewNormal seed consumes ``offset_mode``; the ``dist`` guard
    short-circuits the registry lookup off that path and for non-registry legacy
    slugs.
    """
    return (
        dist == "SkewNormal"
        and get_target_normalization(target_normalization).start_mode_flag == "offset"
    )


def decode_model_params(
    prob_params: pd.DataFrame,
    dist: str,
    X: pd.DataFrame,
    *,
    hist_gate: float,
    offset_meta: dict | None,
    target_normalization: str,
) -> DecodedParams:
    """Decode raw LightGBMLSS output into an absolute mean plus family shape.

    Wraps :func:`~sportstradamus.helpers.distributions.decode_predictive_mean` with
    the target-normalization inverse the count families don't need: only SkewNormal
    trains in a transformed space, so only it routes ``loc``/``scale`` back through
    the registry. Every serving consumer — the market path in
    ``prediction.model_prob`` and the volume-projection path in ``stats.base`` —
    decodes here, so a cell cannot mean one thing scored as a market and another
    scored as a volume dependency.

    Args:
        prob_params: ``predict(pred_type="parameters")`` frame. SkewNormal needs
            ``loc``/``scale``/``alpha``; the count families read their own columns
            and their ``gate`` where zero-inflated.
        dist: Distribution family slug from the model pickle.
        X: The feature frame the params were predicted from, row-aligned with
            ``prob_params``; supplies the decode denominator.
        hist_gate: Empirical zero-rate persisted with the model.
        offset_meta: Pickle-persisted baseline metadata; ``None`` for legacy or
            ratio-strategy models.
        target_normalization: Slug of the baseline strategy the model trained
            against.

    Returns:
        DecodedParams with the absolute ``ev`` plus the family's shape fields.
    """
    if dist != "SkewNormal":
        return decode_predictive_mean(prob_params, dist)

    strategy = get_target_normalization(target_normalization)
    # ratio_projvol persists its projected-volume denominator in offset_meta;
    # serve_denom_col reads it (and the centered prior_* back-compat keys),
    # falling back to the season-mean choice for legacy pickles.
    denom_col = serve_denom_col(offset_meta, hist_gate, X.columns)
    # global_mean snapshot lives in offset_meta for centered strategies; the
    # ratio strategy ignores it (uses MeanYr from features directly).
    global_mean = float((offset_meta or {}).get("global_mean", 0.0))
    return decode_predictive_mean(
        prob_params,
        dist,
        sn_loc=strategy.decode_loc(prob_params["loc"].values, X, global_mean, denom_col),
        sn_scale=strategy.decode_scale(prob_params["scale"].values, X, denom_col),
        hist_gate=hist_gate,
    )
