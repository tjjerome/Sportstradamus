"""Distributional probability scoring for a single league/market/platform batch.

:func:`model_prob` is the hot path of the prediction pipeline: it loads
the trained LightGBMLSS model pickle, runs vectorized inference on the
``playerStats`` feature matrix, blends model and bookmaker predictions
via :func:`fused_loc`, applies temperature-scaling calibration, and
returns a list of scored offer dicts ready for :func:`find_correlation`.
"""

from __future__ import annotations

import hashlib
import json
import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from sportstradamus.helpers import (
    GATE_PUBLISH_THRESHOLD,
    NONZERO_DENOM_GATE,
    UNDERDOG_BOOST_BASELINE,
    LazyArchive,
    apply_cdf_recal,
    apply_temperature,
    book_gate,
    book_skewnormal_shape,
    book_weights,
    combo_props,
    decode_predictive_mean,
    fused_loc,
    get_odds,
    get_push_prob,
    set_model_start_values,
    skewnormal_loc_from_mean,
    stat_cv,
    stat_dist,
    stat_map,
    stat_meta,
    stat_zi,
)
from sportstradamus.helpers.distributions import _DP_PHI_CEILING, dfs_boost_probs
from sportstradamus.helpers.io import market_file_slug, model_pickle_path
from sportstradamus.helpers.training_quotes import (
    COMBO_SUM_SOURCE,
    DFS_PLATFORM_BOOKS,
    TrainingQuote,
    quote_pricing_params,
    resolve_training_quote,
)
from sportstradamus.spiderLogger import logger
from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.group_conditional_cdf import (
    LEGACY_AFFINE_SCHEMA_VERSION,
    AffinePredictive,
    TwoPartLineOutput,
    apply_affine_groupcdf,
    apply_two_part_line,
    serialize_two_part_calibration,
)
from sportstradamus.training.group_conditional_cdf import (
    ROLE_VALUES as TWO_PART_ROLE_VALUES,
)
from sportstradamus.training.group_conditional_cdf import (
    TWO_PART_SCHEMA_VERSION as TWO_PART_CALIBRATION_SCHEMA_VERSION,
)
from sportstradamus.training.model_strategy import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_SERVE,
    MODEL_STRATEGY_MODEL_KEY,
    ArtifactIdentity,
    CellContext,
    InactiveStrategyArtifactError,
    distribution_class,
    get_strategy,
    parse_controls,
    resolve_report_identity,
    validate_strategy_selection,
)
from sportstradamus.training.posthoc import PROB_STAGE, apply_posthoc, correct_fused_mean
from sportstradamus.training.role_specs import RoleSpec, role_spec_for
from sportstradamus.training.ship_config import WITHHELD
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY,
    TWO_PART_STRATEGY,
)

# LazyArchive defers DuckDB lock acquisition until the first attribute
# access. See LazyArchive docstring in helpers/archive.py.
archive = LazyArchive()

# Maximum allowed model confidence before applying a boost.
_MAX_CONFIDENCE = 0.90

# Minimum bookmaker line for a "yards" market to be scored. Combo legs ("vs.")
# are exempt; single-player yards lines at or below this are too low to model
# reliably and are dropped.
_MIN_YARDS_LINE = 8

# Maximum Underdog boost multiplier kept when de-duplicating a player's offers.
# Above this the promo is an outlier (e.g. a discounted special) that distorts
# the per-player distance ranking, so it is filtered out.
_MAX_UNDERDOG_BOOST = 3.65

# A book-fallback quote serves only when its same-line cohort holds at least this many
# real sportsbooks. DFS platforms repost (or discount) consensus rather than price
# independently, so a cohort of platforms alone is the platform quoting itself — the
# Sleeper fake-50/50 poisoning vector.
_MIN_FALLBACK_REAL_BOOKS = 1

# Component sums whose components are not fit to quote, so the sum inherits their defect.
# NBA BLK and STL archive a probability that disagrees with their own stored ev at every
# line (+0.17 to +0.27, with 29%/43% of ev NULL); the sum graded claimed 0.141 against a
# realized 0.439. Repairing the components is what lifts this, not a kernel change.
_COMBO_SERVE_BLOCKED = frozenset({("NBA", "BLST")})

# Drop an unquoted single-player leg when the model disagrees with the payout-implied
# probability by more than this. Underdog prices its own boosts near-fair, so a model
# >15 pts from the only price available has no independent support — these were the
# +51%-edge phantom picks the flat 0.5 prior manufactured.
_UNQUOTED_BOOK_DISAGREEMENT_MAX = 0.15

# A bookmaker projected mean beyond this multiple of its own line is implausible:
# model and book project the same stat, and a sane book mean sits within a small
# factor of the line it was quoted at. Such rows (a pre-fix get_ev zero-inflation
# runaway still in the archive, or an ill-conditioned anytime-TD count-tail
# inversion) trigger the regularizer below — and are warned so corruption never
# inflates predictions silently.
_BOOK_EV_LINE_CAP: float = 10.0

# Runaway book means are shrunk toward the model rather than discarded to the line:
# a book mean is capped at this many model predictive SDs from the model mean
# (|mu_book - mu_hat| <= K*SD). Wide enough (4 SD) that a genuinely disagreeing
# book stays an independent vote; only ill-conditioned-tail / corrupt rows bind.
_BOOK_EV_MODEL_SD_CAP: float = 4.0

# Symmetric guard on the model side: a model mean beyond this multiple of the player's
# own realized scale (max of MeanYr / Mean10 / STDYr) is a sparse-leaf blow-up of the
# unbounded response function, not a projection (realized Result/own-scale p99.9 ~40, so
# 10x is wide headroom). Winsorized toward the cap before the pool so a runaway can't ride
# model_weight into a max-confidence bet. See research/ brief; Mekelburg & Strauss (2024).
_MODEL_EV_OWN_SCALE_CAP: float = 10.0

# Cap floor: the own-scale cap never drops below _MODEL_EV_OWN_SCALE_CAP * this (= 5), so a
# low-volume player with a small legitimate mean is never clamped.
_OWN_SCALE_FLOOR: float = 0.5

# Below this own-scale a player has no informative history to anchor a model mean (a
# debutant, or a position player on a foreign stat line); the clamp can't help, so drop it.
_OWN_SCALE_MIN: float = 0.1

# Training brackets the dispersion fit in scorecard._DISPERSION_C_BOUNDS = (0.1, 10.0); a value
# pinned at either end is a diverged fit, not a measurement, and serving it rescales the
# predictive shape up to 10x too narrow (SkewNormal pins at the floor, count families at the
# cap) — the 2026-08 WNBA overconfident-unders vector. Local constants, not an import: the
# scorecard/sweep modules (click, tabulate, Optuna) stay out of the serving import path. The
# tolerances absorb float wobble around the bounds, mirroring sweep._DIVERGED_DISPERSION_CAL.
# Known gap: DPO's train-time cap is dynamic (min(10, ceiling/max(phi))), so a DPO fit pinned
# below 9.95 escapes detection; the in-family phi re-clip in _dispersion_calibrate bounds it.
_DIVERGED_DISPERSION_FLOOR: float = 0.1005
_DIVERGED_DISPERSION_CEIL: float = 9.95

# Maximum scored offers retained per player after boost-distance deduplication.
_MAX_OFFERS_PER_PLAYER: int = 3

# Serve-time model-version cache keyed by (pickle path, mtime): the legacy synthesis
# hashes the whole pickle file, so it runs once per model per process. See resolve_model_version.
_MODEL_VERSION_CACHE: dict[tuple[str, float], str] = {}

# Attribution id for a leg scored off devigged book odds (no trained model pickle).
_BOOK_FALLBACK_VERSION = "book_fallback"

# The two-part validation operator settles every quoted line from these
# adjacent integer CDF endpoints, independent of the pickle's generic step.
_TWO_PART_SETTLEMENT_STEP = 1.0


def resolve_model_version(filepath: str, filedict: dict) -> str:
    """The stamped ``model_version``, or a stable ``legacy.<sha10>`` for pre-stamp pickles.

    ~53 pickles trained before the train-time stamp carry no ``model_version``;
    for those we synthesize one from a sha1 of the pickle bytes so each legacy
    model still attributes to a distinct, reproducible id. Memoized per
    ``(path, mtime)`` so the file-hash cost is paid once per model per process.
    """
    version = filedict.get("model_version")
    if version is not None:
        return version
    key = (filepath, Path(filepath).stat().st_mtime)
    cached = _MODEL_VERSION_CACHE.get(key)
    if cached is None:
        cached = "legacy." + hashlib.sha1(Path(filepath).read_bytes()).hexdigest()[:10]
        _MODEL_VERSION_CACHE[key] = cached
    return cached


def normalize_market(league: str, market: str, platform: str) -> str:
    """Canonicalize a platform's market label to the league gamelog/model key.

    Applies the per-platform ``stat_map`` alias plus the NHL and NBA/WNBA
    fixups shared by ``match_offers``, ``model_prob``, and ``book_fallback_prob``.
    """
    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
    if league in ("NBA", "WNBA"):
        market = market.replace("underdog", "prizepicks")
    return market


def _resolve_denom_col(offset_meta: dict | None, hist_gate: float, columns) -> str:
    """SkewNormal decode denominator, preferring the value training persisted.

    The denom (``MeanYr`` vs ``MeanYr_nonzero``) is a property of the trained model:
    training picks it from the empirical zero-rate and stores it in the pickle. Reading
    it back keeps serve in lockstep even when the runtime ``stat_zi`` (gitignored,
    per-box) is stale or absent. Falls back to the legacy ``hist_gate`` recompute for
    pre-persist pickles; the ``prior_*`` keys cover already-shipped centered pickles
    without a retrain (mean10 stores the denom as ``prior_fallback_col``, EB as
    ``prior_col``).
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


def _decode_skewnormal(
    prob_params: pd.DataFrame,
    playerStats: pd.DataFrame,
    hist_gate: float,
    offset_meta: dict | None,
    target_normalization: str,
) -> pd.DataFrame:
    """Decode raw SkewNormal model outputs into absolute EV / sigma / skew.

    Dispatches the ``loc`` and ``scale`` inverse transforms through the
    :mod:`sportstradamus.training.baselines` registry so the prediction-side
    decode mirrors the training-side forward transform exactly. Legacy
    pickles without ``offset_meta`` / ``target_normalization`` keys decode
    through the ``ratio_meanyr`` strategy, which is bit-identical to the
    pre-Task-5 hand-rolled ``loc * MeanYr_clipped`` formula.

    Args:
        prob_params: LightGBMLSS ``predict(pred_type="parameters")`` frame
            with ``loc``, ``scale``, and ``alpha`` columns.
        playerStats: Feature DataFrame; supplies ``MeanYr`` /
            ``MeanYr_nonzero`` (and ``GamesPlayed`` for the EB-centered
            strategy).
        hist_gate: Empirical zero-rate from ``stat_zi`` for the market.
        offset_meta: Pickle-persisted baseline metadata, or ``None`` for
            legacy / ratio-strategy models. The ``global_mean`` snapshot
            inside drives the EB prior at decode time.
        target_normalization: Slug of the baseline strategy the model was
            trained against; defaults to ``"ratio_meanyr"``.

    Returns:
        The same ``prob_params`` frame, mutated in place with the absolute
        ``Projection``, ``Model Sigma``, ``Model Skew``, and optional
        ``Model Gate`` columns set.
    """
    strategy = get_target_normalization(target_normalization)
    # ratio_projvol persists its projected-volume denominator in offset_meta;
    # _resolve_denom_col reads it (and the centered prior_* back-compat keys),
    # falling back to the season-mean choice for legacy pickles.
    denom_col = _resolve_denom_col(offset_meta, hist_gate, playerStats.columns)
    # global_mean snapshot lives in offset_meta for centered strategies; the
    # ratio strategy ignores it (uses MeanYr from features directly).
    global_mean = float((offset_meta or {}).get("global_mean", 0.0))

    ev_loc = strategy.decode_loc(prob_params["loc"].values, playerStats, global_mean, denom_col)
    ev_scale = strategy.decode_scale(prob_params["scale"].values, playerStats, denom_col)

    decoded = decode_predictive_mean(
        prob_params, "SkewNormal", sn_loc=ev_loc, sn_scale=ev_scale, hist_gate=hist_gate
    )
    prob_params["Projection"] = decoded.ev
    prob_params["Model Sigma"] = decoded.sigma
    prob_params["Model Skew"] = decoded.skew
    if decoded.gate is not None:
        prob_params["Model Gate"] = decoded.gate
    return prob_params


def _apply_prob_posthoc(
    cal_over: np.ndarray, posthoc_slug: str, posthoc_blob: dict | None
) -> np.ndarray:
    """Apply a probability-stage post-hoc corrector to the calibrated over-prob.

    Mirrors the training-side seam in ``pipeline.train_market`` so the live
    probability matches the offline-gated one. No-op for mean-stage slugs and
    legacy pickles (slug ``"none"`` / blob ``None``).
    """
    if posthoc_slug in PROB_STAGE:
        return apply_posthoc(posthoc_slug, posthoc_blob, cal_over)
    return cal_over


def _apply_cdf_recal_over(raw_over: np.ndarray, pit_recal_blob: dict | None) -> np.ndarray:
    """Apply the §6.1 Rung C whole-CDF map to the raw model over-probability.

    The recalibrated served CDF is ``F* = g∘F``, so the over-prob becomes
    ``1 − g(F(line)) = 1 − g(1 − over_raw)``. Mirrors the training-side warp in
    ``pipeline._step_compute_test_probabilities`` so the live probability matches the
    offline-gated one. A legacy pickle (``pit_recal_blob`` is None) returns ``raw_over``
    untouched — the ``1 − (1 − x)`` round-trip would perturb it at float epsilon.
    """
    if pit_recal_blob is None:
        return raw_over
    return 1.0 - apply_cdf_recal(pit_recal_blob, 1.0 - np.asarray(raw_over, dtype=float))


def _book_cell_params(
    league: str, market: str
) -> tuple[str | None, float | None, float | None, float | None]:
    """Resolve ``(dist, cv, gate, step)`` for a cell from config, no pickle.

    Mirrors the pickle metadata ``model_prob`` reads, sourced instead from the
    committed ``stat_meta`` / runtime calibration. ``dist`` is ``None`` when the
    cell is unknown. ``step`` is approximated by family (the trained value is
    empirical and not recoverable without the pickle).
    """
    dist = stat_dist.get(league, {}).get(market)
    if dist is None:
        return None, None, None, None
    cv = stat_cv[league].get(market, 1)
    gate = book_gate(league, market, dist)
    step = 1.0 if dist in ("NegBin", "ZINB", "Poisson", "DPO") else 0.5
    return dist, cv, gate, step


def _book_over_prob(
    offer_df: pd.DataFrame,
    dist: str,
    cv: float,
    step: float,
    gate: float | None,
    league: str,
    market: str,
) -> pd.Series:
    """Devigged book probability of the over per row, inverted from ``Market Projection``.

    Inverts the composite book EV through the cell distribution at each row's line. Shared by
    :func:`model_prob` and :func:`book_fallback_prob`. For SkewNormal the book ``(sigma, skew)``
    comes from the per-cell fitted shape evaluated at the book mean (:func:`book_skewnormal_shape`);
    an unfitted cell returns ``(mean*cv, 0)``, the legacy symmetric constant-CV read. A NaN
    ``Market Projection`` (no independent quote) stays NaN so ``_finalize_records`` prices the
    row payout-implied.
    """
    quoted = offer_df.loc[offer_df["Market Projection"].notna()]
    if quoted.empty:
        return pd.Series(np.nan, index=offer_df.index)
    offer_df = quoted
    if dist == "SkewNormal":
        # The archive encodes SkewNormal sportsbook EVs without the model's
        # external hurdle. Keep the book endpoint gate-free on decode too.
        gate = None

        def _sn_over(x):
            sigma, skew = book_skewnormal_shape(league, market, x["Market Projection"], cv)
            return 1 - get_odds(
                x["Line"],
                x["Market Projection"],
                dist,
                cv=cv,
                step=step,
                sigma=float(sigma),
                skew_alpha=float(skew),
                gate=gate,
            )

        return offer_df.apply(_sn_over, axis=1)
    return offer_df.apply(
        lambda x: 1 - get_odds(x["Line"], x["Market Projection"], dist, cv, step=step, gate=gate),
        axis=1,
    )


def _annotate_display_shape(offer_df: pd.DataFrame, dist: str) -> None:
    """Set the display-only ``Model Param`` and ``Projection STD`` columns in place.

    Reads the distribution-shape parameters the model emitted (``Model R`` /
    ``Model Phi`` / ``Model Sigma`` / ``Model Alpha``); falls back to NaN when they are absent,
    as on a book-fallback record. These columns feed the dashboard detail popup
    only — no scoring path reads them.
    """
    # style: allow-complexity  flat per-distribution-family dispatch of closed-form
    # display variances; splitting duplicates the family/param-present guards.
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model R"]
    elif dist == "DPO" and "Model Phi" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Phi"]
    elif dist == "SkewNormal" and "Model Sigma" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Sigma"]
    elif "Model Alpha" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Alpha"]
    else:
        offer_df["Model Param"] = np.nan

    _m = offer_df["Projection"].to_numpy(dtype=float)
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        _r = offer_df["Model R"].to_numpy(dtype=float)
        offer_df["Projection STD"] = np.sqrt(np.clip(_m + _m**2 / np.clip(_r, 1e-6, None), 0, None))
    elif dist == "DPO" and "Model Phi" in offer_df.columns:
        _phi = offer_df["Model Phi"].to_numpy(dtype=float)
        offer_df["Projection STD"] = np.sqrt(np.clip(_m / np.clip(_phi, 1e-6, None), 0, None))
    elif dist == "SkewNormal" and "Model Sigma" in offer_df.columns:
        _sg = offer_df["Model Sigma"].to_numpy(dtype=float)
        _sk = (
            offer_df["Model Skew"].to_numpy(dtype=float)
            if "Model Skew" in offer_df.columns
            else np.zeros_like(_sg)
        )
        _delta = _sk / np.sqrt(1 + _sk**2)
        offer_df["Projection STD"] = _sg * np.sqrt(np.clip(1 - 2 * _delta**2 / np.pi, 0, None))
    elif "Model Alpha" in offer_df.columns:
        _a = offer_df["Model Alpha"].to_numpy(dtype=float)
        offer_df["Projection STD"] = _m / np.sqrt(np.clip(_a, 1e-6, None))
    else:
        offer_df["Projection STD"] = np.nan


def _finalize_records(
    offer_df: pd.DataFrame,
    league: str,
    platform: str,
    dist: str,
    cv: float,
    step: float,
    temperature: float | None,
    dispersion_cal: float,
    model_version: str,
    pit_recal_blob: dict | None = None,
) -> list[dict]:
    """Resolve the bet side, apply boosts, and project the export schema.

    Shared tail of :func:`model_prob` and :func:`book_fallback_prob`. Expects
    ``offer_df`` to already carry ``Model Over`` / ``Model Under`` (win
    probabilities for each side), the raw ``Market EV`` over-probability (NaN when no
    book quoted the leg — filled with the payout-implied probability of the chosen
    side, and the unquoted-disagreement gate then drops single-player rows the model
    disagrees with beyond ``_UNQUOTED_BOOK_DISAGREEMENT_MAX``), ``Push Prob``,
    ``Projection`` and ``Market Projection``. Feature-derived passenger columns absent on a
    book-fallback record (built without a feature matrix) are filled neutral so
    correlation and the dashboard read sane values — ``Player position`` in
    particular must stay an int, never NaN, or correlation's position map breaks.
    """
    totals_map = archive.default_totals
    for _col in ("Avg5", "AvgH2H", "H2HPlayed", "Total", "Defense position", "Moneyline"):
        if _col not in offer_df.columns:
            offer_df[_col] = np.nan
    if "Home" not in offer_df.columns:
        # Bool, not NaN: dashboard/narrative.py's match_label does `'v' if home else '@'`,
        # and NaN is truthy in Python — it would render every no-Home row as the host.
        offer_df["Home"] = False
    if "Commence" not in offer_df.columns:
        # Only Underdog threads a tip-off timestamp (books.py::get_ud); Sleeper and
        # book-fallback rows default to empty so the export projection stays uniform
        # (the dashboard coerces "" → NaT → non-urgent countdown).
        offer_df["Commence"] = ""

    offer_df["Model Over"] = offer_df["Model Over"].clip(upper=_MAX_CONFIDENCE)
    offer_df["Model Under"] = offer_df["Model Under"].clip(upper=_MAX_CONFIDENCE)

    offer_df["Win Prob"] = offer_df[["Model Over", "Model Under"]].max(axis=1)
    offer_df["Bet"] = offer_df[["Model Over", "Model Under"]].idxmax(axis=1).str[6:]

    if "Boost" in offer_df.columns:
        offer_df.loc[offer_df["Boost"] == 1, ["Boost_Under", "Boost_Over"]] = 1
    offer_df[["Boost_Under", "Boost_Over"]] = offer_df[["Boost_Under", "Boost_Over"]].fillna(
        0
    ).infer_objects(copy=False) * (UNDERDOG_BOOST_BASELINE if platform == "Underdog" else 1)
    offer_df["Boost"] = offer_df.apply(
        lambda x: (
            (x["Boost_Over"] if x["Bet"] == "Over" else x["Boost_Under"])
            if not np.isnan(x["Boost_Over"])
            else x["Boost"]
        ),
        axis=1,
    )

    offer_df["Model EV"] = offer_df["Win Prob"] * offer_df["Boost"]
    offer_df.loc[(offer_df["Bet"] == "Under"), "Market EV"] = (
        1 - offer_df.loc[(offer_df["Bet"] == "Under"), "Market EV"]
    )
    unquoted = offer_df["Market EV"].isna()
    # Post-conversion Boost_Over/Boost_Under are full decimal odds on every platform, so
    # the payout-implied probability of the chosen side is the platform's own price for
    # the leg — the only price available when no book quoted it.
    implied = offer_df.apply(
        lambda x: dfs_boost_probs(x["Boost_Over"], x["Boost_Under"])[
            1 if x["Bet"] == "Under" else 0
        ],
        axis=1,
    )
    offer_df["Market Prob"] = offer_df["Market EV"].fillna(implied)
    phantom = (
        unquoted
        & ~offer_df["Player"].str.contains(" vs. ", regex=False)
        & ((offer_df["Win Prob"] - offer_df["Market Prob"]).abs() > _UNQUOTED_BOOK_DISAGREEMENT_MAX)
    )
    if phantom.any():
        logger.warning(
            f"{league} {offer_df['Market'].iat[0]}: dropped {int(phantom.sum())} unquoted "
            "offer(s) disagreeing with the payout-implied price"
        )
        offer_df = offer_df.loc[~phantom]
    offer_df["Market EV"] = offer_df["Market Prob"] * offer_df["Boost"]
    offer_df["Kelly"] = (offer_df["Model EV"] - 1) / (offer_df["Boost"] - 1)
    offer_df["Distance"] = offer_df["Boost"] / UNDERDOG_BOOST_BASELINE
    offer_df.loc[offer_df["Distance"] < 1, "Distance"] = (
        1 / offer_df.loc[offer_df["Distance"] < 1, "Distance"]
    )
    offer_df = (
        offer_df.loc[offer_df["Boost"] <= _MAX_UNDERDOG_BOOST]
        .sort_values("Distance", ascending=True)
        .groupby("Player")
        .head(_MAX_OFFERS_PER_PLAYER)
    )

    offer_df["Avg 5"] = offer_df["Avg5"] - offer_df["Line"]
    offer_df["Avg H2H"] = offer_df["AvgH2H"] - offer_df["Line"]
    offer_df.loc[offer_df["H2HPlayed"] == 0, "Avg H2H"] = 0
    offer_df["O/U"] = offer_df["Total"] / totals_map.get(league, 1)
    offer_df["DVPOA"] = offer_df["Defense position"]
    if "Player position" not in offer_df:
        offer_df["Player position"] = -1

    offer_df["Player position"] = offer_df["Player position"].astype("category")
    offer_df["Player position"] = (
        offer_df["Player position"].cat.set_categories(range(-1, 5)).fillna(-1).astype(int)
    )
    _annotate_display_shape(offer_df, dist)

    offer_df["Dist"] = dist
    offer_df["CV"] = cv
    offer_df["Gate"] = offer_df.get("Model Gate", np.nan)
    offer_df["Temperature"] = temperature
    offer_df["Disp Cal"] = dispersion_cal
    offer_df["Step"] = step
    offer_df["Model Version"] = model_version
    # §6.1 Rung C: the whole-CDF recal map g (a cdf_recal_isotonic cell's served
    # warp) so the deep-dive can draw g∘F, the distribution the model actually
    # serves. Null for every other cell — the raw parametric curve is exact there.
    offer_df["Model PIT Recal"] = json.dumps(pit_recal_blob) if pit_recal_blob else None

    return offer_df[
        [
            "League",
            "Date",
            "Commence",
            "Team",
            "Opponent",
            "Home",
            "Player",
            "Market",
            "Line",
            "Boost",
            "Boost_Over",
            "Boost_Under",
            "Bet",
            "Market EV",
            "Model EV",
            "Avg 5",
            "Avg H2H",
            "Moneyline",
            "O/U",
            "DVPOA",
            "Player position",
            "Projection",
            "Model Param",
            "Projection STD",
            "Win Prob",
            "Push Prob",
            "Market Projection",
            "Market Prob",
            "Kelly",
            "Dist",
            "CV",
            "Gate",
            "Temperature",
            "Disp Cal",
            "Step",
            "Model Version",
            "Model PIT Recal",
        ]
    ].to_dict("records")


def _col_or_none(df: pd.DataFrame, col: str, active: bool = True) -> np.ndarray | None:
    return df[col].to_numpy() if active and col in df.columns else None


def _serve_offset_mode(dist: str, target_normalization: str) -> bool:
    """Whether serve seeds the SkewNormal with the centered-residual offset.

    Mirrors training's ``offset_mode = strategy.start_mode_flag == "offset"``
    (``pipeline.train_market``) so a model trained against any offset strategy seeds the
    same way at predict time. Derived from the registry — not a hardcoded
    ``offset_meta["method"]`` string — so a new offset strategy needs no serve-side edit.
    Only the SkewNormal seed consumes ``offset_mode``; the ``dist`` guard short-circuits
    the registry lookup off that path and for non-registry legacy slugs.
    """
    return (
        dist == "SkewNormal"
        and get_target_normalization(target_normalization).start_mode_flag == "offset"
    )


def _build_prob_params(
    filedict: dict,
    market: str,
    stat_data,
    playerStats: pd.DataFrame,
    dist: str,
    normalized: bool,
    target_normalization: str,
    structural_strategy: str,
) -> pd.DataFrame:
    if market in stat_data.volume_stats:
        prob_params = pd.DataFrame(index=playerStats.index)
        prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} mean"])
        if f"proj {market} std" in stat_data.playerProfile.columns:
            prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} std"])
        prob_params.rename(
            columns={f"proj {market} mean": "Projection", f"proj {market} std": "Model Param"},
            inplace=True,
        )
        return prob_params

    model = filedict["model"]
    categories = ["Home", "Player position"]
    if "Player position" not in playerStats.columns:
        categories.remove("Player position")
    for c in categories:
        playerStats[c] = playerStats[c].astype("category")

    def _predict(predict_model, rows: pd.DataFrame) -> pd.DataFrame:
        if getattr(predict_model, "is_hurdle", False):
            # HurdleZINB seeds its internal NegBin from its own X; the external
            # set_model_start_values would treat it as a plain LSS and fail.
            predict_model.set_model_start_values(rows)
        else:
            set_model_start_values(
                predict_model,
                dist,
                rows,
                normalized=normalized,
                offset_mode=_serve_offset_mode(dist, target_normalization),
                sn_param=filedict.get("sn_param", "direct"),
            )
        params = predict_model.predict(rows, pred_type="parameters")
        params.index = rows.index
        return params

    prob_params = _predict(model, playerStats)
    if structural_strategy == AFFINE_STRATEGY:
        _overlay_affine_experts(prob_params, playerStats, filedict.get("expert_models"), _predict)
    return prob_params


def _overlay_affine_experts(
    prob_params: pd.DataFrame,
    playerStats: pd.DataFrame,
    experts: object,
    predict,
) -> None:
    """Overlay each persisted position's expert params onto the pooled serve params.

    The expert code set is recovered from the pickle's own ``expert_models`` keys — the
    market's discovered positions — never a fixed QB/RB pair.
    """
    if not isinstance(experts, dict) or not experts:
        raise ValueError("affine candidate pickle is missing its per-position expert models")
    position = pd.to_numeric(playerStats["Player position"], errors="coerce")
    for code in experts:
        rows = playerStats.loc[position.eq(code)]
        if rows.empty:
            continue
        expert_params = predict(experts[code], rows)
        if list(expert_params.columns) != list(prob_params.columns):
            raise ValueError("affine expert parameter schema drifted from pooled model")
        prob_params.loc[rows.index, :] = expert_params.to_numpy()


def _resolve_serving_strategy(filedict: dict, league: str, market: str) -> ArtifactIdentity:
    """Validate the strategy identity before any strategy-specific serving work."""
    identity = resolve_report_identity(filedict, league, market)
    if MODEL_STRATEGY_MODEL_KEY not in filedict:
        if identity.structural_strategy != BASE_STRUCTURAL_STRATEGY:
            raise ValueError("structural adapter requires canonical generic strategy identity")
        if CAP_SERVE not in get_strategy(identity.strategy_slug).capabilities:
            raise ValueError(f"{identity.strategy_slug}: legacy strategy lacks serve capability")
        return identity
    if identity.status != "active":
        raise InactiveStrategyArtifactError(
            f"{identity.strategy_slug}: inactive strategy artifact cannot serve"
        )
    expected_columns = filedict.get("expected_columns")
    if not isinstance(expected_columns, (list, tuple)) or any(
        not isinstance(column, str) for column in expected_columns
    ):
        raise ValueError(f"{identity.strategy_slug}: generic strategy expected_columns are invalid")
    distribution = filedict.get("distribution")
    if not isinstance(distribution, str):
        raise ValueError(f"{identity.strategy_slug}: generic strategy distribution is invalid")
    controls = parse_controls(identity.controls_json)
    spec = get_strategy(identity.strategy_slug)
    if bool(spec.split_fingerprint_path) != bool(identity.split_fingerprint):
        raise ValueError(
            f"{identity.strategy_slug}: strategy split fingerprint is missing or stale"
        )
    cell = CellContext(
        league,
        market,
        distribution,
        distribution_class(distribution),
        frozenset(expected_columns),
        identity.matrix_hash,
    )
    validate_strategy_selection(cell, spec, controls, required_capabilities=(CAP_SERVE,))
    return identity


def _book_evs_for_players(
    offer_df: pd.DataFrame,
    league: str,
    market: str,
    dist: str,
    cv: float,
    hist_gate: float,
    date_map: dict,
    stat_data,
    players: pd.Index,
) -> tuple[list, list]:
    """Modal-cohort book means and spreads for the model path's book leg.

    Resolves the same admission-gated quotes as the fallback path and inverts each
    at its own cohort line, replacing the old cross-line average of stored per-row
    means (a DFS platform's boundary-line self-quote inverted under a too-narrow
    fitted shape once implied a 2.5-K mean for a 1.1-K batter). A player whose only
    support is the platform's own quote gets NaN: the blend then runs model-only
    and ``_finalize_records`` prices ``Market Prob`` payout-implied, applying the
    unquoted-disagreement drop.

    The spread is ``cv * mean`` for a single-market quote, which is all that family
    asserts, but a component sum carries its own — the kernel added the component
    variances and their correlation, so ``cv * mean`` would substitute the composite
    cell's generic dispersion for a quantity actually computed.
    """
    quotes = _servable_fallback_quotes(offer_df, league, market, date_map, stat_data, dist, cv)
    gate = None if dist == "SkewNormal" else hist_gate or None
    means, sds = [], []
    for player in players:
        quote = quotes.get(player)
        if quote is None:
            means.append(np.nan)
            sds.append(np.nan)
            continue
        mean = quote_pricing_params(quote, league, market, dist, cv, gate=gate).mean
        means.append(mean)
        sds.append(quote.sum_sd if quote.source == COMBO_SUM_SOURCE else cv * mean)
    return means, sds


def _decode_model_params(
    prob_params: pd.DataFrame,
    dist: str,
    playerStats: pd.DataFrame,
    hist_gate: float,
    offset_meta,
    target_normalization: str,
) -> None:
    if dist == "SkewNormal":
        if "loc" in prob_params.columns:
            # Dispatch SkewNormal loc/scale through the baselines registry so the
            # train-side forward transform and predict-side inverse cannot drift.
            _decode_skewnormal(
                prob_params, playerStats, hist_gate, offset_meta, target_normalization
            )
        return
    # The raw-column guards skip the volume_stats branch, which already set Projection
    # from the player profile rather than a fitted distribution.
    raw_col = {
        "NegBin": "total_count",
        "ZINB": "total_count",
        "DPO": "mu",
        "Gamma": "concentration",
        "ZAGamma": "concentration",
    }.get(dist)
    if raw_col is None or raw_col not in prob_params.columns:
        return
    decoded = decode_predictive_mean(prob_params, dist)
    prob_params["Projection"] = decoded.ev
    if decoded.r is not None:
        prob_params["Model R"] = decoded.r
    elif decoded.phi is not None:
        prob_params["Model Phi"] = decoded.phi
    else:
        prob_params["Model Alpha"] = decoded.alpha
    if decoded.gate is not None:
        prob_params["Model Gate"] = decoded.gate


def _clamp_shape_ceiling(offer_df: pd.DataFrame, dist: str, shape_ceiling) -> None:
    # DPO is deliberately absent: its phi is hard-bounded in-family (the DP kernel
    # clips to _DP_PHI_CEILING), so the NB r-ceiling semantics do not apply and DPO
    # pickles persist no shape_ceiling.
    if shape_ceiling is None:
        return
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        offer_df["Model R"] = np.minimum(offer_df["Model R"], shape_ceiling)
    elif dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
        offer_df["Model Alpha"] = np.minimum(offer_df["Model Alpha"], shape_ceiling)


def _zi_kwargs(offer_df: pd.DataFrame, dist: str, hist_gate: float) -> dict:
    if dist == "SkewNormal":
        if "Model Gate" in offer_df.columns:
            return {"gate_model": offer_df["Model Gate"].to_numpy(), "gate_book": 0.0}
        return {"gate_book": hist_gate} if hist_gate > GATE_PUBLISH_THRESHOLD else {}
    if dist in ("ZINB", "ZAGamma") and "Model Gate" in offer_df.columns:
        return {"gate_model": offer_df["Model Gate"].to_numpy(), "gate_book": hist_gate}
    return {}


def _valid_two_part_routing_shape(routing: object) -> bool:
    """Role-by-position routing with string role columns and integer-coded positions."""
    if not isinstance(routing, dict) or routing.get("kind") != "pregame_role_by_position":
        return False
    role_columns = routing.get("role_columns")
    positions = routing.get("positions")
    valid_role_columns = (
        isinstance(role_columns, list)
        and bool(role_columns)
        and all(isinstance(column, str) for column in role_columns)
    )
    valid_positions = (
        isinstance(positions, dict)
        and bool(positions)
        and all(
            isinstance(code, str) and code.isdigit() and isinstance(label, str)
            for code, label in positions.items()
        )
    )
    return valid_role_columns and valid_positions


def _two_part_candidate_state(
    experiment_blob: dict,
) -> tuple[dict, dict]:
    """Return the active two-part calibration and feature-routing state."""
    # style: allow-complexity -- flat serving-blob schema validator; each branch is one
    # persisted-contract invariant, so splitting would only scatter one accept/reject decision.
    if not isinstance(experiment_blob, dict):
        raise ValueError("two-part candidate is missing its algorithm state")
    if (
        experiment_blob.get("schema_version") != TWO_PART_CALIBRATION_SCHEMA_VERSION
        or experiment_blob.get("status") != "active"
        or experiment_blob.get("line_probability_only") is not True
    ):
        raise ValueError("two-part candidate is not active on schema 3")

    routing = experiment_blob.get("routing")
    expected_routing_keys = {
        "kind",
        "thresholds",
        "role_columns",
        "gate_rates",
        "served_gate_rates",
        "gate_model_weight",
        "positions",
    }
    if (
        not isinstance(routing, dict)
        or set(routing) != expected_routing_keys
        or not _valid_two_part_routing_shape(routing)
    ):
        raise ValueError("two-part candidate has invalid feature routing state")
    thresholds = routing.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(routing["positions"]):
        raise ValueError("two-part candidate requires one role threshold per persisted position")
    valid_thresholds = all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        for value in thresholds.values()
    )
    if not valid_thresholds:
        raise ValueError("two-part role thresholds must be finite numbers")

    raw_gate_rates = routing.get("gate_rates")
    served_gate_rates = routing.get("served_gate_rates")
    gate_model_weight = routing.get("gate_model_weight")
    rate_maps = (raw_gate_rates, served_gate_rates)
    valid_rates = all(
        isinstance(rates, dict)
        and set(rates) == set(TWO_PART_ROLE_VALUES)
        and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and 0.0 <= float(value) < 1.0
            for value in rates.values()
        )
        for rates in rate_maps
    )
    valid_weight = (
        isinstance(gate_model_weight, (int, float, np.integer, np.floating))
        and not isinstance(gate_model_weight, (bool, np.bool_))
        and np.isfinite(gate_model_weight)
        and 0.0 <= float(gate_model_weight) <= 1.0
    )
    if not valid_rates or not valid_weight:
        raise ValueError("two-part routing requires finite low/high gate state")
    if any(
        float(served_gate_rates[role]) != float(gate_model_weight) * float(raw_gate_rates[role])
        for role in TWO_PART_ROLE_VALUES
    ):
        raise ValueError("two-part served gates do not match their persisted model weight")

    endpoint_contract = experiment_blob.get("endpoint_contract")
    expected_endpoint_keys = {
        "gate_source",
        "gate_rates_are_model_endpoint",
        "book_endpoint_gate",
        "fallback_gate",
        "dispersion",
    }
    fallback_gate = (
        endpoint_contract.get("fallback_gate") if isinstance(endpoint_contract, dict) else None
    )
    book_endpoint_gate = (
        endpoint_contract.get("book_endpoint_gate") if isinstance(endpoint_contract, dict) else None
    )
    valid_book_endpoint = (
        isinstance(book_endpoint_gate, (int, float, np.integer, np.floating))
        and not isinstance(book_endpoint_gate, (bool, np.bool_))
        and np.isfinite(book_endpoint_gate)
        and float(book_endpoint_gate) == 0.0
    )
    valid_fallback = (
        isinstance(fallback_gate, (int, float, np.integer, np.floating))
        and not isinstance(fallback_gate, (bool, np.bool_))
        and np.isfinite(fallback_gate)
        and 0.0 <= float(fallback_gate) < 1.0
    )
    if (
        not isinstance(endpoint_contract, dict)
        or set(endpoint_contract) != expected_endpoint_keys
        or endpoint_contract.get("gate_source") != "train_result_eq_zero_by_role"
        or endpoint_contract.get("gate_rates_are_model_endpoint") is not True
        or not valid_book_endpoint
        or endpoint_contract.get("dispersion") != "raw_fused_shape"
        or not valid_fallback
    ):
        raise ValueError("two-part candidate has an invalid endpoint contract")

    calibration = experiment_blob.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("two-part candidate is missing calibration state")
    serialize_two_part_calibration(calibration)
    return calibration, routing


def _two_part_candidate_routes(
    offer_df: pd.DataFrame,
    routing: dict,
    spec: RoleSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Tier live pregame rows by the persisted train-only role thresholds."""
    if list(spec.role_columns) != routing["role_columns"]:
        raise ValueError("two-part role columns do not match the persisted routing state")
    if "Player position" not in offer_df:
        raise ValueError("two-part candidate requires Player position")
    raw_position = pd.to_numeric(offer_df["Player position"], errors="coerce").to_numpy()
    if not np.isfinite(raw_position).all():
        raise ValueError("two-part candidate requires integer position codes")
    positions = raw_position.astype(int)
    persisted_codes = {int(code) for code in routing["positions"]}
    if (
        not np.array_equal(raw_position, positions)
        or not set(positions.tolist()) <= persisted_codes
    ):
        raise ValueError(
            "two-part candidate positions must be among its persisted routing position codes"
        )

    role_score = spec.role_score(offer_df).to_numpy(dtype=float)
    thresholds = np.asarray([routing["thresholds"][str(code)] for code in positions], dtype=float)
    roles = np.where(role_score < thresholds, "low", "high")
    return roles, positions


def _two_part_candidate_cdf(
    offer_df: pd.DataFrame,
    base_mean: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate the fitted gated SkewNormal CDF at row-aligned points."""
    mean = np.asarray(base_mean, dtype=float)
    point = np.asarray(points, dtype=float)
    sigma = offer_df["Model Sigma"].to_numpy(dtype=float)
    alpha = offer_df["Model Skew"].to_numpy(dtype=float)
    gate = (
        offer_df["Model Gate"].to_numpy(dtype=float)
        if "Model Gate" in offer_df
        else np.zeros(len(offer_df), dtype=float)
    )
    if (
        any(array.shape != (len(offer_df),) for array in (mean, point, sigma, alpha, gate))
        or not all(np.isfinite(array).all() for array in (mean, point, sigma, alpha, gate))
        or np.any(sigma <= 0.0)
        or np.any((gate < 0.0) | (gate >= 1.0))
    ):
        raise ValueError("two-part predictive inputs are invalid or not row-aligned")
    loc = skewnormal_loc_from_mean(mean, sigma, alpha)
    base_cdf = skewnorm.cdf(point, alpha, loc=loc, scale=sigma)
    cdf = np.where(
        point < 0.0,
        (1.0 - gate) * base_cdf,
        gate + (1.0 - gate) * base_cdf,
    )
    if not np.isfinite(cdf).all():
        raise ValueError("two-part predictive CDF is nonfinite")
    return cdf


def _apply_two_part_candidate_distribution(
    offer_df: pd.DataFrame,
    base_mean: np.ndarray,
    calibration: dict,
    routing: dict,
    spec: RoleSpec,
) -> TwoPartLineOutput:
    """Apply two-part endpoint maps and its fixed authentic-book pool."""
    roles, positions = _two_part_candidate_routes(offer_df, routing, spec)
    served_gate = np.asarray([routing["served_gate_rates"][role] for role in roles], dtype=float)
    offer_df["Model Gate"] = served_gate
    offer_df["Projection"] = (1.0 - served_gate) * np.asarray(base_mean, dtype=float)
    line = offer_df["Line"].to_numpy(dtype=float)
    if not np.isfinite(line).all():
        raise ValueError("two-part candidate requires finite quoted lines")
    low_point = np.ceil(line - _TWO_PART_SETTLEMENT_STEP)
    high_point = np.floor(line + _TWO_PART_SETTLEMENT_STEP)
    f0 = _two_part_candidate_cdf(offer_df, base_mean, np.zeros(len(offer_df)))
    line_low = _two_part_candidate_cdf(offer_df, base_mean, low_point)
    line_high = _two_part_candidate_cdf(offer_df, base_mean, high_point)
    book_over = offer_df["Market EV"].to_numpy(dtype=float)
    authentic = np.isfinite(book_over)
    return apply_two_part_line(
        calibration,
        line_low,
        line_high,
        f0,
        roles,
        positions,
        book_over,
        authentic,
    )


def _affine_candidate_calibration(experiment_blob: dict) -> dict:
    """Return the active affine calibration payload, rejecting partial pickles."""
    if not isinstance(experiment_blob, dict):
        raise ValueError("affine candidate is missing its algorithm state")
    if (
        experiment_blob.get("status") != "active"
        or experiment_blob.get("line_probability_only") is not True
    ):
        raise ValueError("affine/group-CDF candidate is not active and complete")
    calibration = experiment_blob.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("affine candidate pickle is missing calibration state")
    return calibration


def _affine_supported_codes(calibration: dict) -> tuple[int, ...]:
    """Position codes the fitted affine candidate calibrated — recovered from its own blob."""
    return tuple(int(code) for code in calibration["position_cdf"])


def _apply_affine_candidate_distribution(
    offer_df: pd.DataFrame,
    base_mean: np.ndarray,
    calibration: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the affine/CDF head and return conditional mean plus final line P(over)."""
    position = pd.to_numeric(offer_df["Player position"], errors="coerce").to_numpy()
    supported = _affine_supported_codes(calibration)
    legacy = calibration.get("schema_version") == LEGACY_AFFINE_SCHEMA_VERSION
    if legacy and not np.isin(position, tuple(supported)).all():
        raise ValueError("affine candidate received an offer outside its calibrated positions")
    gate = (
        offer_df["Model Gate"].to_numpy(dtype=float)
        if "Model Gate" in offer_df
        else np.zeros(len(offer_df), dtype=float)
    )
    predictive = AffinePredictive(
        marginal_mean=np.asarray(base_mean, dtype=float) * (1.0 - gate),
        sigma=offer_df["Model Sigma"].to_numpy(dtype=float),
        alpha=offer_df["Model Skew"].to_numpy(dtype=float),
        gate=gate,
    )
    authentic = None
    if not legacy:
        if "QuoteAuthenticity" in offer_df:
            provenance = offer_df["QuoteAuthenticity"]
            if (
                provenance.isna().any()
                or not provenance.isin(("authentic", "derived", "synthetic")).all()
            ):
                raise ValueError("affine candidate received invalid quote authenticity")
            authentic = provenance.eq("authentic").to_numpy(dtype=bool)
        else:
            # Live offer frames carry current bookmaker probabilities rather than
            # training provenance. Convert that explicit current-quote contract
            # to the boolean required by the schema-2 application API.
            current_book = offer_df["Market EV"].to_numpy(dtype=float)
            authentic = np.isfinite(current_book) & (current_book > 0.0) & (current_book < 1.0)
    output = apply_affine_groupcdf(
        calibration,
        predictive,
        offer_df["Line"].to_numpy(dtype=float),
        offer_df["Market EV"].to_numpy(dtype=float),
        position,
        authentic,
    )
    offer_df["Projection"] = output.marginal_mean
    offer_df["Model Sigma"] = output.sigma
    offer_df["Model Skew"] = output.alpha
    offer_df["Model Gate"] = output.gate
    return output.conditional_mean, output.pooled_over


def _model_predictive_sd(offer_df: pd.DataFrame, dist: str, model_ev: np.ndarray) -> np.ndarray:
    """Per-row model predictive SD — the scale for the book-EV plausibility band.

    Read off the model's pre-blend shape columns (``Model R`` / ``Model Phi`` /
    ``Model Alpha`` / ``Model Sigma``); falls back to the mean when a shape column
    is absent.
    """
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        r = np.clip(offer_df["Model R"].to_numpy(), 1e-9, None)
        return np.sqrt(model_ev + model_ev**2 / r)
    if dist == "DPO" and "Model Phi" in offer_df.columns:
        phi = np.clip(offer_df["Model Phi"].to_numpy(), 1e-9, None)
        return np.sqrt(model_ev / phi)
    if dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
        alpha = np.clip(offer_df["Model Alpha"].to_numpy(), 1e-9, None)
        return model_ev / np.sqrt(alpha)
    if dist == "SkewNormal" and "Model Sigma" in offer_df.columns:
        return np.clip(offer_df["Model Sigma"].to_numpy(), 1e-9, None)
    return np.clip(model_ev, 0.5, None)


def _sanitize_book_ev(
    books_ev: np.ndarray,
    line: np.ndarray,
    model_ev: np.ndarray,
    model_sd: np.ndarray,
    cell: str = "",
) -> np.ndarray:
    """Regularize book means implausibly far above their line before the blend.

    A book projected mean beyond ``_BOOK_EV_LINE_CAP`` times its line is bad data or
    an ill-conditioned count-tail inversion (``mu = -log(1-p)`` blows up as p->1) and
    would dominate ``fused_loc``'s log-opinion pool. Such runaway rows are shrunk
    toward the model mean — capped at ``model_ev + _BOOK_EV_MODEL_SD_CAP * SD`` —
    rather than discarded to the line. The warning reports the worst offender's
    ev/line ratio so a benign DFS-clamp placeholder (~5x line on a high-zero-rate
    count) reads apart from a pre-clamp archive runaway (>1000x).
    """
    runaway = books_ev > _BOOK_EV_LINE_CAP * np.clip(line, 0.5, None)
    if not runaway.any():
        return books_ev
    band = model_ev + _BOOK_EV_MODEL_SD_CAP * np.clip(model_sd, 1e-9, None)
    ratio = books_ev / np.clip(line, 0.5, None)
    worst = int(np.argmax(np.where(runaway, ratio, -np.inf)))
    logger.warning(
        f"regularized {int(runaway.sum())}/{runaway.size} implausible book EV(s) toward "
        f"the model in the {cell + ' ' if cell else ''}blend (worst {books_ev[worst]:.1f} on "
        f"line {line[worst]:.1f} = {ratio[worst]:.0f}x, cap {_BOOK_EV_LINE_CAP:.0f}x)"
    )
    return np.where(runaway, np.minimum(books_ev, band), books_ev)


def _own_scale(offer_df: pd.DataFrame) -> np.ndarray | None:
    """Per-row book-independent scale: max of the player's MeanYr / Mean10 / STDYr.

    Returns ``None`` when none of the three columns are present (a book-fallback
    frame), so callers no-op rather than fabricate a scale.
    """
    cols = [c for c in ("MeanYr", "Mean10", "STDYr") if c in offer_df.columns]
    if not cols:
        return None
    return np.nan_to_num(offer_df[cols].to_numpy(dtype=float)).max(axis=1)


def _sanitize_model_ev(offer_df: pd.DataFrame, dist: str) -> None:
    """Winsorize a runaway model mean toward the player's own realized scale.

    Symmetric with :func:`_sanitize_book_ev` (which shrinks an implausible *book*
    mean toward the model): a model mean beyond ``_MODEL_EV_OWN_SCALE_CAP`` times the
    own-scale is a sparse-leaf blow-up of the unbounded response function, and the
    fused pool damps it only by ``(1 - model_weight)`` — so a high-model-weight cell
    would ride it into a max-confidence bet. Clamp the mean before the pool; for
    SkewNormal rescale the scale by the same factor to hold the per-row CV (the scale
    co-explodes with the mean). Count families rely on the existing ``shape_ceiling``.
    """
    own = _own_scale(offer_df)
    if own is None:
        return
    model_ev = offer_df["Projection"].to_numpy(dtype=float)
    cap = _MODEL_EV_OWN_SCALE_CAP * np.clip(own, _OWN_SCALE_FLOOR, None)
    runaway = model_ev > cap
    if not runaway.any():
        return
    clamped = np.minimum(model_ev, cap)
    if dist == "SkewNormal" and "Model Sigma" in offer_df.columns:
        shrink = np.where(runaway, clamped / np.clip(model_ev, 1e-9, None), 1.0)
        offer_df["Model Sigma"] = offer_df["Model Sigma"].to_numpy(dtype=float) * shrink
    offer_df["Projection"] = clamped
    logger.warning(
        f"clamped {int(runaway.sum())} runaway model EV(s) toward "
        f"{_MODEL_EV_OWN_SCALE_CAP:g}x own-scale (max {float(np.nanmax(model_ev)):.1f})"
    )


def _drop_no_history_offers(offer_df: pd.DataFrame) -> pd.DataFrame:
    """Drop single-player offers with no informative own-scale history.

    A debutant or a position player forced onto a foreign stat line (a running back
    on a passing-yards line) has ~0 own-scale, so the model mean rests on nothing and
    :func:`_sanitize_model_ev` has no anchor. Player-vs-player combos carry no
    per-player own-scale and are kept.
    """
    own = _own_scale(offer_df)
    if own is None:
        return offer_df
    keep = (own >= _OWN_SCALE_MIN) | offer_df["Player"].str.contains("vs.", regex=False).to_numpy()
    return offer_df.loc[keep]


def _blend_with_book(
    offer_df: pd.DataFrame,
    dist: str,
    model_weight: float,
    cv: float,
    hist_gate: float,
    league: str,
    market: str,
    posthoc_slug: str = "none",
    posthoc_blob: dict | None = None,
) -> np.ndarray:
    model_ev = offer_df["Projection"].to_numpy()
    books_ev = offer_df["Market Projection"].fillna(offer_df["Projection"]).to_numpy()
    model_sd = _model_predictive_sd(offer_df, dist, model_ev)
    books_ev = _sanitize_book_ev(
        books_ev,
        offer_df["Line"].to_numpy(),
        model_ev,
        model_sd,
        f"{league} {market}",
    )
    zi = _zi_kwargs(offer_df, dist, hist_gate)
    if dist == "SkewNormal":
        # The book leg stays the symmetric constant-CV shape (fused_loc default). The WS2
        # settling experiment found the per-cell shaped book loses in the served blend OOS
        # (the model's per-row SkewNormal already carries the shape); the shaped book ships
        # only on the book-only fallback leg. See docs/handoffs/book_shape_ws2_handoff.md.
        base_mean, sigma_blend, skew_blend, gate_blend = fused_loc(
            model_weight,
            model_ev,
            books_ev,
            cv,
            "SkewNormal",
            sigma=_col_or_none(offer_df, "Model Sigma"),
            skew_alpha=_col_or_none(offer_df, "Model Skew"),
            **zi,
        )
        offer_df["Model Sigma"] = sigma_blend
        offer_df["Model Skew"] = skew_blend
    elif dist in ("NegBin", "ZINB"):
        r_blend, p_blend, gate_blend = fused_loc(
            model_weight,
            model_ev,
            books_ev,
            cv,
            "NegBin",
            r=_col_or_none(offer_df, "Model R"),
            **zi,
        )
        base_mean = r_blend * (1 - p_blend) / p_blend
        offer_df["Model R"] = r_blend
    elif dist == "DPO":
        # fused_loc's DPO log-pool returns the exact blended mean directly;
        # get_odds Newton-inverts mean -> mu internally.
        base_mean, phi_blend, gate_blend = fused_loc(
            model_weight,
            model_ev,
            books_ev,
            cv,
            "DPO",
            phi=_col_or_none(offer_df, "Model Phi"),
            **zi,
        )
        offer_df["Model Phi"] = phi_blend
    else:
        alpha_blend, beta_blend, gate_blend = fused_loc(
            model_weight,
            model_ev,
            books_ev,
            cv,
            "Gamma",
            alpha=_col_or_none(offer_df, "Model Alpha"),
            **zi,
        )
        base_mean = alpha_blend / beta_blend
        offer_df["Model Alpha"] = alpha_blend
    # The mean-stage corrector recalibrates the POOL, not its legs: a log opinion pool is
    # not mean-preserving, so a pre-fusion correction is multiplied straight back down by
    # rho^(1-w). Ranjan & Gneiting (2010) Thm 1 — recalibrate the combination. Mirrors
    # pipeline._step_calibrate_and_serve, which corrects the same object one stage later.
    base_mean = correct_fused_mean(posthoc_slug, posthoc_blob, base_mean, gate_blend)
    if gate_blend is not None:
        offer_df["Projection"] = (1 - gate_blend) * base_mean
        offer_df["Model Gate"] = gate_blend
    else:
        offer_df["Projection"] = base_mean
    return base_mean


def _serving_dispersion_cal(
    dispersion_cal: float, skew_cal: float, cell: str
) -> tuple[float, float]:
    """Neutralize a diverged dispersion fit at serve time; healthy values pass through.

    The skew shift resets with it: (c, s) come from the same joint fit, so a
    pinned c means s is not trustworthy either. A skew-only cell (c == 1.0,
    s != 0) is healthy and passes through.
    """
    if dispersion_cal <= _DIVERGED_DISPERSION_FLOOR or dispersion_cal >= _DIVERGED_DISPERSION_CEIL:
        logger.warning(
            f"{cell}: dispersion_cal {dispersion_cal:.4g} pinned at its fit bound — "
            "diverged fit, serving unscaled shape"
        )
        return 1.0, 0.0
    return dispersion_cal, skew_cal


def _dispersion_calibrate(
    offer_df: pd.DataFrame, dist: str, dispersion_cal: float, skew_cal: float
) -> None:
    # The skew shift is derived into loc downstream by get_odds (mean held fixed),
    # so it must land before _model_over_and_push reads Model Skew.
    # A skew-only cell (c == 1, s != 0) must still enter — guard on both knobs.
    if dispersion_cal == 1.0 and skew_cal == 0.0:
        return
    shape_col = {
        "NegBin": "Model R",
        "ZINB": "Model R",
        "DPO": "Model Phi",
        "Gamma": "Model Alpha",
        "ZAGamma": "Model Alpha",
        "SkewNormal": "Model Sigma",
    }.get(dist)
    if shape_col is None or shape_col not in offer_df.columns:
        return
    offer_df[shape_col] = offer_df[shape_col] * dispersion_cal
    if dist == "DPO":
        # phi is hard-bounded in-family; re-clip after scaling so a >1
        # dispersion_cal can't push it past the DP kernel's supported band.
        offer_df["Model Phi"] = np.minimum(offer_df["Model Phi"], _DP_PHI_CEILING)
    elif dist == "SkewNormal":
        offer_df["Model Skew"] = offer_df["Model Skew"] + skew_cal


def _model_over_and_push(offer_df: pd.DataFrame, dist: str, cv: float, step, base_mean):
    r = _col_or_none(offer_df, "Model R", dist in ("NegBin", "ZINB"))
    alpha = _col_or_none(offer_df, "Model Alpha", dist in ("Gamma", "ZAGamma"))
    phi = _col_or_none(offer_df, "Model Phi", dist == "DPO")
    sigma = _col_or_none(offer_df, "Model Sigma", dist == "SkewNormal")
    skew = _col_or_none(offer_df, "Model Skew", dist == "SkewNormal")
    gate = _col_or_none(offer_df, "Model Gate", dist in ("ZINB", "ZAGamma", "SkewNormal"))
    line = offer_df["Line"].to_numpy()
    if dist == "SkewNormal":
        raw_under = get_odds(
            line, base_mean, dist, cv=cv, step=step, sigma=sigma, skew_alpha=skew, gate=gate
        )
    else:
        raw_under = get_odds(
            line, base_mean, dist, cv, alpha=alpha, step=step, r=r, gate=gate, phi=phi
        )
    # Push prob for integer-line discrete markets (continuous dists return zero);
    # correlation.py uses it for the Underdog "push drops one leg" rule.
    push = get_push_prob(
        line, base_mean, dist, cv=cv, r=r, sigma=sigma, skew_alpha=skew, gate=gate, phi=phi
    )
    return 1 - raw_under, push


def model_prob(
    offers: list[dict],
    league: str,
    market: str,
    platform: str,
    stat_data,
    playerStats: pd.DataFrame,
) -> list[dict]:
    """Score a batch of offers with the trained distributional model.

    Loads the model pickle for ``(league, market)``, runs LightGBMLSS
    prediction on ``playerStats``, blends the model distribution with the
    bookmaker-implied distribution via ``fused_loc``, applies temperature
    scaling, and returns a list of offer dicts augmented with scoring
    columns (``Model EV``, ``Market EV``, ``Projection``, ``Bet``, ``Kelly``, etc.).

    Returns an empty list when no model file exists, when the pickle's strategy
    identity no longer validates, or when the joined DataFrame is empty after
    filtering.
    """
    # style: allow-complexity -- top-level serving orchestrator: load pickle, build params,
    # apply posthoc/dispersion/structural calibration, blend with book, emit scoring columns.
    dateMap = {x["Player"]: x["Date"] for x in offers}
    market = normalize_market(league, market, platform)
    filename = market_file_slug(league, market)
    filepath = model_pickle_path(league, market)
    # Withhold is normally enforced by meditate pruning the pickle; a missed meditate leaves
    # a stale pickle serving a withheld cell (two weeks of WNBA PRA, 2026-08). Fail closed.
    if stat_meta.get(league, {}).get(market, {}).get("shipped") == WITHHELD:
        if os.path.isfile(filepath):
            logger.warning(
                f"{filename} withheld but pickle on disk — skipping; run meditate to prune"
            )
        return []
    if not os.path.isfile(filepath):
        logger.warning(f"{filename} missing")
        return []

    offer_df = pd.DataFrame(offers)
    offer_df.index = offer_df.Player
    if "yards" in market:
        offer_df = offer_df.loc[
            (offer_df.Player.str.contains("vs.")) | (offer_df.Line > _MIN_YARDS_LINE)
        ]

    with open(filepath, "rb") as infile:
        filedict = pickle.load(infile)
    model_version = resolve_model_version(filepath, filedict)
    cv = filedict["cv"]
    model_weight = filedict["weight"]
    temperature = filedict.get("temperature", None)
    dispersion_cal, skew_cal = _serving_dispersion_cal(
        filedict.get("dispersion_cal", 1.0), filedict.get("skew_cal", 0.0), filename
    )
    shape_ceiling = filedict.get("shape_ceiling")
    dist = filedict["distribution"]
    try:
        strategy_identity = _resolve_serving_strategy(filedict, league, market)
    except ValueError as e:
        # process_offers scores every league and market inside one try/except per platform, so
        # letting this propagate drops the whole slate over one unservable cell. Skip the cell
        # instead, as the withheld and missing-pickle guards above already do.
        logger.warning(f"{filename} not served: {e}")
        return []
    structural_strategy = strategy_identity.structural_strategy
    step = filedict["step"]
    normalized = filedict.get("normalized", False)
    # Defaults keep legacy pickles (pre-P1, no offset_meta / target_normalization)
    # byte-identical; "target_strategy" is the pre-rename key.
    offset_meta = filedict.get("offset_meta")
    target_normalization = (
        filedict.get("target_normalization", "ratio_meanyr")
        if MODEL_STRATEGY_MODEL_KEY in filedict
        else filedict.get("target_normalization", filedict.get("target_strategy", "ratio_meanyr"))
    )
    posthoc_slug = filedict.get("posthoc", "none")
    posthoc_blob = filedict.get("posthoc_blob", None)
    pit_recal_blob = filedict.get("pit_recal_blob", None)
    structural_calibration = filedict.get("structural_calibration")
    two_part_candidate_state = (
        _two_part_candidate_state(structural_calibration)
        if structural_strategy == TWO_PART_STRATEGY
        else None
    )
    affine_candidate_calibration = (
        _affine_candidate_calibration(structural_calibration)
        if structural_strategy == AFFINE_STRATEGY
        else None
    )
    two_part_role_spec = (
        role_spec_for(league, market) if two_part_candidate_state is not None else None
    )
    if two_part_candidate_state is not None and (
        dist != "SkewNormal" or two_part_role_spec is None
    ):
        raise ValueError(
            "two-part candidate requires a SkewNormal cell with a registered role spec"
        )
    if two_part_candidate_state is not None and (dispersion_cal != 1.0 or skew_cal != 0.0):
        raise ValueError("two-part candidate requires its persisted raw fused shape")
    hist_gate = (
        stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma", "SkewNormal") else 0
    )

    prob_params = _build_prob_params(
        filedict,
        market,
        stat_data,
        playerStats,
        dist,
        normalized,
        target_normalization,
        structural_strategy,
    )
    prob_params.sort_index(inplace=True)
    playerStats.sort_index(inplace=True)
    if "Defense position" not in playerStats:
        playerStats["Defense position"] = playerStats["Defense avg"]

    evs, sds = _book_evs_for_players(
        offer_df, league, market, dist, cv, hist_gate, dateMap, stat_data, playerStats.index
    )
    playerStats["Market Projection"] = evs
    playerStats["Books STD"] = sds

    _decode_model_params(
        prob_params, dist, playerStats, hist_gate, offset_meta, target_normalization
    )

    offer_df = offer_df.join(playerStats).join(prob_params).reset_index(drop=True)
    offer_df = offer_df.loc[~offer_df[["Market Projection", "Projection"]].isna().all(axis=1)]
    offer_df = _drop_no_history_offers(offer_df)
    if affine_candidate_calibration is not None:
        position = pd.to_numeric(offer_df.get("Player position"), errors="coerce")
        supported = position.isin(_affine_supported_codes(affine_candidate_calibration))
        if (~supported).any():
            logger.warning(
                f"dropped {int((~supported).sum())} affine offer(s) outside "
                "the calibrated candidate support"
            )
            offer_df = offer_df.loc[supported]
    if offer_df.empty:
        return []

    offer_df["Market EV"] = _book_over_prob(
        offer_df, dist, cv, step, hist_gate or None, league, market
    )
    _clamp_shape_ceiling(offer_df, dist, shape_ceiling)
    _sanitize_model_ev(offer_df, dist)
    base_mean = _blend_with_book(
        offer_df, dist, model_weight, cv, hist_gate, league, market, posthoc_slug, posthoc_blob
    )

    # ZI dists: book reports the non-zero component EV; scale to marginal EV.
    if hist_gate and dist in ("ZINB", "ZAGamma"):
        offer_df["Market Projection"] = (1 - hist_gate) * offer_df["Market Projection"]
    _dispersion_calibrate(offer_df, dist, dispersion_cal, skew_cal)

    two_part_candidate_output = None
    if two_part_candidate_state is not None:
        two_part_candidate_output = _apply_two_part_candidate_distribution(
            offer_df,
            base_mean,
            *two_part_candidate_state,
            two_part_role_spec,
        )

    affine_candidate_over = None
    if affine_candidate_calibration is not None:
        base_mean, affine_candidate_over = _apply_affine_candidate_distribution(
            offer_df,
            base_mean,
            affine_candidate_calibration,
        )

    raw_over, push = _model_over_and_push(offer_df, dist, cv, step, base_mean)
    offer_df["Push Prob"] = np.asarray(push, dtype=float)

    # §6.1 Rung C reshapes the whole served CDF (over = 1 − g(1 − over_raw)) before the
    # binary-prob temperature/posthoc, matching the training-side warp + Gate-4 PIT.
    raw_over = _apply_cdf_recal_over(raw_over, pit_recal_blob)
    cal_over = apply_temperature(raw_over, temperature)
    cal_over = _apply_prob_posthoc(cal_over, posthoc_slug, posthoc_blob)
    if two_part_candidate_output is not None:
        cal_over = two_part_candidate_output.pooled_over
    if affine_candidate_over is not None:
        cal_over = affine_candidate_over
    offer_df["Model Under"] = 1 - cal_over
    offer_df["Model Over"] = 1 - offer_df["Model Under"]

    return _finalize_records(
        offer_df,
        league,
        platform,
        dist,
        cv,
        step,
        temperature,
        dispersion_cal,
        model_version,
        pit_recal_blob,
    )


def _has_serving_support(quote: TrainingQuote) -> bool:
    """Whether a quote rests on evidence independent of the platform offering the leg.

    A direct cohort quote needs a real sportsbook in it; a component sum already
    demanded one behind every component to exist at all. Everything else — synthetic
    anchors, ev-inversions, DFS-only cohorts — is the platform quoting itself back.
    """
    if quote.source == COMBO_SUM_SOURCE:
        return True
    real_books = sum(book not in DFS_PLATFORM_BOOKS for book in quote.books)
    return quote.source == "book_direct" and real_books >= _MIN_FALLBACK_REAL_BOOKS


def _servable_fallback_quotes(
    offer_df: pd.DataFrame,
    league: str,
    market: str,
    date_map: dict,
    stat_data,
    dist: str,
    cv: float,
) -> dict[str, TrainingQuote]:
    """Resolve per-player book quotes; keep only the ones with independent support.

    Mirrors the training-side consumer (``Stats.resolve_player_market_odds``): one
    modal-line cohort quote per player from :func:`resolve_training_quote`, then a
    second pass over whatever came back synthetic that prices the market as an honest
    component sum through :meth:`Stats.combo_quote`, anchored on the player's own
    offered line. A quote serves when a real sportsbook (non-DFS) sits in its
    same-line cohort, or when it is a component sum — whose own admission already
    demands a sportsbook behind every component. Pure ev-inversions and synthetic
    quotes never serve: a DFS platform reposting (or discounting) a line is not
    independent support.

    Only the simple ``combo_props`` sums take the second pass. Fantasy-score markets
    build a mean from up to 8 weighted components and stay no-served pending their own
    graded verdict; ``_COMBO_SERVE_BLOCKED`` carries the sums whose components are
    themselves unfit to quote.
    """
    weights = book_weights.get(league, {}).get(market, {})
    first_lines = offer_df["Line"].groupby(level=0).first()
    players_by_date: dict[str, list[str]] = {}
    for player in offer_df.index.unique():
        players_by_date.setdefault(date_map.get(player, ""), []).append(player)
    combo_servable = market in combo_props and (league, market) not in _COMBO_SERVE_BLOCKED

    quotes: dict[str, TrainingQuote] = {}
    for date, players in players_by_date.items():
        quote_inputs = archive.get_training_quote_inputs(league, market, date, players)
        unpriced = []
        for player in players:
            rows, legacy_line = quote_inputs.get(player, ([], None))
            quotes[player] = resolve_training_quote(
                rows,
                fallback_ev=None,
                legacy_line=legacy_line,
                # Serving has no Avg10 to mirror base.py's fallback anchor; the
                # player's own offered line anchors the inversion instead.
                fallback_line=max(float(first_lines[player]), 0.5),
                dist=dist,
                cv=cv,
                weights=weights,
            )
            if quotes[player].source in ("neutral_fallback", "model_fallback"):
                unpriced.append(player)
        if unpriced and combo_servable:
            lines = {player: float(first_lines[player]) for player in unpriced}
            quotes.update(stat_data.combo_quote(market, unpriced, date, None, lines=lines))

    return {player: quote for player, quote in quotes.items() if _has_serving_support(quote)}


def _price_offers_at_quotes(
    offer_df: pd.DataFrame,
    quotes: dict[str, TrainingQuote],
    league: str,
    market: str,
    dist: str,
    cv: float,
    step: float,
) -> None:
    """Set ``Market Projection`` and the ``Market EV`` over-probability in place.

    Every offered line for a player is decoded from the one quote-implied mean with
    the same fitted shape (``sigma``/``skew`` or ``phi``) used to invert it, so an
    offer at the quote line reproduces the cohort probability exactly and alternate
    lines price off the same distribution rather than a cross-line average.

    A component sum instead prices every line off the kernel's own CDF, which is the
    whole reason it exists: the composite cell's marginal family has one generic ``cv``
    and no component dispersions, and grading the two across ±1.5 sd offset lines put
    the sum ahead on 9 of 10 markets.
    """
    combo_cdfs = {p: q.under_prob_at for p, q in quotes.items() if q.source == COMBO_SUM_SOURCE}
    priced = {p: quote_pricing_params(q, league, market, dist, cv) for p, q in quotes.items()}
    offer_df["Market Projection"] = offer_df.index.map({p: v.mean for p, v in priced.items()})
    shape_kwargs = {}
    if dist == "SkewNormal":
        shape_kwargs = {
            "sigma": offer_df.index.map({p: v.sigma for p, v in priced.items()}).to_numpy(float),
            "skew_alpha": offer_df.index.map({p: v.skew for p, v in priced.items()}).to_numpy(
                float
            ),
        }
    elif dist == "DPO" and any(v.phi is not None for v in priced.values()):
        shape_kwargs = {
            "phi": offer_df.index.map({p: v.phi for p, v in priced.items()}).to_numpy(float)
        }
    elif dist in ("NegBin", "ZINB") and any(v.r is not None for v in priced.values()):
        shape_kwargs = {
            "r": offer_df.index.map({p: v.r for p, v in priced.items()}).to_numpy(float)
        }
    lines = offer_df["Line"].to_numpy(dtype=float)
    over = 1 - get_odds(
        lines,
        offer_df["Market Projection"].to_numpy(dtype=float),
        dist,
        cv=cv,
        step=step,
        **shape_kwargs,
    )
    if combo_cdfs:
        over = [
            1.0 - combo_cdfs[player](line) if player in combo_cdfs else marginal
            for player, line, marginal in zip(offer_df.index, lines, over, strict=True)
        ]
    offer_df["Market EV"] = over


def book_fallback_prob(
    offers: list[dict],
    league: str,
    market: str,
    platform: str,
    stat_data,
) -> list[dict]:
    """Score offers from the modal-line book cohort when no trained model exists.

    Routed to by :func:`process_offers` whenever model scoring would otherwise be
    empty (missing model pickle, or a model that matched no players). Resolves the
    same per-player quote the training matrix uses — :func:`resolve_training_quote`
    over the archive's latest per-book rows, native under-probabilities averaged
    within one modal-line cohort — then inverts that quote to a mean and prices
    every offered line under the same shape, so the leg still flows through
    correlation, parlay search, and the export with no claimed edge (``Model EV``
    mirrors ``Market EV``). Rows without independent support (no real sportsbook in
    the cohort and no combo consensus) are dropped, not served at a guessed price.
    Returns an empty list when the market is unknown to ``stat_meta`` (no
    distribution to devig with) or nothing is servable.
    """
    market = normalize_market(league, market, platform)
    dist, cv, hist_gate, step = _book_cell_params(league, market)
    if dist is None:
        return []
    gate_arg = hist_gate or None

    date_map = {x["Player"]: x["Date"] for x in offers}
    offer_df = pd.DataFrame(offers)
    offer_df.index = offer_df.Player
    if "yards" in market:
        offer_df = offer_df.loc[
            (offer_df.Player.str.contains("vs.")) | (offer_df.Line > _MIN_YARDS_LINE)
        ]
    if offer_df.empty:
        return []

    quotes = _servable_fallback_quotes(offer_df, league, market, date_map, stat_data, dist, cv)
    servable = offer_df.index.isin(list(quotes))
    if unservable := int(len(offer_df) - servable.sum()):
        logger.warning(
            f"{league} {market}: dropped {unservable} offer row(s) without a servable book quote"
        )
        offer_df = offer_df.loc[servable]
    if offer_df.empty:
        return []

    _price_offers_at_quotes(offer_df, quotes, league, market, dist, cv, step)

    playerStats = stat_data.get_stats(market, offers)
    if not playerStats.empty:
        playerStats = (
            playerStats[~playerStats.index.duplicated(keep="first")]
            .fillna(0)
            .infer_objects(copy=False)
        )
        offer_df = offer_df.join(playerStats)
    offer_df = offer_df.reset_index(drop=True)

    _base_ev = offer_df["Market Projection"].to_numpy()
    offer_df["Push Prob"] = np.asarray(
        get_push_prob(offer_df["Line"].to_numpy(), _base_ev, dist, cv=cv, gate=gate_arg),
        dtype=float,
    )
    if hist_gate:
        offer_df["Market Projection"] = (1 - hist_gate) * offer_df["Market Projection"]

    _over = np.asarray(offer_df["Market EV"], dtype=float)
    offer_df["Model Over"] = _over
    offer_df["Model Under"] = 1 - _over
    offer_df["Projection"] = offer_df["Market Projection"]

    return _finalize_records(
        offer_df, league, platform, dist, cv, step, None, 1.0, _BOOK_FALLBACK_VERSION
    )
