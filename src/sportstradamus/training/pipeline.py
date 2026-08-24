"""Per-market training pipeline: data loading, model fitting, calibration, diagnostics."""

import hashlib
import importlib.resources as pkg_resources
import json
import os
import pickle
import random
import warnings
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.Mixture import Mixture
from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
from lightgbmlss.distributions.ZINB import ZINB
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.utils import softmax_fn
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    roc_auc_score,
)

from sportstradamus import data
from sportstradamus.double_poisson import DoublePoisson
from sportstradamus.helpers import (
    GATE_PUBLISH_THRESHOLD,
    NONZERO_DENOM_GATE,
    apply_cdf_recal,
    apply_temperature,
    decode_predictive_mean,
    dp_crps,
    fused_loc,
    gamma_crps,
    get_ev,
    get_logger,
    get_odds,
    negbin_crps,
    set_model_start_values,
    skewnormal_loc_from_mean,
    stat_cv,
    stat_zi,
)
from sportstradamus.helpers.distributions import _DP_PHI_CEILING, _DP_PHI_FLOOR, _dp_mu_from_mean
from sportstradamus.helpers.io import market_file_slug, model_pickle_path
from sportstradamus.hurdle import HurdleZINB
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.skew_normal_centered import CenteredSkewNormal, _centered_to_direct
from sportstradamus.stats.model_dependencies import DEPENDENCY_NAMESPACE
from sportstradamus.training import baselines, calibration, posthoc
from sportstradamus.training.calibration import fit_book_weights
from sportstradamus.training.config import (
    load_distribution_config,
    save_book_shape_config,
    save_cv_std_config,
    save_zi_config,
)
from sportstradamus.training.data import trim_matrix
from sportstradamus.training.group_conditional_cdf import (
    AffineCalibrationFit,
    AffinePredictive,
    TwoPartCalibrationFit,
    affine_cdf_endpoints,
)
from sportstradamus.training.group_conditional_cdf._pipeline_steps_affine import (
    _step_apply_affine_groupcdf_candidate,
)
from sportstradamus.training.group_conditional_cdf._pipeline_steps_shared import (
    STRUCTURAL_ROW_FIELDS,
    _aligned_rows,
    _persist_structural_columns,
)
from sportstradamus.training.group_conditional_cdf._pipeline_steps_two_part import (
    _step_apply_two_part_groupcdf_candidate,
    pin_two_part_grouping,
)
from sportstradamus.training.hyperparams import _BoundedResponseFn, run_hyper_opt
from sportstradamus.training.lineage import (
    MATRIX_TRIM_SEED,
    validate_matrix_manifest,
    write_matrix_manifest,
)
from sportstradamus.training.matrix_audit import incomplete_provenance_rows
from sportstradamus.training.model_strategy import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_DETERMINISTIC_TRAIN,
    CAP_FULL_HPO,
    MODEL_STRATEGY_MODEL_KEY,
    CellContext,
    artifact_identity_columns,
    artifact_namespace,
    build_artifact_identity,
    distribution_class,
    get_strategy,
    validate_strategy_selection,
)
from sportstradamus.training.role_specs import role_spec_for
from sportstradamus.training.scorecard import (
    _DISPERSION_C_BOUNDS,
    _DISPERSION_SKEW_BOUNDS,
    _gate4_pit_ks_threshold,
    _randomized_pit_draws,
    _randomized_pit_ks,
    _served_sn_pit_ks,
    fit_skewnorm_dispersion_skew,
)
from sportstradamus.training.shap import compute_market_importance
from sportstradamus.training.ship_config import TARGET_NORM_NONE
from sportstradamus.training.structural_context import (
    build_affine_expert_context,
    build_two_part_context,
)
from sportstradamus.training.structural_strategies import (
    AFFINE_EXPERT_EXPERIMENTS,
    AFFINE_STRATEGY,
    TWO_PART_STRATEGY,
)

logger = get_logger(__name__)

# Equal-width bins for expected_calibration_error per Phase 3 §4.b.
_ECE_BINS = 10
# Avoid divide-by-zero when the bookmaker baseline Brier is degenerate.
_BRIER_SKILL_DENOM_FLOOR = 1e-9
# Probability clip used so log_loss / Brier never see exact 0 or 1.
_PROBA_CLIP = 1e-6
# Confidence cutoff for the mode-stats and diagnostic masks: only rows where
# the model's top-class probability exceeds this are counted in precision /
# accuracy / over% statistics.  Mirrors the live scoring path in
# prediction/scoring.py. Value from CLAUDE.md "Performance Table".
_MODE_CONFIDENCE_THRESHOLD: float = 0.54
# Minimum rows in an EV-conditioned, confidence-masked subset before its over-
# rate is reported; thinner slices give a noisy mean, so the diagnostic is NaN.
_MIN_DIAGNOSTIC_ROWS: int = 10
# Temporal train/test split fraction: earliest 70% of the matrix goes to
# training, latest 30% is held out for evaluation.  Temporal ordering (not
# random split) prevents look-ahead leakage of player form.
_TRAIN_FRACTION: float = 0.7
# Mean threshold separating SkewNormal (continuous, high-mean) from count
# distributions (NegBin/ZINB).  Stats with global_mean < 2 are integer-like
# enough that a count family fits better than SkewNormal.
_SKEWNORMAL_MEAN_THRESHOLD: float = 2.0
# Families a cell may pin in stat_meta.json `dist` to force its training branch; an unset / "auto"
# cell defers to the data-driven rule (_data_driven_dist). ZAGamma/Gamma are not produced by
# _step_select_distribution, so they are not forceable here. DPO and Mixture are force-only —
# no data-driven rule routes to them (WS-3 §6.6: zero-deflation count cells pin DPO; the
# boom/bust heavy-right-tail continuous cells pin Mixture — 2-component Gaussian).
_FORCEABLE_DISTS: frozenset[str] = frozenset({"SkewNormal", "ZINB", "NegBin", "DPO", "Mixture"})
# Two Gaussian components: the Mixture family exists to add ONE extra mode for the
# boom/bust upper tail; more components overfit the thin right tail the family targets.
_MIXTURE_COMPONENTS: int = 2
# Constrained-mixture guardrails (Hathaway-style scale constraint): component sigma is
# clamped to [floor, ceiling] x the normalized-label std at fit time. The floor removes
# the sigma->0 likelihood spike a Gaussian-mixture MLE is unbounded under (a component
# collapses onto single points); the ceiling stops the near-dead component's unused
# scale head exp-overflowing to inf, which the blend then propagates as NaN. Both sit
# far outside any honest fit.
_MIXTURE_SCALE_STD_FLOOR: float = 0.02
_MIXTURE_SCALE_STD_CEILING: float = 20.0
# LightGBM guardrails for the mixture fit: a near-dead component's rows carry ~zero
# hessian, so an unregularized Newton leaf step (-G/(H+lambda)) explodes on any leaf
# dominated by them — observed as loc heads at +-1e4 in normalized space. Floor the
# per-leaf hessian mass and the L2 term; both floors sit inside the HPO search range,
# so a searched corner is only ever tightened, never loosened.
_MIXTURE_MIN_CHILD_WEIGHT: float = 0.1
_MIXTURE_LAMBDA_L2_FLOOR: float = 1.0
# Constrained-SkewNormal guardrails (research/briefs/researcher_hpo_objective_alignment.md
# R1) — same Hathaway-style clamp as the Mixture above, but keyed to a ROBUST label scale
# (IQR/1.349): np.std runs up to 47x looser on ratio_meanyr cells whose near-zero MeanYr
# denominators fatten the normalized tail (brief finding 5). Ceiling 10: the corrected
# shape ledger separates converged fits (max 2.46x the outcome SD) from diverged ones
# (min 25x) with an empty band — 4x above the highest honest fit, 2.5x below the mildest
# divergence. Floor 0.02: the Kiefer-Wolfowitz sigma->0 spike is live on shipped NBA FGA
# (SN_Scale min 2.7e-6 vs median 4.53); same constant as the Mixture floor.
_SKEWNORMAL_SCALE_ROBUST_FLOOR: float = 0.02
_SKEWNORMAL_SCALE_ROBUST_CEILING: float = 10.0
# |alpha| clamp for the direct parametrization — parity with the centered branch's
# existing +-0.99 gamma1 tanh bound (implied |alpha| ~ 28.4); gamma1(30) = 0.9907 of the
# attainable 0.99527, so the cap costs <0.5% of the skewness range.
_SKEWNORMAL_ALPHA_BOUND: float = 30.0
# Normal-consistency divisor turning an IQR into a sigma estimate (IQR = 1.349 sigma).
_IQR_TO_SIGMA: float = 1.349
# Minimum coefficient of variation for the SkewNormal branch.  Prevents
# degenerate near-zero CV when all players have nearly identical outcomes.
_SKEWNORMAL_CV_FLOOR: float = 0.05
# Shape parameter cap for the NegBin / ZINB count branch.  Very large R values
# collapse NegBin toward Poisson and destabilize optimization.
_COUNT_BRANCH_R_CAP: int = 50
# Quantile of per-player NegBin R estimates used as the marginal shape prior.
# 95th-percentile trims outlier players without discarding the heavy tail.
_MARGINAL_SHAPE_QUANTILE: float = 0.95
# Floor on the marginal shape prior — avoids a degenerate shape_ceiling of ~0
# when the market has near-zero variance across all players.
_MARGINAL_SHAPE_FLOOR: float = 0.5
# Minimum nonzero-game count for a player to inform a cell's distribution-shape
# estimate. Per-league because season lengths differ; default for unlisted leagues.
_MIN_PLAYER_NONZERO_OBS: dict[str, int] = {"NBA": 60, "NFL": 10, "NHL": 60, "WNBA": 40, "MLB": 60}
_MIN_PLAYER_NONZERO_OBS_DEFAULT: int = 60
# Shape ceiling = marginal_shape * this multiplier.  2× gives the optimizer
# headroom to exceed the prior while preventing runaway over-dispersion.
_SHAPE_CEILING_MULTIPLIER: float = 2.0
# The ``"auto"`` sentinel shared by the loss search axes: it means "use the default for this
# context" (the per-family training loss here; the cell's stat_meta blending in the CLI), so a
# default run reproduces production. The Operation Ship 75 search sweeps nll ↔ crps off these.
LOSS_AUTO: str = "auto"
# Training loss exposed by ``meditate --dist-training-loss`` — crps for the SkewNormal continuous
# branch, nll for the count branch when ``auto``.
DIST_TRAINING_LOSS_CHOICES: tuple[str, ...] = (LOSS_AUTO, "nll", "crps")


def _resolve_loss_fn(family_default: str, override: str) -> str:
    """Shared by the training-loss and blending-loss axes so ``auto`` reproduces production for each context."""
    return family_default if override == LOSS_AUTO else override


# Fixed RNG seed for --deterministic runs (debug/eval only).
DETERMINISTIC_SEED = 1234

# Half of the uint64 identity-hash space goes to structural calibration. Unlike
# a positional RNG split, shared Player/Date rows keep their role as matrices grow.
_VALIDATION_HASH_THRESHOLD: int = 1 << 63

# Cross-fit folds for the holdout-blind search objective. Folds are player-disjoint:
# the ship gates bootstrap player-clustered, so a row-random fold would leak exactly
# the within-player correlation the gate corrects for. 5 matches posthoc's
# _PIT_RECAL_CV_FOLDS, so the PIT map's own inner cross-fit sees a familiar fold size.
_CALIBRATION_FOLDS: int = 5

# Split frames that rotate between the fit and apply roles when cross-fitting; each
# names a `{stem}_validation` / `{stem}_test` pair in the splits dict.
_ROTATED_SPLIT_STEMS: tuple[str, ...] = (
    "X",
    "y",
    "B",
    "players",
    "dates",
    "archived",
    "odds_synthetic",
    "quote_authenticity",
)

# Deterministic-mode model pickles live OUTSIDE the installed package tree so
# the research harness can iterate on them without polluting the production
# model dir (`src/sportstradamus/data/models/`). Resolved off __file__ so the
# path is stable regardless of cwd. parents[3] of
# src/sportstradamus/training/pipeline.py is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DETERMINISTIC_MODEL_ROOT = _REPO_ROOT / "research" / "models" / "deterministic"

# P0.5 deterministic-mode hyperparameters. Replaces the Optuna search when
# --deterministic is set. Deliberately small/fast: the goal is bit-identical
# re-runs for the eval harness, NOT model quality. opt_rounds is read as
# num_boost_round by model.train(...).
DETERMINISTIC_FIXED_PARAMS = {
    "opt_rounds": 30,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "min_child_samples": 50,
    "min_child_weight": 1e-3,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "path_smooth": 0.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_depth": -1,
    "max_bin": 127,
    "num_threads": 8,  # --deterministic overrides to 1; multi-thread histogram reductions are not bit-reproducible
    "feature_pre_filter": False,
}


def seed_everything(seed: int) -> dict[str, int | bool]:
    """Pin Python/NumPy/Torch RNGs and return LightGBM determinism kwargs.

    DEBUGGING / OFFLINE-EVAL USE ONLY. Never publish a model produced while
    these knobs are active — fixed seeds + fixed hyperparameters cripple model
    quality on purpose; the point is bit-identical re-runs, not accuracy.

    Args:
        seed: RNG seed applied to ``random``, ``numpy``, and ``torch``. Also
            used as the value for all three LightGBM seed params.

    Returns:
        LightGBM training params to merge into the params dict: ``seed``,
        ``bagging_seed``, ``feature_fraction_seed`` (all == ``seed``),
        ``deterministic`` (True), ``force_row_wise`` (True), ``num_threads``
        (1 -- overrides the default 8 so histogram reductions stay bit-reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(
        True
    )  # process-global; later non-deterministic torch ops in-process will raise
    return {
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,  # multi-thread histogram reductions are not bit-reproducible
    }


def fit_lss_model(
    dist_obj,
    dist: str,
    X_train: pd.DataFrame,
    y_train_labels: np.ndarray,
    params: dict,
    *,
    normalized: bool,
    shape_ceiling: float,
    seed: int | None = None,
    offset_mode: bool = False,
    sn_param: str = "direct",
) -> LightGBMLSS:
    """Build + train a LightGBMLSS model. Pure (no disk writes).

    When ``seed`` is not None, RNGs are pinned and LightGBM determinism kwargs
    are merged into ``params`` (DEBUG/eval only — never a production model).
    Builds its own ``lgb.Dataset`` from ``X_train``/``y_train_labels``; callers do not pre-construct one.

    Args:
        dist_obj: Distribution object (already shape-bounded by the caller).
        dist: Distribution name string (e.g. ``"NegBin"``, ``"Gamma"``).
        X_train: Training feature matrix.
        y_train_labels: Training target labels for ``lgb.Dataset``.
        params: LightGBM training params; ``opt_rounds`` is read as
            ``num_boost_round``.
        normalized: Whether start values are computed in normalized space.
        shape_ceiling: Upper bound on the distribution shape parameter.
        seed: If not None, pin RNGs and merge determinism kwargs into
            ``params`` (DEBUG/offline-eval only).
        offset_mode: SkewNormal-only — targets are an additive centered
            residual. Forwarded to ``set_model_start_values``; ignored for
            other distributions.
        sn_param: SkewNormal-only head parametrization for start-value
            seeding — ``"direct"`` or ``"centered"``. Must match
            ``dist_obj``'s parametrization; forwarded to
            ``set_model_start_values`` (no-op for other distributions).

    Returns:
        A trained ``LightGBMLSS`` model.
    """
    if seed is not None:
        params = {**params, **seed_everything(seed)}
    if dist == "Mixture":
        params = {
            **params,
            "min_child_weight": max(params.get("min_child_weight", 0.0), _MIXTURE_MIN_CHILD_WEIGHT),
            "lambda_l2": max(params.get("lambda_l2", 0.0), _MIXTURE_LAMBDA_L2_FLOOR),
        }
    dtrain = lgb.Dataset(X_train, label=y_train_labels)
    model = LightGBMLSS(dist_obj)
    set_model_start_values(
        model,
        dist,
        X_train,
        shape_ceiling=shape_ceiling,
        normalized=normalized,
        offset_mode=offset_mode,
        sn_param=sn_param,
    )
    model.train(params, dtrain, num_boost_round=params["opt_rounds"])
    return model


def predict_lss_params(
    model: LightGBMLSS,
    dist: str,
    X: pd.DataFrame,
    *,
    normalized: bool,
    offset_mode: bool = False,
    sn_param: str = "direct",
) -> pd.DataFrame:
    """Predict raw distribution parameters for X, preserving its index.

    Safe to call on the same X that was just fit — ``set_model_start_values`` is idempotent.

    Args:
        model: A trained ``LightGBMLSS`` model.
        dist: Distribution name string.
        X: Feature matrix to predict on.
        normalized: Whether start values are computed in normalized space.
        offset_mode: SkewNormal-only — targets are an additive centered
            residual. Forwarded to ``set_model_start_values``; ignored for
            other distributions.
        sn_param: SkewNormal-only head parametrization — ``"direct"`` or
            ``"centered"``. Must match how the model was trained; the seeds
            are per-row predict-time margins, so a mismatch shifts the skew
            head. Forwarded to ``set_model_start_values``.

    Returns:
        DataFrame of raw predicted distribution parameters, indexed like ``X``.
    """
    set_model_start_values(
        model, dist, X, normalized=normalized, offset_mode=offset_mode, sn_param=sn_param
    )
    preds = model.predict(X, pred_type="parameters")
    preds.index = X.index
    return preds


def predict_cv_oof_params(
    cvbooster,
    folds,
    opt_rounds: int,
    X: pd.DataFrame,
    start_values: np.ndarray,
    dist_obj,
) -> pd.DataFrame:
    """Pooled out-of-fold raw distribution parameters from ``model.cv``'s fold boosters.

    The raw-booster equivalent of :func:`predict_lss_params` for a ``CVBooster``: each fold's
    held-out rows are predicted by that fold's booster capped at ``num_iteration=opt_rounds``,
    the per-row predict-time margins are added, and the distribution's response functions are
    applied in ``param_dict`` order — exactly what ``LightGBMLSS.predict`` does via
    ``dist.predict_dist``, which exposes no iteration cap (why this cannot just call it). Same
    column layout and float32 dtype as :func:`predict_lss_params`; the frame is indexed by the
    pooled held-out rows in fold order.

    Args:
        cvbooster: ``CVBooster`` from ``model.cv(..., return_cvbooster=True)``;
            ``boosters[k]`` pairs with ``folds[k]``.
        folds: The explicit ``(train_idx, test_idx)`` positional index pairs the cv ran on.
        opt_rounds: Boosting rounds at the trial's CV-loss argmin.
        X: The full training frame the cv ``Dataset`` was built from.
        start_values: Per-row raw-space start values for ``X`` — the array
            ``set_model_start_values`` computes, shape ``(len(X), n_dist_param)``.
        dist_obj: Distribution object whose ``param_dict`` response functions apply
            (including any ``_BoundedResponseFn`` shape clamp already installed on it).
    """
    frames = [
        _response_param_frame(
            booster.predict(X.iloc[test_idx], raw_score=True, num_iteration=opt_rounds),
            start_values[test_idx],
            dist_obj,
            X.iloc[test_idx].index,
        )
        for booster, (_, test_idx) in zip(cvbooster.boosters, folds, strict=True)
    ]
    return pd.concat(frames)


def _response_param_frame(raw_margins, start_values: np.ndarray, dist_obj, index) -> pd.DataFrame:
    """Raw booster margins + start values → response-space param frame for one row block."""
    predt = torch.tensor(raw_margins, dtype=torch.float32).reshape(-1, dist_obj.n_dist_param)
    init = torch.tensor(start_values, dtype=torch.float32)
    params = np.concatenate(
        [
            response_fn(predt[:, i].reshape(-1, 1) + init[:, i].reshape(-1, 1)).numpy()
            for i, response_fn in enumerate(dist_obj.param_dict.values())
        ],
        axis=1,
    )
    frame = pd.DataFrame(params, columns=list(dist_obj.param_dict.keys()), index=index)
    if isinstance(dist_obj, CenteredSkewNormal):
        # Mirror CenteredSkewNormal.predict_dist: downstream consumers only ever see the
        # direct (loc, scale, alpha) frame.
        loc, scale, alpha = _centered_to_direct(
            frame["mean"].to_numpy(), frame["sd"].to_numpy(), frame["gamma1"].to_numpy(), np
        )
        frame = pd.DataFrame(
            {"loc": loc, "scale": scale, "alpha": alpha}, index=index, dtype=np.float32
        )
    return frame


def predict_cv_ensemble_params(
    cvbooster, opt_rounds: int, X: pd.DataFrame, start_values: np.ndarray, dist_obj
) -> pd.DataFrame:
    """Ensemble raw params for rows no fold booster trained on (e.g. the validation split).

    Averages the fold boosters' raw margins at ``num_iteration=opt_rounds`` — the standard
    cv-ensemble predictor — then applies start values and response functions exactly as
    :func:`predict_cv_oof_params` does for held-out fold rows.
    """
    raw = np.mean(
        [b.predict(X, raw_score=True, num_iteration=opt_rounds) for b in cvbooster.boosters],
        axis=0,
    )
    return _response_param_frame(raw, start_values, dist_obj, X.index)


def fit_predict_params(
    dist_obj,
    dist: str,
    X_train: pd.DataFrame,
    y_train_labels: np.ndarray,
    X_predict: pd.DataFrame,
    params: dict,
    *,
    normalized: bool,
    shape_ceiling: float,
    seed: int | None = None,
    offset_mode: bool = False,
    sn_param: str = "direct",
) -> pd.DataFrame:
    """Fit one LightGBMLSS model and return raw predicted params for X_predict.

    Pure: no disk writes, no Optuna. When ``seed`` is not None, RNGs are pinned
    and LightGBM determinism kwargs are merged into ``params`` (DEBUG/eval only).

    Args:
        dist_obj: Distribution object (already shape-bounded by the caller).
        dist: Distribution name string.
        X_train: Training feature matrix.
        y_train_labels: Training target labels for ``lgb.Dataset``.
        X_predict: Feature matrix to predict on after fitting.
        params: LightGBM training params; ``opt_rounds`` is read as
            ``num_boost_round``.
        normalized: Whether start values are computed in normalized space.
        shape_ceiling: Upper bound on the distribution shape parameter.
        seed: If not None, pin RNGs and merge determinism kwargs into
            ``params`` (DEBUG/offline-eval only).
        offset_mode: SkewNormal-only — targets are an additive centered
            residual. Forwarded to ``set_model_start_values``; ignored for
            other distributions.
        sn_param: SkewNormal-only head parametrization — ``"direct"`` or
            ``"centered"``. Forwarded to both the fit and predict seeding.

    Returns:
        DataFrame of raw predicted distribution parameters for ``X_predict``.
    """
    model = fit_lss_model(
        dist_obj,
        dist,
        X_train,
        y_train_labels,
        params,
        normalized=normalized,
        shape_ceiling=shape_ceiling,
        seed=seed,
        offset_mode=offset_mode,
        sn_param=sn_param,
    )
    return predict_lss_params(
        model, dist, X_predict, normalized=normalized, offset_mode=offset_mode, sn_param=sn_param
    )


def fit_hurdle_model(
    X_train: pd.DataFrame,
    y_train_labels: np.ndarray,
    params: dict,
    *,
    shape_ceiling: float | None = None,
    seed: int | None = None,
) -> HurdleZINB:
    """Build + train a HurdleZINB model. Pure (no disk writes).

    Parallel to ``fit_lss_model`` for the ZINB-hurdle path. When ``seed`` is
    not None, ``HurdleZINB.fit`` pins RNGs internally — caller doesn't merge
    determinism kwargs into ``params``.

    Args:
        X_train: Training feature matrix.
        y_train_labels: Training target labels (may contain zeros).
        params: LightGBM training params; ``opt_rounds`` is read as
            ``num_boost_round``.
        shape_ceiling: Upper bound on NegBin ``total_count`` shape.
        seed: If not None, pin RNGs for bit-reproducible training (DEBUG /
            offline-eval only).

    Returns:
        A trained ``HurdleZINB``.
    """
    rounds = int(params["opt_rounds"])
    model = HurdleZINB()
    model.fit(
        X_train, y_train_labels, hp=params, rounds=rounds, shape_ceiling=shape_ceiling, seed=seed
    )
    return model


def predict_hurdle_params(model: HurdleZINB, X: pd.DataFrame) -> pd.DataFrame:
    """Predict ZINB-compatible parameters from a HurdleZINB, preserving index.

    Args:
        model: A fitted ``HurdleZINB``.
        X: Feature matrix to predict on.

    Returns:
        DataFrame with columns ``["total_count", "probs", "gate"]``,
        indexed like ``X``.
    """
    preds = model.predict(X, pred_type="parameters")
    preds.index = X.index
    return preds


def fit_predict_hurdle_params(
    X_train: pd.DataFrame,
    y_train_labels: np.ndarray,
    X_predict: pd.DataFrame,
    params: dict,
    *,
    shape_ceiling: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Fit one HurdleZINB and return its predicted params for X_predict.

    Convenience wrapper mirroring ``fit_predict_params`` for the hurdle path.

    Args:
        X_train: Training feature matrix.
        y_train_labels: Training target labels.
        X_predict: Feature matrix to predict on after fitting.
        params: LightGBM training params; ``opt_rounds`` is read as rounds.
        shape_ceiling: Upper bound on NegBin ``total_count`` shape.
        seed: If not None, pin RNGs (DEBUG/eval only).

    Returns:
        DataFrame of predicted distribution parameters for ``X_predict``.
    """
    model = fit_hurdle_model(
        X_train,
        y_train_labels,
        params,
        shape_ceiling=shape_ceiling,
        seed=seed,
    )
    return predict_hurdle_params(model, X_predict)


def _expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = _ECE_BINS) -> float:
    """10-bin equal-width ECE: weighted |avg_pred - avg_actual| across bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    total = len(probs)
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(float(probs[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _compute_metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Raw classification metrics for binary over/under predictions."""
    probs = np.clip(np.asarray(probs, dtype=float), _PROBA_CLIP, 1 - _PROBA_CLIP)
    y = np.asarray(y).astype(int)
    pred = (probs > 0.5).astype(int)
    n_classes = len(np.unique(y))
    over_n = int((pred == 1).sum())
    under_n = int((pred == 0).sum())
    return {
        "brier_score": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probs)) if n_classes > 1 else float("nan"),
        "expected_calibration_error": _expected_calibration_error(probs, y),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_over": (
            float(precision_score(y, pred, pos_label=1, zero_division=0))
            if over_n
            else float("nan")
        ),
        "precision_under": (
            float(precision_score(y, pred, pos_label=0, zero_division=0))
            if under_n
            else float("nan")
        ),
        "predicted_over_rate": float(pred.mean()),
        "empirical_over_rate": float(y.mean()),
        "prediction_std": float(probs.std()),
        "nll": float(log_loss(y, probs, labels=[0, 1])),
    }


def _step_init_market(
    league: str,
    market: str,
    stat_data,
    archive,
    *,
    deterministic: bool,
    full_rebuild: bool = False,
    isolated: bool = False,
) -> dict:
    """Validate, load distribution config + book weights + existing pickle.

    Args:
        league: League slug.
        market: Market name.
        stat_data: League-specific ``Stats`` instance.
        archive: ``Archive`` singleton (passed through to book-weight fitting).
        deterministic: When True, skip the archive-hitting book-weight fit and
            the config write — ``train_market`` never consumes
            ``init["book_weights"]``, and a deterministic run must not mutate
            production config.

    Returns:
        Dict with: ``filedict`` (loaded model state or {}), ``dist`` (existing
        distribution name or None), ``cv`` (existing coefficient of variation),
        ``step`` (existing rounding step), ``need_model`` (True if no pickle
        exists), ``book_weights`` (full book-weights dict written to disk).
    """
    stat_dist = load_distribution_config()
    stat_dist.setdefault(league, {})
    stat_zi.setdefault(league, {})

    if os.path.isfile(pkg_resources.files(data) / "config" / "book_weights.json"):
        with open(pkg_resources.files(data) / "config" / "book_weights.json") as infile:
            book_weights = json.load(infile)
    else:
        book_weights = {}

    if not deterministic and not full_rebuild and not isolated:
        book_weights.setdefault(league, {}).setdefault(market, {})
        book_weights[league][market] = fit_book_weights(
            league, market, stat_data, archive, book_weights
        )

        with open(pkg_resources.files(data) / "config" / "book_weights.json", "w") as outfile:
            json.dump(book_weights, outfile, indent=4)

    filename = market_file_slug(league, market)
    filepath = model_pickle_path(league, market)
    need_model = True
    if full_rebuild or isolated:
        filedict = {}
        dist = stat_dist[league].get(market)
        cv = stat_cv[league].get(market, 1)
        step = None
    elif os.path.isfile(filepath):
        with open(filepath, "rb") as infile:
            filedict = pickle.load(infile)
            dist = filedict["distribution"]
            cv = filedict["cv"]
            step = filedict["step"]
        need_model = False
    else:
        filedict = {}
        dist = None
        cv = stat_cv[league].get(market, 1)
        step = None

    logger.info("Training %s - %s", league, market)
    cv = stat_cv[league].get(market, 1)
    return {
        "filedict": filedict,
        "dist": dist,
        "cv": cv,
        "step": step,
        "need_model": need_model,
        "book_weights": book_weights,
        "filename": filename,
    }


def _step_load_matrix(
    filename: str,
    league_start_date,
    stat_data,
    league: str,
    market: str,
    *,
    deterministic: bool,
    force: bool,
    need_model: bool,
    full_rebuild: bool = False,
    matrix_output: Path | None = None,
    dependency_root: Path | None = None,
    dependency_namespace: str = DEPENDENCY_NAMESPACE,
    matrix_input: Path | None = None,
) -> tuple[pd.DataFrame, object] | None:
    """Load cached training parquet, fetch new rows, concat. Returns None on early-exit.

    Args:
        filename: Slugified market identifier (used to derive parquet path).
        league_start_date: Earliest cutoff date for training rows.
        stat_data: ``Stats`` instance (provides ``get_training_matrix``).
        league: League slug (used only for the no-data message).
        market: Market name.
        deterministic: If True, freeze input to cached parquet only.
        force: If True, train even when no new rows arrived.
        need_model: True if no existing model pickle was found.

    Returns:
        ``(M, training_data_path)`` tuple where ``M`` is the combined matrix,
        or ``None`` when there is nothing to train on.
    """
    # style: allow-complexity — four mutually exclusive matrix sources (frozen input, full
    # rebuild, incremental cache, cold start) then one shared assemble-and-validate tail.
    canonical_path = Path(str(pkg_resources.files(data) / f"training_data/{filename}.parquet"))
    if matrix_input is not None:
        filepath = matrix_input
        if not filepath.is_file():
            raise FileNotFoundError(f"frozen matrix input does not exist: {filepath}")
        M = pd.read_parquet(filepath)
        validate_matrix_manifest(filepath, M)
        cutoff_date = league_start_date
        stat_data.snapshot_only_rebuild = True
    elif full_rebuild:
        if league == "NFL":
            from sportstradamus.stats import nfl_fp_team_weekly, nfl_fp_weekly

            nfl_fp_weekly.enable_snapshot_cache()
            nfl_fp_team_weekly.enable_snapshot_cache()
        if matrix_output is None:
            raise ValueError("full rebuild requires a quarantine matrix output directory")
        canonical_root = canonical_path.parent.resolve()
        output_root = matrix_output.resolve()
        if output_root == canonical_root or canonical_root in output_root.parents:
            raise ValueError("full rebuild output must be outside canonical training_data")
        filepath = matrix_output / f"{filename}.parquet"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        M = pd.DataFrame()
        cutoff_date = league_start_date
        stat_data.model_dependency_root = dependency_root or Path(
            str(pkg_resources.files(data) / "model_dependencies")
        )
        stat_data.model_dependency_namespace = dependency_namespace
        stat_data.snapshot_only_rebuild = True
    elif os.path.isfile(canonical_path):
        filepath = canonical_path
        M = pd.read_parquet(filepath)
        cutoff_date = pd.to_datetime(M["Date"]).max().date()
        M = M.loc[
            (pd.to_datetime(M.Date).dt.date <= cutoff_date)
            & (pd.to_datetime(M.Date).dt.date > league_start_date)
        ]
    else:
        filepath = canonical_path
        cutoff_date = league_start_date
        M = pd.DataFrame()

    new_M = (
        pd.DataFrame()
        if deterministic or matrix_input is not None
        else stat_data.get_training_matrix(market, cutoff_date)
    )

    if new_M.empty and not force and not need_model:
        return None

    M = pd.concat([M, new_M], ignore_index=True)
    if M.empty:
        logger.warning("  No usable training data for %s %s, skipping", league, market)
        return None
    M.Date = pd.to_datetime(M.Date, format="mixed")
    if "Player" in M.columns:
        M = M.drop_duplicates(subset=["Player", "Date"], keep="last")
    unpriced = incomplete_provenance_rows(M)
    if unpriced:
        # Raise before the caller persists: cached rows predating the provenance block concat
        # against fresh rows carrying it, and pd.concat back-fills all of history with NaN. Any
        # reader would have to guess, and _split_quote_authenticity_mask's legacy fallback guesses
        # "authentic", letting synthetic quotes buy book evidence.
        raise ValueError(
            f"{league} {market}: {unpriced} of {len(M)} rows carry a half-populated odds-provenance "
            "block, so the matrix mixes two incompatible eras. Repair it with "
            "`python -m sportstradamus.scripts.inject_backfilled_odds "
            f'--league {league} --markets "{market}"`, which re-resolves the block from the '
            "archive through the training join's own resolver, then retrain."
        )
    return M, filepath


def _step_synthesize_odds(
    M: pd.DataFrame, league: str, market: str, dist: str | None, cv: float
) -> tuple[pd.DataFrame, float]:
    """Fill missing/zero Odds rows with synthesized values from EV (or 0.5 fallback).

    The historical-zero gate is consulted only when ``dist`` is already a
    zero-inflated kind (``ZINB``/``ZAGamma``); first-time training uses 0.

    Args:
        M: The combined training matrix (mutated in place via ``.loc``).
        league: League slug (for ``stat_zi`` lookup).
        market: Market name (for ``stat_zi`` lookup).
        dist: Existing distribution name from the pickle, if any.
        cv: Coefficient of variation used to invert the EV→odds mapping.

    Returns:
        ``(M, step)`` where ``step`` is the smallest positive gap between
        unique ``Result`` values (the rounding granularity of the stat).
    """
    step = M["Result"].drop_duplicates().sort_values().diff().min()
    _prep_gate = stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0
    # Synthetic odds are a cv-based book stand-in for rows lacking a real line, generated
    # BEFORE any model is fit — so there are no fitted mixture params to pass. Route Mixture
    # through the SkewNormal cv-fallback (get_odds derives sigma=ev*cv, skew=0): a plain
    # continuous prior, which is all a fabricated line needs.
    synth_dist = "SkewNormal" if dist == "Mixture" else dist
    synthetic_mask = M.Odds.isna() | (M.Odds == 0)
    for i, row in M.loc[synthetic_mask].iterrows():
        if np.isnan(row["EV"]) or row["EV"] <= 0:
            M.loc[i, "Odds"] = 0.5
            M.loc[i, "EV"] = get_ev(
                M.loc[i, "Line"], 0.5, cv=cv, dist=synth_dist, gate=_prep_gate or None
            )
            M.loc[i, "Odds_synthetic"] = True
            if "QuoteSource" in M.columns:
                M.loc[i, "QuoteSource"] = "neutral_fallback"
                M.loc[i, "QuoteAuthenticity"] = "synthetic"
                M.loc[i, "QuoteSyntheticReason"] = "no_usable_book_probability"
                M.loc[i, "Archived"] = False
        else:
            M.loc[i, "Odds"] = 1 - get_odds(
                row["Line"], row["EV"], synth_dist, cv=cv, step=step, gate=_prep_gate or None
            )
            M.loc[i, "Odds_synthetic"] = True
            if "QuoteSource" in M.columns:
                M.loc[i, "QuoteSource"] = "pipeline_ev_inversion"
                M.loc[i, "QuoteAuthenticity"] = "derived"
                M.loc[i, "QuoteSyntheticReason"] = "ev_inversion"
                M.loc[i, "Archived"] = False
    return M, step


_PRETRIM_LINE_COLUMN = "__PreTrimLine"


def _reconcile_clipped_neutral_quotes(
    M: pd.DataFrame,
    *,
    league: str,
    market: str,
    dist: str | None,
    cv: float,
) -> pd.DataFrame:
    """Keep neutral synthetic quote tuples coherent after line clipping.

    ``trim_matrix`` may clip a fallback line to the observed market range. A
    neutral quote has no bookmaker probability or EV to preserve, so its EV
    must be regenerated from the clipped line instead of retaining the
    pre-clip value under misleading neutral provenance.
    """
    if _PRETRIM_LINE_COLUMN not in M:
        return M
    changed_line = ~np.isclose(
        pd.to_numeric(M["Line"], errors="coerce"),
        pd.to_numeric(M[_PRETRIM_LINE_COLUMN], errors="coerce"),
        equal_nan=True,
    )
    if {"QuoteSource", "QuoteAuthenticity"} <= set(M.columns):
        neutral = (
            changed_line
            & M["QuoteAuthenticity"].eq("synthetic")
            & M["QuoteSource"].isin(("neutral_fallback", "model_fallback"))
        )
        if neutral.any():
            prep_gate = stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0
            synth_dist = "SkewNormal" if dist == "Mixture" else dist
            M.loc[neutral, "Odds"] = 0.5
            M.loc[neutral, "EV"] = [
                get_ev(line, 0.5, cv=cv, dist=synth_dist, gate=prep_gate or None)
                for line in M.loc[neutral, "Line"]
            ]
    return M.drop(columns=_PRETRIM_LINE_COLUMN)


def _step_persist_matrix_and_comps(
    M: pd.DataFrame,
    filepath,
    stat_data,
    *,
    deterministic: bool,
    full_rebuild: bool = False,
    immutable_input: bool = False,
    league: str | None = None,
    market: str | None = None,
    dist: str | None = None,
    cv: float = 1.0,
) -> pd.DataFrame:
    """Trim the matrix, write parquet, save comps. Deterministic mode skips I/O."""
    if immutable_input:
        return M
    if _PRETRIM_LINE_COLUMN in M:
        raise ValueError(f"reserved matrix column is present: {_PRETRIM_LINE_COLUMN}")
    M[_PRETRIM_LINE_COLUMN] = M["Line"]
    M = trim_matrix(M, 15000, seed=MATRIX_TRIM_SEED)
    if league is not None and market is not None:
        M = _reconcile_clipped_neutral_quotes(
            M,
            league=league,
            market=market,
            dist=dist,
            cv=cv,
        )
    else:
        M = M.drop(columns=_PRETRIM_LINE_COLUMN)
    if full_rebuild:
        M = M.reindex(sorted(M.columns), axis=1)
    if not deterministic:
        M.to_parquet(filepath, compression="zstd", index=True)
        if not full_rebuild:
            stat_data.save_comps()
    return M


# Must survive pruning even when constant: `set_model_start_values`
# (helpers/distributions.py) reads these unconditionally. Canonical
# failure: NBA MIN has ZeroYr ≡ 0 (gamelog excludes DNPs → minutes
# never zero). LightGBM cost of carrying a constant feature is negligible.
_SEEDING_REQUIRED_COLUMNS: tuple[str, ...] = ("MeanYr", "STDYr", "ZeroYr")


def _prune_uninformative_features(X_train: pd.DataFrame, categorical_cols: list[str]) -> list[str]:
    """Return the subset of ``X_train.columns`` LightGBM can split on.

    A column is dropped when ``X_train`` shows it is entirely NaN or has
    fewer than two distinct non-NaN values (zero variance). Exact ``overall``
    snapshot aliases are also dropped when the unsuffixed canonical column
    exists with identical train values. Such columns cannot add a split that
    the retained data do not already provide. Removing only the explicit alias
    preserves every other feature and its feature-subsampling weight.

    Categorical columns and ``_SEEDING_REQUIRED_COLUMNS`` are always kept.
    LightGBM treats categoricals as a special split type, and the seeding
    columns are consumed unconditionally by ``set_model_start_values``
    regardless of whether they have splittable variance.
    """
    keep: list[str] = []
    for col in X_train.columns:
        if col in categorical_cols or col in _SEEDING_REQUIRED_COLUMNS:
            keep.append(col)
            continue
        s = X_train[col]
        if s.isna().all():
            continue
        if s.nunique(dropna=True) < 2:
            continue
        canonical = col.replace("_overall_", "_")
        if canonical != col and canonical in X_train and s.equals(X_train[canonical]):
            continue
        keep.append(col)
    return keep


def _step_build_splits(
    M: pd.DataFrame,
    stat_data,
    market: str,
    target_normalization: str = "ratio_meanyr",
    structural_strategy: str = BASE_STRUCTURAL_STRATEGY,
    *,
    holdout_blind: bool = False,
) -> dict:
    """Build feature matrix, temporal 70/30 split, then 50/50 test/validation.

    Args:
        M: Trimmed training matrix.
        stat_data: Stats instance (provides ``get_stat_columns``).
        market: Market name.
        target_normalization: Normalization slug. ``ratio_projvol`` needs its
            projected-volume denominator carried into ``X`` (see below).
        holdout_blind: Drop the held-out rows and evaluate on the validation
            frame instead — the search mode; see ``training/cli.py``.

    Returns:
        Dict with: ``X``, ``y``, ``X_train``, ``X_test``, ``X_validation``,
        ``y_train``, ``y_test``, ``y_validation``, ``B_train``, ``B_test``,
        ``B_validation``, ``y_train_labels``.
    """
    # style: allow-complexity -- orchestrates matrix build, temporal split, feature prune, and role-column protection
    y = M[["Result"]]
    # ``reindex`` over ``M[cols]`` so a stale cached parquet that hasn't been
    # regenerated since the last ``feature_filter.json`` schema addition fills
    # the new columns with NaN instead of raising KeyError. LightGBM treats
    # NaN as a missing-value category deterministically, so the deterministic
    # gate (``test_deterministic_mode_hurdle_is_bit_reproducible_*``) remains
    # bit-reproducible when running against an older cache. Production
    # non-``--deterministic`` runs rebuild new_M with the current schema and
    # concat-fill old cached rows the same way, so behavior is identical on
    # the happy path.
    X = M.reindex(columns=stat_data.get_stat_columns(market))

    # ratio_projvol divides by a projected-volume column that base_profile does
    # not emit, so get_stat_columns omits it and the reindex above drops it.
    # Carry it in from M (the matrix build persisted it) so the denominator
    # reaches every split, the model's expected_columns, and the served dump —
    # the same denominator-as-feature shape ratio_meanyr has with MeanYr.
    projvol_denom = None
    if target_normalization == "ratio_projvol":
        projvol_denom = baselines.resolve_denom_col(
            target_normalization, stat_data.league, market, M.columns, zero_inflated=False
        )
        X[projvol_denom] = M[projvol_denom]

    categories = ["Home", "Player position"]
    if "Player position" not in X.columns:
        categories.remove("Player position")
    for c in categories:
        X[c] = X[c].astype("category")

    M_sorted = M.sort_values("Date")
    n = len(M_sorted)
    n_train = int(n * _TRAIN_FRACTION)
    train_idx = M_sorted.index[:n_train]
    test_idx = M_sorted.index[n_train:]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    # Drop columns LightGBM cannot meaningfully split on. The mask is
    # computed on ``X_train`` only so test rows never influence which
    # features survive. The post-prune column list is persisted as
    # ``expected_columns`` in the model pickle (see ``_build_filedict``);
    # the inference path slices each offer's feature frame to that list
    # before ``model.predict`` (see ``match_offers`` in
    # ``prediction/scoring.py``), so pruning at train time propagates
    # cleanly to serving without any inference-side change.
    kept_cols = _prune_uninformative_features(X_train, categories)
    if structural_strategy == TWO_PART_STRATEGY:
        # Routing is part of the candidate's auditable feature contract even
        # when a cached train partition makes one role input constant.
        for column in role_spec_for(stat_data.league, market).role_columns:
            if column in X.columns and column not in kept_cols:
                kept_cols.append(column)
    # The projvol denominator is load-bearing for decode even if its train-split
    # variance is low — never let pruning drop it.
    if projvol_denom is not None and projvol_denom not in kept_cols:
        kept_cols.append(projvol_denom)
    if len(kept_cols) < len(X.columns):
        X = X[kept_cols]
        X_train = X_train[kept_cols]
        X_test = X_test[kept_cols]

    held_out_identity = M.loc[test_idx, ["Player", "Date"]]
    identity_hash = pd.util.hash_pandas_object(held_out_identity, index=False).to_numpy(
        dtype=np.uint64
    )
    validation_mask = identity_hash < _VALIDATION_HASH_THRESHOLD
    # A recipe search must never see the frame its own gate is scored on. Under
    # holdout_blind the evaluation frame IS the validation frame, so the true
    # held-out rows enter neither and are absent from the run rather than merely
    # unread; _step_crossfit_calibrate_and_serve then scores them out-of-fold.
    eval_mask = validation_mask if holdout_blind else ~validation_mask
    X_validation = X_test.loc[validation_mask]
    y_validation = y_test.loc[validation_mask]
    X_test = X_test.loc[eval_mask]
    y_test = y_test.loc[eval_mask]

    B_train = M.loc[X_train.index, ["Line", "Odds", "EV"]]
    B_test = M.loc[X_test.index, ["Line", "Odds", "EV"]]
    B_validation = M.loc[X_validation.index, ["Line", "Odds", "EV"]]

    y_train_labels = np.ravel(y_train.to_numpy())
    # Keep metadata index-bearing: ``_step_predict_splits`` sorts ``X_test`` by
    # index before persistence, and a positional array would silently attach
    # each player/date to whichever row moved into that position.
    players_train = M.loc[X_train.index, "Player"] if "Player" in M.columns else None
    players_validation = M.loc[X_validation.index, "Player"] if "Player" in M.columns else None
    players_test = M.loc[X_test.index, "Player"] if "Player" in M.columns else None
    dates_validation = M.loc[X_validation.index, "Date"]
    dates_test = M.loc[X_test.index, "Date"]
    archived_validation = M.loc[X_validation.index, "Archived"] if "Archived" in M.columns else None
    archived_test = M.loc[X_test.index, "Archived"] if "Archived" in M.columns else None
    odds_synthetic_validation = (
        M.loc[X_validation.index, "Odds_synthetic"] if "Odds_synthetic" in M.columns else None
    )
    odds_synthetic_test = (
        M.loc[X_test.index, "Odds_synthetic"] if "Odds_synthetic" in M.columns else None
    )
    quote_authenticity_validation = (
        M.loc[X_validation.index, "QuoteAuthenticity"] if "QuoteAuthenticity" in M.columns else None
    )
    quote_authenticity_test = (
        M.loc[X_test.index, "QuoteAuthenticity"] if "QuoteAuthenticity" in M.columns else None
    )
    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "X_validation": X_validation,
        "y_train": y_train,
        "y_test": y_test,
        "y_validation": y_validation,
        "B_train": B_train,
        "B_test": B_test,
        "B_validation": B_validation,
        "y_train_labels": y_train_labels,
        "players_train": players_train,
        "players_validation": players_validation,
        "players_test": players_test,
        "dates_validation": dates_validation,
        "dates_test": dates_test,
        "archived_validation": archived_validation,
        "archived_test": archived_test,
        "odds_synthetic_validation": odds_synthetic_validation,
        "odds_synthetic_test": odds_synthetic_test,
        "quote_authenticity_validation": quote_authenticity_validation,
        "quote_authenticity_test": quote_authenticity_test,
    }


def _step_build_lss_model(
    dist: str,
    dist_obj,
    X_train,
    shape_ceiling,
    *,
    normalize: bool,
    offset_mode: bool,
    use_hurdle: bool,
    sn_param: str,
):
    """Apply the bounded shape response and build an unfit LightGBMLSS model.

    Returns ``None`` for the hurdle path (the model is built+fit later by
    ``fit_hurdle_model``).
    """
    if use_hurdle:
        return None
    # Bound shape response function (safety net) — hurdle applies it internally.
    if dist in ("NegBin", "ZINB"):
        dist_obj.param_dict["total_count"] = _BoundedResponseFn(
            dist_obj.param_dict["total_count"], shape_ceiling
        )
    elif dist in ("Gamma", "ZAGamma"):
        dist_obj.param_dict["concentration"] = _BoundedResponseFn(
            dist_obj.param_dict["concentration"], shape_ceiling
        )

    model = LightGBMLSS(dist_obj)
    set_model_start_values(
        model,
        dist,
        X_train,
        shape_ceiling=shape_ceiling,
        normalized=normalize,
        offset_mode=offset_mode,
        sn_param=sn_param,
    )
    return model


# Asymptotic SD of the Kolmogorov statistic × √n — one sampling SE of a trial's measured PIT-KS.
_KS_SAMPLING_SD_COEF = 0.2606

# Per-trial constraint cost cap: the OOF pool is subsampled to this many rows before the
# KS fit — every KS eval prices the predictive CDF over the whole pool (x 25 draws on the
# lattice), and an uncapped 8-10k-row pool pushed one Optuna trial to ~17 min (exp2 NHL
# points: 4 trials/hour). At 4000 rows the KS sampling SE is 0.021, still tighter than the
# validation frame the retired v2 measure used at every cell size seen in the confirm ledger.
_OOF_KS_MAX_ROWS = 4000
_OOF_KS_SUBSAMPLE_SEED = 4517

# Per-trial minimize_scalar tolerance for the SkewNormal (c, s) and count dispersion-c fits
# inside the OOF constraint: a coarse ranking measure, not the full-precision serve fit — see
# the per-branch comments below for each fit's specific eval cost.
_TRIAL_XATOL = 1e-2


def _pick_calibrated_candidate(candidates: list[dict], threshold: float, n_rows: int) -> dict:
    """Calibration-constrained HP selection (Lever 1): the lowest-CRPS trial whose OOF
    PIT-KS clears the noise-aware effective threshold.

    Operationalizes Gneiting-Balabdaoui-Raftery 2007's "sharpness subject to calibration"
    as an ε-constraint under the one-standard-error rule (ESL §7.10):
    ``tau_eff = max(threshold, min_pit_ks + SE)`` with ``SE = _KS_SAMPLING_SD_COEF / √n_rows``,
    then min ``cv_loss`` over ``pit_ks <= tau_eff`` (never empty — ``min_pit_ks`` qualifies).
    When no trial clears the raw Gate-4 threshold the pick is the sharpest trial statistically
    tied with the best-calibrated one, never the bare min-KS trial — the single most
    winner's-cursed of ~300 noisy KS draws (Lever-1-v3 brief, R3).

    Also logs the constraint's headroom (min PIT-KS, feasible count at the raw threshold,
    ``tau_eff``, and the CRPS price of the pick over the loss-only best) — the probe of
    whether the constraint is non-vacuous.

    Args:
        candidates: Completed trials tagged with ``cv_loss`` and ``pit_ks``.
        threshold: The cell's raw Gate-4 PIT-KS bar.
        n_rows: Rows in the pooled OOF frame each trial's ``pit_ks`` was measured on.
    """
    min_pit_ks = min(c["pit_ks"] for c in candidates)
    tau_eff = max(threshold, min_pit_ks + _KS_SAMPLING_SD_COEF / np.sqrt(n_rows))
    picked = min((c for c in candidates if c["pit_ks"] <= tau_eff), key=lambda c: c["cv_loss"])
    logger.info(
        "calibrated HP headroom: min_pit_ks=%.4f feasible=%d/%d threshold=%.4f "
        "tau_eff=%.4f crps_price=%.5f",
        min_pit_ks,
        sum(c["pit_ks"] < threshold for c in candidates),
        len(candidates),
        threshold,
        tau_eff,
        picked["cv_loss"] - min(c["cv_loss"] for c in candidates),
    )
    return picked


def _blend_oof_skewnormal(decoded, sn_scale, skew, book_ev, y_pool, dist_info, blending):
    """Blend the OOF model predictive with the book the way serving does.

    exp2 MLB hits allowed: at model_weight ~0.12 the model-only measure anti-correlates
    with the served gate (280 trials walked g4 from 0.02 to 0.15), so the constraint must
    score the fused predictive. Per-trial ``fit_blend_weight`` mirrors the production fit;
    rows without a real book EV blend at weight 1 (the authenticity fallback). Returns the
    blended ``(mean, sigma, alpha, gate)``.
    """
    book_ok = np.isfinite(book_ev) & (book_ev > 0)
    cv = dist_info["cv"]
    gate_kwargs = (
        {"gate_model": decoded.gate, "gate_book": 0.0}
        if dist_info["hist_gate"] > GATE_PUBLISH_THRESHOLD and decoded.gate is not None
        else {}
    )
    if book_ok.any():
        w = calibration.fit_blend_weight(
            blending,
            decoded.ev[book_ok],
            book_ev[book_ok],
            y_pool[book_ok],
            "SkewNormal",
            cv=cv,
            model_sigma=sn_scale[book_ok],
            model_skew_alpha=skew[book_ok],
            **{
                key: (value[book_ok] if isinstance(value, np.ndarray) else value)
                for key, value in gate_kwargs.items()
            },
        )
    else:
        w = 1.0
    mean, sigma_blend, alpha_blend, gate_blend = fused_loc(
        np.where(book_ok, w, 1.0),
        decoded.ev,
        np.where(book_ok, book_ev, decoded.ev),
        cv,
        "SkewNormal",
        sigma=sn_scale,
        skew_alpha=skew,
        **gate_kwargs,
    )
    return mean, sigma_blend, alpha_blend, gate_blend if gate_kwargs else decoded.gate


def _calibration_penalty(
    splits: dict, dist_info: dict, *, blending: str = calibration.DEFAULT_BLENDING
):
    """Build the per-trial served-PIT-KS evaluator that search-gates the HPO (Lever 1).

    Returns ``evaluate(params, *, cvbooster, folds, opt_rounds)`` mapping one trial's
    already-trained CV fold boosters to a served randomized-PIT KS — SkewNormal via its
    joint (scale, skew) fit, every other family (NegBin / ZINB joint / DPO / Gamma /
    ZAGamma) via the scalar dispersion-``c`` fit. No refit anywhere: fold held-out rows
    come from that fold's booster at ``num_iteration=opt_rounds`` (~4.7× the validation
    rows, cutting the KS sampling SD 2.16× and removing the +33% per-trial refit —
    Lever-1-v3 brief, R2), and the validation rows from the fold-booster ensemble mean.
    The SkewNormal statistic is the **max over the OOF and validation frames** (the R2
    era caveat, escalated after exp2): both frames blend with the book per trial
    (:func:`_blend_oof_skewnormal` — mandatory at low model_weight) and the validation
    leg vetoes trials whose in-era recal-fixability does not survive to the era the
    serve fit and gate actually use. Deliberate proxy simplifications: (a) both frames
    skip the post-hoc stages (≤0.005 KS); the count branch stays model-only OOF (no
    demonstrated harm; its risk pins are all SN); (b) the count branch always fits ``c``
    on the PIT-KS
    objective — a proxy for the production dispersion fit even when the cell's
    ``count_dispersion_objective`` is crps — because the closure's output IS the
    served-KS ranking statistic; (c) the measure is budgeted — the pool is capped at
    :data:`_OOF_KS_MAX_ROWS` rows, the SN fit runs sequential (c then s), and the count
    c-fit at 0.01 tolerance — a per-trial constraint must cost seconds, not the ~17 min
    the production-grade fit took on an 8k-row pool.
    """
    dist = dist_info["dist"]
    strategy = dist_info["target_normalization"]
    global_mean = dist_info["global_mean"]
    denom_col = dist_info["denom_col"]
    dist_obj = dist_info["dist_obj"]
    X_train = splits["X_train"]
    y_train_result = splits["y_train"]["Result"]
    # The per-row predict-time margins predict_lss_params would seed for these rows; they
    # depend only on the frame, so compute once. SimpleNamespace stands in for the model
    # object set_model_start_values assigns onto.
    carrier = SimpleNamespace()
    set_model_start_values(
        carrier,
        dist,
        X_train,
        shape_ceiling=dist_info["shape_ceiling"],
        normalized=dist_info["normalize"],
        offset_mode=dist_info["offset_mode"],
        sn_param=dist_info["sn_param"],
    )
    start_values = carrier.start_values

    def oof_frames(cvbooster, folds, opt_rounds):
        preds = predict_cv_oof_params(cvbooster, folds, opt_rounds, X_train, start_values, dist_obj)
        if len(preds) > _OOF_KS_MAX_ROWS:
            keep = np.sort(
                np.random.default_rng(_OOF_KS_SUBSAMPLE_SEED).choice(
                    len(preds), _OOF_KS_MAX_ROWS, replace=False
                )
            )
            preds = preds.iloc[keep]
        # y_train is the raw pre-transform frame and X_train keeps its original index
        # through the SkewNormal nonzero filter, so .loc realigns the raw outcomes.
        y_pool = y_train_result.loc[preds.index].to_numpy(dtype=float)
        return preds, X_train.loc[preds.index], y_pool

    def sn_frame_ks(preds, X_pool, y_pool, book_ev):
        sn_loc = strategy.decode_loc(preds["loc"].to_numpy(), X_pool, global_mean, denom_col)
        sn_scale = strategy.decode_scale(preds["scale"].to_numpy(), X_pool, denom_col)
        skew = preds["alpha"].to_numpy()
        decoded = decode_predictive_mean(
            preds,
            dist,
            sn_loc=sn_loc,
            sn_scale=sn_scale,
            hist_gate=dist_info["hist_gate"],
        )
        mean, sn_scale, skew, gate = _blend_oof_skewnormal(
            decoded, sn_scale, skew, book_ev, y_pool, dist_info, blending
        )

        # Coarse sequential (c, s) at 0.01 tolerance — a ranking measure, not the serve
        # fit. The production fit_skewnorm_dispersion_skew burns ~50 full-precision KS
        # evals per call (~187 s/trial on a ZI SN pool); ~20 coarse evals order the
        # trials identically.
        def ks_at(c, s):
            return _served_sn_pit_ks(mean, sn_scale, skew, y_pool, c, s, gate)

        c = minimize_scalar(
            lambda c: ks_at(c, 0.0),
            bounds=_DISPERSION_C_BOUNDS,
            method="bounded",
            options={"xatol": _TRIAL_XATOL},
        ).x
        s = minimize_scalar(
            lambda s: ks_at(c, s),
            bounds=_DISPERSION_SKEW_BOUNDS,
            method="bounded",
            options={"xatol": _TRIAL_XATOL},
        ).x
        return ks_at(c, s)

    def served_pit_ks(params: dict, *, cvbooster, folds, opt_rounds) -> float:
        preds, X_pool, y_pool = oof_frames(cvbooster, folds, opt_rounds)
        ks_oof = sn_frame_ks(
            preds, X_pool, y_pool, splits["B_train"]["EV"].loc[preds.index].to_numpy(dtype=float)
        )
        # Two-frame constraint (the R2 era caveat, escalated): the OOF pool is train-era, the
        # gate is later-era. exp2's MLB rerun showed a trial can be recal-fixable in-era yet
        # demand an extreme, non-transferring (c, s) on the validation era (serve fit
        # c=1.80/s=-2.82 → test g4 0.104 at model_weight 0.05). The validation leg prices each
        # trial on the same frame the serve fit will use; the fold-booster ensemble keeps it
        # refit-free.
        preds_val = predict_cv_ensemble_params(
            cvbooster, opt_rounds, splits["X_validation"], start_values_val, dist_obj
        )
        ks_val = sn_frame_ks(
            preds_val,
            splits["X_validation"],
            splits["y_validation"]["Result"].to_numpy(dtype=float),
            splits["B_validation"]["EV"].to_numpy(dtype=float),
        )
        return max(ks_oof, ks_val)

    def count_pit_ks(params: dict, *, cvbooster, folds, opt_rounds) -> float:
        preds, _, y_pool = oof_frames(cvbooster, folds, opt_rounds)
        decoded = decode_predictive_mean(preds, dist)
        shape_ceiling = dist_info["shape_ceiling"]
        if dist == "DPO":
            upper = min(10.0, _DP_PHI_CEILING / float(np.max(decoded.phi)))
        else:
            mean_shape = np.mean(decoded.r if dist in ("NegBin", "ZINB") else decoded.alpha)
            if mean_shape > 0 and shape_ceiling is not None and not np.isnan(shape_ceiling):
                upper = min(10.0, shape_ceiling / mean_shape)
            else:
                upper = 10.0
        c_opt = minimize_scalar(
            lambda c: _dispersion_pit_ks_loss(
                c,
                dist=dist,
                y_val_arr=y_pool,
                val_weighted_mean=decoded.ev,
                gate_blend_val=decoded.gate,
                r_blend_val=decoded.r,
                alpha_blend_val=decoded.alpha,
                phi_blend_val=decoded.phi,
            ),
            bounds=(0.1, upper),
            method="bounded",
            # xatol 0.01: each loss eval prices the whole lattice CDF over the pool x 25
            # draws; default tolerance burns ~3x the evals refining c digits the
            # constraint ranking never uses. The production serve fit stays full precision.
            options={"xatol": _TRIAL_XATOL},
        ).x
        frame = _count_pit_frame(
            dist, c_opt, decoded.ev, decoded.gate, decoded.r, decoded.alpha, decoded.phi
        )
        return _randomized_pit_ks(frame, dist, y_pool, strategy=TARGET_NORM_NONE)

    if dist == "SkewNormal":
        carrier_val = SimpleNamespace()
        set_model_start_values(
            carrier_val,
            dist,
            splits["X_validation"],
            shape_ceiling=dist_info["shape_ceiling"],
            normalized=dist_info["normalize"],
            offset_mode=dist_info["offset_mode"],
            sn_param=dist_info["sn_param"],
        )
        start_values_val = carrier_val.start_values
        return served_pit_ks
    return count_pit_ks


def _step_select_hyperparams(
    X_train,
    dist: str,
    model,
    opt_params_in: dict | None,
    dtrain,
    *,
    use_hurdle: bool,
    deterministic: bool,
    calibration_penalty=None,
    penalty_threshold=None,
) -> tuple[dict | list[dict], list[int]]:
    """Pick Optuna-tuned params, warm-start, or the deterministic fixed set.

    When ``calibration_penalty`` is given (Lever 1, calibrated HP selection) the Optuna
    sampler is feasibility-constrained on the out-of-fold PIT-KS and the paths return every
    completed trial (a list) for the caller's :func:`_pick_calibrated_candidate` final pick;
    every other case returns the single best param dict as before.

    Args:
        X_train: Training feature matrix (used to size ``min_child_weight`` upper).
        dist: Distribution name (controls monotone constraint on ``MeanYr``).
        model: Built LightGBMLSS for the LSS path; ``None`` for hurdle.
        use_hurdle: True when the hurdle path will fit the model later.
        deterministic: If True, use ``DETERMINISTIC_FIXED_PARAMS`` verbatim.
        calibration_penalty: Per-trial PIT-KS evaluator returned by
            :func:`_calibration_penalty`; ``None`` for the plain-CRPS path.
        penalty_threshold: Gate-4 PIT-KS threshold paired with
            ``calibration_penalty``; ``None`` when penalty is None.
        opt_params_in: Warm-start params from the existing pickle, if any.
        dtrain: ``lgb.Dataset`` for the LSS Optuna path.

    Returns:
        ``(opt_params, monotone)`` — hyperparam dict (or list of dicts when
        ``calibration_penalty`` is given) and the monotone constraint vector.
    """
    col_list = list(X_train.columns)
    monotone = [0] * len(col_list)
    if dist in ("Gamma", "ZAGamma", "SkewNormal", "Mixture") and "MeanYr" in col_list:
        monotone[col_list.index("MeanYr")] = 1

    hp_search_space = {
        "feature_pre_filter": ["none", [False]],
        "num_threads": ["none", [8]],
        "max_depth": ["none", [-1]],
        "max_bin": ["none", [127]],
        "hist_pool_size": ["none", [9 * 1024]],
        "monotone_constraints": ["none", [monotone]],
        "path_smooth": ["float", {"low": 0, "high": 20, "log": False}],
        "num_leaves": ["int", {"low": 8, "high": 127, "log": False}],
        "lambda_l1": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "lambda_l2": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "min_child_samples": ["int", {"low": 30, "high": 150, "log": False}],
        "min_child_weight": [
            "float",
            {"low": 1e-3, "high": 0.75 * len(X_train) / 1000, "log": True},
        ],
        "learning_rate": ["float", {"low": 0.01, "high": 0.15, "log": True}],
        "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
        "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
        "bagging_freq": ["none", [1]],
    }

    opt_params = opt_params_in
    if deterministic:
        opt_params = {**DETERMINISTIC_FIXED_PARAMS, "monotone_constraints": monotone}
    elif use_hurdle:
        # Hurdle skips Optuna (LightGBMLSS hyper_opt does not apply to the
        # two-stage architecture). Reuse warm-start params from the pickle
        # if present; otherwise use DETERMINISTIC_FIXED_PARAMS as a sane
        # default but bump rounds to 200 for non-deterministic quality.
        if opt_params is None or opt_params.get("opt_rounds") is None:
            opt_params = {
                **DETERMINISTIC_FIXED_PARAMS,
                "monotone_constraints": monotone,
                "opt_rounds": 200,
            }
        else:
            # Unlike the Optuna paths below, this one hands the pickle's params straight to
            # LightGBM, which aborts the fit unless the constraint vector has exactly one entry
            # per training column. The pickle's vector is sized to the feature set it was trained
            # on, so a warm start after any feature-set change dies; recompute instead of
            # carrying it forward, since it is a pure function of the current column list.
            opt_params = {**opt_params, "monotone_constraints": monotone}
    elif opt_params is None or opt_params.get("opt_rounds") is None:
        opt_params = run_hyper_opt(
            model,
            hp_search_space,
            dtrain,
            num_boost_round=999,
            nfold=4,
            early_stopping_rounds=50,
            max_minutes=60,
            n_trials=300,
            calibration_penalty=calibration_penalty,
            penalty_threshold=penalty_threshold,
        )
    else:
        opt_params = run_hyper_opt(
            model,
            hp_search_space,
            dtrain,
            initial_params=opt_params,
            n_trials=150,
            max_minutes=5,
            calibration_penalty=calibration_penalty,
            penalty_threshold=penalty_threshold,
        )
    return opt_params, monotone


def _step_fit_model(
    dist: str,
    dist_obj,
    X_train,
    y_train_labels,
    opt_params: dict,
    *,
    use_hurdle: bool,
    normalize: bool,
    offset_mode: bool,
    shape_ceiling,
    deterministic: bool,
    sn_param: str,
):
    """Dispatch ``fit_hurdle_model`` vs ``fit_lss_model`` based on ``use_hurdle``."""
    seed = DETERMINISTIC_SEED if deterministic else None
    if use_hurdle:
        return fit_hurdle_model(
            X_train, y_train_labels, opt_params, shape_ceiling=shape_ceiling, seed=seed
        )
    return fit_lss_model(
        dist_obj,
        dist,
        X_train,
        y_train_labels,
        opt_params,
        normalized=normalize,
        shape_ceiling=shape_ceiling,
        seed=seed,
        offset_mode=offset_mode,
        sn_param=sn_param,
    )


def _kill_experiment_context(context: dict, splits: dict, reason: str) -> dict:
    """Mark a research candidate killed and route every split to its pooled fallback."""
    return {
        **context,
        "status": "killed_fallback",
        "kill_reason": reason,
        "routes": {
            split: pd.Series(
                "pooled_fallback",
                index=splits[f"X_{split}"].index,
                dtype="object",
            )
            for split in ("train", "validation", "test")
        },
    }


def _step_fit_affine_experts(
    context: dict,
    *,
    dist: str,
    dist_obj,
    splits: dict,
    opt_params: dict,
    use_hurdle: bool,
    normalize: bool,
    offset_mode: bool,
    shape_ceiling,
    deterministic: bool,
    sn_param: str,
) -> tuple[dict[int, object], dict]:
    """Fit one expert per discovered position with the pooled columns, labels, and params."""
    if context["status"] != "active":
        return {}, context

    X_train = splits["X_train"]
    positions = pd.to_numeric(X_train["Player position"], errors="coerce")
    y_train_labels = np.asarray(splits["y_train_labels"])
    expert_models: dict[int, object] = {}
    for code in context["experts"]:
        mask = positions.eq(code).to_numpy()
        if not mask.any():
            return {}, _kill_experiment_context(
                context,
                splits,
                f"missing eligible train rows for position code {code}",
            )
        expert_models[code] = _step_fit_model(
            dist,
            dist_obj,
            X_train.loc[positions.eq(code)],
            y_train_labels[mask],
            opt_params,
            use_hurdle=use_hurdle,
            normalize=normalize,
            offset_mode=offset_mode,
            shape_ceiling=shape_ceiling,
            deterministic=deterministic,
            sn_param=sn_param,
        )
        if expert_models[code] is None:
            return {}, _kill_experiment_context(
                context,
                splits,
                f"missing fitted expert for position code {code}",
            )
    return expert_models, context


def _predict_split_params(
    predict_model,
    dist: str,
    X_part: pd.DataFrame,
    *,
    normalize: bool,
    offset_mode: bool,
    sn_param: str,
) -> pd.DataFrame:
    """Raw distribution params for one split, routing hurdle vs LSS heads."""
    if getattr(predict_model, "is_hurdle", False):
        return predict_hurdle_params(predict_model, X_part)
    return predict_lss_params(
        predict_model,
        dist,
        X_part,
        normalized=normalize,
        offset_mode=offset_mode,
        sn_param=sn_param,
    )


def _stitch_one_split(
    split: str,
    pooled: pd.DataFrame,
    X_part: pd.DataFrame,
    routes: pd.Series,
    experts: dict[int, object],
    labels: dict[int, str],
    dist: str,
    *,
    normalize: bool,
    offset_mode: bool,
    sn_param: str,
) -> pd.DataFrame:
    """Overlay each discovered position's expert params onto one pooled split."""
    stitched = pooled.copy()
    position = pd.to_numeric(X_part["Player position"], errors="coerce")
    eligible_fallback = position.isin(labels) & routes.eq("pooled_fallback")
    if eligible_fallback.any():
        raise ValueError(f"eligible expert fallback rows on {split}")

    for code in experts:
        eligible_index = X_part.index[position.eq(code)]
        expert_params = _predict_split_params(
            experts[code],
            dist,
            X_part.loc[eligible_index],
            normalize=normalize,
            offset_mode=offset_mode,
            sn_param=sn_param,
        )
        valid = (
            not expert_params.index.has_duplicates
            and expert_params.index.equals(eligible_index)
            and list(expert_params.columns) == list(pooled.columns)
            and np.isfinite(expert_params.to_numpy(dtype=float)).all()
        )
        if not valid:
            raise ValueError(f"invalid {labels[code]} expert parameters on {split}")
        stitched.loc[eligible_index, :] = expert_params.to_numpy()
    return stitched


def _stitch_affine_expert_params(
    pooled_params: dict[str, pd.DataFrame],
    dist: str,
    splits: dict,
    context: dict | None,
    experts: dict[int, object],
    *,
    normalize: bool,
    offset_mode: bool,
    sn_param: str,
) -> tuple[dict[str, pd.DataFrame], dict | None]:
    """Overlay each discovered position's expert params onto the pooled predictions.

    Returns ``(stitched_params, context)``. When the candidate is not an active
    affine experiment, the pooled params pass through unchanged. A missing
    expert or an invalid stitch kills the experiment and routes every split to
    its pooled fallback.
    """
    if (
        context is None
        or context.get("slug") not in AFFINE_EXPERT_EXPERIMENTS
        or context["status"] != "active"
    ):
        return pooled_params, context
    labels: dict[int, str] = context["experts"]
    if any(code not in experts for code in labels):
        return pooled_params, _kill_experiment_context(
            context,
            splits,
            "missing fitted expert for a discovered position",
        )

    try:
        stitched = {
            split: _stitch_one_split(
                split,
                pooled_params[split],
                splits[f"X_{split}"],
                context["routes"][split].reindex(splits[f"X_{split}"].index),
                experts,
                labels,
                dist,
                normalize=normalize,
                offset_mode=offset_mode,
                sn_param=sn_param,
            )
            for split in ("train", "validation", "test")
        }
    except ValueError as exc:
        return pooled_params, _kill_experiment_context(context, splits, str(exc))
    return stitched, context


def _step_predict_splits(
    model,
    dist: str,
    splits: dict,
    *,
    normalize: bool,
    offset_mode: bool,
    sn_param: str,
    expert_models: dict[int, object] | None = None,
    structural_context: dict | None = None,
) -> dict:
    """Predict raw distribution params on train/validation/test splits.

    Also sorts indices on every split + the B_* frames + replaces ``Odds == 0``
    with 0.5 in B_*. Mutates ``splits`` to keep the sorted views consistent.

    Returns:
        Dict with: ``prob_params_train``, ``prob_params_validation``, ``prob_params``.
    """
    experts = expert_models or {}
    pooled_params = {
        split: _predict_split_params(
            model,
            dist,
            splits[f"X_{split}"],
            normalize=normalize,
            offset_mode=offset_mode,
            sn_param=sn_param,
        )
        for split in ("train", "validation", "test")
    }
    stitched_params, context = _stitch_affine_expert_params(
        pooled_params,
        dist,
        splits,
        structural_context,
        experts,
        normalize=normalize,
        offset_mode=offset_mode,
        sn_param=sn_param,
    )

    prob_params_train = stitched_params["train"]
    prob_params_validation = stitched_params["validation"]
    prob_params = stitched_params["test"]

    prob_params_train.sort_index(inplace=True)
    prob_params_train["result"] = splits["y_train"]["Result"]
    prob_params_validation.sort_index(inplace=True)
    prob_params_validation["result"] = splits["y_validation"]["Result"]
    prob_params.sort_index(inplace=True)
    prob_params["result"] = splits["y_test"]["Result"]
    for key in (
        "X_train",
        "B_train",
        "y_train",
        "X_test",
        "B_test",
        "y_test",
        "X_validation",
        "B_validation",
        "y_validation",
    ):
        splits[key].sort_index(inplace=True)
    for key in ("B_train", "B_test", "B_validation"):
        splits[key].loc[splits[key]["Odds"] == 0, "Odds"] = 0.5

    return {
        "prob_params_train": prob_params_train,
        "prob_params_validation": prob_params_validation,
        "prob_params": prob_params,
        "structural_context": context,
    }


def _step_compute_skill_metrics(
    val_calibrated: np.ndarray,
    y_class_val: np.ndarray,
    val_book_proba: np.ndarray,
    league: str,
    market: str,
) -> dict:
    """Compute model + book metrics, brier_skill_score, kelly_shrinkage.

    Returns:
        Dict with: ``model_metrics``, ``book_metrics`` (or None),
        ``brier_skill_score``, ``kelly_shrinkage``.
    """
    if len(val_book_proba) == 0:
        metric_names = (
            "brier_score",
            "log_loss",
            "roc_auc",
            "expected_calibration_error",
            "accuracy",
            "precision_over",
            "precision_under",
            "predicted_over_rate",
            "empirical_over_rate",
            "prediction_std",
            "nll",
        )
        logger.warning(
            "no authentic book evidence for %s/%s; skill metrics set to nan",
            league,
            market,
        )
        return {
            "model_metrics": dict.fromkeys(metric_names, float("nan")),
            "book_metrics": None,
            "brier_skill_score": float("nan"),
            "kelly_shrinkage": float("nan"),
        }
    book_proba_available = np.isfinite(val_book_proba).all()
    model_metrics = _compute_metrics(val_calibrated, y_class_val)
    if book_proba_available:
        book_metrics = _compute_metrics(val_book_proba, y_class_val)
        brier_skill_score = 1 - (
            model_metrics["brier_score"]
            / max(book_metrics["brier_score"], _BRIER_SKILL_DENOM_FLOOR)
        )
    else:
        logger.warning(
            "book baseline unavailable for %s/%s; skill score set to nan",
            league,
            market,
        )
        book_metrics = None
        brier_skill_score = float("nan")
    kelly_shrinkage = (
        float(np.clip(brier_skill_score, 0.0, 1.0))
        if np.isfinite(brier_skill_score)
        else float("nan")
    )
    return {
        "model_metrics": model_metrics,
        "book_metrics": book_metrics,
        "brier_skill_score": float(brier_skill_score),
        "kelly_shrinkage": kelly_shrinkage,
    }


def _step_compute_mode_stats(
    y_proba_raw: np.ndarray,
    y_proba_no_filt: np.ndarray,
    y_proba_filt: np.ndarray,
    y_class: np.ndarray,
) -> dict:
    """Compute the legacy prec/acc/sharp/ll/over_pct/under_prec arrays (length-3).

    Index 0 = raw, 1 = no_filt (post-blend, pre-temp), 2 = filt (post-temp).
    Confidence mask is ``max(proba) > _MODE_CONFIDENCE_THRESHOLD``.
    """
    prec = np.zeros(3)
    acc = np.zeros(3)
    sharp = np.zeros(3)
    ll = np.zeros(3)
    over_pct = np.zeros(3)
    under_prec = np.zeros(3)
    for i, y_proba in enumerate([y_proba_raw, y_proba_no_filt, y_proba_filt]):
        y_pred = (y_proba > 0.5).astype(int)[:, 1]
        mask = np.max(y_proba, axis=1) > _MODE_CONFIDENCE_THRESHOLD
        prec[i] = precision_score(y_class[mask], y_pred[mask])
        acc[i] = accuracy_score(y_class[mask], y_pred[mask])
        sharp[i] = np.std(y_proba[:, 1])
        ll[i] = log_loss(y_class, y_proba[:, 1])
        over_pct[i] = y_pred[mask].mean() / mask.mean() if mask.sum() > 0 else np.nan
        under_mask = mask & (y_pred == 0)
        under_prec[i] = (y_class[under_mask] == 0).mean() if under_mask.sum() > 0 else np.nan
    return {
        "prec": prec,
        "acc": acc,
        "sharp": sharp,
        "ll": ll,
        "over_pct": over_pct,
        "under_prec": under_prec,
    }


def _zero_inflated_outcome_mean(
    base_mean: np.ndarray, dist: str, gate: np.ndarray | None
) -> np.ndarray:
    """E[Y] = (1-π)·μ for zero-adjusted cells; the base mean otherwise.

    ZINB/ZAGamma store the base-distribution mean — the betting convention factors
    the gate out and reapplies it in ``get_odds``. The EV/bias diagnostics compare
    against zero-INCLUSIVE outcomes (Result, Line), so they must reapply it. Gated
    SkewNormal is also trained on positive rows and carries a historical gate; plain
    SkewNormal, NegBin, and Gamma pass through. Mirrors
    ``training.scorecard._zero_inflated_mean``.
    """
    if dist in ("ZINB", "ZAGamma", "SkewNormal") and gate is not None:
        return base_mean * (1.0 - gate)
    return base_mean


def _diag_shape(
    dist,
    cv,
    prob_params,
    test_mean_yr,
    test_std_yr,
    strategy,
    X_test,
    denom_col,
    y_test,
    player_stats,
):
    """(shape_label, start_shape, model_shape, empirical_shape) for the cell's family.

    Continuous families report sigma over sigma: ``model_shape`` is the mean predictive
    sigma decoded through the strategy's own ``decode_scale`` (identity for additive
    normalizations, per-row denominator multiply for ratio_*) and ``empirical_shape`` is
    the marginal outcome SD, so ``shape_ratio`` separates converged fits (~0.6-2.5) from
    diverged ones (>=25) — researcher_hpo_objective_alignment brief, finding 3.
    """
    if dist in ("SkewNormal", "Mixture"):
        diag_start_shape = float(cv)
        if dist == "SkewNormal":
            sigma_norm = prob_params["scale"].to_numpy(dtype=float)
        else:
            _, sigma_norm = _mixture_moments(
                prob_params["mix_prob_1"].to_numpy(dtype=float),
                prob_params["loc_1"].to_numpy(dtype=float),
                prob_params["scale_1"].to_numpy(dtype=float),
                prob_params["loc_2"].to_numpy(dtype=float),
                prob_params["scale_2"].to_numpy(dtype=float),
            )
        diag_model_shape = float(strategy.decode_scale(sigma_norm, X_test, denom_col).mean())
        diag_empirical_shape = float(y_test["Result"].to_numpy().std())
        diag_shape_label = "scale"
    elif dist in ("Gamma", "ZAGamma"):
        diag_start_shape = float(np.clip((test_mean_yr / max(test_std_yr, 1e-6)) ** 2, 0.1, 100))
        diag_model_shape = float(prob_params["concentration"].mean())
        per_player_emp_alpha = (player_stats.mean() / np.maximum(player_stats.std(), 0.01)) ** 2
        diag_empirical_shape = float(np.median(per_player_emp_alpha))
        diag_shape_label = "alpha"
    elif dist in ("NegBin", "ZINB"):
        diag_start_shape = float(
            np.clip(
                test_mean_yr**2 / max(test_std_yr**2 - test_mean_yr, 1e-6),
                1,
                _COUNT_BRANCH_R_CAP,
            )
        )
        diag_model_shape = float(prob_params["total_count"].mean())
        per_player_emp_r = player_stats.mean() ** 2 / np.maximum(
            player_stats.var() - player_stats.mean(), 0.01
        )
        per_player_emp_r = np.minimum(per_player_emp_r, _COUNT_BRANCH_R_CAP)
        diag_empirical_shape = float(np.median(per_player_emp_r))
        diag_shape_label = "r"
    elif dist == "DPO":
        # Moment estimator: var = mu/phi ⇒ phi = mu/var, clipped to the family's hard bounds.
        diag_start_shape = float(
            np.clip(test_mean_yr / max(test_std_yr**2, 1e-6), _DP_PHI_FLOOR, _DP_PHI_CEILING)
        )
        diag_model_shape = float(prob_params["phi"].mean())
        per_player_emp_phi = player_stats.mean() / np.maximum(player_stats.var(), 0.01)
        per_player_emp_phi = np.minimum(per_player_emp_phi, _DP_PHI_CEILING)
        diag_empirical_shape = float(np.median(per_player_emp_phi))
        diag_shape_label = "phi"
    return diag_shape_label, diag_start_shape, diag_model_shape, diag_empirical_shape


def _diag_over_pcts(y_class, ev_minus_line_arr, y_proba_no_filt):
    """Realized over-rate among confident EV>line / EV<=line picks (NaN below the floor)."""
    ev_gt_mask = ev_minus_line_arr > 0
    ev_lt_mask = ev_minus_line_arr <= 0
    conf_mask = np.max(y_proba_no_filt, axis=1) > _MODE_CONFIDENCE_THRESHOLD
    over_pct_ev_gt = (
        float(y_class[ev_gt_mask & conf_mask].mean())
        if (ev_gt_mask & conf_mask).sum() > _MIN_DIAGNOSTIC_ROWS
        else float("nan")
    )
    over_pct_ev_lt = (
        float(y_class[ev_lt_mask & conf_mask].mean())
        if (ev_lt_mask & conf_mask).sum() > _MIN_DIAGNOSTIC_ROWS
        else float("nan")
    )
    return over_pct_ev_gt, over_pct_ev_lt


def _diag_cf_over_pct(dist, diag_empirical_shape, weighted_mean, B_test, gate_blend_test, step):
    """Counterfactual over-rate if the model used the empirical (not fitted) shape.

    NaN for SkewNormal/Mixture (no counterfactual shape) and when the empirical shape
    is unusable (NaN or non-positive) or too few confident rows clear the floor.
    """
    if dist in ("SkewNormal", "Mixture"):
        return float("nan")
    if np.isnan(diag_empirical_shape) or diag_empirical_shape <= 0:
        return float("nan")
    emp_shape = np.full_like(weighted_mean, diag_empirical_shape)
    if dist in ("NegBin", "ZINB"):
        cf_under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            r=emp_shape,
            gate=gate_blend_test,
        )
    elif dist == "DPO":
        cf_under = get_odds(B_test["Line"].to_numpy(), weighted_mean, dist, phi=emp_shape)
    else:
        cf_under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            alpha=emp_shape,
            step=step,
            gate=gate_blend_test,
        )
    cf_over = 1 - cf_under
    cf_pred = (cf_over > 0.5).astype(int)
    cf_mask = np.maximum(cf_under, cf_over) > _MODE_CONFIDENCE_THRESHOLD
    return (
        float(cf_pred[cf_mask].mean() / cf_mask.mean())
        if cf_mask.sum() > _MIN_DIAGNOSTIC_ROWS
        else float("nan")
    )


def _step_compute_diagnostics(
    splits: dict,
    prob_params: pd.DataFrame,
    weighted_mean: np.ndarray,
    y_proba_no_filt: np.ndarray,
    y_class: np.ndarray,
    gate_blend_test: np.ndarray | None,
    dist: str,
    cv: float,
    denom_col: str,
    strategy,
    player_stats,
    step,
) -> dict:
    """Compute the diag_* values written into ``filedict["diagnostics"]``.

    Returns:
        Dict matching the diagnostics keys (``shape_label``, ``start_shape``,
        ``model_shape``, ``empirical_shape``, ``start_mean``, ``model_ev``,
        ``mean_line``, ``ev_minus_line``, ``result_mean``, ``median_ev_diff``,
        ``frac_ev_gt_line``, ``over_pct_ev_gt``, ``over_pct_ev_lt``,
        ``cf_over_pct``, ``ev_meanyr_corr``, ``result_meanyr_corr``).
    """
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    B_test = splits["B_test"]

    test_mean_yr = X_test["MeanYr"].mean()
    test_std_yr = X_test["STDYr"].mean()

    diag_shape_label, diag_start_shape, diag_model_shape, diag_empirical_shape = _diag_shape(
        dist,
        cv,
        prob_params,
        test_mean_yr,
        test_std_yr,
        strategy,
        X_test,
        denom_col,
        y_test,
        player_stats,
    )

    diag_start_mean = float(test_mean_yr)
    # E[Y] = (1-π)·μ — the EV/bias diagnostics below compare against zero-INCLUSIVE
    # outcomes (Result, Line), so zero-inflated count cells reapply the gate that
    # weighted_mean (base-μ betting convention) factors out. The get_odds/cf path
    # further down keeps using weighted_mean and reapplies the gate itself.
    ev_full = _zero_inflated_outcome_mean(weighted_mean, dist, gate_blend_test)
    diag_model_ev = float(ev_full.mean())
    diag_mean_line = float(B_test["Line"].mean())
    diag_ev_minus_line = float((ev_full - B_test["Line"].to_numpy()).mean())
    diag_result_mean = float(y_test["Result"].mean())

    _meanyr_arr = X_test["MeanYr"].to_numpy()
    _result_arr = y_test["Result"].to_numpy()
    diag_ev_meanyr_corr = float(np.corrcoef(_meanyr_arr, ev_full - _meanyr_arr)[0, 1])
    diag_result_meanyr_corr = float(np.corrcoef(_meanyr_arr, _result_arr - _meanyr_arr)[0, 1])

    ev_minus_line_arr = ev_full - B_test["Line"].to_numpy()
    diag_median_ev_diff = float(np.median(ev_minus_line_arr))
    diag_frac_ev_gt_line = float((ev_minus_line_arr > 0).mean())

    diag_over_pct_ev_gt, diag_over_pct_ev_lt = _diag_over_pcts(
        y_class, ev_minus_line_arr, y_proba_no_filt
    )
    diag_cf_over_pct = _diag_cf_over_pct(
        dist, diag_empirical_shape, weighted_mean, B_test, gate_blend_test, step
    )

    return {
        "shape_label": diag_shape_label,
        "start_shape": diag_start_shape,
        "model_shape": diag_model_shape,
        "empirical_shape": diag_empirical_shape,
        "start_mean": diag_start_mean,
        "model_ev": diag_model_ev,
        "mean_line": diag_mean_line,
        "ev_minus_line": diag_ev_minus_line,
        "result_mean": diag_result_mean,
        "median_ev_diff": diag_median_ev_diff,
        "frac_ev_gt_line": diag_frac_ev_gt_line,
        "over_pct_ev_gt": diag_over_pct_ev_gt,
        "over_pct_ev_lt": diag_over_pct_ev_lt,
        "cf_over_pct": diag_cf_over_pct,
        "ev_meanyr_corr": diag_ev_meanyr_corr,
        "result_meanyr_corr": diag_result_meanyr_corr,
    }


def _build_y_proba_raw(B_test, decoded: dict, dist: str, step) -> np.ndarray:
    """Compute the pre-blend, pre-calibration test probabilities (Nx2)."""
    line = B_test["Line"].to_numpy()
    ev = decoded["ev"]
    gate_test = decoded["gate_test"]
    if dist == "SkewNormal":
        under = get_odds(
            line,
            ev,
            "SkewNormal",
            sigma=decoded["sn_sigma_test"],
            skew_alpha=decoded["sn_alpha_test"],
            gate=gate_test,
        )
    elif dist == "Mixture":
        under = get_odds(line, ev, dist, mix=decoded["mix_test"])
    elif dist in ("NegBin", "ZINB"):
        under = get_odds(line, ev, dist, r=decoded["r"], gate=gate_test)
    elif dist == "DPO":
        under = get_odds(line, ev, dist, phi=decoded["phi"])
    else:
        under = get_odds(line, ev, dist, alpha=decoded["alpha"], step=step, gate=gate_test)
    under = np.clip(under, 0, 1)
    return np.array([under, 1 - under]).transpose()


def _model_version(
    trained_at: str,
    opt_params: dict,
    dist: str,
    cv: float,
    step,
    norm: str,
    *,
    structural_strategy: str = BASE_STRUCTURAL_STRATEGY,
) -> str:
    """Deterministic ``{yyyymmdd}.{norm-slug}.{sha8}`` identity for a trained model.

    ``sha8`` is a sha1 over the sorted repr of the model's stable identity
    (hyperparams + distribution + cv + step), so re-running :func:`train_market`
    on the same data and config reproduces the same version. WS-1 joins live
    reads on this string to attribute predictions to the model that made them.
    """
    identity_fields = [
        *opt_params.items(),
        ("distribution", dist),
        ("cv", cv),
        ("step", step),
    ]
    if structural_strategy != BASE_STRUCTURAL_STRATEGY:
        identity_fields.append(("structural_strategy", structural_strategy))
    identity = repr(sorted(identity_fields))
    sha8 = hashlib.sha1(identity.encode()).hexdigest()[:8]
    return f"{trained_at[:10].replace('-', '')}.{norm or 'none'}.{sha8}"


def _build_filedict(
    *,
    model,
    step,
    mode_stats: dict,
    skill: dict,
    diag: dict,
    c_opt: float,
    skew_cal: float,
    shape_ceiling,
    marginal_shape,
    opt_params: dict,
    dist: str,
    cv: float,
    y,
    T_opt: float,
    model_weight,
    model_calib,
    hist_gate: float,
    normalize: bool,
    strategy,
    global_mean,
    denom_col: str,
    target_normalization: str,
    posthoc_slug: str,
    posthoc_blob: dict | None,
    pit_recal_blob: dict | None,
    zinb_mode: str,
    sn_param: str,
    X,
    league: str,
    market: str,
    matrix_hash: str,
    selected_controls: dict[str, str],
    structural_strategy: str = BASE_STRUCTURAL_STRATEGY,
    algorithm_payload: dict | None = None,
    expert_models: dict[int, object] | None = None,
) -> dict:
    """Assemble the model pickle dict. Key order is load-bearing for byte parity."""
    trained_at = datetime.now(UTC).isoformat()
    filedict = {
        "model": model,
        "step": step,
        "stats": {
            "Accuracy": mode_stats["acc"],
            "Over Prec": mode_stats["prec"],
            "Under Prec": mode_stats["under_prec"],
            "Over%": mode_stats["over_pct"],
            "Sharpness": mode_stats["sharp"],
            "NLL": mode_stats["ll"],
        },
        "metrics": {
            "model": skill["model_metrics"],
            "book_baseline": skill["book_metrics"],
            "brier_skill_score": skill["brier_skill_score"],
            "kelly_shrinkage": skill["kelly_shrinkage"],
        },
        "diagnostics": {
            "model_weight": model_weight,
            "model_calib": model_calib,
            "brier_skill_score": skill["brier_skill_score"],
            "kelly_shrinkage": skill["kelly_shrinkage"],
            "shape_label": diag["shape_label"],
            "start_shape": diag["start_shape"],
            "model_shape": diag["model_shape"],
            "empirical_shape": diag["empirical_shape"],
            "start_mean": diag["start_mean"],
            "model_ev": diag["model_ev"],
            "mean_line": diag["mean_line"],
            "ev_minus_line": diag["ev_minus_line"],
            "result_mean": diag["result_mean"],
            "dispersion_cal": c_opt,
            "skew_cal": skew_cal,
            "median_ev_diff": diag["median_ev_diff"],
            "frac_ev_gt_line": diag["frac_ev_gt_line"],
            "over_pct_ev_gt": diag["over_pct_ev_gt"],
            "over_pct_ev_lt": diag["over_pct_ev_lt"],
            "cf_over_pct": diag["cf_over_pct"],
            "ev_meanyr_corr": diag["ev_meanyr_corr"],
            "result_meanyr_corr": diag["result_meanyr_corr"],
            "shape_ceiling": shape_ceiling,
            "marginal_shape": marginal_shape,
        },
        "params": opt_params,
        "distribution": dist,
        "cv": cv,
        "std": y.Result.std(),
        "temperature": T_opt,
        "dispersion_cal": c_opt,
        "skew_cal": skew_cal,
        "weight": model_weight,
        "r_book": None,
        "hist_gate": hist_gate,
        "shape_ceiling": shape_ceiling,
        "normalized": normalize,
        "offset_meta": (
            strategy.offset_meta(global_mean, denom_col) if strategy is not None else None
        ),
        "target_normalization": target_normalization,
        "posthoc": posthoc_slug,
        "posthoc_blob": posthoc_blob,
        "pit_recal_blob": pit_recal_blob,
        "zinb_mode": zinb_mode,
        "sn_param": sn_param,
        "is_hurdle": bool(getattr(model, "is_hurdle", False)),
        "expected_columns": list(X.columns),
        "trained_at": trained_at,
        "model_version": _model_version(
            trained_at,
            opt_params,
            dist,
            cv,
            step,
            target_normalization,
            structural_strategy=structural_strategy,
        ),
    }
    if algorithm_payload is not None:
        filedict["structural_calibration"] = algorithm_payload
    strategy_slug = structural_strategy if structural_strategy != BASE_STRUCTURAL_STRATEGY else dist
    filedict[MODEL_STRATEGY_MODEL_KEY] = build_artifact_identity(
        strategy_slug,
        league,
        market,
        selected_controls,
        algorithm_payload,
        matrix_hash=matrix_hash,
    ).as_model_blob()
    if expert_models is not None:
        filedict["expert_models"] = expert_models
    return filedict


def _persist_player_metadata(X_test: pd.DataFrame, splits: dict) -> None:
    # Player key + game date enable the offline scorecard's player-clustered Gate-1
    # bootstrap; the i.i.d. bootstrap over-credits repeated-player panels without them.
    # Guard is real — some leagues (e.g. team-level markets) omit "Player" entirely.
    if splits.get("players_test") is not None:
        X_test["Player"] = splits["players_test"]
    X_test["Date"] = splits["dates_test"]


def _deterministic_dump_suffix(dist: str, zinb_mode: str) -> str:
    """Deterministic dump subdir suffix, keyed on the *trained* architecture.

    The hurdle two-stage model engages only when ``dist == "ZINB" and zinb_mode == "hurdle"``
    (see ``use_hurdle`` in :func:`train_market`). A cell pins ``zinb_mode`` in stat_meta
    independent of its trained dist, so a ``--dist NegBin`` override on a hurdle-pinned cell
    trains plain NegBin and must dump to the non-hurdle path — keying on the raw flag would
    strand the sweep scorer on a stale ``_hurdle`` subdir.
    """
    return "_hurdle" if dist == "ZINB" and zinb_mode == "hurdle" else ""


def _deterministic_dump_subdir(
    dist: str,
    zinb_mode: str,
    target_normalization: str,
    structural_strategy: str = BASE_STRUCTURAL_STRATEGY,
) -> str:
    """Research artifact namespace, preserving the legacy default exactly."""
    subdir = f"{target_normalization}{_deterministic_dump_suffix(dist, zinb_mode)}"
    strategy_slug = structural_strategy if structural_strategy != BASE_STRUCTURAL_STRATEGY else dist
    return artifact_namespace(subdir, get_strategy(strategy_slug))


def _stage_family_shape_columns(
    X_test: pd.DataFrame,
    *,
    dist: str,
    prob_params: pd.DataFrame,
    weighted_mean: np.ndarray,
    sn_scale_test: np.ndarray | None,
    sn_skew_test: np.ndarray | None,
    mix_test: dict | None,
    r_test: np.ndarray | None,
    gate_blend_test: np.ndarray | float | None,
    phi_test: np.ndarray | None,
    target_normalization: str,
    global_mean: float,
    denom_col: str,
    hist_gate: float,
) -> None:
    """Stage the family's served shape columns onto the test-set frame.

    Gate 4 must score the SERVED predictive (blended params × dispersion c), the
    same distribution inference prices. Continuous families re-encode the served
    params to the model's normalized space, while count families persist their
    post-fusion, post-dispersion natural parameters directly. The scorecard's
    decode therefore recovers the priced distribution byte-for-byte. Encode is
    the exact inverse of decode only when both use this cell's denom_col, which
    the scorecard recovers from the persisted DenomCol.
    """
    # style: allow-complexity -- assembles per-family (SkewNormal/Mixture/ZINB) shape columns for scoring
    if dist == "SkewNormal":
        # Derive scipy loc via skewnormal_loc_from_mean — the single authoritative
        # formula shared with the betting path and the scorecard fit.
        strat = baselines.get_target_normalization(target_normalization)
        served_loc = skewnormal_loc_from_mean(weighted_mean, sn_scale_test, sn_skew_test)
        X_test["SN_Loc"] = strat.encode_loc(served_loc, X_test, global_mean, denom_col)
        X_test["SN_Scale"] = strat.encode_scale(sn_scale_test, X_test, denom_col)
        X_test["SN_Alpha"] = sn_skew_test
        # Zero-inflated cells encode against MeanYr_nonzero, not MeanYr; persist the
        # choice so the scorecard's Gate-4 decode reads the same denominator the
        # betting path serves with (else its dispersion is mis-scaled and g4 fails).
        X_test["DenomCol"] = denom_col
        # The EB-prior normalizations decode loc by re-adding an empirical-Bayes prior that
        # shrinks toward global_mean; persist it so the scorecard's `_decode_sn_loc_scale`
        # recovers the served loc instead of shrinking toward 0. Ratio/Mean10 decodes ignore it.
        X_test["GlobalMean"] = global_mean
        if gate_blend_test is not None:
            X_test["Gate"] = gate_blend_test
    elif dist == "Mixture":
        # Same served-predictive contract as the SkewNormal branch; the mixture
        # weight needs no encode — it is scale-free.
        strat = baselines.get_target_normalization(target_normalization)
        X_test["MIX_Loc1"] = strat.encode_loc(mix_test["loc1"], X_test, global_mean, denom_col)
        X_test["MIX_Scale1"] = strat.encode_scale(mix_test["scale1"], X_test, denom_col)
        X_test["MIX_Loc2"] = strat.encode_loc(mix_test["loc2"], X_test, global_mean, denom_col)
        X_test["MIX_Scale2"] = strat.encode_scale(mix_test["scale2"], X_test, denom_col)
        X_test["MIX_W1"] = mix_test["w1"]
        X_test["DenomCol"] = denom_col
        X_test["GlobalMean"] = global_mean
    elif dist in ("NegBin", "ZINB"):
        if r_test is None:
            raise ValueError(f"{dist} artifact persistence requires served r_test")
        X_test["R"] = r_test
        X_test["NB_P"] = weighted_mean / (r_test + weighted_mean)
        if dist == "ZINB":
            if gate_blend_test is None:
                raise ValueError("ZINB artifact persistence requires served gate_blend_test")
            X_test["Gate"] = gate_blend_test
    elif dist == "DPO":
        if phi_test is None:
            raise ValueError("DPO artifact persistence requires served phi_test")
        # DP_MU is the natural parameter, not the mean. Invert the served mean
        # after fusion while holding the served, dispersion-calibrated phi fixed.
        X_test["DP_MU"] = _dp_mu_from_mean(weighted_mean, phi_test)
        X_test["DP_PHI"] = phi_test
    elif dist in ("Gamma", "ZAGamma"):
        if dist == "ZAGamma":
            X_test["Gate"] = prob_params["gate"]
        X_test["Alpha"] = prob_params["concentration"]


def _step_persist_artifacts(
    *,
    filedict: dict,
    splits: dict,
    prob_params: pd.DataFrame,
    decoded: dict,
    weighted_mean: np.ndarray,
    y_proba_filt: np.ndarray,
    y_proba_raw: np.ndarray,
    dist: str,
    hist_gate: float,
    filename: str,
    deterministic: bool,
    target_normalization: str,
    zinb_mode: str,
    sn_scale_test: np.ndarray | None,
    sn_skew_test: np.ndarray | None,
    mix_test: dict | None,
    r_test: np.ndarray | None,
    gate_blend_test: np.ndarray | float | None,
    phi_test: np.ndarray | None,
    global_mean: float,
    denom_col: str,
    structural_rows: dict[str, pd.Series | None],
    structural_strategy: str = BASE_STRUCTURAL_STRATEGY,
    artifact_output: Path | None = None,
) -> None:
    """Write the scorecard-shaped test-set CSV and the model pickle.

    ``B_test["Odds"]`` enters in the training-layer book-over convention and is
    converted here to the scorecard's book-under convention. Family shape args
    are the served post-fusion, post-dispersion values, not raw model outputs.

    Deterministic mode redirects the model pickle to
    ``research/models/deterministic/{strategy}{_hurdle?}/`` at the repo root so
    the research harness can iterate without overwriting the production model
    dir. The test-set CSV still lives under the package ``data/test_sets/``.
    """
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    B_test = splits["B_test"]
    ev = decoded["ev"]

    X_test["Result"] = y_test["Result"]
    X_test["Line"] = B_test["Line"].values
    X_test["Blended_EV"] = weighted_mean
    # Internal B_* frames use book OVER probability for validation/blending;
    # scorecard-shaped CSV artifacts define Odds as book UNDER probability.
    X_test["Odds"] = 1.0 - B_test["Odds"].values
    # EV is the base mean the blend used, mean-corrected when a mean-stage
    # corrector is active, so the bias gates (Gate 2/3) read the corrected value.
    # The served shape columns below retain their separately calibrated
    # dispersion, which the mean-stage corrector intentionally leaves alone.
    X_test["EV"] = ev
    _stage_family_shape_columns(
        X_test,
        dist=dist,
        prob_params=prob_params,
        weighted_mean=weighted_mean,
        sn_scale_test=sn_scale_test,
        sn_skew_test=sn_skew_test,
        mix_test=mix_test,
        r_test=r_test,
        gate_blend_test=gate_blend_test,
        phi_test=phi_test,
        target_normalization=target_normalization,
        global_mean=global_mean,
        denom_col=denom_col,
        hist_gate=hist_gate,
    )

    if structural_rows["pit_recal_by_row"] is not None:
        # §6.1 Rung C map. One constant JSON payload from a whole-fit run, one payload per
        # cross-fit fold otherwise, so the CSV-only scorecard can apply g to the PIT before
        # the Gate-4 KS (mirrors DenomCol/GlobalMean). Absent column => no warp =>
        # byte-identical to a pre-Rung-C cell.
        X_test["PITRecalKnots"] = _aligned_rows(
            structural_rows, "pit_recal_by_row", X_test.index
        ).to_numpy()

    X_test["P"] = y_proba_filt[:, 1]
    if structural_rows["prepool_over"] is not None:
        prepool = _aligned_rows(structural_rows, "prepool_over", X_test.index).to_numpy(dtype=float)
        if not np.isfinite(prepool).all():
            raise ValueError("pre-pool probability must be finite and row-aligned")
        X_test["P_PrePool"] = prepool
    # Pre-blend (model-only) over-probability — lets the offline scorecard report a
    # standalone Gate-1 CI alongside the fused one, attributing the pass to model vs book.
    X_test["P_standalone"] = y_proba_raw[:, 1]
    quote_authenticity = splits.get("quote_authenticity_test")
    if quote_authenticity is not None:
        aligned_authenticity = pd.Series(quote_authenticity).reindex(X_test.index)
        if aligned_authenticity.isna().any():
            raise ValueError("test quote authenticity does not align to persisted rows")
        X_test["QuoteAuthenticity"] = aligned_authenticity.to_numpy()
    _persist_player_metadata(X_test, splits)
    for column, value in artifact_identity_columns(filedict[MODEL_STRATEGY_MODEL_KEY]).items():
        X_test[column] = value
    _persist_structural_columns(
        X_test, structural_strategy=structural_strategy, rows=structural_rows
    )

    # Under --deterministic, redirect to a `deterministic/` subdir so the
    # scorecard harness can score artifacts without overwriting production.
    # The training-data parquet stays suppressed (input is unchanged under
    # input-freeze); the whole-suite report() runs at the CLI's league
    # boundary and skips deterministic runs there for the same reason.
    # Test-set CSVs remain inside the package data tree; only the model
    # pickle moves to the repo-root research dir so the package install
    # never carries the research artifacts.
    if artifact_output is not None:
        csv_filepath = artifact_output / "test_sets" / f"{filename}.csv"
        mdl_filepath = artifact_output / f"{filename}.mdl"
    elif deterministic:
        strategy_subdir = _deterministic_dump_subdir(
            dist,
            zinb_mode,
            target_normalization,
            structural_strategy,
        )
        csv_subdir = f"deterministic/{strategy_subdir}/"
        mdl_dir = _DETERMINISTIC_MODEL_ROOT / strategy_subdir
    else:
        csv_subdir = ""
        mdl_dir = Path(str(pkg_resources.files(data) / "models"))
    if artifact_output is None:
        csv_filepath = pkg_resources.files(data) / f"test_sets/{csv_subdir}{filename}.csv"
        mdl_filepath = mdl_dir / f"{filename}.mdl"
    Path(str(csv_filepath.parent)).mkdir(parents=True, exist_ok=True)
    X_test.to_csv(csv_filepath)

    mdl_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(mdl_filepath, "wb") as outfile:
        pickle.dump(filedict, outfile, -1)


def _dispersion_crps_loss(
    c: float,
    *,
    dist: str,
    y_val_arr: np.ndarray,
    val_weighted_mean: np.ndarray,
    gate_blend_val: np.ndarray | None,
    r_blend_val: np.ndarray | None = None,
    alpha_blend_val: np.ndarray | None = None,
    phi_blend_val: np.ndarray | None = None,
) -> float:
    """CRPS-style dispersion calibration loss for a single scaling factor ``c``.

    Pure: no closures over module state. Used by ``minimize_scalar`` inside
    ``_step_calibrate_dispersion``.

    Args:
        c: Multiplicative scale on the model shape (NegBin r, Gamma α, DPO φ —
            var = μ/φ, so c > 1 narrows the predictive, same direction as r·c).
        dist: Distribution name.
        y_val_arr: Validation outcomes (1-D).
        val_weighted_mean: Per-row blended mean on the validation split.
        gate_blend_val: Per-row gate when ZI; None otherwise.
        r_blend_val: NegBin/ZINB ``r`` per row; None for other families.
        alpha_blend_val: Gamma α per row; None for other families.
        phi_blend_val: DPO φ per row; None for other families.

    Returns:
        CRPS + log-c² regularization.
    """
    if dist == "DPO":
        phi_cal = phi_blend_val * c
        # dp_crps takes the natural (mu, phi) — re-invert mu at each c so the
        # served mean is held fixed while only the dispersion moves.
        crps = dp_crps(y_val_arr, _dp_mu_from_mean(val_weighted_mean, phi_cal), phi_cal)
    elif dist in ("NegBin", "ZINB"):
        r_cal = r_blend_val * c
        p_cal = r_cal / (r_cal + val_weighted_mean)
        crps = negbin_crps(y_val_arr, r_cal, p_cal, gate=gate_blend_val)
    else:
        alpha_cal = alpha_blend_val * c
        scale_cal = val_weighted_mean / alpha_cal
        crps = gamma_crps(y_val_arr, alpha_cal, scale_cal, gate=gate_blend_val)
    reg = 0.01 * np.log(c) ** 2
    return np.mean(crps) + reg


def _count_pit_frame(
    dist: str,
    c: float,
    val_weighted_mean: np.ndarray,
    gate_blend_val: np.ndarray | None,
    r_blend_val: np.ndarray | None = None,
    alpha_blend_val: np.ndarray | None = None,
    phi_blend_val: np.ndarray | None = None,
) -> pd.DataFrame:
    """Scorecard param frame of the count/gamma predictive at dispersion scale ``c``.

    Shared by the dispersion fit's KS objective and the Lever-1 per-trial evaluator: the
    mean is held fixed while the family shape (NegBin ``r``, Gamma ``alpha``, DPO ``phi``)
    scales by ``c``.
    """
    if dist == "DPO":
        phi_cal = phi_blend_val * c
        # DP_MU/DP_PHI is the scorecard's DPO param-frame contract; natural mu is
        # re-inverted at each c so the served mean stays fixed (like NB_P below).
        params = pd.DataFrame(
            {"DP_MU": _dp_mu_from_mean(val_weighted_mean, phi_cal), "DP_PHI": phi_cal}
        )
    elif dist in ("NegBin", "ZINB"):
        r_cal = r_blend_val * c
        # `_pred_cdf_pmf` reads NB_P as the failure prob (scipy success = 1 − NB_P), so the
        # served mean is r·NB_P/(1−NB_P); hold it at val_weighted_mean ⇒ NB_P = mean/(r+mean).
        # (The CRPS path's `p_cal = r/(r+mean)` is `negbin_crps`'s reciprocal convention.)
        nb_p = val_weighted_mean / (r_cal + val_weighted_mean)
        params = pd.DataFrame({"R": r_cal, "NB_P": nb_p})
    else:
        params = pd.DataFrame({"Alpha": alpha_blend_val * c, "EV": val_weighted_mean})
    if gate_blend_val is not None:
        params["Gate"] = gate_blend_val
    return params


def _dispersion_pit_ks_loss(
    c: float,
    *,
    dist: str,
    y_val_arr: np.ndarray,
    val_weighted_mean: np.ndarray,
    gate_blend_val: np.ndarray | None,
    r_blend_val: np.ndarray | None = None,
    alpha_blend_val: np.ndarray | None = None,
    phi_blend_val: np.ndarray | None = None,
) -> float:
    """Gate-4 randomized-PIT KS dispersion loss for one count-branch scale ``c`` (Lever 2).

    The PIT-KS counterpart of :func:`_dispersion_crps_loss`: it minimizes the exact statistic
    Gate 4 scores instead of CRPS, mirroring the SkewNormal branch (which already fits its
    scale on the served PIT-KS). The same ``0.01·log(c)²`` brake as the CRPS path keeps a
    flat objective from driving ``c`` to an extreme (research brief open-question 2 guard).
    """
    params = _count_pit_frame(
        dist, c, val_weighted_mean, gate_blend_val, r_blend_val, alpha_blend_val, phi_blend_val
    )
    ks = _randomized_pit_ks(params, dist, y_val_arr, strategy=TARGET_NORM_NONE)
    reg = 0.01 * np.log(c) ** 2
    return ks + reg


def _brier_temperature_loss(T: float, val_logits: np.ndarray, y_class_val: np.ndarray) -> float:
    """Brier + (T-1)² regularization at temperature ``T``. Pure."""
    cal = expit(val_logits / T)
    brier = np.mean((cal - y_class_val) ** 2)
    reg = 0.01 * (T - 1) ** 2
    return brier + reg


def _blended_val_pit(fused: dict, y_val: np.ndarray) -> np.ndarray:
    """Per-row served PIT ``F(y)`` of the RAW blended SkewNormal validation predictive.

    The Rung C fit input: builds the same served-space df as :func:`_served_sn_pit_ks`
    (so the map is fit on the exact predictive Gate 4 scores) and averages the
    randomized-PIT draws to one value per row (a single ``F(y)`` for the continuous head).
    """
    mean = fused["weighted_mean_val"]
    scale = fused["sn_sigma_blend_val"]
    alpha = fused["sn_alpha_blend_val"]
    params = pd.DataFrame(
        {
            "SN_Loc": skewnormal_loc_from_mean(mean, scale, alpha),
            "SN_Scale": scale,
            "SN_Alpha": alpha,
        }
    )
    if fused.get("gate_blend_val") is not None:
        params["Gate"] = fused["gate_blend_val"]
    # A gated continuous head has a genuine atom at zero. Averaging randomized draws
    # would collapse that jump back to a non-uniform mid-PIT. Preserve a draw-by-row
    # matrix so Rung C can keep every row's draws together in cross-validation.
    draws = _randomized_pit_draws(params, "SkewNormal", y_val, strategy=TARGET_NORM_NONE)
    return draws[0] if len(draws) == 1 else np.stack(draws)


def _mixture_pit_frame(mix: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MIX_Loc1": mix["loc1"],
            "MIX_Scale1": mix["scale1"],
            "MIX_Loc2": mix["loc2"],
            "MIX_Scale2": mix["scale2"],
            "MIX_W1": mix["w1"],
        }
    )


def _calibrate_mixture_dispersion(out: dict, fused: dict, y_val_arr: np.ndarray) -> dict:
    """Mixture scalar dispersion fit: hold both locs, scale both component scales by ``c``.

    Objective is the served Gate-4 randomized-PIT KS of the blended validation
    predictive — the mixture has no closed-form CRPS, and g4 is the gate the family
    exists to clear. One shared ``c`` preserves the fitted component structure
    (separation and weights); per-component scaling would let the optimizer trade
    tail mass between modes and undo the mixture fit.
    """
    mix_val = fused["mix_blend_val"]

    def _served_pit_ks(c: float) -> float:
        return _randomized_pit_ks(
            _mixture_pit_frame(_scale_mixture(mix_val, c)),
            "Mixture",
            y_val_arr,
            strategy=TARGET_NORM_NONE,
        )

    disp_result = minimize_scalar(_served_pit_ks, bounds=(0.1, 10.0), method="bounded")
    c_opt = disp_result.x
    out["c_opt"] = c_opt
    out["mix_blend_test"] = _scale_mixture(fused["mix_blend_test"], c_opt)
    out["mix_blend_val"] = _scale_mixture(mix_val, c_opt)
    out["val_weighted_mean_val"] = fused["weighted_mean_val"]
    return out


def _calibrate_dpo_dispersion(
    out: dict, fused: dict, y_val_arr: np.ndarray, count_dispersion_objective: str
) -> dict:
    """DPO scalar dispersion fit: hold the blended mean, scale ``phi`` by ``c``.

    Same objective pair (CRPS / Gate-4 PIT-KS) as the NB/Gamma branch; ``c > 1``
    NARROWS the predictive (var = μ/φ), the same direction as NegBin ``r·c``. The
    bracket caps ``c`` so the scaled ``phi`` stays inside the family's hard
    ceiling (``phi_max·c ≤ _DP_PHI_CEILING``) — the NB shape_ceiling does not apply.
    """
    phi_blend_val = fused["phi_blend_val"]
    val_weighted_mean = fused["weighted_mean_val"]
    upper_bound = min(10.0, _DP_PHI_CEILING / float(np.max(phi_blend_val)))
    disp_loss = (
        _dispersion_pit_ks_loss if count_dispersion_objective == "pit_ks" else _dispersion_crps_loss
    )
    disp_result = minimize_scalar(
        lambda c: disp_loss(
            c,
            dist="DPO",
            y_val_arr=y_val_arr,
            val_weighted_mean=val_weighted_mean,
            gate_blend_val=None,
            phi_blend_val=phi_blend_val,
        ),
        bounds=(0.1, upper_bound),
        method="bounded",
    )
    c_opt = disp_result.x
    out["c_opt"] = c_opt
    out["phi_test"] = fused["phi_test"] * c_opt
    out["phi_blend_val"] = phi_blend_val * c_opt
    out["val_weighted_mean"] = val_weighted_mean
    return out


def _calibrate_count_dispersion(
    out: dict,
    fused: dict,
    dist: str,
    y_val_arr: np.ndarray,
    shape_ceiling,
    count_dispersion_objective: str,
) -> dict:
    """NegBin/ZINB/Gamma scalar dispersion fit: scale ``r``/``alpha`` by ``c``."""
    r_blend_val = fused["r_blend_val"]
    alpha_blend_val = fused["alpha_blend_val"]
    beta_blend_val = fused["beta_blend_val"]
    p_val = fused["p_val"]
    gate_blend_val = fused["gate_blend_val"]

    val_weighted_mean = (
        r_blend_val * (1 - p_val) / p_val
        if dist in ("NegBin", "ZINB")
        else alpha_blend_val / beta_blend_val
    )

    mean_shape = np.mean(r_blend_val) if dist in ("NegBin", "ZINB") else np.mean(alpha_blend_val)
    max_c = shape_ceiling / mean_shape if mean_shape > 0 else 10.0
    upper_bound = min(10.0, max_c)

    # Lever 2: minimize the served Gate-4 PIT-KS instead of CRPS when opted in. Same
    # signature, so the objective is a drop-in swap; default keeps the CRPS production path.
    disp_loss = (
        _dispersion_pit_ks_loss if count_dispersion_objective == "pit_ks" else _dispersion_crps_loss
    )
    disp_result = minimize_scalar(
        lambda c: disp_loss(
            c,
            dist=dist,
            y_val_arr=y_val_arr,
            val_weighted_mean=val_weighted_mean,
            gate_blend_val=gate_blend_val,
            r_blend_val=r_blend_val,
            alpha_blend_val=alpha_blend_val,
        ),
        bounds=(0.1, upper_bound),
        method="bounded",
    )
    c_opt = disp_result.x

    if dist in ("NegBin", "ZINB"):
        out["r_test"] = fused["r_test"] * c_opt
        out["r_blend_val"] = r_blend_val * c_opt
    else:
        out["alpha_blend"] = fused["alpha_blend"] * c_opt
        out["alpha_blend_val"] = alpha_blend_val * c_opt
        out["beta_blend_val"] = (alpha_blend_val * c_opt) / val_weighted_mean
    out["c_opt"] = c_opt
    out["val_weighted_mean"] = val_weighted_mean
    return out


def _step_calibrate_dispersion(
    decoded: dict,
    fused: dict,
    splits: dict,
    dist: str,
    cv: float,
    hist_gate: float,
    shape_ceiling,
    model_weight,
    *,
    count_dispersion_objective: str = "crps",
    posthoc_slug: str = "none",
) -> dict:
    """Fit dispersion scaling factor ``c_opt`` and apply it to blended params.

    Pure: returns a new dict with updated ``r_test``, ``r_blend_val``,
    ``alpha_blend``, ``alpha_blend_val``, ``beta_blend_val``, ``c_opt``,
    ``val_weighted_mean`` (count branch) or ``sn_sigma_blend_test`` (= blended
    test sigma × ``c_opt``), ``skew_cal`` + ``sn_alpha_blend_{test,val}`` (= blended
    alpha + ``skew_cal``) + ``val_weighted_mean_val`` (SkewNormal). The SkewNormal
    branch fits ``(c_opt, skew_cal)`` jointly (Lever 1 scale + Lever 4a additive skew
    shift) to minimize the Gate-4 randomized-PIT KS of the served (blended) predictive;
    ``skew_cal = 0`` recovers pure scale calibration. The count branch minimizes CRPS.
    Hurdle ZINB participates via the NegBin branch — gate passed through as ``gate_blend_val``.
    DPO scales the blended ``phi_test``/``phi_blend_val`` instead (mean held fixed).
    """
    y_val_arr = splits["y_validation"]["Result"].to_numpy()

    out = {
        "c_opt": 1.0,
        "skew_cal": 0.0,
        "r_test": fused["r_test"],
        "r_blend_val": fused["r_blend_val"],
        "alpha_blend": fused["alpha_blend"],
        "alpha_blend_val": fused["alpha_blend_val"],
        "beta_blend_val": fused["beta_blend_val"],
        "phi_test": fused["phi_test"],
        "phi_blend_val": fused["phi_blend_val"],
        "sn_sigma_blend_test": fused["sn_sigma_blend_test"],
        "sn_sigma_blend_val": fused["sn_sigma_blend_val"],
        "sn_alpha_blend_test": fused["sn_alpha_blend_test"],
        "sn_alpha_blend_val": fused["sn_alpha_blend_val"],
        "mix_blend_test": fused["mix_blend_test"],
        "mix_blend_val": fused["mix_blend_val"],
        "val_weighted_mean": None,
        "val_weighted_mean_val": None,
        "pit_recal_blob": None,
    }

    # §6.1 Rung C (posthoc CDF_STAGE) replaces the scalar (c, skew_cal): serve the RAW
    # blended params (c=1, s=0, already in `out`) and recalibrate the whole CDF via a
    # monotone PIT map fit — cross-fit lambda — on the validation PIT. Count families keep
    # the scalar (their port is deferred, plan §Out of scope), so the slug never breaks them.
    if posthoc_slug in posthoc.CDF_STAGE and dist == "SkewNormal":
        # The scalar (c, skew) is bypassed, but the temperature + test-prob steps still read
        # the served val mean off this dict — the raw sigma/alpha are already set above.
        out["val_weighted_mean_val"] = fused["weighted_mean_val"]
        blob, cv_ks = posthoc.select_pit_recal(_blended_val_pit(fused, y_val_arr))
        out["pit_recal_blob"] = blob
        if blob is not None:
            logger.info("Rung C: lambda=%.2f cross-fit pit_ks=%.4f", blob["lam"], cv_ks)
        return out

    if dist == "SkewNormal":
        # Fit c (Lever 1, scale) and s (Lever 4a, additive skew shift) jointly against the
        # served (blended) predictive — the exact thing inference prices and Gate 4 scores:
        # mean held fixed at the blended val mean, scale scaled by c, shape shifted by s.
        # Reuses the gate's own randomized-PIT KS (no PIT-math dup); s = 0 ⇒ pure Lever 1.
        c_opt, s_opt = fit_skewnorm_dispersion_skew(
            fused["weighted_mean_val"],
            fused["sn_sigma_blend_val"],
            fused["sn_alpha_blend_val"],
            y_val_arr,
            gate=fused["gate_blend_val"],
        )
        out["c_opt"] = c_opt
        out["skew_cal"] = s_opt
        out["sn_sigma_blend_test"] = fused["sn_sigma_blend_test"] * c_opt
        out["sn_sigma_blend_val"] = fused["sn_sigma_blend_val"] * c_opt
        out["sn_alpha_blend_test"] = fused["sn_alpha_blend_test"] + s_opt
        out["sn_alpha_blend_val"] = fused["sn_alpha_blend_val"] + s_opt
        out["val_weighted_mean_val"] = fused["weighted_mean_val"]
        return out

    if dist == "Mixture":
        return _calibrate_mixture_dispersion(out, fused, y_val_arr)

    if dist == "DPO":
        return _calibrate_dpo_dispersion(out, fused, y_val_arr, count_dispersion_objective)
    return _calibrate_count_dispersion(
        out, fused, dist, y_val_arr, shape_ceiling, count_dispersion_objective
    )


def _step_compute_test_probabilities(
    fused: dict,
    calibrated: dict,
    splits: dict,
    decoded: dict,
    dist: str,
    step,
) -> np.ndarray:
    """Compute ``y_proba_no_filt`` (Nx2) for the test set after dispersion calibration."""
    B_test = splits["B_test"]
    weighted_mean = fused["weighted_mean"]
    if dist == "SkewNormal":
        under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            "SkewNormal",
            sigma=calibrated["sn_sigma_blend_test"],
            skew_alpha=calibrated["sn_alpha_blend_test"],
            gate=fused["gate_blend_test"],
        )
    elif dist == "Mixture":
        under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            mix=calibrated["mix_blend_test"],
        )
    elif dist in ("NegBin", "ZINB"):
        under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            r=calibrated["r_test"],
            gate=fused["gate_blend_test"],
        )
    elif dist == "DPO":
        under = get_odds(B_test["Line"].to_numpy(), weighted_mean, dist, phi=calibrated["phi_test"])
    else:
        under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            alpha=calibrated["alpha_blend"],
            step=step,
            gate=fused["gate_blend_test"],
        )
    # §6.1 Rung C: serve the recalibrated CDF F*=g∘F, so the persisted over-prob (Gate 1/5)
    # matches what model_prob serves. Identity (no copy) for a cell without a map.
    under = apply_cdf_recal(calibrated.get("pit_recal_blob"), under)
    return np.array([under, 1 - under]).transpose()


def _step_calibrate_temperature(
    fused: dict,
    calibrated: dict,
    splits: dict,
    decoded: dict,
    dist: str,
    step,
) -> tuple[float, np.ndarray, float]:
    """Fit temperature ``T_opt`` on validation, return calibrated val probs + model_calib."""
    B_validation = splits["B_validation"]
    y_class_val = (splits["y_validation"]["Result"] >= B_validation["Line"]).astype(int).to_numpy()

    if dist == "SkewNormal":
        val_raw_under = get_odds(
            B_validation["Line"].to_numpy(),
            calibrated["val_weighted_mean_val"],
            "SkewNormal",
            sigma=calibrated["sn_sigma_blend_val"],
            skew_alpha=calibrated["sn_alpha_blend_val"],
            gate=fused["gate_blend_val"],
        )
    elif dist == "Mixture":
        val_raw_under = get_odds(
            B_validation["Line"].to_numpy(),
            calibrated["val_weighted_mean_val"],
            dist,
            mix=calibrated["mix_blend_val"],
        )
    else:
        _r_val = calibrated["r_blend_val"] if dist in ("NegBin", "ZINB") else None
        _alpha_val = calibrated["alpha_blend_val"] if dist in ("Gamma", "ZAGamma") else None
        _gate_val = fused["gate_blend_val"] if dist in ("ZINB", "ZAGamma") else None
        _phi_val = calibrated["phi_blend_val"] if dist == "DPO" else None
        val_raw_under = get_odds(
            B_validation["Line"].to_numpy(),
            calibrated["val_weighted_mean"],
            dist,
            alpha=_alpha_val,
            step=step,
            r=_r_val,
            gate=_gate_val,
            phi=_phi_val,
        )
    # §6.1 Rung C: temperature calibrates the SERVED over-prob, so fit it on the warped CDF.
    val_raw_under = apply_cdf_recal(calibrated.get("pit_recal_blob"), val_raw_under)
    val_raw_over_clipped = np.clip(1 - val_raw_under, 1e-6, 1 - 1e-6)
    val_logits = logit(val_raw_over_clipped)
    result_ts = minimize_scalar(
        lambda T: _brier_temperature_loss(T, val_logits, y_class_val),
        bounds=(1.0, 10.0),
        method="bounded",
    )
    T_opt = result_ts.x
    val_calibrated = apply_temperature(1 - val_raw_under, T_opt)
    model_calib = 1 - np.mean((val_calibrated - y_class_val) ** 2)
    return T_opt, val_calibrated, model_calib, y_class_val


def _mixture_moments(w1, loc1, scale1, loc2, scale2):
    """Mean and sd of a 2-component Gaussian mixture (same-length arrays)."""
    mean = w1 * loc1 + (1 - w1) * loc2
    second = w1 * (scale1**2 + loc1**2) + (1 - w1) * (scale2**2 + loc2**2)
    return mean, np.sqrt(np.clip(second - mean**2, 1e-12, None))


def _decode_mixture_frame(prob_params, X, strategy, global_mean, denom_col) -> dict:
    """Decode raw Mixture params to EV-space ``{w1, loc1, scale1, loc2, scale2}``.

    Both component locations/scales run through the same strategy registry decode
    the SkewNormal branch uses, so the mixture is expressed in the space the
    gates and the betting path price.
    """
    return {
        "w1": prob_params["mix_prob_1"].to_numpy(dtype=float),
        "loc1": strategy.decode_loc(
            prob_params["loc_1"].to_numpy(dtype=float), X, global_mean, denom_col
        ),
        "scale1": strategy.decode_scale(prob_params["scale_1"].to_numpy(dtype=float), X, denom_col),
        "loc2": strategy.decode_loc(
            prob_params["loc_2"].to_numpy(dtype=float), X, global_mean, denom_col
        ),
        "scale2": strategy.decode_scale(prob_params["scale_2"].to_numpy(dtype=float), X, denom_col),
    }


def _shift_mixture(mix: dict, shift: np.ndarray) -> dict:
    """Location-shift a decoded mixture dict (blend moves the mean, shape stays)."""
    return {**mix, "loc1": mix["loc1"] + shift, "loc2": mix["loc2"] + shift}


def _scale_mixture(mix: dict, c: float) -> dict:
    """Dispersion-calibrate a decoded mixture dict (both component scales by ``c``)."""
    return {**mix, "scale1": mix["scale1"] * c, "scale2": mix["scale2"] * c}


def _step_decode_predictions(
    prob_params: pd.DataFrame,
    prob_params_validation: pd.DataFrame,
    X_test,
    X_validation,
    dist: str,
    strategy,
    global_mean: float,
    denom_col: str,
    hist_gate: float,
) -> dict:
    """Decode raw distribution parameters to per-row EVs and shape vectors.

    SkewNormal applies the strategy's decode_loc/decode_scale, then its family
    mean adjustment. NegBin/ZINB: EV = r·p/(1−p).
    DPO: EV is the exact series mean (``mu`` is only approximate) with ``phi`` alongside.
    Gamma/ZAGamma: EV = α/β. Synthesizes a constant ``gate_*`` vector for
    SkewNormal when ``hist_gate > GATE_PUBLISH_THRESHOLD`` (no per-row gate from the model).

    Returns:
        Dict with: ``ev``, ``ev_validation``, ``gate_test``, ``gate_validation``,
        ``sn_sigma_test``, ``sn_sigma_val``, ``sn_alpha_test``, ``sn_alpha_val``,
        ``r``, ``p``, ``r_validation``, ``p_validation``, ``phi``, ``phi_validation``,
        ``alpha``, ``beta``, ``alpha_validation``, ``beta_validation``.
        Unused fields are None.
    """
    out = {
        "ev": None,
        "ev_validation": None,
        "gate_test": None,
        "gate_validation": None,
        "sn_sigma_test": None,
        "sn_sigma_val": None,
        "sn_alpha_test": None,
        "sn_alpha_val": None,
        "r": None,
        "p": None,
        "r_validation": None,
        "p_validation": None,
        "phi": None,
        "phi_validation": None,
        "alpha": None,
        "beta": None,
        "alpha_validation": None,
        "beta_validation": None,
        "mix_test": None,
        "mix_val": None,
    }
    if dist == "Mixture":
        mix_test = _decode_mixture_frame(prob_params, X_test, strategy, global_mean, denom_col)
        mix_val = _decode_mixture_frame(
            prob_params_validation, X_validation, strategy, global_mean, denom_col
        )
        out["mix_test"] = mix_test
        out["mix_val"] = mix_val
        out["ev"], _ = _mixture_moments(**mix_test)
        out["ev_validation"], _ = _mixture_moments(**mix_val)
        out["gate_test"] = None
        out["gate_validation"] = None
        return out
    if dist == "SkewNormal":
        # Location/scale come from the baselines registry (feature/strategy
        # dependent); the shared kernel applies the skew mean-adjustment and gate.
        ev_loc = strategy.decode_loc(prob_params["loc"].to_numpy(), X_test, global_mean, denom_col)
        ev_scale = strategy.decode_scale(prob_params["scale"].to_numpy(), X_test, denom_col)
        ev_loc_val = strategy.decode_loc(
            prob_params_validation["loc"].to_numpy(), X_validation, global_mean, denom_col
        )
        ev_scale_val = strategy.decode_scale(
            prob_params_validation["scale"].to_numpy(), X_validation, denom_col
        )
        decoded_test = decode_predictive_mean(
            prob_params, dist, sn_loc=ev_loc, sn_scale=ev_scale, hist_gate=hist_gate
        )
        decoded_val = decode_predictive_mean(
            prob_params_validation,
            dist,
            sn_loc=ev_loc_val,
            sn_scale=ev_scale_val,
            hist_gate=hist_gate,
        )
        out["sn_sigma_test"] = decoded_test.sigma
        out["sn_sigma_val"] = decoded_val.sigma
        out["sn_alpha_test"] = decoded_test.skew
        out["sn_alpha_val"] = decoded_val.skew
    elif dist in ("NegBin", "ZINB"):
        decoded_test = decode_predictive_mean(prob_params, dist)
        decoded_val = decode_predictive_mean(prob_params_validation, dist)
        out["r"] = decoded_test.r
        out["p"] = prob_params["probs"].to_numpy()
        out["r_validation"] = decoded_val.r
        out["p_validation"] = prob_params_validation["probs"].to_numpy()
    elif dist == "DPO":
        decoded_test = decode_predictive_mean(prob_params, dist)
        decoded_val = decode_predictive_mean(prob_params_validation, dist)
        out["phi"] = decoded_test.phi
        out["phi_validation"] = decoded_val.phi
    else:
        decoded_test = decode_predictive_mean(prob_params, dist)
        decoded_val = decode_predictive_mean(prob_params_validation, dist)
        out["alpha"] = decoded_test.alpha
        out["beta"] = prob_params["rate"].to_numpy()
        out["alpha_validation"] = decoded_val.alpha
        out["beta_validation"] = prob_params_validation["rate"].to_numpy()

    out["ev"] = decoded_test.ev
    out["ev_validation"] = decoded_val.ev
    out["gate_test"] = decoded_test.gate
    out["gate_validation"] = decoded_val.gate
    return out


def _split_quote_authenticity_mask(splits: dict, split: str) -> np.ndarray:
    """Return a row-aligned mask for quotes that may contribute book evidence.

    Current matrices carry the explicit three-state provenance contract. Legacy
    matrices fall back to their archived/synthetic compatibility columns, and
    truly provenance-free inputs retain their historical all-priced behavior.
    """
    index = splits[f"B_{split}"].index
    authenticity = splits.get(f"quote_authenticity_{split}")
    if authenticity is not None:
        values = pd.Series(authenticity).reindex(index)
        if values.isna().any() or not values.isin(("authentic", "derived", "synthetic")).all():
            raise ValueError(f"{split} quote authenticity is missing, unaligned, or unsupported")
        return values.eq("authentic").to_numpy(dtype=bool)

    archived = splits.get(f"archived_{split}")
    synthetic = splits.get(f"odds_synthetic_{split}")
    if archived is None and synthetic is None:
        return np.ones(len(index), dtype=bool)

    archived_values = (
        pd.Series(archived).reindex(index).fillna(False).astype(bool)
        if archived is not None
        else pd.Series(True, index=index)
    )
    synthetic_values = (
        pd.Series(synthetic).reindex(index).fillna(True).astype(bool)
        if synthetic is not None
        else pd.Series(False, index=index)
    )
    return (archived_values & ~synthetic_values).to_numpy(dtype=bool)


def _fuse_skewnormal(out, decoded, splits, cv, hist_gate, blending):
    """SkewNormal branch: fit model_weight, blend sigma / skew on test + validation."""
    ev = decoded["ev"]
    ev_validation = decoded["ev_validation"]
    book_ev_test = splits["B_test"]["EV"].to_numpy()
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    y_val_result = splits["y_validation"]["Result"].to_numpy()
    authentic_test = _split_quote_authenticity_mask(splits, "test")
    authentic_val = _split_quote_authenticity_mask(splits, "validation")

    # SkewNormal book EVs are archived without a zero gate. The external hurdle
    # belongs to the positive-only model head, so the two blend endpoints are
    # model_gate and zero. This preserves the ungated sportsbook endpoint at w=0.
    _fit_gate_kwargs = (
        {"gate_model": decoded["gate_validation"], "gate_book": 0.0}
        if hist_gate > GATE_PUBLISH_THRESHOLD and decoded["gate_validation"] is not None
        else {}
    )
    if authentic_val.any():
        model_weight = calibration.fit_blend_weight(
            blending,
            ev_validation[authentic_val],
            book_ev_val[authentic_val],
            y_val_result[authentic_val],
            "SkewNormal",
            cv=cv,
            model_sigma=decoded["sn_sigma_val"][authentic_val],
            model_skew_alpha=decoded["sn_alpha_val"][authentic_val],
            **{
                key: (value[authentic_val] if isinstance(value, np.ndarray) else value)
                for key, value in _fit_gate_kwargs.items()
            },
        )
    else:
        model_weight = 1.0
    _test_gate_kwargs = (
        {"gate_model": decoded["gate_test"], "gate_book": 0.0}
        if hist_gate > GATE_PUBLISH_THRESHOLD and decoded["gate_test"] is not None
        else {}
    )
    _val_gate_kwargs = (
        {"gate_model": decoded["gate_validation"], "gate_book": 0.0}
        if hist_gate > GATE_PUBLISH_THRESHOLD and decoded["gate_validation"] is not None
        else {}
    )
    test_weight = np.where(authentic_test, model_weight, 1.0)
    val_weight = np.where(authentic_val, model_weight, 1.0)
    weighted_mean, sn_sigma_blend_test, sn_alpha_blend_test, gate_blend_test = fused_loc(
        test_weight,
        ev,
        book_ev_test,
        cv,
        "SkewNormal",
        sigma=decoded["sn_sigma_test"],
        skew_alpha=decoded["sn_alpha_test"],
        **_test_gate_kwargs,
    )
    weighted_mean_val, sn_sigma_blend_val, sn_alpha_blend_val, gate_blend_val = fused_loc(
        val_weight,
        ev_validation,
        book_ev_val,
        cv,
        "SkewNormal",
        sigma=decoded["sn_sigma_val"],
        skew_alpha=decoded["sn_alpha_val"],
        **_val_gate_kwargs,
    )
    out.update(
        {
            "model_weight": model_weight,
            "weighted_mean": weighted_mean,
            "weighted_mean_val": weighted_mean_val,
            "gate_blend_test": gate_blend_test,
            "gate_blend_val": gate_blend_val,
            "sn_sigma_blend_test": sn_sigma_blend_test,
            "sn_sigma_blend_val": sn_sigma_blend_val,
            "sn_alpha_blend_test": sn_alpha_blend_test,
            "sn_alpha_blend_val": sn_alpha_blend_val,
        }
    )
    return out


def _fuse_mixture(out, decoded, splits, cv, blending):
    """Mixture branch: fit model_weight on moment-matched normals, blend by location shift.

    The weight fit scores blended means through the SkewNormal machinery with the
    mixture's per-row sd and zero skew — a moment-matched normal approximation used
    ONLY to pick the scalar blend weight. The served predictive keeps the full
    mixture shape: the blend moves both component locations by the pooled-mean
    delta and leaves scales/weight untouched.
    """
    ev = decoded["ev"]
    ev_validation = decoded["ev_validation"]
    book_ev_test = splits["B_test"]["EV"].to_numpy()
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    y_val_result = splits["y_validation"]["Result"].to_numpy()
    _, mix_sd_val = _mixture_moments(**decoded["mix_val"])
    _, mix_sd_test = _mixture_moments(**decoded["mix_test"])
    authentic_test = _split_quote_authenticity_mask(splits, "test")
    authentic_val = _split_quote_authenticity_mask(splits, "validation")

    if authentic_val.any():
        model_weight = calibration.fit_blend_weight(
            blending,
            ev_validation[authentic_val],
            book_ev_val[authentic_val],
            y_val_result[authentic_val],
            "SkewNormal",
            cv=cv,
            model_sigma=mix_sd_val[authentic_val],
            model_skew_alpha=np.zeros(int(authentic_val.sum()), dtype=float),
        )
    else:
        model_weight = 1.0
    weighted_mean, _, _, _ = fused_loc(
        np.where(authentic_test, model_weight, 1.0),
        ev,
        book_ev_test,
        cv,
        "SkewNormal",
        sigma=mix_sd_test,
        skew_alpha=np.zeros_like(mix_sd_test),
    )
    weighted_mean_val, _, _, _ = fused_loc(
        np.where(authentic_val, model_weight, 1.0),
        ev_validation,
        book_ev_val,
        cv,
        "SkewNormal",
        sigma=mix_sd_val,
        skew_alpha=np.zeros_like(mix_sd_val),
    )
    out.update(
        {
            "model_weight": model_weight,
            "weighted_mean": weighted_mean,
            "weighted_mean_val": weighted_mean_val,
            "mix_blend_test": _shift_mixture(decoded["mix_test"], weighted_mean - ev),
            "mix_blend_val": _shift_mixture(decoded["mix_val"], weighted_mean_val - ev_validation),
        }
    )
    return out


def _fit_nonsn_weight(out, decoded, splits, base_dist, dist, cv, hist_gate, blending):
    """Fit model_weight for the NegBin / Gamma families (shared preamble)."""
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    y_val_result = splits["y_validation"]["Result"].to_numpy()
    authentic_val = _split_quote_authenticity_mask(splits, "validation")

    _zi_kwargs = {}
    if dist in ("ZINB", "ZAGamma") and hist_gate > 0:
        _zi_kwargs = {"gate_model": decoded["gate_validation"], "gate_book": hist_gate}
    if authentic_val.any():
        model_weight = calibration.fit_blend_weight(
            blending,
            decoded["ev_validation"][authentic_val],
            book_ev_val[authentic_val],
            y_val_result[authentic_val],
            base_dist,
            model_alpha=(
                decoded["alpha_validation"][authentic_val]
                if decoded["alpha_validation"] is not None
                else None
            ),
            model_r=(
                decoded["r_validation"][authentic_val]
                if decoded["r_validation"] is not None
                else None
            ),
            cv=cv,
            **{
                key: (value[authentic_val] if isinstance(value, np.ndarray) else value)
                for key, value in _zi_kwargs.items()
            },
        )
    else:
        model_weight = 1.0
    out["model_weight"] = model_weight
    return model_weight


def _fuse_negbin(out, decoded, splits, model_weight, cv, hist_gate, dist):
    """NegBin / ZINB branch: blend r + derive p on test + validation."""
    ev = decoded["ev"]
    ev_validation = decoded["ev_validation"]
    book_ev_test = splits["B_test"]["EV"].to_numpy()
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    test_weight = np.where(_split_quote_authenticity_mask(splits, "test"), model_weight, 1.0)
    val_weight = np.where(_split_quote_authenticity_mask(splits, "validation"), model_weight, 1.0)

    _zi_test = (
        {"gate_model": decoded["gate_test"], "gate_book": hist_gate} if dist == "ZINB" else {}
    )
    _zi_val = (
        {"gate_model": decoded["gate_validation"], "gate_book": hist_gate} if dist == "ZINB" else {}
    )
    r_blend_test, p_test, gate_blend_test = fused_loc(
        test_weight,
        ev,
        book_ev_test,
        cv,
        "NegBin",
        r=decoded["r"],
        **_zi_test,
    )
    r_blend_val, p_val, gate_blend_val = fused_loc(
        val_weight,
        ev_validation,
        book_ev_val,
        cv,
        "NegBin",
        r=decoded["r_validation"],
        **_zi_val,
    )
    out.update(
        {
            "weighted_mean": r_blend_test * (1 - p_test) / p_test,
            "r_test": r_blend_test,
            "r_blend_val": r_blend_val,
            "p_test": p_test,
            "p_val": p_val,
            "gate_blend_test": gate_blend_test,
            "gate_blend_val": gate_blend_val,
        }
    )
    return out


def _fuse_dpo(out, decoded, splits, cv, blending):
    """DPO branch: fit the blend weight, then log-pool exact mean + phi on test + validation.

    Gate-free by design (no ZI kwargs), mirroring plain NegBin; ``fused_loc`` returns the
    blended MEAN directly — ``get_odds`` re-inverts mean → ``mu`` internally.
    """
    book_ev_test = splits["B_test"]["EV"].to_numpy()
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    y_val_result = splits["y_validation"]["Result"].to_numpy()
    authentic_test = _split_quote_authenticity_mask(splits, "test")
    authentic_val = _split_quote_authenticity_mask(splits, "validation")

    if authentic_val.any():
        model_weight = calibration.fit_dpo_weight(
            decoded["ev_validation"][authentic_val],
            book_ev_val[authentic_val],
            y_val_result[authentic_val],
            decoded["phi_validation"][authentic_val],
            cv,
            blending,
        )
    else:
        model_weight = 1.0
    weighted_mean, phi_blend_test, _ = fused_loc(
        np.where(authentic_test, model_weight, 1.0),
        decoded["ev"],
        book_ev_test,
        cv,
        "DPO",
        phi=decoded["phi"],
    )
    weighted_mean_val, phi_blend_val, _ = fused_loc(
        np.where(authentic_val, model_weight, 1.0),
        decoded["ev_validation"],
        book_ev_val,
        cv,
        "DPO",
        phi=decoded["phi_validation"],
    )
    out.update(
        {
            "model_weight": model_weight,
            "weighted_mean": weighted_mean,
            "weighted_mean_val": weighted_mean_val,
            "phi_test": phi_blend_test,
            "phi_blend_val": phi_blend_val,
        }
    )
    return out


def _fuse_gamma(out, decoded, splits, model_weight, cv, hist_gate, dist):
    """Gamma / ZAGamma branch: blend alpha + beta on test + validation."""
    ev = decoded["ev"]
    ev_validation = decoded["ev_validation"]
    book_ev_test = splits["B_test"]["EV"].to_numpy()
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    test_weight = np.where(_split_quote_authenticity_mask(splits, "test"), model_weight, 1.0)
    val_weight = np.where(_split_quote_authenticity_mask(splits, "validation"), model_weight, 1.0)

    _zi_test = (
        {"gate_model": decoded["gate_test"], "gate_book": hist_gate} if dist == "ZAGamma" else {}
    )
    _zi_val = (
        {"gate_model": decoded["gate_validation"], "gate_book": hist_gate}
        if dist == "ZAGamma"
        else {}
    )
    alpha_blend, beta_blend, gate_blend_test = fused_loc(
        test_weight,
        ev,
        book_ev_test,
        cv,
        "Gamma",
        alpha=decoded["alpha"],
        **_zi_test,
    )
    alpha_blend_val, beta_blend_val, gate_blend_val = fused_loc(
        val_weight,
        ev_validation,
        book_ev_val,
        cv,
        "Gamma",
        alpha=decoded["alpha_validation"],
        **_zi_val,
    )
    out.update(
        {
            "weighted_mean": alpha_blend / beta_blend,
            "alpha_blend": alpha_blend,
            "alpha_blend_val": alpha_blend_val,
            "beta_blend": beta_blend,
            "beta_blend_val": beta_blend_val,
            "gate_blend_test": gate_blend_test,
            "gate_blend_val": gate_blend_val,
        }
    )
    return out


def _step_fuse_predictions(
    decoded: dict,
    splits: dict,
    dist: str,
    cv: float,
    hist_gate: float,
    blending: str = calibration.DEFAULT_BLENDING,
) -> dict:
    """Fit model_weight, then fuse model+book predictions on test and validation.

    Args:
        decoded: Output of ``_step_decode_predictions``.
        splits: Output of ``_step_build_splits`` (post-prediction sorted).
        dist: Distribution name.
        cv: Coefficient of variation from ``_step_select_distribution``.
        hist_gate: Historical zero rate.
        blending: Blending loss slug forwarded to ``calibration.fit_blend_weight``.

    Returns:
        Dict with: ``model_weight``, ``weighted_mean``, ``gate_blend_test``,
        ``gate_blend_val``, ``r_test``, ``r_blend_val``, ``p_test``, ``p_val``,
        ``phi_test``, ``phi_blend_val``, ``alpha_blend``, ``alpha_blend_val``,
        ``beta_blend``, ``beta_blend_val``, ``sn_sigma_blend_test``,
        ``sn_sigma_blend_val``, ``sn_alpha_blend_test``, ``sn_alpha_blend_val``.
        Unused fields are None.
    """
    base_dist = (
        "SkewNormal"
        if dist == "SkewNormal"
        else ("NegBin" if dist in ("NegBin", "ZINB") else "Gamma")
    )

    out = {
        "model_weight": None,
        "weighted_mean": None,
        "weighted_mean_val": None,
        "gate_blend_test": None,
        "gate_blend_val": None,
        "r_test": None,
        "r_blend_val": None,
        "p_test": None,
        "p_val": None,
        "phi_test": None,
        "phi_blend_val": None,
        "alpha_blend": None,
        "alpha_blend_val": None,
        "beta_blend": None,
        "beta_blend_val": None,
        "sn_sigma_blend_test": None,
        "sn_sigma_blend_val": None,
        "sn_alpha_blend_test": None,
        "sn_alpha_blend_val": None,
        "mix_blend_test": None,
        "mix_blend_val": None,
    }

    if dist == "Mixture":
        return _fuse_mixture(out, decoded, splits, cv, blending)
    if dist == "SkewNormal":
        return _fuse_skewnormal(out, decoded, splits, cv, hist_gate, blending)
    if dist == "DPO":
        return _fuse_dpo(out, decoded, splits, cv, blending)

    model_weight = _fit_nonsn_weight(out, decoded, splits, base_dist, dist, cv, hist_gate, blending)
    if dist in ("NegBin", "ZINB"):
        return _fuse_negbin(out, decoded, splits, model_weight, cv, hist_gate, dist)
    return _fuse_gamma(out, decoded, splits, model_weight, cv, hist_gate, dist)


def _data_driven_dist(global_mean: float, hist_gate: float) -> str:
    """The distribution the data implies: ``SkewNormal`` at ``global_mean >= _SKEWNORMAL_MEAN_THRESHOLD``,
    else the count branch — ``ZINB`` when the zero rate clears ``GATE_PUBLISH_THRESHOLD``, else ``NegBin``.
    """
    if global_mean >= _SKEWNORMAL_MEAN_THRESHOLD:
        return "SkewNormal"
    return "ZINB" if hist_gate > GATE_PUBLISH_THRESHOLD else "NegBin"


def _resolve_dist(configured: str | None, data_dist: str, league: str, market: str) -> str:
    """The distribution family to train: the configured one (authoritative — ``--dist`` flag or
    stat_meta, where the sweep persists its gate-validated pick), or ``data_dist`` when nothing
    is pinned / ``"auto"``.

    Disagreement with the mean/zero-rate heuristic is expected, not warned — DPO and Mixture are
    force-only, so every such cell would warn weekly. An unrecognized family is a config error:
    fail loud.
    """
    if configured in (None, "auto"):
        return data_dist
    if configured not in _FORCEABLE_DISTS:
        raise ValueError(
            f"{league} {market}: stat_meta dist {configured!r} is not a forceable family; "
            f"use one of {sorted(_FORCEABLE_DISTS)} or 'auto'"
        )
    return configured


def _skewnormal_dist_obj(
    sn_param: str, stabilization: str, dist_training_loss: str, y_train_labels
):
    """The continuous-branch distribution object for the requested parametrization.

    ``"centered"`` boosts (mean, sd, gamma1) heads via :class:`CenteredSkewNormal`,
    whose ``predict_dist`` re-emits direct (loc, scale, alpha) columns — the family
    string and every ``dist == "SkewNormal"`` consumer downstream stay unchanged.

    Both parametrizations clamp the sigma-like head to the robust normalized-label
    scale, and the direct alpha head to ``+-_SKEWNORMAL_ALPHA_BOUND`` — the boosted
    skew-normal likelihood otherwise diverges inside one-sided leaves (monotone in
    alpha, unbounded ridge in the exp sigma response). The wrapped ``param_dict``
    rides the pickle, so the final refit, the warm cron, and inference share the
    bound.
    """
    loss_fn = _resolve_loss_fn("crps", dist_training_loss)
    q75, q25 = np.percentile(y_train_labels, [75, 25])
    label_scale = (q75 - q25) / _IQR_TO_SIGMA
    scale_ceiling = _SKEWNORMAL_SCALE_ROBUST_CEILING * label_scale
    scale_floor = _SKEWNORMAL_SCALE_ROBUST_FLOOR * label_scale
    if sn_param == "centered":
        # Unstabilized centered heads NaN out within ~4 rounds (mapped-alpha hessians
        # explode near the gamma1 bound), so the module-default "None" upgrades to the
        # family's own L2 default; an explicit MAD/L2 from the caller is honored.
        if stabilization == "None":
            stabilization = "L2"
        sn = CenteredSkewNormal(stabilization=stabilization, loss_fn=loss_fn)
        # gamma1 is already tanh-bounded to +-0.99; the mean head stays unconstrained.
        sn.param_dict["sd"] = _BoundedResponseFn(
            sn.param_dict["sd"], scale_ceiling, floor=scale_floor
        )
        return sn
    sn = SkewNormalDist(stabilization=stabilization, loss_fn=loss_fn)
    sn.param_dict["scale"] = _BoundedResponseFn(
        sn.param_dict["scale"], scale_ceiling, floor=scale_floor
    )
    sn.param_dict["alpha"] = _BoundedResponseFn(
        sn.param_dict["alpha"], _SKEWNORMAL_ALPHA_BOUND, floor=-_SKEWNORMAL_ALPHA_BOUND
    )
    return sn


def _continuous_dist_obj(
    dist: str, sn_param: str, stabilization: str, dist_training_loss: str, y_train_labels
):
    """Distribution object for the continuous branch.

    ``y_train_labels`` must already be in the strategy's normalized space — both
    families derive their scale-head clamp from it (robust IQR/1.349 scale for
    SkewNormal, std for the Mixture, whose cells are all additive).
    """
    if dist == "SkewNormal":
        return _skewnormal_dist_obj(sn_param, stabilization, dist_training_loss, y_train_labels)
    # Unstabilized mixture heads run away — same failure mode and fix as the centered
    # SkewNormal: upgrade the module-default "None" to L2, honor an explicit choice.
    if stabilization == "None":
        stabilization = "L2"
    # Component loss must be nll — lightgbmlss Mixture rejects crps components, so a
    # forced crps corner fails loudly in the ctor (DPO precedent), never silently remaps.
    mix = Mixture(
        Gaussian(
            stabilization=stabilization,
            loss_fn=_resolve_loss_fn("nll", dist_training_loss),
        ),
        M=_MIXTURE_COMPONENTS,
    )
    # LightGBMLSS defaults mixture weights to a sampled Gumbel-Softmax response,
    # so an identical row receives different weights when its batch position
    # changes. Replace it before the object reaches either fit or predict: the
    # deterministic categorical probabilities must define both paths.
    mix.param_dict["mix_prob"] = softmax_fn
    label_std = float(np.std(y_train_labels))
    mix.param_dict["scale"] = _BoundedResponseFn(
        mix.param_dict["scale"],
        _MIXTURE_SCALE_STD_CEILING * label_std,
        floor=_MIXTURE_SCALE_STD_FLOOR * label_std,
    )
    return mix


def _step_select_distribution(
    splits: dict,
    stat_data,
    market: str,
    league: str,
    target_normalization: str,
    zinb_mode: str,
    *,
    deterministic: bool,
    isolated: bool = False,
    dist_training_loss: str = LOSS_AUTO,
    stabilization: str = "None",
    dist_override: str = LOSS_AUTO,
    sn_param: str = "direct",
) -> dict:
    """Choose distribution family + apply target transform + compute shape priors.

    Family selection: ``dist_override`` (the ``meditate --dist`` sweep axis) wins over every
    cell when not ``"auto"``; otherwise the cell's stat_meta ``dist`` is authoritative when set
    (:func:`_resolve_dist` forces that branch); an unset / ``"auto"`` cell falls back to the
    data-driven rule (:func:`_data_driven_dist`): ``global_mean >= _SKEWNORMAL_MEAN_THRESHOLD``
    → SkewNormal, else NegBin escalated to ZINB when ``hist_gate > GATE_PUBLISH_THRESHOLD``. For
    SkewNormal, drops zero rows when ``hist_gate > NONZERO_DENOM_GATE`` and applies the
    strategy's forward transform.

    Mutates ``splits["X_train"]`` and ``splits["y_train_labels"]`` for the
    SkewNormal nonzero path. Also writes ``stat_zi[league][market]`` and
    ``stat_cv[league][market]``; persists to JSON unless ``deterministic`` or
    ``isolated``.

    Args:
        splits: Output of ``_step_build_splits``.
        stat_data: Stats instance (gamelog + log_strings).
        market: Market name.
        league: League slug.
        target_normalization: Continuous-family slug for
            ``baselines.get_target_normalization``, or ``"none"`` for count families.
        deterministic: If True, skip persisting cv/zi to stat_calibration.json.
        isolated: If True, keep the fitted cv/zi in process memory without
            mutating canonical calibration config.
        sn_param: ``"direct"`` or ``"centered"`` — SkewNormal head
            parametrization (see :func:`_skewnormal_dist_obj`); ignored by the
            count families.

    Returns:
        Dict with: ``dist``, ``dist_obj`` (None on hurdle path until built),
        ``cv``, ``shape_ceiling``, ``marginal_shape``, ``normalize``,
        ``offset_mode``, ``denom_col``, ``strategy`` (None for count families), ``hist_gate``,
        ``player_stats``, ``global_mean``, ``sn_param``.
    """
    # style: allow-complexity -- selects the distribution family and its training controls across families
    y_train_labels = splits["y_train_labels"]
    X_train = splits["X_train"]

    threshold = _MIN_PLAYER_NONZERO_OBS.get(league, _MIN_PLAYER_NONZERO_OBS_DEFAULT)
    player_stats = (
        stat_data.gamelog.groupby(stat_data.log_strings.get("player"))
        .filter(lambda x: x[market].gt(0).sum() > threshold)
        .groupby(stat_data.log_strings.get("player"))[market]
    )

    zero_mask = y_train_labels == 0
    hist_gate = zero_mask.sum() / len(y_train_labels) if len(y_train_labels) > 0 else 0

    stat_zi[league][market] = hist_gate
    # In deterministic mode the in-memory update still flows downstream (so
    # this run's branch selection sees the gate), but skip persisting to
    # stat_calibration.json — deterministic runs use crippled hyperparameters
    # and must never mutate production config.
    if not deterministic and not isolated:
        save_zi_config(stat_zi)

    player_stats = player_stats.apply(lambda x: x[x != 0]).groupby(level=0)

    global_mean = y_train_labels.mean()
    normalize = False
    offset_mode = False
    denom_col = "MeanYr"
    strategy = None
    dist_obj = None

    configured = (
        dist_override
        if dist_override != LOSS_AUTO
        else load_distribution_config().get(league, {}).get(market)
    )
    dist = _resolve_dist(configured, _data_driven_dist(global_mean, hist_gate), league, market)
    if dist in ("SkewNormal", "Mixture"):
        strategy = baselines.get_target_normalization(target_normalization)
        cv = (
            player_stats.std()
            / player_stats.mean()
            * player_stats.count()
            / player_stats.count().sum()
        ).sum()
        cv = max(cv, _SKEWNORMAL_CV_FLOOR)
        shape_ceiling = None
        # NaN not None: count-branch marginal-shape doesn't apply here, and
        # float(None) raises TypeError in _wide_row's _diag() helper.
        marginal_shape = float("nan")

        zero_adjusted_continuous = dist in ("SkewNormal", "Mixture")
        if zero_adjusted_continuous and hist_gate > NONZERO_DENOM_GATE:
            nonzero_mask = y_train_labels > 0
            X_train = X_train[nonzero_mask]
            y_train_labels = y_train_labels[nonzero_mask]

        denom_col = baselines.resolve_denom_col(
            target_normalization,
            league,
            market,
            X_train.columns,
            zero_inflated=zero_adjusted_continuous and hist_gate > NONZERO_DENOM_GATE,
        )
        normalize = strategy.start_mode_flag == "normalized"
        offset_mode = strategy.start_mode_flag == "offset"
        y_train_labels = strategy.forward(y_train_labels, X_train, global_mean, denom_col)
        dist_obj = _continuous_dist_obj(
            dist, sn_param, stabilization, dist_training_loss, y_train_labels
        )
    else:
        # dist is resolved to a count family: "ZINB"/"NegBin" (forced, or the data-driven
        # zero-rate pick) or the force-only "DPO".
        if dist == "NegBin":
            dist_obj = NegativeBinomial(
                stabilization=stabilization, loss_fn=_resolve_loss_fn("nll", dist_training_loss)
            )
        elif dist == "DPO":
            # nll is the family default; a forced crps corner must fail loudly in the
            # DoublePoisson ctor (no rsample → no crps path), never silently remap.
            dist_obj = DoublePoisson(
                stabilization=stabilization, loss_fn=_resolve_loss_fn("nll", dist_training_loss)
            )
        elif zinb_mode == "joint":
            # Legacy jointly-fit LightGBMLSS ZINB path — byte-identical to pre-P2.B.
            dist_obj = ZINB(
                stabilization=stabilization, loss_fn=_resolve_loss_fn("nll", dist_training_loss)
            )
        # else: hurdle path — dist_obj is not constructed; HurdleZINB is built at fit time.

        per_player_r = player_stats.mean() ** 2 / np.maximum(
            player_stats.var() - player_stats.mean(), 0.01
        )
        per_player_r = np.minimum(per_player_r, _COUNT_BRANCH_R_CAP)

        marginal_shape = max(
            float(np.quantile(per_player_r, _MARGINAL_SHAPE_QUANTILE)),
            _MARGINAL_SHAPE_FLOOR,
        )
        shape_ceiling = marginal_shape * _SHAPE_CEILING_MULTIPLIER

        cv = (
            player_stats.std()
            / player_stats.mean()
            * player_stats.count()
            / player_stats.count().sum()
        ).sum()
        cv = max(cv, 1 / np.sqrt(shape_ceiling))
        if dist == "DPO":
            # cv above keeps the shared count formula (same book-side variance as an
            # NB corner on the same cell), but the NB r-ceiling machinery must not
            # bound or report phi: it is hard-bounded inside DoublePoissonTorch, and
            # serve's _clamp_shape_ceiling expects DPO pickles to persist None.
            shape_ceiling = None
            marginal_shape = float("nan")

    stat_cv[league][market] = cv
    if not deterministic and not isolated:
        save_cv_std_config({league: {market: cv}}, {})

    # Reflect SkewNormal nonzero filtering back into splits.
    splits["X_train"] = X_train
    splits["y_train_labels"] = y_train_labels

    return {
        "dist": dist,
        "dist_obj": dist_obj,
        "cv": cv,
        "shape_ceiling": shape_ceiling,
        "marginal_shape": marginal_shape,
        "normalize": normalize,
        "offset_mode": offset_mode,
        "denom_col": denom_col,
        "target_normalization": strategy,
        "hist_gate": hist_gate,
        "player_stats": player_stats,
        "global_mean": global_mean,
        "sn_param": sn_param,
    }


def _step_persist_book_shape(
    M: pd.DataFrame,
    league: str,
    market: str,
    dist: str,
    *,
    deterministic: bool,
    isolated: bool,
) -> None:
    """Fit canonical book-shape config only for a live, mutable training run."""
    if deterministic or isolated or dist != "SkewNormal":
        return
    save_book_shape_config(
        {league: {market: calibration.fit_book_shape(league, market, M["Result"], M["Line"])}}
    )


def _structural_gate_inputs(
    structural_strategy: str,
    *,
    calibrated: dict,
    fused: dict,
    splits: dict,
    decoded: dict,
    dist: str,
    step,
    experiment_context,
    posthoc_slug: str,
    mean_posthoc_blob,
) -> dict:
    """Return the unified gate-input tuple for the standard and structural paths.

    The structural candidates carry their own already-pooled probabilities, so
    they bypass ``_step_calibrate_temperature`` and the prob-stage post-hoc; the
    standard path fits temperature and layers the post-hoc corrector. Both yield
    the same keys the downstream scorecard consumes.
    """
    if structural_strategy == TWO_PART_STRATEGY:
        calibrated, _ = _step_apply_two_part_groupcdf_candidate(
            calibrated, fused, splits, experiment_context
        )
        receiving_fit: TwoPartCalibrationFit = calibrated["receiving_candidate_fit"]
        receiving_test = calibrated["receiving_candidate_test"]
        y_proba_no_filt = np.column_stack(
            (receiving_test.mapped_under, 1.0 - receiving_test.mapped_under)
        )
        T_opt = float(receiving_fit.blob["temperature"])
        val_calibrated = receiving_fit.oof_pooled_over
        posthoc_blob = None
        test_calibrated_over = receiving_test.pooled_over
    elif structural_strategy == AFFINE_STRATEGY:
        calibrated, experiment_context = _step_apply_affine_groupcdf_candidate(
            calibrated, fused, splits, experiment_context
        )
        rushing_fit: AffineCalibrationFit = calibrated["rushing_candidate_fit"]
        rushing_test = calibrated["rushing_candidate_test"]
        test_positions = (
            experiment_context["positions"]["test"]
            .reindex(splits["X_test"].index)
            .to_numpy(dtype=float)
        )
        base_predictive: AffinePredictive = calibrated["rushing_candidate_test_base"]
        _, test_mapped_under = affine_cdf_endpoints(
            rushing_fit.blob,
            base_predictive,
            splits["B_test"]["Line"].to_numpy(dtype=float),
            test_positions,
        )
        y_proba_no_filt = np.column_stack((test_mapped_under, 1.0 - test_mapped_under))
        T_opt = float(rushing_fit.blob["temperature"])
        val_calibrated = rushing_fit.oof_pooled_over
        posthoc_blob = None
        test_calibrated_over = rushing_test.pooled_over
    else:
        y_proba_no_filt = _step_compute_test_probabilities(
            fused, calibrated, splits, decoded, dist, step
        )
        T_opt, val_calibrated, model_calib, y_class_val = _step_calibrate_temperature(
            fused, calibrated, splits, decoded, dist, step
        )
        posthoc_blob = mean_posthoc_blob  # None unless the slug is a MEAN_STAGE corrector
        if posthoc_slug in posthoc.PROB_STAGE:
            posthoc_blob = posthoc.fit_posthoc(posthoc_slug, val_calibrated, y_class_val)
            val_calibrated = posthoc.apply_posthoc(posthoc_slug, posthoc_blob, val_calibrated)
        test_calibrated_over = apply_temperature(y_proba_no_filt[:, 1], T_opt)
        if posthoc_slug in posthoc.PROB_STAGE:
            test_calibrated_over = posthoc.apply_posthoc(
                posthoc_slug, posthoc_blob, test_calibrated_over
            )
        return {
            "calibrated": calibrated,
            "y_proba_no_filt": y_proba_no_filt,
            "T_opt": T_opt,
            "val_calibrated": val_calibrated,
            "model_calib": model_calib,
            "y_class_val": y_class_val,
            "posthoc_blob": posthoc_blob,
            "test_calibrated_over": test_calibrated_over,
        }
    y_class_val = (
        splits["y_validation"]["Result"].to_numpy(dtype=float)
        >= splits["B_validation"]["Line"].to_numpy(dtype=float)
    ).astype(int)
    model_calib = 1.0 - float(np.mean((val_calibrated - y_class_val) ** 2))
    return {
        "calibrated": calibrated,
        "y_proba_no_filt": y_proba_no_filt,
        "T_opt": T_opt,
        "val_calibrated": val_calibrated,
        "model_calib": model_calib,
        "y_class_val": y_class_val,
        "posthoc_blob": posthoc_blob,
        "test_calibrated_over": test_calibrated_over,
    }


def _structural_rows(
    structural_strategy: str, calibrated: dict, index: pd.Index
) -> dict[str, pd.Series | None]:
    """Every fit-dependent served value the CSV carries per row, as row-indexed Series.

    One shape for all three paths — base, two-part, affine — so the cross-fit stitcher does
    not have to know which method produced them; a field the active method never fits is
    ``None``. The values are what the persisted CSV must show *per row*, which under
    cross-fitting differs from the whole-fit constants the pickle keeps.
    """
    routes = calibrated.get("structural_routes")
    rows: dict[str, pd.Series | None] = dict.fromkeys(STRUCTURAL_ROW_FIELDS)
    rows["test_route"] = None if routes is None else routes["test"].reindex(index)
    if structural_strategy == TWO_PART_STRATEGY:
        test_output = calibrated["receiving_candidate_test"]
        rows["prepool_over"] = pd.Series(test_output.candidate_over, index=index)
        rows["two_part_calibration_by_row"] = pd.Series(
            calibrated["receiving_candidate_calibration_payload"], index=index
        )
        rows["two_part_f0"] = pd.Series(calibrated["receiving_candidate_test_f0"], index=index)
        rows["two_part_role"] = calibrated["receiving_candidate_test_role"]
        rows["two_part_position"] = calibrated["receiving_candidate_test_position"]
        return rows
    if structural_strategy == AFFINE_STRATEGY:
        rows["prepool_over"] = pd.Series(
            calibrated["rushing_candidate_test"].candidate_over, index=index
        )
        rows["pit_recal_by_row"] = calibrated["structural_pit_maps"]
        return rows
    blob = calibrated.get("pit_recal_blob")
    if blob is not None:
        rows["pit_recal_by_row"] = pd.Series(json.dumps(blob), index=index)
    return rows


def _crossfit_structural_rows(
    per_fold: list[dict], order: pd.Index, target: pd.Index
) -> dict[str, pd.Series | None]:
    """Stitch each fold's structural rows back into the caller's row order.

    A field either has a fold-specific value everywhere or nowhere: a calibrator that
    collapsed on one fold and not another would leave the frame half-warped, which reads as
    a scored artifact rather than the failure it is.
    """
    rows: dict[str, pd.Series | None] = {}
    for field in STRUCTURAL_ROW_FIELDS:
        parts = [fold["structural_rows"][field] for fold in per_fold]
        if any(part is None for part in parts) and any(part is not None for part in parts):
            raise ValueError(f"cross-fit structural {field} collapsed on at least one fold")
        rows[field] = _concat_folds(parts, order, target)
    return rows


def _step_calibrate_and_serve(
    splits: dict,
    preds: dict,
    dist_info: dict,
    *,
    dist: str,
    cv: float,
    hist_gate: float,
    shape_ceiling,
    step,
    blending: str,
    count_dispersion_objective: str,
    posthoc_slug: str,
    structural_strategy: str,
    experiment_context,
) -> dict:
    """Decode, correct, fuse and calibrate one model's predictions into served values.

    Every calibrator in this chain — the mean-stage corrector, the blend weight, the
    dispersion/skew scalars, the PIT recalibration map, the temperature, and the
    prob-stage corrector — is fit on ``splits["*_validation"]`` and applied to
    ``splits["*_test"]``. Nothing here refits the GBDT: the model's predictions arrive
    ready in ``preds``. That is what lets :func:`_step_crossfit_calibrate_and_serve`
    rotate the two frames over folds of a single split and pay only this chain.
    """
    prob_params = preds["prob_params"]
    decoded = _step_decode_predictions(
        prob_params,
        preds["prob_params_validation"],
        splits["X_test"],
        splits["X_validation"],
        dist,
        dist_info["target_normalization"],
        dist_info["global_mean"],
        dist_info["denom_col"],
        hist_gate,
    )
    # Mean-stage post-hoc (orthogonal to target_normalization and to the
    # prob-stage corrector below): fit on the validation decoded mean, then
    # correct both test and validation means BEFORE fusion so the correction
    # flows through the blend into P (Gate 1/5) and into the persisted EV
    # (Gate 2/3). roe_mean undoes leaf-averaging compression; it deliberately
    # does not touch dispersion (Gate 4).
    mean_posthoc_blob = None
    if posthoc_slug in posthoc.MEAN_STAGE:
        val_result = splits["y_validation"]["Result"].to_numpy(dtype=float)
        mean_posthoc_blob = posthoc.fit_posthoc(posthoc_slug, decoded["ev_validation"], val_result)
        decoded["ev"] = posthoc.apply_posthoc(posthoc_slug, mean_posthoc_blob, decoded["ev"])
        decoded["ev_validation"] = posthoc.apply_posthoc(
            posthoc_slug, mean_posthoc_blob, decoded["ev_validation"]
        )

    fused = _step_fuse_predictions(decoded, splits, dist, cv, hist_gate, blending=blending)
    calibrated = _step_calibrate_dispersion(
        decoded,
        fused,
        splits,
        dist,
        cv,
        hist_gate,
        shape_ceiling,
        fused["model_weight"],
        count_dispersion_objective=count_dispersion_objective,
        posthoc_slug=posthoc_slug,
    )
    gate_inputs = _structural_gate_inputs(
        structural_strategy,
        calibrated=calibrated,
        fused=fused,
        splits=splits,
        decoded=decoded,
        dist=dist,
        step=step,
        experiment_context=experiment_context,
        posthoc_slug=posthoc_slug,
        mean_posthoc_blob=mean_posthoc_blob,
    )
    test_calibrated_over = gate_inputs["test_calibrated_over"]
    return {
        "prob_params": prob_params,
        "decoded": decoded,
        "fused": fused,
        "structural_rows": _structural_rows(
            structural_strategy, gate_inputs["calibrated"], splits["X_test"].index
        ),
        "calibrated": gate_inputs["calibrated"],
        "y_proba_no_filt": gate_inputs["y_proba_no_filt"],
        "T_opt": gate_inputs["T_opt"],
        "val_calibrated": gate_inputs["val_calibrated"],
        "model_calib": gate_inputs["model_calib"],
        "y_class_val": gate_inputs["y_class_val"],
        "posthoc_blob": gate_inputs["posthoc_blob"],
        "y_proba_filt": np.array([1 - test_calibrated_over, test_calibrated_over]).transpose(),
        "y_proba_raw": _build_y_proba_raw(splits["B_test"], decoded, dist, step),
    }


def _calibration_folds(splits: dict) -> pd.Series:
    """Player-disjoint cross-fit fold id for every validation row.

    Hashing the player (not the row) keeps a player's games together, and reuses
    ``_step_build_splits``' own ``hash_pandas_object`` idiom so a player keeps its
    fold as the matrix grows. Team-level markets carry no ``Player`` and fall back
    to the date, which is the finest identity those cells have.
    """
    key = splits["players_validation"]
    if key is None:
        key = splits["dates_validation"]
    digest = pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)
    return pd.Series(digest % _CALIBRATION_FOLDS, index=key.index)


def _calibration_fit_partitions(splits: dict) -> list[pd.Index]:
    """Every row set a cross-fit run fits a calibrator on: the whole frame, then each fold's rest."""
    folds = _calibration_folds(splits)
    return [folds.index, *(folds.index[folds.ne(fold)] for fold in sorted(folds.unique()))]


def _fold_splits_view(splits: dict, fit_index, apply_index) -> dict:
    """Re-point a holdout-blind splits dict at one fold: validation fits, test applies.

    Only the validation-sourced frames move. ``X_train``/``y_train_labels`` pass
    through untouched, which is the whole point — the GBDT is never refit per fold.
    """
    view = dict(splits)
    for stem in _ROTATED_SPLIT_STEMS:
        source = splits[f"{stem}_validation"]
        view[f"{stem}_validation"] = None if source is None else source.loc[fit_index]
        view[f"{stem}_test"] = None if source is None else source.loc[apply_index]
    return view


def _concat_folds(parts: list, order: pd.Index, target: pd.Index):
    """Stitch per-fold apply-row payloads back into the caller's original row order.

    A decoded mixture rides as a ``{w1, loc1, scale1, loc2, scale2}`` dict of per-row arrays
    rather than one array, so dicts merge per key.
    """
    if parts[0] is None:
        return None
    if isinstance(parts[0], (pd.DataFrame, pd.Series)):
        return pd.concat(parts).reindex(target)
    if isinstance(parts[0], dict):
        return {
            key: _concat_folds([part[key] for part in parts], order, target) for key in parts[0]
        }
    stacked = np.concatenate([np.asarray(part) for part in parts])
    return stacked[order.get_indexer(target)]


def _step_crossfit_calibrate_and_serve(
    splits: dict, preds: dict, dist_info: dict, **kwargs
) -> dict:
    """Out-of-fold calibration over the whole validation frame.

    Each fold fits the calibration chain on the other folds and applies it to its own
    rows, so no served value was calibrated on the row it scores. The GBDT is untouched:
    every fold reuses the same validation predictions. A final unrotated pass supplies
    the pickle's own constants, which are reported context rather than scored values.

    Returns the same keys as :func:`_step_calibrate_and_serve`, in the caller's row order.
    """
    folds = _calibration_folds(splits)
    target = splits["X_test"].index
    validation_params = preds["prob_params_validation"]
    per_fold: list[dict] = []
    applied: list[pd.Index] = []
    for fold in sorted(folds.unique()):
        apply_index = folds.index[folds.eq(fold)]
        fit_index = folds.index[folds.ne(fold)]
        fold_preds = {
            "prob_params": validation_params.loc[apply_index],
            "prob_params_validation": validation_params.loc[fit_index],
        }
        per_fold.append(
            _step_calibrate_and_serve(
                _fold_splits_view(splits, fit_index, apply_index), fold_preds, dist_info, **kwargs
            )
        )
        applied.append(apply_index)

    order = applied[0].append(applied[1:]) if len(applied) > 1 else applied[0]
    whole = _step_calibrate_and_serve(
        _fold_splits_view(splits, folds.index, folds.index),
        {"prob_params": validation_params, "prob_params_validation": validation_params},
        dist_info,
        **kwargs,
    )
    served = dict(whole)
    for key in ("prob_params", "y_proba_filt", "y_proba_raw", "y_proba_no_filt"):
        served[key] = _concat_folds([fold[key] for fold in per_fold], order, target)
    served["decoded"] = {
        **whole["decoded"],
        "ev": _concat_folds([fold["decoded"]["ev"] for fold in per_fold], order, target),
    }
    served["fused"] = {
        **whole["fused"],
        "weighted_mean": _concat_folds(
            [fold["fused"]["weighted_mean"] for fold in per_fold], order, target
        ),
        "gate_blend_test": _concat_folds(
            [fold["fused"]["gate_blend_test"] for fold in per_fold], order, target
        ),
    }
    served["calibrated"] = {
        **whole["calibrated"],
        **{
            key: _concat_folds([fold["calibrated"][key] for fold in per_fold], order, target)
            for key in (
                "sn_sigma_blend_test",
                "sn_alpha_blend_test",
                "mix_blend_test",
                "r_test",
                "phi_test",
            )
        },
    }
    # The served rows ARE the validation rows here, so the out-of-fold probabilities
    # are also the honest validation-side numbers the skill metrics should report.
    served["val_calibrated"] = served["y_proba_filt"][:, 1]
    # Every fit-dependent structural value the CSV shows becomes each row's own fold value —
    # the PIT warp Gate 4 re-derives, the two-part calibration map, the pre-pool probability.
    # The whole-fit blobs still ride the pickle as its constants, but scoring a row against
    # state fitted on it is the leak this mode removes.
    served["structural_rows"] = _crossfit_structural_rows(per_fold, order, target)
    return served


def train_market(
    league: str,
    market: str,
    stat_data,
    archive,
    league_start_date,
    *,
    force: bool,
    deterministic: bool = False,
    target_normalization: str = "ratio_meanyr",
    posthoc_slug: str = "none",
    blending: str = calibration.DEFAULT_BLENDING,
    zinb_mode: str = "joint",
    dist_training_loss: str = LOSS_AUTO,
    dist: str = LOSS_AUTO,
    stabilization: str = "None",
    hpo_selection: str = "loss",
    count_dispersion_objective: str = "crps",
    sn_param: str = "direct",
    matrix_only: bool = False,
    full_rebuild: bool = False,
    matrix_output: Path | None = None,
    dependency_root: Path | None = None,
    matrix_input: Path | None = None,
    artifact_output: Path | None = None,
    dependency_namespace: str | None = None,
    holdout_blind: bool = False,
) -> None:
    """Train or retrain one LightGBMLSS model for a single league/market pair.

    Loads the training matrix, selects the distribution, runs Optuna hyperparameter
    search, fits the model, applies dispersion calibration and temperature scaling,
    evaluates on the held-out test set, and saves the model pickle + training report.

    DEBUG / OFFLINE-EVAL ONLY when ``deterministic=True``: seeds all RNGs,
    replaces the Optuna search with fixed fast hyperparameters, and freezes
    input to the cached training parquet (no incremental fetch). Produces
    bit-identical re-runs for the P0 compression harness. A model trained
    with this flag MUST NEVER be published as a production model.

    Args:
        league: League slug (``"NBA"``, ``"NFL"``, etc.).
        market: Market name (e.g. ``"FGA"``, ``"PTS"``).
        stat_data: League-specific ``Stats`` instance.
        force: Retrain even when no new training rows arrived.
        archive: ``Archive`` instance (passed through for book weights).
        league_start_date: Earliest cutoff for training rows.
        deterministic: Pin RNGs and replace Optuna with fixed hyperparams for
            bit-identical re-runs. Writes go to
            ``research/models/deterministic/{strategy}/`` at the repo root.
            Debug/offline-eval only — never publish such a model.
        target_normalization: Slug from
            :data:`sportstradamus.training.baselines.TARGET_NORMALIZATION_SLUGS`. Selects
            the SkewNormal forward target transform and the matching decode.
            Default ``"ratio_meanyr"`` reproduces legacy production behavior
            byte-for-byte.
        posthoc_slug: Calibration-method slug from
            :data:`sportstradamus.training.posthoc.POSTHOC_SLUGS`. A
            probability-stage slug recalibrates the over-probability after
            temperature scaling; a structural slug (from
            :data:`sportstradamus.training.posthoc.STRUCTURAL_STAGE`) names the
            group-conditional-CDF method this cell trains and the structural
            stage replaces the corrector; ``"none"`` (default) is a no-op.
        zinb_mode: Either ``"joint"`` (legacy LightGBMLSS ZINB; the default) or
            ``"hurdle"`` (use ``HurdleZINB`` from ``sportstradamus.hurdle`` — a
            two-stage model with a derived-π gate; see
            docs/OVERCONFIDENCE_INVESTIGATION.md §2 and
            docs/CENTERED_TARGET_NEGATIVE_RESULT.md for context). Only consulted
            when the count-branch chooses ``dist == "ZINB"``. ``"joint"`` is
            byte-identical to pre-P2.B production behavior.
        dist: Distribution-family override (the WS-3 ``meditate --dist`` sweep axis).
            ``"auto"`` (default) honors each cell's stat_meta ``dist`` (else the
            data-driven rule); an explicit forceable family overrides every cell via
            :func:`_resolve_dist`, byte-unchanged from production under ``"auto"``.
        sn_param: SkewNormal parametrization — ``"direct"`` (loc/scale/alpha heads;
            the production default) or ``"centered"`` (mean/sd/gamma1 heads via
            :class:`CenteredSkewNormal`, re-emitting direct params at predict —
            zero serving delta). Arrives resolved: the CLI maps ``"auto"`` to the
            cell's stat_meta before calling. Consulted only on the SkewNormal
            branch and in start-value seeding; persisted in the pickle.
    """
    # style: allow-length  pre-existing research orchestrator (§2.8/§18.9): flag,
    # don't split. Already over the limit before the FBT keyword-only conversion.
    # style: allow-complexity  the per-cell training orchestrator: the Lever-1 candidate-list
    # collapse is one more sequential stage, not nested logic.
    if zinb_mode not in {"joint", "hurdle"}:
        raise ValueError(f"zinb_mode must be 'joint' or 'hurdle', got {zinb_mode!r}")
    # A structural calibration method rides the single-valued ``posthoc`` pool: when that
    # slug names a structural method it selects the group-conditional-CDF stage, and the
    # corrector stages read the same slug as a no-op (mutual exclusivity by construction).
    structural_strategy = (
        posthoc_slug if posthoc_slug in posthoc.STRUCTURAL_STAGE else BASE_STRUCTURAL_STRATEGY
    )

    dist_override = dist
    isolated_run = matrix_input is not None or artifact_output is not None
    init = _step_init_market(
        league,
        market,
        stat_data,
        archive,
        deterministic=deterministic,
        full_rebuild=full_rebuild,
        isolated=isolated_run,
    )
    filedict = init["filedict"]
    # init["dist"] is the existing pickle's family, used only for the pre-selection odds synth
    # below; the authoritative training dist (and the --dist override) is applied at
    # _step_select_distribution, which overwrites this local.
    dist = init["dist"]
    cv = init["cv"]
    filename = init["filename"]

    loaded = _step_load_matrix(
        filename,
        league_start_date,
        stat_data,
        league,
        market,
        deterministic=deterministic,
        force=force,
        need_model=init["need_model"],
        full_rebuild=full_rebuild,
        matrix_output=matrix_output,
        dependency_root=dependency_root,
        dependency_namespace=dependency_namespace or DEPENDENCY_NAMESPACE,
        matrix_input=matrix_input,
    )
    if loaded is None:
        return
    M, training_data_path = loaded

    M, step = _step_synthesize_odds(M, league, market, dist, cv)
    M = _step_persist_matrix_and_comps(
        M,
        training_data_path,
        stat_data,
        deterministic=deterministic,
        full_rebuild=full_rebuild,
        immutable_input=matrix_input is not None,
        league=league,
        market=market,
        dist=dist,
        cv=cv,
    )
    if full_rebuild:
        repo_root = Path(__file__).resolve().parents[3]
        write_matrix_manifest(
            Path(training_data_path),
            M,
            stat_data,
            league=league,
            market=market,
            cutoff_date=league_start_date,
            repo_root=repo_root,
        )
    if matrix_only:
        return
    with open(training_data_path, "rb") as matrix_stream:
        matrix_hash = hashlib.file_digest(matrix_stream, "sha256").hexdigest()

    splits = _step_build_splits(
        M,
        stat_data,
        market,
        target_normalization,
        structural_strategy,
        holdout_blind=holdout_blind,
    )
    if structural_strategy == TWO_PART_STRATEGY:
        # Routes, thresholds and gate rates come off the original train/full-validation
        # split, so every calibration fold shares them. The positive-map granularity is the
        # one piece the support audit would otherwise re-decide per fold, so a cross-fit run
        # pins it across all of them up front.
        experiment_context = build_two_part_context(
            splits,
            league=league,
            market=market,
            slug=structural_strategy,
        )
        if holdout_blind and experiment_context["status"] == "active":
            experiment_context["pinned_grouping"] = pin_two_part_grouping(
                splits, experiment_context, _calibration_fit_partitions(splits)
            )
    elif structural_strategy in AFFINE_EXPERT_EXPERIMENTS:
        experiment_context = build_affine_expert_context(
            splits,
            league=league,
            slug=structural_strategy,
        )
    else:
        experiment_context = None
    dist_info = _step_select_distribution(
        splits,
        stat_data,
        market,
        league,
        target_normalization,
        zinb_mode,
        deterministic=deterministic,
        isolated=isolated_run,
        dist_training_loss=dist_training_loss,
        stabilization=stabilization,
        dist_override=dist_override,
        sn_param=sn_param,
    )
    dist = dist_info["dist"]
    cv = dist_info["cv"]
    hist_gate = dist_info["hist_gate"]
    shape_ceiling = dist_info["shape_ceiling"]
    use_hurdle = dist == "ZINB" and zinb_mode == "hurdle"
    strategy_slug = structural_strategy if structural_strategy != BASE_STRUCTURAL_STRATEGY else dist
    selected_spec = get_strategy(strategy_slug)
    runtime_controls = {
        "dist": dist,
        "normalization": target_normalization,
        # Report the loss the dist object was actually built with. The count constructors
        # resolve `auto` against their own "nll" default, which a bare "crps" here
        # contradicted -- every count cell then failed the corner check on a control the
        # spec fixes. Families that leave it unpinned (SkewNormal) keep the "crps" default.
        "dist_training_loss": _resolve_loss_fn(
            selected_spec.fixed_controls.get("dist_training_loss", "crps"), dist_training_loss
        ),
        "sn_param": sn_param,
        "blending_loss_fn": blending,
        "zinb_mode": zinb_mode,
        "count_dispersion_objective": count_dispersion_objective,
        "hpo_selection": hpo_selection,
        "stabilization": stabilization,
        "posthoc": posthoc_slug,
    }
    selected_controls = {
        name: runtime_controls[name]
        for name in (*selected_spec.fixed_controls, *selected_spec.axes)
    }
    cell = CellContext(
        league=league,
        market=market,
        distribution=dist,
        distribution_class=distribution_class(dist),
        data_columns=frozenset(M.columns),
        matrix_sha256=matrix_hash,
    )
    required_capability = CAP_DETERMINISTIC_TRAIN if deterministic else CAP_FULL_HPO
    validate_strategy_selection(
        cell,
        selected_spec,
        selected_controls,
        required_capabilities={required_capability},
    )
    persisted_settings = {
        "dist": dist,
        "target_normalization": target_normalization,
        "dist_training_loss": runtime_controls["dist_training_loss"],
        "sn_param": sn_param,
        "blending": blending,
        "zinb_mode": zinb_mode,
        "count_dispersion_objective": count_dispersion_objective,
        "hpo_selection": hpo_selection,
        "stabilization": stabilization,
        "posthoc": posthoc_slug,
    }
    fixed_mismatches = [
        f"{field}={persisted_settings[field]!r} (requires {expected!r})"
        for field, expected in selected_spec.fixed_persist.items()
        if persisted_settings[field] != expected
    ]
    if fixed_mismatches:
        raise ValueError(
            f"{selected_spec.slug} fixed-control recipe mismatch: {', '.join(fixed_mismatches)}"
        )

    # WS2: fit the book's conditional (var, skew) curves and persist them for the book-only
    # fallback leg; book_skewnormal_shape reads them there only — the served blend stays
    # symmetric per the settling verdict. fit_book_shape's per-line-bin row floor self-filters
    # the synthetic fallback lines, so binning all rows still conditions on real book lines.
    _step_persist_book_shape(
        M,
        league,
        market,
        dist,
        deterministic=deterministic,
        isolated=isolated_run,
    )

    model = _step_build_lss_model(
        dist,
        dist_info["dist_obj"],
        splits["X_train"],
        shape_ceiling,
        normalize=dist_info["normalize"],
        offset_mode=dist_info["offset_mode"],
        use_hurdle=use_hurdle,
        sn_param=sn_param,
    )
    dtrain = lgb.Dataset(splits["X_train"], label=splits["y_train_labels"])
    opt_params_in = filedict.get("params")
    penalty = penalty_threshold = None
    if hpo_selection == "calibrated" and not use_hurdle and dist != "Mixture":
        # Hurdle skips Optuna entirely (_step_select_hyperparams), so there is nothing to
        # select. Mixture has no per-trial served-KS closure (its params frame doesn't fit
        # either _calibration_penalty branch); calibrated falls back to loss selection.
        penalty = _calibration_penalty(splits, dist_info, blending=blending)
        penalty_threshold = _gate4_pit_ks_threshold(len(splits["y_validation"]))
    elif hpo_selection == "calibrated":
        logger.warning(
            "%s %s: hpo_selection 'calibrated' is inert on the %s path — no HP search "
            "consults it; drop the stat_meta pin or route the cell off this path",
            league,
            market,
            "hurdle" if use_hurdle else "Mixture",
        )
    opt_params, _ = _step_select_hyperparams(
        splits["X_train"],
        dist,
        model,
        opt_params_in,
        dtrain,
        use_hurdle=use_hurdle,
        deterministic=deterministic,
        calibration_penalty=penalty,
        penalty_threshold=penalty_threshold,
    )
    if isinstance(opt_params, list):
        picked = _pick_calibrated_candidate(
            opt_params, penalty_threshold, min(len(splits["X_train"]), _OOF_KS_MAX_ROWS)
        )
        opt_params = {k: v for k, v in picked.items() if k not in ("cv_loss", "pit_ks")}
    model = _step_fit_model(
        dist,
        dist_info["dist_obj"],
        splits["X_train"],
        splits["y_train_labels"],
        opt_params,
        use_hurdle=use_hurdle,
        normalize=dist_info["normalize"],
        offset_mode=dist_info["offset_mode"],
        shape_ceiling=shape_ceiling,
        deterministic=deterministic,
        sn_param=sn_param,
    )
    expert_models: dict[int, object] = {}
    if structural_strategy in AFFINE_EXPERT_EXPERIMENTS:
        expert_models, experiment_context = _step_fit_affine_experts(
            experiment_context,
            dist=dist,
            dist_obj=dist_info["dist_obj"],
            splits=splits,
            opt_params=opt_params,
            use_hurdle=use_hurdle,
            normalize=dist_info["normalize"],
            offset_mode=dist_info["offset_mode"],
            shape_ceiling=shape_ceiling,
            deterministic=deterministic,
            sn_param=sn_param,
        )

    preds = _step_predict_splits(
        model,
        dist,
        splits,
        normalize=dist_info["normalize"],
        offset_mode=dist_info["offset_mode"],
        sn_param=sn_param,
        expert_models=(expert_models if structural_strategy in AFFINE_EXPERT_EXPERIMENTS else None),
        structural_context=(
            experiment_context if structural_strategy in AFFINE_EXPERT_EXPERIMENTS else None
        ),
    )
    if structural_strategy in AFFINE_EXPERT_EXPERIMENTS:
        experiment_context = preds["structural_context"]
    calibrate = _step_crossfit_calibrate_and_serve if holdout_blind else _step_calibrate_and_serve
    served = calibrate(
        splits,
        preds,
        dist_info,
        dist=dist,
        cv=cv,
        hist_gate=hist_gate,
        shape_ceiling=shape_ceiling,
        step=step,
        blending=blending,
        count_dispersion_objective=count_dispersion_objective,
        posthoc_slug=posthoc_slug,
        structural_strategy=structural_strategy,
        experiment_context=experiment_context,
    )
    prob_params = served["prob_params"]
    decoded = served["decoded"]
    fused = served["fused"]
    calibrated = served["calibrated"]
    y_proba_no_filt = served["y_proba_no_filt"]
    T_opt = served["T_opt"]
    val_calibrated = served["val_calibrated"]
    model_calib = served["model_calib"]
    y_class_val = served["y_class_val"]
    posthoc_blob = served["posthoc_blob"]
    y_proba_filt = served["y_proba_filt"]
    y_proba_raw = served["y_proba_raw"]

    val_book_proba = splits["B_validation"]["Odds"].to_numpy(dtype=float)
    authentic_val = _split_quote_authenticity_mask(splits, "validation")
    skill = _step_compute_skill_metrics(
        val_calibrated[authentic_val],
        y_class_val[authentic_val],
        val_book_proba[authentic_val],
        league,
        market,
    )
    y_class = np.ravel(
        (splits["y_test"]["Result"] >= splits["B_test"]["Line"]).astype(int).to_numpy()
    )

    mode_stats = _step_compute_mode_stats(y_proba_raw, y_proba_no_filt, y_proba_filt, y_class)
    served_weighted_mean = fused["weighted_mean"]
    diag = _step_compute_diagnostics(
        splits,
        prob_params,
        served_weighted_mean,
        y_proba_no_filt,
        y_class,
        fused["gate_blend_test"],
        dist,
        cv,
        dist_info["denom_col"],
        dist_info["target_normalization"],
        dist_info["player_stats"],
        step,
    )

    filedict = _build_filedict(
        model=model,
        step=step,
        mode_stats=mode_stats,
        skill=skill,
        diag=diag,
        c_opt=calibrated["c_opt"],
        skew_cal=calibrated["skew_cal"],
        shape_ceiling=shape_ceiling,
        marginal_shape=dist_info["marginal_shape"],
        opt_params=opt_params,
        dist=dist,
        cv=cv,
        y=splits["y"],
        T_opt=T_opt,
        model_weight=fused["model_weight"],
        model_calib=model_calib,
        hist_gate=hist_gate,
        normalize=dist_info["normalize"],
        strategy=dist_info["target_normalization"],
        global_mean=dist_info["global_mean"],
        denom_col=dist_info["denom_col"],
        target_normalization=target_normalization,
        posthoc_slug=posthoc_slug,
        posthoc_blob=posthoc_blob,
        pit_recal_blob=calibrated.get("pit_recal_blob"),
        zinb_mode=zinb_mode,
        sn_param=sn_param,
        X=splits["X"],
        league=league,
        market=market,
        matrix_hash=matrix_hash,
        selected_controls=selected_controls,
        structural_strategy=structural_strategy,
        algorithm_payload=calibrated.get("structural_calibration_blob"),
        expert_models=(expert_models if structural_strategy in AFFINE_EXPERT_EXPERIMENTS else None),
    )
    if dependency_namespace is not None:
        filedict["dependency_identity"] = {
            "namespace": dependency_namespace,
            "schema_version": 1,
            "league": league,
            "market": market,
            "training_cutoff": pd.to_datetime(M["Date"]).max().date().isoformat(),
            "matrix_sha256": matrix_hash,
        }

    _step_persist_artifacts(
        filedict=filedict,
        splits=splits,
        prob_params=prob_params,
        decoded=decoded,
        weighted_mean=served_weighted_mean,
        y_proba_filt=y_proba_filt,
        y_proba_raw=y_proba_raw,
        dist=dist,
        hist_gate=hist_gate,
        filename=filename,
        deterministic=deterministic,
        target_normalization=target_normalization,
        zinb_mode=zinb_mode,
        sn_scale_test=calibrated["sn_sigma_blend_test"],
        sn_skew_test=calibrated["sn_alpha_blend_test"],
        mix_test=calibrated["mix_blend_test"],
        r_test=calibrated["r_test"],
        gate_blend_test=fused["gate_blend_test"],
        phi_test=calibrated["phi_test"],
        global_mean=dist_info["global_mean"],
        denom_col=dist_info["denom_col"],
        structural_rows=served["structural_rows"],
        structural_strategy=structural_strategy,
        artifact_output=artifact_output,
    )

    # Drift-monitoring SHAP: write per-cell |SHAP| + corr columns to the
    # training/feature_importances.csv + feature_correlations.csv. After the
    # 2026-05-27 no-filter rewire these CSVs are no longer used for selection;
    # they survive purely so the dashboard can show importance drift over time.
    # Skip in deterministic mode (artifacts must not leak from eval runs) and
    # skip hurdle (HurdleZINB has two separate boosters; SHAP would need a
    # custom path — defer to a follow-up if hurdle drift becomes interesting).
    if not deterministic and artifact_output is None and not use_hurdle:
        test_df = splits["X_test"].copy()
        test_df["Result"] = splits["y_test"]["Result"].to_numpy()
        compute_market_importance(league, market, model, test_df)

    del filedict
    del model
