"""Per-market training pipeline: data loading, model fitting, calibration, diagnostics."""

import importlib.resources as pkg_resources
import json
import os
import pickle
import random
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
from lightgbmlss.distributions.ZINB import ZINB
from lightgbmlss.model import LightGBMLSS
from scipy.optimize import minimize_scalar
from scipy.special import beta as beta_fn
from scipy.special import expit, logit
from scipy.stats import gamma, nbinom, norm, skewnorm
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from sportstradamus import data
from sportstradamus.helpers import (
    GATE_PUBLISH_THRESHOLD,
    NONZERO_DENOM_GATE,
    apply_temperature,
    decode_predictive_mean,
    fused_loc,
    get_ev,
    get_logger,
    get_odds,
    set_model_start_values,
    stat_cv,
    stat_zi,
)
from sportstradamus.helpers.io import market_file_slug, model_pickle_path
from sportstradamus.hurdle import HurdleZINB
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.training import baselines, posthoc
from sportstradamus.training.calibration import fit_book_weights, fit_model_weight
from sportstradamus.training.config import (
    load_distribution_config,
    save_cv_std_config,
    save_zi_config,
)
from sportstradamus.training.data import trim_matrix
from sportstradamus.training.hyperparams import _BoundedResponseFn, warm_start_hyper_opt
from sportstradamus.training.report import report
from sportstradamus.training.shap import compute_market_importance

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


# Fixed RNG seed for --deterministic runs (debug/eval only).
DETERMINISTIC_SEED = 1234

# RNG seed for the val/test random split inside ``_step_build_splits``.
# Arbitrary but fixed so the split boundary is stable across reruns on the
# same dataset.
_VAL_SPLIT_RANDOM_STATE: int = 25

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
    "num_threads": 1,  # multi-thread LightGBM histogram reductions are not bit-reproducible even with deterministic=True
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
        ``deterministic`` (True), ``force_row_wise`` (True).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(
        True
    )  # NOTE: process-global; later non-deterministic torch ops in-process will raise
    return {
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
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

    Returns:
        A trained ``LightGBMLSS`` model.
    """
    if seed is not None:
        params = {**params, **seed_everything(seed)}
    dtrain = lgb.Dataset(X_train, label=y_train_labels)
    model = LightGBMLSS(dist_obj)
    set_model_start_values(
        model,
        dist,
        X_train,
        shape_ceiling=shape_ceiling,
        normalized=normalized,
        offset_mode=offset_mode,
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

    Returns:
        DataFrame of raw predicted distribution parameters, indexed like ``X``.
    """
    set_model_start_values(model, dist, X, normalized=normalized, offset_mode=offset_mode)
    preds = model.predict(X, pred_type="parameters")
    preds.index = X.index
    return preds


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
    )
    return predict_lss_params(
        model, dist, X_predict, normalized=normalized, offset_mode=offset_mode
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


def _step_init_market(league: str, market: str, stat_data, archive) -> dict:
    """Validate, load distribution config + book weights + existing pickle.

    Args:
        league: League slug.
        market: Market name.
        stat_data: League-specific ``Stats`` instance.
        archive: ``Archive`` singleton (passed through to book-weight fitting).

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

    book_weights.setdefault(league, {}).setdefault(market, {})
    book_weights[league][market] = fit_book_weights(
        league, market, stat_data, archive, book_weights
    )

    with open(pkg_resources.files(data) / "config" / "book_weights.json", "w") as outfile:
        json.dump(book_weights, outfile, indent=4)

    filename = market_file_slug(league, market)
    filepath = model_pickle_path(league, market)
    need_model = True
    if os.path.isfile(filepath):
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
    filepath = pkg_resources.files(data) / (f"training_data/{filename}.parquet")
    if os.path.isfile(filepath):
        M = pd.read_parquet(filepath)
        cutoff_date = pd.to_datetime(M["Date"]).max().date()
        M = M.loc[
            (pd.to_datetime(M.Date).dt.date <= cutoff_date)
            & (pd.to_datetime(M.Date).dt.date > league_start_date)
        ]
    else:
        cutoff_date = league_start_date
        M = pd.DataFrame()

    new_M = pd.DataFrame() if deterministic else stat_data.get_training_matrix(market, cutoff_date)

    if new_M.empty and not force and not need_model:
        return None

    M = pd.concat([M, new_M], ignore_index=True)
    if M.empty:
        logger.warning("  No usable training data for %s %s, skipping", league, market)
        return None
    M.Date = pd.to_datetime(M.Date, format="mixed")
    if "Player" in M.columns:
        M = M.drop_duplicates(subset=["Player", "Date"], keep="last")
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
    synthetic_mask = M.Odds.isna() | (M.Odds == 0)
    if "Odds_synthetic" in M.columns:
        synthetic_mask |= M["Odds_synthetic"].fillna(False)
    for i, row in M.loc[synthetic_mask].iterrows():
        if np.isnan(row["EV"]) or row["EV"] <= 0:
            M.loc[i, "Odds"] = 0.5
            M.loc[i, "EV"] = get_ev(
                M.loc[i, "Line"], 0.5, cv=cv, dist=dist, gate=_prep_gate or None
            )
            M.loc[i, "Odds_synthetic"] = True
        else:
            M.loc[i, "Odds"] = 1 - get_odds(
                row["Line"], row["EV"], dist, cv=cv, step=step, gate=_prep_gate or None
            )
            M.loc[i, "Odds_synthetic"] = False
    return M, step


def _step_persist_matrix_and_comps(
    M: pd.DataFrame, filepath, stat_data, *, deterministic: bool
) -> pd.DataFrame:
    """Trim the matrix, write parquet, save comps. Deterministic mode skips I/O."""
    M = trim_matrix(M, 15000)
    if not deterministic:
        M.to_parquet(filepath, compression="zstd", index=True)
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
    fewer than two distinct non-NaN values (zero variance). Such columns
    can never improve the loss — LightGBM has nothing to split on — so
    dropping them is mathematically lossless. The win is wall-time:
    per-trial Optuna cost scales linearly with feature count, and the
    2026-05-27 no-filter rewire ballooned the candidate set to ~440
    features per NFL cell, of which a non-trivial slice is sparse on a
    given cell.

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
        keep.append(col)
    return keep


def _step_build_splits(M: pd.DataFrame, stat_data, market: str) -> dict:
    """Build feature matrix, temporal 70/30 split, then 50/50 test/validation.

    Args:
        M: Trimmed training matrix.
        stat_data: Stats instance (provides ``get_stat_columns``).
        market: Market name.

    Returns:
        Dict with: ``X``, ``y``, ``X_train``, ``X_test``, ``X_validation``,
        ``y_train``, ``y_test``, ``y_validation``, ``B_train``, ``B_test``,
        ``B_validation``, ``y_train_labels``.
    """
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
    if len(kept_cols) < len(X.columns):
        X = X[kept_cols]
        X_train = X_train[kept_cols]
        X_test = X_test[kept_cols]

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5, random_state=_VAL_SPLIT_RANDOM_STATE
    )

    B_train = M.loc[X_train.index, ["Line", "Odds", "EV"]]
    B_test = M.loc[X_test.index, ["Line", "Odds", "EV"]]
    B_validation = M.loc[X_validation.index, ["Line", "Odds", "EV"]]

    y_train_labels = np.ravel(y_train.to_numpy())
    players_test = M.loc[X_test.index, "Player"].values if "Player" in M.columns else None
    dates_test = M.loc[X_test.index, "Date"].values
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
        "players_test": players_test,
        "dates_test": dates_test,
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
    )
    return model


def _step_select_hyperparams(
    X_train,
    dist: str,
    model,
    opt_params_in: dict | None,
    dtrain,
    *,
    use_hurdle: bool,
    deterministic: bool,
) -> tuple[dict, list[int]]:
    """Pick Optuna-tuned params, warm-start, or the deterministic fixed set.

    Args:
        X_train: Training feature matrix (used to size ``min_child_weight`` upper).
        dist: Distribution name (controls monotone constraint on ``MeanYr``).
        model: Built LightGBMLSS for the LSS path; ``None`` for hurdle.
        use_hurdle: True when the hurdle path will fit the model later.
        deterministic: If True, use ``DETERMINISTIC_FIXED_PARAMS`` verbatim.
        opt_params_in: Warm-start params from the existing pickle, if any.
        dtrain: ``lgb.Dataset`` for the LSS Optuna path.

    Returns:
        ``(opt_params, monotone)`` — final hyperparam dict and the monotone
        constraint vector.
    """
    col_list = list(X_train.columns)
    monotone = [0] * len(col_list)
    if dist in ("Gamma", "ZAGamma", "SkewNormal") and "MeanYr" in col_list:
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
    elif opt_params is None or opt_params.get("opt_rounds") is None:
        opt_params = model.hyper_opt(
            hp_search_space,
            dtrain,
            num_boost_round=999,
            nfold=4,
            early_stopping_rounds=50,
            max_minutes=60,
            n_trials=300,
            silence=True,
        )
    else:
        opt_params = warm_start_hyper_opt(
            model, hp_search_space, dtrain, opt_params, n_trials=150, max_minutes=5
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
    )


def _step_predict_splits(
    model, dist: str, splits: dict, *, normalize: bool, offset_mode: bool
) -> dict:
    """Predict raw distribution params on train/validation/test splits.

    Also sorts indices on every split + the B_* frames + replaces ``Odds == 0``
    with 0.5 in B_*. Mutates ``splits`` to keep the sorted views consistent.

    Returns:
        Dict with: ``prob_params_train``, ``prob_params_validation``, ``prob_params``.
    """

    def _predict(X_part: pd.DataFrame) -> pd.DataFrame:
        if getattr(model, "is_hurdle", False):
            return predict_hurdle_params(model, X_part)
        return predict_lss_params(
            model, dist, X_part, normalized=normalize, offset_mode=offset_mode
        )

    prob_params_train = _predict(splits["X_train"])
    prob_params_validation = _predict(splits["X_validation"])
    prob_params = _predict(splits["X_test"])

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
    """E[Y] = (1-π)·μ for zero-inflated count cells; the base mean otherwise.

    ZINB/ZAGamma store the base-distribution mean — the betting convention factors
    the gate out and reapplies it in ``get_odds``. The EV/bias diagnostics compare
    against zero-INCLUSIVE outcomes (Result, Line), so they must reapply it. SkewNormal
    EV is already a full mean and NegBin/Gamma carry no gate — both pass through. Mirrors
    ``training.scorecard._zero_inflated_mean``.
    """
    if dist in ("ZINB", "ZAGamma") and gate is not None:
        return base_mean * (1.0 - gate)
    return base_mean


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
    test_denom_mean = X_test[denom_col].mean() if dist == "SkewNormal" else test_mean_yr

    if dist == "SkewNormal":
        diag_start_shape = float(cv)
        scale_norm_mean = float(prob_params["scale"].mean())
        diag_model_shape = scale_norm_mean * test_denom_mean
        result_arr = y_test["Result"].to_numpy()
        diag_empirical_shape = float(result_arr.std() / max(result_arr.mean(), 1e-6))
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

    ev_gt_mask = ev_minus_line_arr > 0
    ev_lt_mask = ev_minus_line_arr <= 0
    conf_mask = np.max(y_proba_no_filt, axis=1) > _MODE_CONFIDENCE_THRESHOLD
    diag_over_pct_ev_gt = (
        float(y_class[ev_gt_mask & conf_mask].mean())
        if (ev_gt_mask & conf_mask).sum() > _MIN_DIAGNOSTIC_ROWS
        else float("nan")
    )
    diag_over_pct_ev_lt = (
        float(y_class[ev_lt_mask & conf_mask].mean())
        if (ev_lt_mask & conf_mask).sum() > _MIN_DIAGNOSTIC_ROWS
        else float("nan")
    )

    if dist == "SkewNormal":
        diag_cf_over_pct = float("nan")
    elif not np.isnan(diag_empirical_shape) and diag_empirical_shape > 0:
        emp_shape = np.full_like(weighted_mean, diag_empirical_shape)
        if dist in ("NegBin", "ZINB"):
            cf_under = get_odds(
                B_test["Line"].to_numpy(),
                weighted_mean,
                dist,
                r=emp_shape,
                gate=gate_blend_test,
            )
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
        diag_cf_over_pct = (
            float(cf_pred[cf_mask].mean() / cf_mask.mean())
            if cf_mask.sum() > _MIN_DIAGNOSTIC_ROWS
            else float("nan")
        )
    else:
        diag_cf_over_pct = float("nan")

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
    elif dist in ("NegBin", "ZINB"):
        under = get_odds(line, ev, dist, r=decoded["r"], gate=gate_test)
    else:
        under = get_odds(line, ev, dist, alpha=decoded["alpha"], step=step, gate=gate_test)
    under = np.clip(under, 0, 1)
    return np.array([under, 1 - under]).transpose()


def _build_filedict(
    *,
    model,
    step,
    mode_stats: dict,
    skill: dict,
    diag: dict,
    c_opt: float,
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
    zinb_mode: str,
    X,
) -> dict:
    """Assemble the model pickle dict. Key order is load-bearing for byte parity."""
    return {
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
        "weight": model_weight,
        "r_book": None,
        "hist_gate": hist_gate,
        "shape_ceiling": shape_ceiling,
        "normalized": normalize,
        "offset_meta": strategy.offset_meta(global_mean, denom_col),
        "target_normalization": target_normalization,
        "posthoc": posthoc_slug,
        "posthoc_blob": posthoc_blob,
        "zinb_mode": zinb_mode,
        "is_hurdle": bool(getattr(model, "is_hurdle", False)),
        "expected_columns": list(X.columns),
    }


def _persist_player_metadata(X_test: pd.DataFrame, splits: dict) -> None:
    # Player key + game date enable the offline scorecard's player-clustered Gate-1
    # bootstrap; the i.i.d. bootstrap over-credits repeated-player panels without them.
    # Guard is real — some leagues (e.g. team-level markets) omit "Player" entirely.
    if splits.get("players_test") is not None:
        X_test["Player"] = splits["players_test"]
    X_test["Date"] = splits["dates_test"]


def _step_persist_artifacts(
    *,
    filedict: dict,
    splits: dict,
    prob_params: pd.DataFrame,
    decoded: dict,
    weighted_mean: np.ndarray,
    y_proba_filt: np.ndarray,
    dist: str,
    hist_gate: float,
    filename: str,
    deterministic: bool,
    target_normalization: str,
    zinb_mode: str,
) -> None:
    """Write the test-set CSV and the model pickle.

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
    X_test["Odds"] = B_test["Odds"].values
    # EV is the base mean the blend used, mean-corrected when a mean-stage
    # corrector is active, so the bias gates (Gate 2/3) read the corrected value.
    # The native shape columns below stay uncorrected — Gate 4 reads them and
    # measures dispersion, which the corrector intentionally leaves alone.
    X_test["EV"] = ev
    if dist == "SkewNormal":
        X_test["SN_Loc"] = prob_params["loc"]
        X_test["SN_Scale"] = prob_params["scale"]
        X_test["SN_Alpha"] = prob_params["alpha"]
        if hist_gate > GATE_PUBLISH_THRESHOLD:
            X_test["Gate"] = hist_gate
    elif dist in ("NegBin", "ZINB"):
        if dist == "ZINB":
            X_test["Gate"] = prob_params["gate"]
        X_test["R"] = prob_params["total_count"]
        X_test["NB_P"] = prob_params["probs"]
    elif dist in ("Gamma", "ZAGamma"):
        if dist == "ZAGamma":
            X_test["Gate"] = prob_params["gate"]
        X_test["Alpha"] = prob_params["concentration"]

    X_test["P"] = y_proba_filt[:, 1]
    _persist_player_metadata(X_test, splits)

    # Under --deterministic, redirect to a `deterministic/` subdir so the
    # scorecard harness can score artifacts without overwriting production.
    # Training-data parquet and the whole-suite report() stay suppressed
    # (input is unchanged under input-freeze; report() is not per-market and
    # would clobber the production data/training/model_stats.{parquet,csv}).
    # Test-set CSVs remain inside the package data tree; only the model
    # pickle moves to the repo-root research dir so the package install
    # never carries the research artifacts.
    if deterministic:
        suffix = "_hurdle" if zinb_mode == "hurdle" else ""
        strategy_subdir = f"{target_normalization}{suffix}"
        csv_subdir = f"deterministic/{strategy_subdir}/"
        mdl_dir = _DETERMINISTIC_MODEL_ROOT / strategy_subdir
    else:
        csv_subdir = ""
        mdl_dir = Path(str(pkg_resources.files(data) / "models"))
    csv_filepath = pkg_resources.files(data) / f"test_sets/{csv_subdir}{filename}.csv"
    Path(str(csv_filepath.parent)).mkdir(parents=True, exist_ok=True)
    X_test.to_csv(csv_filepath)

    mdl_filepath = mdl_dir / f"{filename}.mdl"
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
) -> float:
    """CRPS-style dispersion calibration loss for a single scaling factor ``c``.

    Pure: no closures over module state. Used by ``minimize_scalar`` inside
    ``_step_calibrate_dispersion``.

    Args:
        c: Multiplicative scale on the model shape (NegBin r, Gamma α).
        dist: Distribution name.
        y_val_arr: Validation outcomes (1-D).
        val_weighted_mean: Per-row blended mean on the validation split.
        gate_blend_val: Per-row gate when ZI; None otherwise.
        r_blend_val: NegBin/ZINB ``r`` per row; None for Gamma.
        alpha_blend_val: Gamma α per row; None for NegBin.

    Returns:
        CRPS + log-c² regularization.
    """
    if dist in ("NegBin", "ZINB"):
        r_cal = r_blend_val * c
        p_cal = r_cal / (r_cal + val_weighted_mean)
        k_max = int(max(y_val_arr.max() * 2, np.mean(val_weighted_mean) * 4, 30))
        k_vals = np.arange(k_max + 1)
        cdf = nbinom.cdf(k_vals[:, None], r_cal[None, :], p_cal[None, :])
        if gate_blend_val is not None:
            cdf = gate_blend_val[None, :] + (1 - gate_blend_val[None, :]) * cdf
        indicator = (y_val_arr[None, :] <= k_vals[:, None]).astype(float)
        crps = np.sum((cdf - indicator) ** 2, axis=0)
    else:
        alpha_cal = alpha_blend_val * c
        scale_cal = val_weighted_mean / alpha_cal
        if gate_blend_val is not None:
            x_max = max(y_val_arr.max() * 2, np.mean(val_weighted_mean) * 4)
            x_grid = np.linspace(0, x_max, 500)
            dx = x_grid[1] - x_grid[0]
            cdf_grid = gamma.cdf(x_grid[:, None], alpha_cal[None, :], scale=scale_cal[None, :])
            cdf_grid = gate_blend_val[None, :] + (1 - gate_blend_val[None, :]) * cdf_grid
            indicator = (y_val_arr[None, :] <= x_grid[:, None]).astype(float)
            crps = np.sum((cdf_grid - indicator) ** 2, axis=0) * dx
        else:
            F_y = gamma.cdf(y_val_arr, alpha_cal, scale=scale_cal)
            F_y_a1 = gamma.cdf(y_val_arr, alpha_cal + 1, scale=scale_cal)
            crps = (
                y_val_arr * (2 * F_y - 1)
                - val_weighted_mean * (2 * F_y_a1 - 1)
                - scale_cal / beta_fn(0.5, alpha_cal)
            )
    reg = 0.01 * np.log(c) ** 2
    return np.mean(crps) + reg


def _brier_temperature_loss(T: float, val_logits: np.ndarray, y_class_val: np.ndarray) -> float:
    """Brier + (T-1)² regularization at temperature ``T``. Pure."""
    cal = expit(val_logits / T)
    brier = np.mean((cal - y_class_val) ** 2)
    reg = 0.01 * (T - 1) ** 2
    return brier + reg


def _step_calibrate_dispersion(
    decoded: dict,
    fused: dict,
    splits: dict,
    dist: str,
    cv: float,
    hist_gate: float,
    shape_ceiling,
    model_weight,
) -> dict:
    """Fit dispersion scaling factor ``c_opt`` and apply it to blended params.

    Pure: returns a new dict with updated ``r_test``, ``r_blend_val``,
    ``alpha_blend``, ``alpha_blend_val``, ``beta_blend_val``, ``c_opt``,
    ``val_weighted_mean`` (count branch) or ``val_weighted_mean_val``
    (SkewNormal). Hurdle ZINB participates in dispersion calibration via the
    NegBin branch — gate is passed through as ``gate_blend_val``.
    """
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    y_val_arr = splits["y_validation"]["Result"].to_numpy()

    out = {
        "c_opt": 1.0,
        "r_test": fused["r_test"],
        "r_blend_val": fused["r_blend_val"],
        "alpha_blend": fused["alpha_blend"],
        "alpha_blend_val": fused["alpha_blend_val"],
        "beta_blend_val": fused["beta_blend_val"],
        "val_weighted_mean": None,
        "val_weighted_mean_val": None,
    }

    if dist == "SkewNormal":
        val_weighted_mean_val, _, _, _ = fused_loc(
            model_weight,
            decoded["ev_validation"],
            book_ev_val,
            cv,
            "SkewNormal",
            sigma=decoded["sn_sigma_val"],
            skew_alpha=decoded["sn_alpha_val"],
            **({"gate_book": hist_gate} if hist_gate > GATE_PUBLISH_THRESHOLD else {}),
        )
        out["val_weighted_mean_val"] = val_weighted_mean_val
        return out

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

    disp_result = minimize_scalar(
        lambda c: _dispersion_crps_loss(
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
            sigma=fused["sn_sigma_blend_test"],
            skew_alpha=fused["sn_alpha_blend_test"],
            gate=fused["gate_blend_test"],
        )
    elif dist in ("NegBin", "ZINB"):
        under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            r=calibrated["r_test"],
            gate=fused["gate_blend_test"],
        )
    else:
        under = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            alpha=calibrated["alpha_blend"],
            step=step,
            gate=fused["gate_blend_test"],
        )
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
            sigma=fused["sn_sigma_blend_val"],
            skew_alpha=fused["sn_alpha_blend_val"],
            gate=fused["gate_blend_val"],
        )
    else:
        _r_val = calibrated["r_blend_val"] if dist in ("NegBin", "ZINB") else None
        _alpha_val = calibrated["alpha_blend_val"] if dist in ("Gamma", "ZAGamma") else None
        _gate_val = fused["gate_blend_val"] if dist in ("ZINB", "ZAGamma") else None
        val_raw_under = get_odds(
            B_validation["Line"].to_numpy(),
            calibrated["val_weighted_mean"],
            dist,
            alpha=_alpha_val,
            step=step,
            r=_r_val,
            gate=_gate_val,
        )
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

    SkewNormal: applies the strategy's decode_loc/decode_scale then adds the
    skew-normal mean adjustment ``delta * sqrt(2/pi)``. NegBin/ZINB: EV = r·p/(1−p).
    Gamma/ZAGamma: EV = α/β. Synthesizes a constant ``gate_*`` vector for
    SkewNormal when ``hist_gate > GATE_PUBLISH_THRESHOLD`` (no per-row gate from the model).

    Returns:
        Dict with: ``ev``, ``ev_validation``, ``gate_test``, ``gate_validation``,
        ``sn_sigma_test``, ``sn_sigma_val``, ``sn_alpha_test``, ``sn_alpha_val``,
        ``r``, ``p``, ``r_validation``, ``p_validation``, ``alpha``, ``beta``,
        ``alpha_validation``, ``beta_validation``. Unused fields are None.
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
        "alpha": None,
        "beta": None,
        "alpha_validation": None,
        "beta_validation": None,
    }
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


def _step_fuse_predictions(
    decoded: dict,
    splits: dict,
    dist: str,
    cv: float,
    hist_gate: float,
) -> dict:
    """Fit model_weight, then fuse model+book predictions on test and validation.

    Args:
        decoded: Output of ``_step_decode_predictions``.
        splits: Output of ``_step_build_splits`` (post-prediction sorted).
        dist: Distribution name.
        cv: Coefficient of variation from ``_step_select_distribution``.
        hist_gate: Historical zero rate.

    Returns:
        Dict with: ``model_weight``, ``weighted_mean``, ``gate_blend_test``,
        ``gate_blend_val``, ``r_test``, ``r_blend_val``, ``p_test``, ``p_val``,
        ``alpha_blend``, ``alpha_blend_val``, ``beta_blend``, ``beta_blend_val``,
        ``sn_sigma_blend_test``, ``sn_sigma_blend_val``, ``sn_alpha_blend_test``,
        ``sn_alpha_blend_val``. Unused fields are None.
    """
    ev = decoded["ev"]
    ev_validation = decoded["ev_validation"]
    book_ev_test = splits["B_test"]["EV"].to_numpy()
    book_ev_val = splits["B_validation"]["EV"].to_numpy()
    y_val_result = splits["y_validation"]["Result"].to_numpy()

    base_dist = (
        "SkewNormal"
        if dist == "SkewNormal"
        else ("NegBin" if dist in ("NegBin", "ZINB") else "Gamma")
    )

    out = {
        "model_weight": None,
        "weighted_mean": None,
        "gate_blend_test": None,
        "gate_blend_val": None,
        "r_test": None,
        "r_blend_val": None,
        "p_test": None,
        "p_val": None,
        "alpha_blend": None,
        "alpha_blend_val": None,
        "beta_blend": None,
        "beta_blend_val": None,
        "sn_sigma_blend_test": None,
        "sn_sigma_blend_val": None,
        "sn_alpha_blend_test": None,
        "sn_alpha_blend_val": None,
    }

    if dist == "SkewNormal":
        _zi_kwargs = {"gate_book": hist_gate} if hist_gate > GATE_PUBLISH_THRESHOLD else {}
        model_weight = fit_model_weight(
            ev_validation,
            book_ev_val,
            y_val_result,
            "SkewNormal",
            cv=cv,
            model_sigma=decoded["sn_sigma_val"],
            model_skew_alpha=decoded["sn_alpha_val"],
            **_zi_kwargs,
        )
        weighted_mean, sn_sigma_blend_test, sn_alpha_blend_test, gate_blend_test = fused_loc(
            model_weight,
            ev,
            book_ev_test,
            cv,
            "SkewNormal",
            sigma=decoded["sn_sigma_test"],
            skew_alpha=decoded["sn_alpha_test"],
            **_zi_kwargs,
        )
        _, sn_sigma_blend_val, sn_alpha_blend_val, gate_blend_val = fused_loc(
            model_weight,
            ev_validation,
            book_ev_val,
            cv,
            "SkewNormal",
            sigma=decoded["sn_sigma_val"],
            skew_alpha=decoded["sn_alpha_val"],
            **_zi_kwargs,
        )
        out.update(
            {
                "model_weight": model_weight,
                "weighted_mean": weighted_mean,
                "gate_blend_test": gate_blend_test,
                "gate_blend_val": gate_blend_val,
                "sn_sigma_blend_test": sn_sigma_blend_test,
                "sn_sigma_blend_val": sn_sigma_blend_val,
                "sn_alpha_blend_test": sn_alpha_blend_test,
                "sn_alpha_blend_val": sn_alpha_blend_val,
            }
        )
        return out

    _zi_kwargs = {}
    if dist in ("ZINB", "ZAGamma") and hist_gate > 0:
        _zi_kwargs = {"gate_model": decoded["gate_validation"], "gate_book": hist_gate}
    model_weight = fit_model_weight(
        ev_validation,
        book_ev_val,
        y_val_result,
        base_dist,
        model_alpha=decoded["alpha_validation"],
        model_r=decoded["r_validation"],
        cv=cv,
        **_zi_kwargs,
    )
    out["model_weight"] = model_weight

    if dist in ("NegBin", "ZINB"):
        _zi_test = (
            {"gate_model": decoded["gate_test"], "gate_book": hist_gate} if dist == "ZINB" else {}
        )
        _zi_val = (
            {"gate_model": decoded["gate_validation"], "gate_book": hist_gate}
            if dist == "ZINB"
            else {}
        )
        r_blend_test, p_test, gate_blend_test = fused_loc(
            model_weight,
            ev,
            book_ev_test,
            cv,
            "NegBin",
            r=decoded["r"],
            **_zi_test,
        )
        r_blend_val, p_val, gate_blend_val = fused_loc(
            model_weight,
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

    # Gamma / ZAGamma
    _zi_test = (
        {"gate_model": decoded["gate_test"], "gate_book": hist_gate} if dist == "ZAGamma" else {}
    )
    _zi_val = (
        {"gate_model": decoded["gate_validation"], "gate_book": hist_gate}
        if dist == "ZAGamma"
        else {}
    )
    alpha_blend, beta_blend, gate_blend_test = fused_loc(
        model_weight,
        ev,
        book_ev_test,
        cv,
        "Gamma",
        alpha=decoded["alpha"],
        **_zi_test,
    )
    alpha_blend_val, beta_blend_val, gate_blend_val = fused_loc(
        model_weight,
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


def _step_select_distribution(
    splits: dict,
    stat_data,
    market: str,
    league: str,
    target_normalization: str,
    zinb_mode: str,
    *,
    deterministic: bool,
) -> dict:
    """Choose distribution family + apply target transform + compute shape priors.

    Branch logic: ``global_mean >= _SKEWNORMAL_MEAN_THRESHOLD`` → SkewNormal, otherwise
    NegBin (escalated to ZINB when ``hist_gate > GATE_PUBLISH_THRESHOLD``). For SkewNormal,
    drops zero rows when ``hist_gate > NONZERO_DENOM_GATE`` and applies the
    strategy's forward transform.

    Mutates ``splits["X_train"]`` and ``splits["y_train_labels"]`` for the
    SkewNormal nonzero path. Also writes ``stat_zi[league][market]`` and
    ``stat_cv[league][market]``; persists to JSON unless ``deterministic``.

    Args:
        splits: Output of ``_step_build_splits``.
        stat_data: Stats instance (gamelog + log_strings).
        market: Market name.
        league: League slug.
        target_normalization: Slug for ``baselines.get_target_normalization``.
        deterministic: If True, skip persisting cv/zi to stat_calibration.json.

    Returns:
        Dict with: ``dist``, ``dist_obj`` (None on hurdle path until built),
        ``cv``, ``shape_ceiling``, ``marginal_shape``, ``normalize``,
        ``offset_mode``, ``denom_col``, ``strategy``, ``hist_gate``,
        ``player_stats``, ``global_mean``.
    """
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
    if not deterministic:
        save_zi_config(stat_zi)

    player_stats = player_stats.apply(lambda x: x[x != 0]).groupby(level=0)

    global_mean = y_train_labels.mean()
    normalize = False
    offset_mode = False
    denom_col = "MeanYr"
    strategy = baselines.get_target_normalization(target_normalization)
    dist_obj = None

    if global_mean >= _SKEWNORMAL_MEAN_THRESHOLD:
        dist = "SkewNormal"
        dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")

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

        if hist_gate > NONZERO_DENOM_GATE:
            nonzero_mask = y_train_labels > 0
            X_train = X_train[nonzero_mask]
            y_train_labels = y_train_labels[nonzero_mask]
            denom_col = "MeanYr_nonzero" if "MeanYr_nonzero" in X_train.columns else "MeanYr"
        else:
            denom_col = "MeanYr"

        normalize = strategy.start_mode_flag == "normalized"
        offset_mode = strategy.start_mode_flag == "offset"
        y_train_labels = strategy.forward(y_train_labels, X_train, global_mean, denom_col)
    else:
        dist = "NegBin"
        if hist_gate > GATE_PUBLISH_THRESHOLD:
            dist = "ZINB"
        if dist == "NegBin":
            dist_obj = NegativeBinomial(stabilization="None", loss_fn="nll")
        elif zinb_mode == "joint":
            # Legacy jointly-fit LightGBMLSS ZINB path — byte-identical to pre-P2.B.
            dist_obj = ZINB(stabilization="None", loss_fn="nll")
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

    stat_cv[league][market] = cv
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
    }


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
    zinb_mode: str = "joint",
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
        posthoc_slug: Post-hoc corrector from
            :data:`sportstradamus.training.posthoc.POSTHOC_SLUGS`. A
            probability-stage slug recalibrates the over-probability after
            temperature scaling; ``"none"`` (default) is a no-op.
        zinb_mode: Either ``"joint"`` (legacy LightGBMLSS ZINB; the default) or
            ``"hurdle"`` (use ``HurdleZINB`` from ``sportstradamus.hurdle`` — a
            two-stage model with a derived-π gate; see
            docs/OVERCONFIDENCE_INVESTIGATION.md §2 and
            docs/CENTERED_TARGET_NEGATIVE_RESULT.md for context). Only consulted
            when the count-branch chooses ``dist == "ZINB"``. ``"joint"`` is
            byte-identical to pre-P2.B production behavior.
    """
    # style: allow-length  pre-existing research orchestrator (§2.8/§18.9): flag,
    # don't split. Already over the limit before the FBT keyword-only conversion.
    if zinb_mode not in {"joint", "hurdle"}:
        raise ValueError(f"zinb_mode must be 'joint' or 'hurdle', got {zinb_mode!r}")

    init = _step_init_market(league, market, stat_data, archive)
    filedict = init["filedict"]
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
    )
    if loaded is None:
        return
    M, training_data_path = loaded

    M, step = _step_synthesize_odds(M, league, market, dist, cv)
    M = _step_persist_matrix_and_comps(
        M, training_data_path, stat_data, deterministic=deterministic
    )

    splits = _step_build_splits(M, stat_data, market)
    dist_info = _step_select_distribution(
        splits, stat_data, market, league, target_normalization, zinb_mode, deterministic=deterministic
    )
    dist = dist_info["dist"]
    cv = dist_info["cv"]
    hist_gate = dist_info["hist_gate"]
    shape_ceiling = dist_info["shape_ceiling"]
    use_hurdle = dist == "ZINB" and zinb_mode == "hurdle"

    model = _step_build_lss_model(
        dist,
        dist_info["dist_obj"],
        splits["X_train"],
        shape_ceiling,
        normalize=dist_info["normalize"],
        offset_mode=dist_info["offset_mode"],
        use_hurdle=use_hurdle,
    )
    dtrain = lgb.Dataset(splits["X_train"], label=splits["y_train_labels"])
    opt_params_in = filedict.get("params")
    opt_params, _ = _step_select_hyperparams(
        splits["X_train"],
        dist,
        model,
        opt_params_in,
        dtrain,
        use_hurdle=use_hurdle,
        deterministic=deterministic,
    )
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
    )

    preds = _step_predict_splits(
        model, dist, splits, normalize=dist_info["normalize"], offset_mode=dist_info["offset_mode"]
    )
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
    fused = _step_fuse_predictions(decoded, splits, dist, cv, hist_gate)
    calibrated = _step_calibrate_dispersion(
        decoded,
        fused,
        splits,
        dist,
        cv,
        hist_gate,
        shape_ceiling,
        fused["model_weight"],
    )

    y_proba_no_filt = _step_compute_test_probabilities(
        fused, calibrated, splits, decoded, dist, step
    )
    T_opt, val_calibrated, model_calib, y_class_val = _step_calibrate_temperature(
        fused, calibrated, splits, decoded, dist, step
    )

    # Post-hoc probability recalibration (orthogonal to target_normalization):
    # fit on the temperature-calibrated validation over-probs, then layer it onto
    # both the validation probs (so skill reflects it) and the persisted test
    # probs (so the offline ship gates see the corrected cell). No-op when the
    # cell's posthoc slug is "none" or not a probability-stage corrector.
    posthoc_blob = None
    if posthoc_slug in posthoc.PROB_STAGE:
        posthoc_blob = posthoc.fit_posthoc(posthoc_slug, val_calibrated, y_class_val)
        val_calibrated = posthoc.apply_posthoc(posthoc_slug, posthoc_blob, val_calibrated)

    val_book_proba = splits["B_validation"]["Odds"].to_numpy(dtype=float)
    skill = _step_compute_skill_metrics(val_calibrated, y_class_val, val_book_proba, league, market)

    test_calibrated_over = apply_temperature(y_proba_no_filt[:, 1], T_opt)
    if posthoc_slug in posthoc.PROB_STAGE:
        test_calibrated_over = posthoc.apply_posthoc(posthoc_slug, posthoc_blob, test_calibrated_over)
    y_proba_filt = np.array([1 - test_calibrated_over, test_calibrated_over]).transpose()

    y_class = np.ravel(
        (splits["y_test"]["Result"] >= splits["B_test"]["Line"]).astype(int).to_numpy()
    )
    y_proba_raw = _build_y_proba_raw(splits["B_test"], decoded, dist, step)

    mode_stats = _step_compute_mode_stats(y_proba_raw, y_proba_no_filt, y_proba_filt, y_class)
    diag = _step_compute_diagnostics(
        splits,
        prob_params,
        fused["weighted_mean"],
        y_proba_no_filt,
        y_class,
        fused["gate_blend_test"],
        dist,
        cv,
        dist_info["denom_col"],
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
        zinb_mode=zinb_mode,
        X=splits["X"],
    )

    _step_persist_artifacts(
        filedict=filedict,
        splits=splits,
        prob_params=prob_params,
        decoded=decoded,
        weighted_mean=fused["weighted_mean"],
        y_proba_filt=y_proba_filt,
        dist=dist,
        hist_gate=hist_gate,
        filename=filename,
        deterministic=deterministic,
        target_normalization=target_normalization,
        zinb_mode=zinb_mode,
    )

    # Drift-monitoring SHAP: write per-cell |SHAP| + corr columns to the
    # training/feature_importances.csv + feature_correlations.csv. After the
    # 2026-05-27 no-filter rewire these CSVs are no longer used for selection;
    # they survive purely so the dashboard can show importance drift over time.
    # Skip in deterministic mode (artifacts must not leak from eval runs) and
    # skip hurdle (HurdleZINB has two separate boosters; SHAP would need a
    # custom path — defer to a follow-up if hurdle drift becomes interesting).
    if not deterministic and not use_hurdle:
        test_df = splits["X_test"].copy()
        test_df["Result"] = splits["y_test"]["Result"].to_numpy()
        compute_market_importance(league, market, model, test_df)

    if not deterministic:
        report()

    del filedict
    del model
