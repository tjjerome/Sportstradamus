"""Post-hoc per-cell corrections fit on the validation split and reapplied at inference.

Orthogonal to the target-normalization strategy in :mod:`baselines`: a strategy
reshapes the GBDT *target* (and its loc/scale decode); a post-hoc corrector adjusts
either the decoded *mean* (``roe_mean`` / ``isotonic_mean``) or the final
over-*probability* (``prob_recal_isotonic`` / ``prob_recal_platt``) after the
distribution is already formed. Selected per cell via the ``posthoc`` field in
``stat_meta.json``; ``"none"`` is a no-op.

The fitted state is a plain-typed ``dict`` (lists/floats) so model pickles stay
portable, and :func:`apply_posthoc` reproduces the fit's transform exactly — the
training-side test CSV and the live ``model_prob`` path must agree event-for-event.
"""

import numpy as np
from scipy.special import expit, logit
from scipy.stats import kstest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Correctors that transform a calibrated over-probability in [0, 1].
PROB_STAGE: frozenset[str] = frozenset({"prob_recal_isotonic", "prob_recal_platt"})
# Correctors that transform a decoded mean prediction (non-negative stat units).
MEAN_STAGE: frozenset[str] = frozenset({"roe_mean", "isotonic_mean"})
# §6.1 Rung C — whole-CDF recalibration. Where PROB_STAGE recalibrates a single-line
# over-probability and MEAN_STAGE shifts the decoded mean, this reshapes the entire
# predictive CDF via a monotone map on the PIT (fit by select_pit_recal, applied by
# apply_cdf_recal), subsuming both the scalar dispersion calibration and prob_recal_*.
# The posthoc field is single-valued, so selecting it is structurally exclusive with
# every other corrector — at most one corrector per cell, never two stacked.
CDF_STAGE: frozenset[str] = frozenset({"cdf_recal_isotonic"})
POSTHOC_SLUGS: frozenset[str] = PROB_STAGE | MEAN_STAGE | CDF_STAGE | {"none"}

# Below this many finite rows a corrector overfits more than it calibrates; fall
# back to identity (the offline ship gate then judges the uncorrected cell).
_MIN_FIT_ROWS: int = 10
_PROB_CLIP: float = 1e-4
# Effectively unregularized Platt scaling — we want the calibration MLE, not a
# shrunk-toward-zero slope.
_PLATT_C: float = 1e6

# A whole-CDF map needs few knots to flatten a smooth PIT and overfits with many, so
# B equal-mass bins cap its degrees of freedom (B≈10 matches the g5 ECE binning). The
# shrink weight lambda blends the empirical map toward the identity
# (g_lambda = lambda*g + (1-lambda)*id); the cross-fit below selects lambda per cell.
_PIT_RECAL_BINS: int = 10
# lambda is chosen per cell by honest K-fold cross-fit — a single split flatters a
# high-DOF map — from no-recal to full-recal; K matches the project's other CV uses.
# The fold permutation is seeded so a shipped map reproduces on retrain.
_PIT_RECAL_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
_PIT_RECAL_CV_FOLDS: int = 5
_PIT_RECAL_CV_SEED: int = 0


def fit_posthoc(slug: str, x: np.ndarray, y: np.ndarray) -> dict | None:
    """Fit a corrector on validation data; ``None`` means "apply nothing".

    Args:
        slug: One of :data:`POSTHOC_SLUGS`.
        x: Validation predictions — over-probability for :data:`PROB_STAGE`,
            decoded mean for :data:`MEAN_STAGE`.
        y: Validation outcomes — binary over/under for :data:`PROB_STAGE`,
            raw result for :data:`MEAN_STAGE`.
    """
    if slug not in POSTHOC_SLUGS:
        raise ValueError(f"Unknown posthoc slug {slug!r}; valid: {sorted(POSTHOC_SLUGS)}")
    if slug == "none":
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < _MIN_FIT_ROWS or np.ptp(x) == 0.0:
        return None
    if slug == "prob_recal_platt":
        return _fit_platt(x, y)
    if slug in ("prob_recal_isotonic", "isotonic_mean"):
        return _fit_isotonic(x, y)
    return _fit_affine(x, y)


def apply_posthoc(slug: str, blob: dict | None, x: np.ndarray) -> np.ndarray:
    """Apply a fitted corrector to test/live predictions; identity when ``blob`` is None."""
    x = np.asarray(x, dtype=float)
    if blob is None:
        return x
    kind = blob["kind"]
    if kind == "isotonic":
        out = np.interp(x, blob["x"], blob["y"])
    elif kind == "platt":
        out = expit(blob["a"] * logit(np.clip(x, _PROB_CLIP, 1 - _PROB_CLIP)) + blob["b"])
    else:  # affine
        out = blob["a"] + blob["b"] * x
    if slug in MEAN_STAGE:
        return np.clip(out, 0.0, None)
    return np.clip(out, _PROB_CLIP, 1 - _PROB_CLIP)


def fit_isotonic_pit(
    pit: np.ndarray, *, n_bins: int = _PIT_RECAL_BINS, lam: float = 1.0
) -> dict | None:
    """Fit the §6.1 Rung C recalibration map ``g`` on validation PIT values.

    ``g`` is the empirical CDF of ``pit`` over ``n_bins`` equal-mass bins (its degrees of
    freedom capped), shrunk toward the identity as ``g_lam = lam*g + (1-lam)*id``: ``lam=1``
    is full recalibration, ``lam=0`` an exact no-op. Applying it to a cell's PIT uniformizes
    it; applying it to any CDF value ``u = F(line)`` yields the recalibrated ``g(u)``. Returns
    ``None`` (apply nothing) below the fit-row floor.
    """
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    if len(pit) < _MIN_FIT_ROWS:
        return None
    n = len(pit)
    sorted_pit = np.sort(pit, kind="stable")
    coverage = (np.arange(n) + 0.5) / n  # empirical-CDF plotting positions
    bins = min(max(n_bins, 2), n)
    x_knots, y_knots = [0.0], [0.0]
    for grp_pit, grp_cov in zip(
        np.array_split(sorted_pit, bins), np.array_split(coverage, bins), strict=True
    ):
        x = float(grp_pit.mean())
        if x <= x_knots[-1]:  # np.interp needs a strictly increasing domain
            continue
        x_knots.append(x)
        y_knots.append(lam * float(grp_cov.mean()) + (1.0 - lam) * x)
    x_knots.append(1.0)
    y_knots.append(1.0)
    return {"kind": "isotonic_pit", "x": x_knots, "y": y_knots}


def apply_cdf_recal(blob: dict | None, u: np.ndarray) -> np.ndarray:
    """Apply a Rung C map to CDF value(s) ``u = F(point)``; identity when ``blob`` is None."""
    u = np.asarray(u, dtype=float)
    if blob is None:
        return u
    return np.clip(np.interp(u, blob["x"], blob["y"]), 0.0, 1.0)


def select_pit_recal(
    pit: np.ndarray,
    *,
    lambdas: tuple[float, ...] = _PIT_RECAL_LAMBDA_GRID,
    k: int = _PIT_RECAL_CV_FOLDS,
) -> tuple[dict | None, float]:
    """Select the Rung C shrink ``lam`` by honest K-fold cross-fit, then fit the served map.

    For each candidate ``lam`` the map is fit on K−1 folds and applied to the held-out fold;
    the concatenated out-of-fold ``g_lam(Z)`` gives an optimism-free Gate-4 KS. The ``lam``
    minimizing that KS is taken (ties broken toward more recalibration) and the shipped map
    re-fit on the full split at that ``lam``. Returns ``(blob, cross_fit_ks)`` — ``cross_fit_ks``
    is the number to gate on; ``blob`` is ``None`` (apply nothing) below the fit-row floor.
    """
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    if len(pit) < _MIN_FIT_ROWS:
        return None, float("nan")
    best_lam, best_ks = lambdas[0], float("inf")
    for lam in lambdas:
        ks = _crossfit_pit_ks(pit, lam, k)
        if ks < best_ks or (ks == best_ks and lam > best_lam):
            best_lam, best_ks = lam, ks
    blob = fit_isotonic_pit(pit, lam=best_lam)
    if blob is not None:
        blob["lam"] = best_lam
    return blob, best_ks


def _crossfit_pit_ks(pit: np.ndarray, lam: float, k: int) -> float:
    """Out-of-fold KS-from-Uniform of ``g_lam(Z)`` — the honest (non-in-sample) Gate-4 estimate."""
    n = len(pit)
    folds_n = max(2, min(k, n // _MIN_FIT_ROWS))
    if folds_n < 2:
        return _pit_ks(apply_cdf_recal(fit_isotonic_pit(pit, lam=lam), pit))
    rng = np.random.default_rng(_PIT_RECAL_CV_SEED)
    oof = np.empty(n)
    for fold in np.array_split(rng.permutation(n), folds_n):
        train = np.setdiff1d(np.arange(n), fold, assume_unique=True)
        oof[fold] = apply_cdf_recal(fit_isotonic_pit(pit[train], lam=lam), pit[fold])
    return _pit_ks(oof)


def _pit_ks(u: np.ndarray) -> float:
    return float(kstest(np.clip(u, 0.0, 1.0), "uniform").statistic)


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> dict:
    iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    return {
        "kind": "isotonic",
        "x": [float(v) for v in iso.X_thresholds_],
        "y": [float(v) for v in iso.y_thresholds_],
    }


def _fit_platt(x: np.ndarray, y: np.ndarray) -> dict | None:
    if len(np.unique(y)) < 2:
        return None
    feat = logit(np.clip(x, _PROB_CLIP, 1 - _PROB_CLIP)).reshape(-1, 1)
    lr = LogisticRegression(C=_PLATT_C).fit(feat, y)
    return {"kind": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def _fit_affine(x: np.ndarray, y: np.ndarray) -> dict:
    b, a = np.polyfit(x, y, 1)
    return {"kind": "affine", "a": float(a), "b": float(b)}
