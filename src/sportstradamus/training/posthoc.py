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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Correctors that transform a calibrated over-probability in [0, 1].
PROB_STAGE: frozenset[str] = frozenset({"prob_recal_isotonic", "prob_recal_platt"})
# Correctors that transform a decoded mean prediction (non-negative stat units).
MEAN_STAGE: frozenset[str] = frozenset({"roe_mean", "isotonic_mean"})
POSTHOC_SLUGS: frozenset[str] = PROB_STAGE | MEAN_STAGE | {"none"}

# Below this many finite rows a corrector overfits more than it calibrates; fall
# back to identity (the offline ship gate then judges the uncorrected cell).
_MIN_FIT_ROWS: int = 10
_PROB_CLIP: float = 1e-4
# Effectively unregularized Platt scaling — we want the calibration MLE, not a
# shrunk-toward-zero slope.
_PLATT_C: float = 1e6


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
