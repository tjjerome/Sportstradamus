"""Per-cell calibration-method selector, fit on the validation split and reapplied at inference.

The single-valued ``posthoc`` field in ``stat_meta.json`` names **at most one**
calibration method for a cell — a light post-distribution corrector *or* a structural
group-conditional-CDF method, never both — so the field structurally enforces mutual
exclusivity. ``"none"`` is a no-op.

*Light correctors* (this module's :func:`fit_posthoc` / :func:`apply_posthoc`) are
orthogonal to the target-normalization strategy in :mod:`baselines`: a strategy reshapes
the GBDT *target* (and its loc/scale decode); a corrector adjusts either the decoded
*mean* (``roe_mean`` / ``isotonic_mean``) or the final over-*probability*
(``prob_recal_isotonic`` / ``prob_recal_platt`` / ``prob_recal_platt_cv``) after the
distribution is already formed.

*Structural methods* (:data:`STRUCTURAL_STAGE`) reshape the target/CDF earlier in the
fit rather than after it, so they are dispatched by ``training.pipeline`` to their own
group-conditional-CDF stage (context build + fit/apply) — never through
:func:`fit_posthoc`, which treats a structural slug as identity. Their fitted state
persists under the ``structural_calibration`` pickle key; serving dispatches on the
resolved strategy identity, not on this field.

The light correctors' fitted state is a plain-typed ``dict`` (lists/floats) so model
pickles stay portable, and :func:`apply_posthoc` reproduces the fit's transform exactly —
the training-side test CSV and the live ``model_prob`` path must agree event-for-event.
"""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import kstest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sportstradamus.helpers.distributions import apply_cdf_recal
from sportstradamus.training.structural_strategies import AFFINE_STRATEGY, TWO_PART_STRATEGY

# Correctors that transform a calibrated over-probability in [0, 1].
PROB_STAGE: frozenset[str] = frozenset(
    {"prob_recal_isotonic", "prob_recal_platt", "prob_recal_platt_cv"}
)
# Correctors that transform a decoded mean prediction (non-negative stat units).
MEAN_STAGE: frozenset[str] = frozenset({"roe_mean", "isotonic_mean"})
# §6.1 Rung C — whole-CDF recalibration. Where PROB_STAGE recalibrates a single-line
# over-probability and MEAN_STAGE shifts the decoded mean, this reshapes the entire
# predictive CDF via a monotone map on the PIT (fit here by select_pit_recal, applied
# by helpers.distributions.apply_cdf_recal), subsuming both the scalar dispersion
# calibration and prob_recal_*. The posthoc field is single-valued, so selecting it is
# structurally exclusive with every other corrector — at most one corrector per cell,
# never two stacked.
CDF_STAGE: frozenset[str] = frozenset({"cdf_recal_isotonic"})
# Structural group-conditional-CDF methods graduated into this selector. Unlike the
# corrector stages above, these do not run through fit_posthoc/apply_posthoc — the
# pipeline routes a structural slug to its structural stage. Listed here so the field
# validates and the sweep/CLI Choice enumerate them as mutually-exclusive pool members.
STRUCTURAL_STAGE: frozenset[str] = frozenset({TWO_PART_STRATEGY, AFFINE_STRATEGY})
POSTHOC_SLUGS: frozenset[str] = PROB_STAGE | MEAN_STAGE | CDF_STAGE | STRUCTURAL_STAGE | {"none"}

# Below this many finite rows a corrector overfits more than it calibrates; fall
# back to identity (the offline ship gate then judges the uncorrected cell).
_MIN_FIT_ROWS: int = 10
_PROB_CLIP: float = 1e-4
# Effectively unregularized Platt scaling — we want the calibration MLE, not a
# shrunk-toward-zero slope.
_PLATT_C: float = 1e6

# Intercept-penalty grid for prob_recal_platt_cv: the Platt intercept estimates the
# fit-split over-rate, which is unlearnable at small n and transfers a Brier cost
# across the val/holdout over-rate gap, while the slope carries the transferable
# shrinkage. 0 recovers the unpenalised map, inf is intercept-free; the interior
# points are log-spaced. Out-of-fold log-loss picks per cell (folds/seed reuse the
# _PIT_RECAL_CV_* constants below).
_PLATT_CV_LAMBDA_GRID: tuple[float, ...] = (0.0, 1.0, 5.0, 20.0, 100.0, math.inf)
# Below this many rows a fold's held-out log-loss is noise and a single-class train
# split cannot fit a map, so the selection degrades to the unpenalised lambda=0 fit.
_PLATT_CV_MIN_FOLD_ROWS: int = 20

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


def fit_posthoc(
    slug: str, x: np.ndarray, y: np.ndarray, clusters: np.ndarray | None = None
) -> dict | None:
    """Fit a corrector on validation data; ``None`` means "apply nothing".

    Args:
        slug: One of :data:`POSTHOC_SLUGS`.
        x: Validation predictions — over-probability for :data:`PROB_STAGE`,
            decoded mean for :data:`MEAN_STAGE`.
        y: Validation outcomes — binary over/under for :data:`PROB_STAGE`,
            raw result for :data:`MEAN_STAGE`.
        clusters: Optional per-row group labels (player identity) aligned with ``x``.
            Only ``prob_recal_platt_cv`` reads them, to keep its CV folds group-disjoint.
    """
    if slug not in POSTHOC_SLUGS:
        raise ValueError(f"Unknown posthoc slug {slug!r}; valid: {sorted(POSTHOC_SLUGS)}")
    if slug == "none" or slug in STRUCTURAL_STAGE:
        # A structural method reshapes the CDF at its own pipeline stage, not here.
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < _MIN_FIT_ROWS or np.ptp(x) == 0.0:
        return None
    if slug == "prob_recal_platt_cv":
        return _fit_platt_cv(x, y, None if clusters is None else np.asarray(clusters)[finite])
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


def served_mean(base_mean: np.ndarray, gate: np.ndarray | float | None) -> np.ndarray:
    """The mean the ship gates score: the base mean deflated by a zero-inflation gate.

    ZINB/ZAGamma and gated SkewNormal carry the BASE-distribution mean, with the gate
    reapplied only when pricing over/under probabilities, so ``E[Y] = (1 - gate) * base``.
    ``gate`` is ``None`` for a gate-free family, which is what ``fused_loc`` returns.
    """
    base_mean = np.asarray(base_mean, dtype=float)
    if gate is None:
        return base_mean
    return base_mean * (1.0 - np.asarray(gate, dtype=float))


def correct_fused_mean(
    slug: str, blob: dict | None, base_mean: np.ndarray, gate: np.ndarray | float | None
) -> np.ndarray:
    """Apply a :data:`MEAN_STAGE` corrector to the SERVED mean; return the corrected BASE mean.

    Ranjan & Gneiting (2010, doi:10.1111/j.1467-9868.2009.00726.x) Thm 1: a non-trivial
    pool of calibrated components is itself uncalibrated, so the corrector belongs on the
    model-book combination rather than on either leg. It therefore acts on the object the
    gates score — ``(1 - gate) * base`` on a zero-inflated family, the base mean otherwise
    — and the gate divides back out so the family keeps its base-mean parameterization.
    Training and inference both call this, which is what keeps the two stages identical.

    Args:
        slug: The cell's ``posthoc`` value; anything outside :data:`MEAN_STAGE` is identity.
        blob: The fitted corrector, or ``None`` for identity.
        base_mean: Post-fusion base-distribution mean.
        gate: Blended zero-inflation probability, or ``None`` for a gate-free family.
    """
    if slug not in MEAN_STAGE:
        return np.asarray(base_mean, dtype=float)
    corrected = apply_posthoc(slug, blob, served_mean(base_mean, gate))
    if gate is None:
        return corrected
    return corrected / (1.0 - np.asarray(gate, dtype=float))


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
    if x_knots[-1] == 1.0:
        # A terminal atom can make the final equal-mass bin's mean exactly one.
        # Preserve the bin's fitted coverage immediately to the left of one,
        # then add the true CDF endpoint. This keeps the empirical map intact
        # while giving np.interp the strictly increasing domain it requires.
        left_of_one = float(np.nextafter(1.0, 0.0))
        y_knots[-1] -= (1.0 - lam) * (1.0 - left_of_one)
        x_knots[-1] = left_of_one
    x_knots.append(1.0)
    y_knots.append(1.0)
    return {"kind": "isotonic_pit", "x": x_knots, "y": y_knots}


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
    draws = _pit_draw_matrix(pit)
    if draws.shape[1] < _MIN_FIT_ROWS:
        return None, float("nan")
    best_lam, best_ks = lambdas[0], float("inf")
    for lam in lambdas:
        ks = _crossfit_pit_ks(draws, lam, k)
        if ks < best_ks or (ks == best_ks and lam > best_lam):
            best_lam, best_ks = lam, ks
    blob = fit_isotonic_pit(draws.reshape(-1), lam=best_lam)
    if blob is not None:
        blob["lam"] = best_lam
    return blob, best_ks


def _crossfit_pit_ks(pit: np.ndarray, lam: float, k: int) -> float:
    """Row-grouped out-of-fold KS of ``g_lam(Z)``.

    ``pit`` may be one PIT per row or a ``(draw, row)`` matrix for a predictive
    with atoms. Every randomized draw for a source row stays in the same fold;
    otherwise duplicate positive-row PITs leak into the training folds and make
    the cross-fit estimate optimistically pseudoreplicated.
    """
    draws = _pit_draw_matrix(pit)
    n = draws.shape[1]
    folds_n = max(2, min(k, n // _MIN_FIT_ROWS))
    if folds_n < 2:
        blob = fit_isotonic_pit(draws.reshape(-1), lam=lam)
        return float(np.mean([_pit_ks(apply_cdf_recal(blob, draw)) for draw in draws]))
    rng = np.random.default_rng(_PIT_RECAL_CV_SEED)
    oof = np.empty_like(draws)
    for fold in np.array_split(rng.permutation(n), folds_n):
        train = np.setdiff1d(np.arange(n), fold, assume_unique=True)
        blob = fit_isotonic_pit(draws[:, train].reshape(-1), lam=lam)
        oof[:, fold] = apply_cdf_recal(blob, draws[:, fold])
    return float(np.mean([_pit_ks(draw) for draw in oof]))


def _pit_draw_matrix(pit: np.ndarray) -> np.ndarray:
    """Return finite PIT values as ``(draw, source-row)`` without splitting row groups."""
    draws = np.asarray(pit, dtype=float)
    if draws.ndim == 1:
        draws = draws[None, :]
    elif draws.ndim != 2:
        raise ValueError("PIT input must be one-dimensional or a (draw, row) matrix")
    return draws[:, np.all(np.isfinite(draws), axis=0)]


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
    a, b = _platt_coeffs(logit(np.clip(x, _PROB_CLIP, 1 - _PROB_CLIP)), y, 0.0)
    return {"kind": "platt", "a": a, "b": b}


def _fit_platt_cv(x: np.ndarray, y: np.ndarray, clusters: np.ndarray | None = None) -> dict | None:
    """Platt map with the intercept penalty selected by out-of-fold log-loss.

    Folds are group-disjoint over ``clusters`` when given, so a repeat player cannot
    leak the fit split's over-rate into the held-out score. Ties go to the larger
    penalty; the winning ``lam`` is refit on all rows and recorded in the blob, which
    :func:`apply_posthoc` treats as a plain ``"platt"`` map.
    """
    if len(np.unique(y)) < 2:
        return None
    feat = logit(np.clip(x, _PROB_CLIP, 1 - _PROB_CLIP))
    folds = _platt_cv_folds(len(feat), clusters)
    if any(len(f) < _PLATT_CV_MIN_FOLD_ROWS or len(np.unique(y[f])) < 2 for f in folds):
        return {**_fit_platt(x, y), "lam": 0.0}
    best_lam, best_loss = _PLATT_CV_LAMBDA_GRID[0], float("inf")
    for lam in _PLATT_CV_LAMBDA_GRID:
        loss = _platt_cv_oof_loss(feat, y, folds, lam)
        if loss < best_loss or (loss == best_loss and lam > best_lam):
            best_lam, best_loss = lam, loss
    a, b = _platt_coeffs(feat, y, best_lam)
    return {"kind": "platt", "a": a, "b": b, "lam": float(best_lam)}


def _platt_cv_folds(n: int, clusters: np.ndarray | None) -> list[np.ndarray]:
    """Seeded row-index folds, group-disjoint over ``clusters`` when given."""
    rng = np.random.default_rng(_PIT_RECAL_CV_SEED)
    if clusters is None:
        return np.array_split(rng.permutation(n), _PIT_RECAL_CV_FOLDS)
    groups = np.array_split(rng.permutation(np.unique(clusters)), _PIT_RECAL_CV_FOLDS)
    return [np.flatnonzero(np.isin(clusters, group)) for group in groups]


def _platt_cv_oof_loss(
    feat: np.ndarray, y: np.ndarray, folds: list[np.ndarray], lam: float
) -> float:
    oof = np.empty_like(feat)
    for fold in folds:
        train = np.setdiff1d(np.arange(len(feat)), fold, assume_unique=True)
        a, b = _platt_coeffs(feat[train], y[train], lam)
        oof[fold] = a * feat[fold] + b
    return _bernoulli_nll(oof, y)


def _platt_coeffs(feat: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, float]:
    """Platt ``(slope, intercept)`` on a logit feature under intercept penalty ``lam * b**2``.

    ``lam == 0`` is the exact unpenalised ``prob_recal_platt`` fit; ``lam == inf`` drops
    the intercept; an interior ``lam`` solves the penalised MLE from the unpenalised start.
    """
    if lam == 0.0:
        lr = LogisticRegression(C=_PLATT_C).fit(feat.reshape(-1, 1), y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if math.isinf(lam):
        lr = LogisticRegression(C=_PLATT_C, fit_intercept=False).fit(feat.reshape(-1, 1), y)
        return float(lr.coef_[0, 0]), 0.0

    def objective(params: np.ndarray) -> float:
        return _bernoulli_nll(params[0] * feat + params[1], y) + lam * params[1] ** 2

    res = minimize(objective, np.array(_platt_coeffs(feat, y, 0.0)), method="BFGS")
    if not res.success:
        res = minimize(objective, res.x, method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def _bernoulli_nll(logits: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(expit(logits), _PROB_CLIP, 1 - _PROB_CLIP)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_affine(x: np.ndarray, y: np.ndarray) -> dict:
    b, a = np.polyfit(x, y, 1)
    return {"kind": "affine", "a": float(a), "b": float(b)}
