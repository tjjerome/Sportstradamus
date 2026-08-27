"""Pin the intercept-penalised Platt recalibrator (``prob_recal_platt_cv``).

The lambda=0 corner must reproduce ``prob_recal_platt`` exactly, lambda=inf must be
intercept-free, small-n base-rate noise must drive the CV to a positive penalty, and
degenerate inputs must degrade to the unpenalised map instead of raising.
"""

import math

import numpy as np
from scipy.special import expit, logit

from sportstradamus.training import posthoc
from sportstradamus.training.model_strategy import get_strategy


def _synthetic(n: int, seed: int, slope: float = 0.6, offset: float = 0.0):
    """Candidate probs with a miscalibrated slope and a small base-rate offset vs truth."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.2, n)
    y = (rng.random(n) < expit(z)).astype(float)
    return expit(slope * z + offset), y


def _feat(x: np.ndarray) -> np.ndarray:
    return logit(np.clip(x, posthoc._PROB_CLIP, 1 - posthoc._PROB_CLIP))


def test_lambda_zero_reproduces_plain_platt():
    x, y = _synthetic(300, seed=7)
    a, b = posthoc._platt_coeffs(_feat(x), y, 0.0)
    plain = posthoc._fit_platt(x, y)
    assert (a, b) == (plain["a"], plain["b"])


def test_lambda_inf_is_intercept_free():
    x, y = _synthetic(300, seed=7)
    a, b = posthoc._platt_coeffs(_feat(x), y, math.inf)
    assert b == 0.0
    assert math.isfinite(a) and a > 0.0


def test_small_n_base_rate_noise_selects_positive_penalty():
    x, y = _synthetic(250, seed=11, offset=0.15)
    blob = posthoc._fit_platt_cv(x, y)
    assert blob["lam"] > 0.0
    folds = posthoc._platt_cv_folds(len(x), None)
    selected = posthoc._platt_cv_oof_loss(_feat(x), y, folds, blob["lam"])
    baseline = posthoc._platt_cv_oof_loss(_feat(x), y, folds, 0.0)
    assert selected <= baseline


def test_single_class_returns_none():
    x = np.linspace(0.2, 0.8, 40)
    y = np.ones(40)
    assert posthoc._fit_platt_cv(x, y) is None
    assert posthoc.fit_posthoc("prob_recal_platt_cv", x, y) is None


def test_tiny_n_degrades_to_unpenalised_fallback():
    x, y = _synthetic(30, seed=3)
    blob = posthoc._fit_platt_cv(x, y)
    assert blob == {**posthoc._fit_platt(x, y), "lam": 0.0}
    # apply_posthoc must treat the blob as a plain platt map, ignoring the lam key.
    applied = posthoc.apply_posthoc("prob_recal_platt_cv", blob, x)
    expected = expit(blob["a"] * _feat(x) + blob["b"])
    assert np.allclose(applied, np.clip(expected, posthoc._PROB_CLIP, 1 - posthoc._PROB_CLIP))


def test_cluster_folds_are_group_disjoint():
    clusters = np.repeat([f"p{i}" for i in range(25)], 12)
    folds = posthoc._platt_cv_folds(len(clusters), clusters)
    owner: dict[str, int] = {}
    for i, fold in enumerate(folds):
        for label in np.unique(clusters[fold]):
            assert label not in owner
            owner[label] = i
    assert sum(len(fold) for fold in folds) == len(clusters)
    # fit_posthoc drops non-finite rows and must subset clusters by the same mask.
    x, y = _synthetic(300, seed=5)
    x[5] = np.nan
    blob = posthoc.fit_posthoc("prob_recal_platt_cv", x, y, clusters=clusters)
    assert blob is not None and blob["kind"] == "platt"


def test_registry_exposes_the_new_slug():
    assert "prob_recal_platt_cv" in posthoc.PROB_STAGE
    assert "prob_recal_platt_cv" in posthoc.POSTHOC_SLUGS
    # Out of the sweep axis pool after the 2026-08-27 live kill (specs._POSTHOC comment);
    # the slug stays registered so a stat_meta cell carrying it still fits and serves.
    assert "prob_recal_platt_cv" not in get_strategy("SkewNormal").axes["posthoc"]
    assert "prob_recal_platt_cv" not in get_strategy("NegBin").axes["posthoc"]
