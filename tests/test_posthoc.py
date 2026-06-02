"""Unit tests for the post-hoc corrector registry (training/posthoc.py)."""
import numpy as np
import pytest
from scipy.special import expit, logit

from sportstradamus.training import posthoc


def _miscalibrated_probs(n=4000, seed=0):
    """Over-probabilities that are systematically over-confident vs outcomes."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < true_p).astype(float)
    # report a sharpened (over-confident) probability: pushes away from 0.5
    p_reported = np.clip(expit(1.6 * logit(true_p)), 1e-4, 1 - 1e-4)
    return p_reported, y, true_p


def test_none_is_identity():
    assert posthoc.fit_posthoc("none", np.array([0.2, 0.8]), np.array([0.0, 1.0])) is None
    x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(posthoc.apply_posthoc("none", None, x), x)


def test_unknown_slug_raises():
    with pytest.raises(ValueError):
        posthoc.fit_posthoc("not_a_slug", np.array([0.2]), np.array([1.0]))


def test_slug_buckets_partition():
    assert posthoc.PROB_STAGE.isdisjoint(posthoc.MEAN_STAGE)
    assert "none" in posthoc.POSTHOC_SLUGS
    assert posthoc.PROB_STAGE <= posthoc.POSTHOC_SLUGS
    assert posthoc.MEAN_STAGE <= posthoc.POSTHOC_SLUGS


@pytest.mark.parametrize("slug", ["prob_recal_isotonic", "prob_recal_platt"])
def test_prob_recal_reduces_brier(slug):
    p, y, _ = _miscalibrated_probs()
    blob = posthoc.fit_posthoc(slug, p, y)
    p_corr = posthoc.apply_posthoc(slug, blob, p)
    assert ((p_corr >= 0) & (p_corr <= 1)).all()
    brier_before = np.mean((p - y) ** 2)
    brier_after = np.mean((p_corr - y) ** 2)
    assert brier_after < brier_before


def test_prob_recal_blob_is_plain_types():
    """Blob must be JSON/pickle-safe plain floats/lists so model pickles stay portable."""
    p, y, _ = _miscalibrated_probs()
    for slug in ("prob_recal_isotonic", "prob_recal_platt"):
        blob = posthoc.fit_posthoc(slug, p, y)
        assert isinstance(blob, dict)
        for v in blob.values():
            assert isinstance(v, (str, float, int, list))


def test_isotonic_apply_matches_interp():
    p, y, _ = _miscalibrated_probs()
    blob = posthoc.fit_posthoc("prob_recal_isotonic", p, y)
    x = np.linspace(0.01, 0.99, 50)
    expected = np.clip(np.interp(x, blob["x"], blob["y"]), 1e-4, 1 - 1e-4)
    assert np.allclose(posthoc.apply_posthoc("prob_recal_isotonic", blob, x), expected)


def test_roe_mean_recovers_affine():
    rng = np.random.default_rng(1)
    pred = rng.uniform(1, 30, 3000)
    # outcomes are a known affine function of the prediction plus noise
    actual = 2.0 + 0.5 * pred + rng.normal(0, 1, pred.size)
    blob = posthoc.fit_posthoc("roe_mean", pred, actual)
    assert blob["kind"] == "affine"
    assert blob["a"] == pytest.approx(2.0, abs=0.3)
    assert blob["b"] == pytest.approx(0.5, abs=0.05)
    corr = posthoc.apply_posthoc("roe_mean", blob, np.array([10.0]))
    assert corr[0] == pytest.approx(7.0, abs=0.5)


def test_mean_stage_output_nonnegative():
    rng = np.random.default_rng(2)
    pred = rng.uniform(0, 5, 500)
    actual = np.clip(pred - 3.0, 0, None)  # correction would push some negative
    blob = posthoc.fit_posthoc("roe_mean", pred, actual)
    out = posthoc.apply_posthoc("roe_mean", blob, np.array([0.0, 0.1, 0.5]))
    assert (out >= 0).all()


def test_degenerate_fit_is_noop():
    # constant prediction => no slope to fit => identity (None blob)
    x = np.full(50, 0.4)
    y = np.zeros(50)
    assert posthoc.fit_posthoc("roe_mean", x, y) is None
    # tiny sample => identity
    assert posthoc.fit_posthoc("prob_recal_isotonic", x[:3], y[:3]) is None
