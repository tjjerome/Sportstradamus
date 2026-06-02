"""The live inference seam must reproduce the training-side post-hoc transform.

``pipeline.train_market`` applies ``posthoc.apply_posthoc`` to the test-set
over-probabilities (what the offline ship gates score); ``model_prob`` applies
``_apply_prob_posthoc`` to the live over-probabilities. If they diverge, a cell
ships offline but behaves differently in production. These tests pin them together.
"""
import numpy as np

from sportstradamus.prediction.model_prob import _apply_prob_posthoc
from sportstradamus.training import posthoc


def _fitted_blob(slug):
    rng = np.random.default_rng(0)
    p = np.clip(rng.uniform(0.05, 0.95, 2000), 1e-4, 1 - 1e-4)
    y = (rng.uniform(size=p.size) < p).astype(float)
    return posthoc.fit_posthoc(slug, p, y)


def test_inference_matches_training_for_prob_stage():
    cal_over = np.linspace(0.02, 0.98, 200)
    for slug in posthoc.PROB_STAGE:
        blob = _fitted_blob(slug)
        training_side = posthoc.apply_posthoc(slug, blob, cal_over)
        inference_side = _apply_prob_posthoc(cal_over, slug, blob)
        np.testing.assert_array_equal(training_side, inference_side)


def test_inference_is_identity_for_legacy_pickle():
    """slug 'none' / blob None (pickles written before posthoc) must pass through."""
    cal_over = np.linspace(0.02, 0.98, 50)
    np.testing.assert_array_equal(_apply_prob_posthoc(cal_over, "none", None), cal_over)


def test_inference_ignores_mean_stage_at_prob_seam():
    """A mean-stage corrector adjusts the decoded mean, never the over-prob seam."""
    cal_over = np.linspace(0.02, 0.98, 50)
    blob = {"kind": "affine", "a": 1.0, "b": 2.0}
    for slug in posthoc.MEAN_STAGE:
        np.testing.assert_array_equal(_apply_prob_posthoc(cal_over, slug, blob), cal_over)
