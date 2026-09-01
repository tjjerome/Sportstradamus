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
    book = np.clip(p + rng.normal(0.0, 0.05, p.size), 0.05, 0.95)
    return posthoc.fit_posthoc(slug, p, y, book=book)


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


def test_mean_stage_seam_matches_across_train_and_serve():
    """Both paths correct the FUSED mean, so the same blob lands on the same object.

    The training side pools in ``_step_fuse_predictions`` and corrects in
    ``_step_correct_fused_mean``; the live side does both inside ``_blend_with_book``.
    Feeding the training side the pool the live side computed isolates the seam: if
    either moved the corrector back to the model leg, these would diverge by the log
    pool's ``rho^(1-w)`` haircut.
    """
    import pandas as pd

    from sportstradamus.prediction.model_prob import _blend_with_book
    from sportstradamus.training.pipeline import _step_correct_fused_mean

    rng = np.random.default_rng(7)
    model_mean = rng.uniform(0.4, 3.0, 400)
    book_mean = model_mean * rng.uniform(0.3, 0.6, 400)
    result = rng.poisson(model_mean * 1.25).astype(float)

    def _serve(slug, blob):
        offer_df = pd.DataFrame(
            {
                "Projection": model_mean,
                "Market Projection": book_mean,
                "Line": np.full(model_mean.size, 1.5),
                "Model R": np.full(model_mean.size, 3.0),
            }
        )
        return np.asarray(
            _blend_with_book(offer_df, "NegBin", 0.6, 1.0, 0.0, "NBA", "BLK", slug, blob),
            dtype=float,
        )

    pooled = _serve("none", None)
    for slug in sorted(posthoc.MEAN_STAGE):
        fused = {
            "weighted_mean": pooled.copy(),
            "weighted_mean_val": pooled.copy(),
            "gate_blend_test": None,
            "gate_blend_val": None,
        }
        splits = {"y_validation": pd.DataFrame({"Result": result})}
        blob = _step_correct_fused_mean(fused, splits, "NegBin", slug)
        np.testing.assert_allclose(_serve(slug, blob), fused["weighted_mean"], rtol=1e-12)
