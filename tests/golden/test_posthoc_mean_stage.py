import numpy as np

from sportstradamus.training import posthoc


def test_roe_mean_corrects_compression_on_validation():
    # Compressed model: predicted mean is shrunk toward the grand mean (slope < 1
    # vs the truth), exactly the leaf-averaging failure roe_mean targets.
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 4, size=500)
    grand = truth.mean()
    compressed_mu = grand + 0.6 * (truth - grand)  # slope 0.6 => compressed
    blob = posthoc.fit_posthoc("roe_mean", compressed_mu, truth)
    assert blob is not None and blob["kind"] == "affine"
    corrected = posthoc.apply_posthoc("roe_mean", blob, compressed_mu)
    # The affine fit should restore the decompressed slope: corrected tracks truth
    # with slope ~1.
    slope = np.polyfit(corrected, truth, 1)[0]
    assert 0.9 < slope < 1.1
    # MEAN_STAGE output is clipped non-negative.
    assert (corrected >= 0).all()


def test_mean_stage_is_noop_when_slug_is_none():
    # apply with blob=None is identity; the train_market guard only fits when the
    # slug is in MEAN_STAGE, so a "none"/prob-stage cell never touches decoded ev.
    mu = np.array([1.0, 2.0, 3.0])
    assert np.allclose(posthoc.apply_posthoc("none", None, mu), mu)
    assert "roe_mean" in posthoc.MEAN_STAGE
    assert "roe_mean" not in posthoc.PROB_STAGE
