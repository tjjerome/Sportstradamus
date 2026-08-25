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


def test_correct_fused_mean_is_plain_apply_without_a_gate():
    mu = np.array([0.5, 1.5, 2.5, 3.5])
    blob = {"kind": "affine", "a": 0.2, "b": 1.3}
    got = posthoc.correct_fused_mean("roe_mean", blob, mu, None)
    assert np.allclose(got, posthoc.apply_posthoc("roe_mean", blob, mu))
    # a prob-stage slug must never touch the mean path, and a legacy pickle with no
    # blob decodes to identity
    assert np.allclose(posthoc.correct_fused_mean("prob_recal_platt", blob, mu, None), mu)
    assert np.allclose(posthoc.correct_fused_mean("none", None, mu, None), mu)


def test_correct_fused_mean_applies_the_gate_exactly_once():
    # The mis-contract this replaces fit the corrector on the gate-EXCLUDED base mean
    # against the zero-INCLUSIVE Result, then let the gate hit the corrected value a
    # second time downstream. Here the corrector sees the served mean and the gate
    # divides straight back out, so the served mean is the corrected served mean.
    base = np.array([0.4, 1.2, 2.0, 3.1])
    gate = np.array([0.0, 0.25, 0.5, 0.75])
    blob = {"kind": "affine", "a": 0.1, "b": 1.4}
    corrected_base = posthoc.correct_fused_mean("roe_mean", blob, base, gate)
    served_in = posthoc.served_mean(base, gate)
    assert np.allclose(
        posthoc.served_mean(corrected_base, gate),
        posthoc.apply_posthoc("roe_mean", blob, served_in),
    )
    # and the un-gated rows are untouched by the gate round trip
    assert np.isclose(corrected_base[0], posthoc.apply_posthoc("roe_mean", blob, base[:1])[0])
