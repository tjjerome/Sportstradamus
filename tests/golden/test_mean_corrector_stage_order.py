"""The mean-stage corrector sits on the FUSED side of the model-book pool.

A logarithmic opinion pool is not mean-preserving, so a corrector fitted before the
pool is multiplied straight back down by ``rho^(1-w)``. Ranjan & Gneiting (2010) Thm 1:
recalibrate the combination, not its components. These tests pin the stage on both the
training and the inference path, and pin the one case where the two stages must agree
exactly — a cell with no authentic validation quote, which fuses at ``w = 1``.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.distributions import fused_loc
from sportstradamus.prediction.model_prob import _blend_with_book
from sportstradamus.training import posthoc

_SLUG = "roe_mean"
_BLOB = {"kind": "affine", "a": 0.15, "b": 1.25}
_EV_MODEL = np.array([0.20, 0.55, 1.10, 2.40, 4.75])
_EV_BOOK = np.array([0.05, 0.30, 1.60, 1.90, 6.10])
_CV = 0.8


def _fused_mean(dist, weight, ev_model):
    """The blended mean each family's ``fused_loc`` branch reports."""
    if dist == "SkewNormal":
        mean, _, _, _ = fused_loc(
            weight, ev_model, _EV_BOOK, _CV, dist, sigma=ev_model * 0.6, skew_alpha=np.full(5, 0.4)
        )
        return mean
    if dist == "NegBin":
        r, p, _ = fused_loc(weight, ev_model, _EV_BOOK, _CV, dist, r=np.full(5, 3.0))
        return r * (1 - p) / p
    if dist == "DPO":
        mean, _, _ = fused_loc(weight, ev_model, _EV_BOOK, _CV, dist, phi=np.full(5, 1.4))
        return mean
    alpha, beta, _ = fused_loc(weight, ev_model, _EV_BOOK, _CV, "Gamma", alpha=np.full(5, 2.0))
    return alpha / beta


@pytest.mark.parametrize("dist", ["SkewNormal", "NegBin", "DPO", "Gamma"])
def test_corrector_stage_is_a_noop_at_weight_one(dist):
    # No authentic validation quote => _fit_nonsn_weight returns w = 1.0 and the fused
    # mean IS the model mean, so pre-fusion and post-fusion correction must agree bit for
    # bit. If this fails the w = 1 path has a bug and no other result in the lane holds.
    pre = _fused_mean(dist, 1.0, posthoc.apply_posthoc(_SLUG, _BLOB, _EV_MODEL))
    post = posthoc.correct_fused_mean(_SLUG, _BLOB, _fused_mean(dist, 1.0, _EV_MODEL), None)
    assert np.allclose(pre, post, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("dist", ["SkewNormal", "NegBin", "DPO", "Gamma"])
def test_correction_survives_the_pool_below_weight_one(dist):
    # The defect this change fixes: at w < 1 a pre-fusion correction is discounted by the
    # pool, so it under-delivers exactly where the two legs disagree.
    pre = _fused_mean(dist, 0.9, posthoc.apply_posthoc(_SLUG, _BLOB, _EV_MODEL))
    post = posthoc.correct_fused_mean(_SLUG, _BLOB, _fused_mean(dist, 0.9, _EV_MODEL), None)
    want = posthoc.apply_posthoc(_SLUG, _BLOB, _fused_mean(dist, 0.9, _EV_MODEL))
    assert np.allclose(post, want)
    assert not np.allclose(pre, post)


def _offer_frame(gate=None):
    df = pd.DataFrame(
        {
            "Projection": _EV_MODEL.copy(),
            "Market Projection": _EV_BOOK.copy(),
            "Line": np.array([0.5, 0.5, 1.5, 2.5, 4.5]),
            "Model R": np.full(5, 3.0),
        }
    )
    if gate is not None:
        df["Model Gate"] = gate
    return df


@pytest.mark.parametrize("dist,gate", [("NegBin", None), ("ZINB", np.linspace(0.05, 0.6, 5))])
def test_inference_corrects_the_pool_output_not_the_model_leg(dist, gate):
    plain = _offer_frame(gate)
    uncorrected = _blend_with_book(plain, dist, 0.85, _CV, 0.3, "NFL", "tds")
    # _blend_with_book overwrites Model Gate with the BLENDED gate; that is the one the
    # served mean is deflated by, so it is the one the corrector must round-trip through.
    gate_blend = plain["Model Gate"].to_numpy() if gate is not None else None

    frame = _offer_frame(gate)
    corrected = _blend_with_book(frame, dist, 0.85, _CV, 0.3, "NFL", "tds", _SLUG, _BLOB)
    assert np.allclose(corrected, posthoc.correct_fused_mean(_SLUG, _BLOB, uncorrected, gate_blend))

    # Projection is the served mean, and the gate lands on it exactly once.
    served_in = posthoc.served_mean(uncorrected, gate_blend)
    assert np.allclose(frame["Projection"], posthoc.apply_posthoc(_SLUG, _BLOB, served_in))
    # The pool is untouched: correcting after it never moves the blended shape.
    assert np.allclose(plain["Model R"], frame["Model R"])
