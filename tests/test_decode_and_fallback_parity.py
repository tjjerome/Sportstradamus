"""Parity tests for the prediction↔training scoring consolidation.

Two invariants are pinned here so a future edit cannot silently drift the two
pipelines apart:

1. **Book-odds fallback ≡ blend with model_weight 0.** The missing-model
   fallback scores a leg from the book params directly (via ``_book_over_prob``).
   Blending model and book through :func:`fused_loc` with ``w=0`` must collapse
   to those same book params for every distribution family, so the fallback is
   exactly the weight-0 blend (the user-stated equivalence).

2. **Decode parity.** :func:`decode_predictive_mean` is the single decode kernel
   shared by ``prediction.model_prob`` and ``training.pipeline``; it must
   reproduce the closed-form means/shape the two pipelines used inline before the
   consolidation.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import (
    GATE_PUBLISH_THRESHOLD,
    decode_predictive_mean,
    fused_loc,
    get_odds,
)

mp = importlib.import_module("sportstradamus.prediction.model_prob")


def _book_over(dist, line, book_ev, cv, step, gate):
    """Over-probability the fallback assigns — the real ``_book_over_prob``."""
    offer_df = pd.DataFrame({"Line": [line], "Market Projection": [book_ev]})
    # Synthetic, never-fitted cell: the SkewNormal book shape stays the symmetric
    # constant-CV (mean*cv) read the w=0 reference uses, so the parity holds at the
    # passed cv regardless of any production cell that gains a fitted book_shape.
    return mp._book_over_prob(offer_df, dist, cv, step, gate, "PARITY", "UNFITTED").to_numpy()


def _blend0_over(dist, line, model_ev, book_ev, cv, step, *, shape, gate=None):
    """Over-probability of the model path (model_prob) at ``model_weight == 0``.

    Mirrors model_prob's blend → get_odds for each family. ``shape`` carries the
    model's per-row dispersion (``r`` / ``alpha`` / ``(sigma, skew)``) — all of
    which w=0 must discard in favor of the book's implied shape.
    """
    a = np.array([float(model_ev)])
    b = np.array([float(book_ev)])
    # Count/Gamma zero inflation belongs to both endpoints' historical book
    # contract.  SkewNormal is different: its archived book EV is ungated, while
    # a positive-only model head may still carry a model-side zero atom.  At
    # w=0 that model gate must therefore blend to the ungated book endpoint.
    zi = {"gate_book": gate} if gate and dist != "SkewNormal" else {}
    if dist in ("NegBin", "ZINB"):
        r_b, p_b, g_b = fused_loc(0.0, a, b, cv, "NegBin", r=np.array([shape]), **zi)
        mean = r_b * (1 - p_b) / p_b
        under = get_odds(np.array([line]), mean, dist, cv, step=step, r=r_b, gate=g_b)
    elif dist in ("Gamma", "ZAGamma"):
        a_b, beta_b, g_b = fused_loc(0.0, a, b, cv, "Gamma", alpha=np.array([shape]), **zi)
        mean = a_b / beta_b
        under = get_odds(np.array([line]), mean, dist, cv, step=step, alpha=a_b, gate=g_b)
    else:
        sigma, skew = shape
        if gate:
            zi = {"gate_model": np.array([gate]), "gate_book": 0.0}
        ev_b, sig_b, sk_b, g_b = fused_loc(
            0.0, a, b, cv, "SkewNormal", sigma=np.array([sigma]), skew_alpha=np.array([skew]), **zi
        )
        under = get_odds(
            np.array([line]),
            ev_b,
            "SkewNormal",
            cv,
            step=step,
            sigma=sig_b,
            skew_alpha=sk_b,
            gate=g_b,
        )
    return 1 - under


@pytest.mark.parametrize(
    "dist, line, book_ev, cv, step, shape, gate",
    [
        ("NegBin", 9.5, 10.0, 0.5, 1.0, 4.0, None),
        ("ZINB", 7.5, 9.0, 0.6, 1.0, 3.0, 0.25),
        ("Gamma", 24.5, 22.0, 0.4, 0.5, 6.0, None),
        ("ZAGamma", 18.5, 20.0, 0.5, 0.5, 5.0, 0.20),
        ("SkewNormal", 29.5, 28.0, 0.3, 0.5, (9.0, 0.5), None),
        ("SkewNormal", 31.5, 30.0, 0.3, 0.5, (8.0, -0.4), 0.10),
    ],
)
def test_book_fallback_equals_weight_zero_blend(dist, line, book_ev, cv, step, shape, gate):
    book = _book_over(dist, line, book_ev, cv, step, gate)
    # model_ev deliberately differs from book_ev; w=0 must ignore it entirely.
    blend = _blend0_over(
        dist,
        line,
        model_ev=book_ev * 1.5,
        book_ev=book_ev,
        cv=cv,
        step=step,
        shape=shape,
        gate=gate,
    )
    np.testing.assert_allclose(blend, book, rtol=0, atol=1e-9)


def test_decode_negbin_matches_inline():
    pp = pd.DataFrame({"total_count": [5.0, 10.0, 2.5], "probs": [0.4, 0.6, 0.3]})
    d = decode_predictive_mean(pp, "NegBin")
    np.testing.assert_allclose(d.ev, pp["total_count"] * pp["probs"] / (1 - pp["probs"]))
    np.testing.assert_allclose(d.r, pp["total_count"])
    assert d.gate is None and d.alpha is None and d.sigma is None


def test_decode_zinb_carries_gate():
    pp = pd.DataFrame({"total_count": [5.0], "probs": [0.4], "gate": [0.3]})
    d = decode_predictive_mean(pp, "ZINB")
    np.testing.assert_allclose(d.gate, [0.3])


def test_decode_gamma_matches_inline():
    pp = pd.DataFrame({"concentration": [2.0, 6.0], "rate": [0.5, 1.5]})
    d = decode_predictive_mean(pp, "Gamma")
    np.testing.assert_allclose(d.ev, pp["concentration"] / pp["rate"])
    np.testing.assert_allclose(d.alpha, pp["concentration"])
    assert d.gate is None


def test_decode_zagamma_carries_gate():
    pp = pd.DataFrame({"concentration": [2.0], "rate": [0.5], "gate": [0.2]})
    d = decode_predictive_mean(pp, "ZAGamma")
    np.testing.assert_allclose(d.gate, [0.2])


def test_decode_skewnormal_matches_inline():
    loc = np.array([10.0, 20.0])
    scale = np.array([3.0, 4.0])
    alpha = np.array([0.5, -0.5])
    pp = pd.DataFrame({"alpha": alpha})
    d = decode_predictive_mean(pp, "SkewNormal", sn_loc=loc, sn_scale=scale, hist_gate=0.10)
    delta = alpha / np.sqrt(1 + alpha**2)
    np.testing.assert_allclose(d.ev, loc + scale * delta * np.sqrt(2 / np.pi))
    np.testing.assert_allclose(d.sigma, scale)
    np.testing.assert_allclose(d.skew, alpha)
    # hist_gate above the publish threshold -> per-row gate broadcast.
    np.testing.assert_allclose(d.gate, [0.10, 0.10])


def test_decode_skewnormal_no_gate_below_threshold():
    pp = pd.DataFrame({"alpha": [0.0]})
    d = decode_predictive_mean(
        pp,
        "SkewNormal",
        sn_loc=np.array([10.0]),
        sn_scale=np.array([3.0]),
        hist_gate=GATE_PUBLISH_THRESHOLD / 2,
    )
    assert d.gate is None
