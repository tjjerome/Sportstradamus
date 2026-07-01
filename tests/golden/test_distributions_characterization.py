"""Characterization pins for the three over-CC functions in ``helpers/distributions.py``:
``get_ev`` (CC 11), ``get_odds`` (CC 16), ``set_model_start_values`` (CC 11).

All three are pure numeric functions (book-line inversion, CDF evaluation, LightGBMLSS
seed-value derivation). These pins fix the exact output across every distribution
family + zero-inflation gate so the CC decomposition into per-family helpers is
provably behavior-preserving. Existing tests only exercise one round-trip case each;
this covers Gamma / ZAGamma / NegBin / ZINB / Poisson / SkewNormal and the
SkewNormal normalized / offset_mode seeding branches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers.distributions import (
    SN_MAX_MEAN_FACTOR,
    get_ev,
    get_odds,
    set_model_start_values,
    skewnormal_loc_from_mean,
    skewnormal_params_from_moments,
)


@pytest.mark.parametrize(
    "mean,var,skew,sigma_ref,alpha_ref",
    [
        (4.262, 2.179, -0.631, 2.235, -2.784),  # WS2 brief: WNBA DREB line 6.0 (solved exactly)
        (4.110, 2.345, -0.557, 2.266, -2.413),  # WS2 brief: NBA DREB line 5.5
    ],
)
def test_skewnormal_params_from_moments_matches_brief(mean, var, skew, sigma_ref, alpha_ref):
    """The book-shape moment-match reproduces the research brief's hand-solved DREB bins, and
    round-trips: a mean-pinned SkewNormal with the solved (sigma, alpha) has the target moments."""
    sigma, alpha = skewnormal_params_from_moments(var, skew)
    assert float(sigma) == pytest.approx(sigma_ref, abs=2e-3)
    assert float(alpha) == pytest.approx(alpha_ref, abs=2e-3)
    loc = float(skewnormal_loc_from_mean(mean, sigma, alpha))
    m, v, s = skewnorm.stats(float(alpha), loc=loc, scale=float(sigma), moments="mvs")
    assert (float(m), float(v), float(s)) == pytest.approx((mean, var, skew), abs=2e-3)


def test_skewnormal_params_from_moments_edges():
    """skew=0 -> symmetric Normal (alpha=0, sigma=sqrt(var)); a skew past the SkewNormal bound
    clamps alpha finite while preserving the target variance (the low-mean count regime)."""
    sigma0, alpha0 = skewnormal_params_from_moments(4.0, 0.0)
    assert float(alpha0) == pytest.approx(0.0, abs=1e-9)
    assert float(sigma0) == pytest.approx(2.0, abs=1e-9)
    # skew 1.26 > 0.9953 bound: alpha finite, realized skew clamps, variance still met
    sigma1, alpha1 = skewnormal_params_from_moments(1.354, 1.26)
    assert np.isfinite(float(alpha1))
    loc = float(skewnormal_loc_from_mean(1.04, sigma1, alpha1))
    _, v, s = skewnorm.stats(float(alpha1), loc=loc, scale=float(sigma1), moments="mvs")
    assert float(v) == pytest.approx(1.354, abs=2e-3)
    assert float(s) < 0.9953


def test_get_ev_by_family():
    assert get_ev(10.5, 0.55, cv=0.8, dist="Gamma") == pytest.approx(11.8187948708, abs=1e-6)
    assert get_ev(10.5, 0.55, cv=0.8, dist="ZAGamma", gate=0.2) == pytest.approx(
        15.1645576946, abs=1e-6
    )
    # under-prob below the ZI mass -> unrepresentable -> clamps to the plausibility cap
    assert get_ev(5.5, 0.10, cv=0.8, dist="ZAGamma", gate=0.2) == pytest.approx(
        SN_MAX_MEAN_FACTOR * 5.5, abs=1e-6
    )
    assert get_ev(7.5, 0.6, cv=0.5, dist="NegBin") == pytest.approx(7.4215818099, abs=1e-6)
    assert get_ev(7.5, 0.6, cv=0.5, dist="ZINB", gate=0.15) == pytest.approx(8.5907941045, abs=1e-6)
    assert get_ev(4.5, 0.45, cv=1, dist="Poisson") == pytest.approx(4.9461078629, abs=1e-6)
    assert get_ev(20.5, 0.5, cv=0.4, dist="SkewNormal") == pytest.approx(20.5, abs=1e-6)
    assert get_ev(20.5, 0.5, cv=0.4, dist="SkewNormal", skew_alpha=2.0) == pytest.approx(
        20.9849997782, abs=1e-6
    )


def test_get_odds_by_family():
    assert get_odds(4.5, 5.0, "NegBin", cv=1) == pytest.approx(0.4404932851, abs=1e-9)
    assert get_odds(4.5, 5.0, "Poisson") == pytest.approx(0.4404932851, abs=1e-9)
    assert get_odds(7.5, 6.0, "NegBin", cv=0.5) == pytest.approx(0.6996612549, abs=1e-9)
    assert get_odds(7.5, 6.0, "ZINB", cv=0.5, gate=0.15) == pytest.approx(0.7447120667, abs=1e-9)
    assert get_odds(20.5, 18.0, "SkewNormal", cv=0.4) == pytest.approx(0.6354737025, abs=1e-9)
    assert get_odds(20.5, 18.0, "SkewNormal", cv=0.4, sigma=7.0, skew_alpha=2.0) == pytest.approx(
        0.7157037365, abs=1e-9
    )
    assert get_odds(10.5, 11.0, "Gamma", cv=0.8) == pytest.approx(0.5837214195, abs=1e-9)
    assert get_odds(10.5, 11.0, "ZAGamma", cv=0.8, gate=0.2) == pytest.approx(
        0.6669771356, abs=1e-9
    )
    assert get_odds(10.0, 11.0, "Gamma", cv=0.8, step=0.5) == pytest.approx(0.5607245607, abs=1e-9)


class _Model:
    pass


_X = pd.DataFrame(
    {
        "MeanYr": [5.0, 12.0, 0.5, 25.0],
        "STDYr": [3.0, 4.0, 1.0, 6.0],
        "ZeroYr": [0.1, 0.0, 0.6, 0.05],
    }
)

_SV_EXPECTED = {
    ("SkewNormal", "default"): [
        [5.0, 1.09861229, 0.0],
        [12.0, 1.38629436, 0.0],
        [0.5, 0.0, 0.0],
        [25.0, 1.79175947, 0.0],
    ],
    ("NegBin", "default"): [
        [6.25, -0.22314355],
        [36.0, -1.09861229],
        [0.5, 0.0],
        [50.0, -0.69314718],
    ],
    ("ZINB", "default"): [
        [8.73869742, -0.45993063, -2.26919002],
        [36.0, -1.09861229, -4.59511985],
        [0.5, 0.0, -4.59511985],
        [50.0, -0.64185389, -2.94443898],
    ],
    ("Gamma", "default"): [
        [2.71358424, -0.29718172],
        [8.99987658, 0.11064653],
        [-1.25869155, -0.43275213],
        [17.36111108, 0.00259285],
    ],
    ("ZAGamma", "default"): [
        [3.39641069, -0.15795764, -2.19722458],
        [8.99987658, 0.11064653, -4.59511985],
        [1.32726946, 0.91242043, 0.40546511],
        [19.23668821, 0.07431363, -2.94443898],
    ],
}


@pytest.mark.parametrize("dist", ["SkewNormal", "NegBin", "ZINB", "Gamma", "ZAGamma"])
def test_set_model_start_values_by_family(dist):
    model = _Model()
    set_model_start_values(model, dist, _X)
    np.testing.assert_allclose(model.start_values, _SV_EXPECTED[(dist, "default")], atol=1e-6)


def test_set_model_start_values_skewnormal_branches():
    m = _Model()
    set_model_start_values(m, "SkewNormal", _X, normalized=True)
    np.testing.assert_allclose(
        m.start_values,
        [
            [1.0, -0.51082562, 0.0],
            [1.0, -1.09861229, 0.0],
            [1.0, 0.69314718, 0.0],
            [1.0, -1.42711636, 0.0],
        ],
        atol=1e-6,
    )
    m = _Model()
    set_model_start_values(m, "SkewNormal", _X, offset_mode=True)
    np.testing.assert_allclose(
        m.start_values,
        [[0.0, 1.09861229, 0.0], [0.0, 1.38629436, 0.0], [0.0, 0.0, 0.0], [0.0, 1.79175947, 0.0]],
        atol=1e-6,
    )
    # offset_mode loc/alpha must be per-row (n,) arrays, not degenerate scalars
    assert m.start_values.shape == (4, 3)


def test_set_model_start_values_shape_ceiling():
    m = _Model()
    set_model_start_values(m, "NegBin", _X, shape_ceiling=20)
    np.testing.assert_allclose(
        m.start_values,
        [[6.25, -0.22314355], [20.0, -0.51082562], [0.5, 0.0], [20.0, 0.22314355]],
        atol=1e-6,
    )
