"""Golden pins for the 2-component Gaussian mixture ("Mixture") serve-path math.

The numpy kernels here are the serve-side counterpart of the lightgbmlss
Mixture(Gaussian, M=2) head (param frame columns loc_1, loc_2, scale_1,
scale_2, mix_prob_1, mix_prob_2); torch stays out of the serve import graph,
same rule as the DPO kernel.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from sportstradamus.helpers.distributions import (
    _mixnorm_cdf,
    _mixnorm_odds,
    _mixnorm_ppf,
    get_odds,
    get_push_prob,
    set_model_start_values,
)

# (w1, loc1, scale1, loc2, scale2)
MIX_ARGS = (0.6, 2.0, 1.0, 7.0, 2.5)

# Last two rows are well-separated components: loc2 = loc1 + 5·scale1.
PPF_GRID = [
    (0.5, 0.0, 1.0, 0.0, 1.0),
    (0.3, 1.0, 0.5, 2.0, 1.5),
    (0.7, 4.0, 2.0, 14.0, 2.0),
    (0.9, -2.0, 0.8, 2.0, 0.8),
]

X_MOMENTS = pd.DataFrame({"MeanYr": [10.0, 4.0], "STDYr": [3.0, 2.0], "ZeroYr": [0.05, 0.6]})


def test_cdf_limits_and_monotone():
    assert _mixnorm_cdf(-1e9, *MIX_ARGS) == pytest.approx(0.0, abs=1e-12)
    assert _mixnorm_cdf(1e9, *MIX_ARGS) == pytest.approx(1.0, abs=1e-12)
    ys = np.linspace(-10.0, 25.0, 301)
    assert np.all(np.diff(_mixnorm_cdf(ys, *MIX_ARGS)) >= 0)


def test_cdf_degenerates_to_plain_normal():
    ys = np.linspace(-3.0, 8.0, 51)
    np.testing.assert_allclose(
        _mixnorm_cdf(ys, 1.0, 2.0, 1.5, 9.0, 0.3), norm.cdf(ys, loc=2.0, scale=1.5), atol=1e-12
    )
    np.testing.assert_allclose(
        _mixnorm_cdf(ys, 0.35, 2.0, 1.5, 2.0, 1.5), norm.cdf(ys, loc=2.0, scale=1.5), atol=1e-12
    )


@pytest.mark.parametrize("w1,loc1,scale1,loc2,scale2", PPF_GRID)
def test_ppf_round_trips_cdf(w1, loc1, scale1, loc2, scale2):
    params = tuple(np.full(3, v) for v in (w1, loc1, scale1, loc2, scale2))
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        y = _mixnorm_ppf(q, *params)
        assert y.shape == (3,)
        np.testing.assert_allclose(_mixnorm_cdf(y, *params), q, atol=1e-4)


def test_ppf_median_of_symmetric_equal_mixture_is_midpoint():
    params = tuple(np.full(2, v) for v in (0.5, 1.0, 0.7, 5.0, 0.7))
    np.testing.assert_allclose(_mixnorm_ppf(0.5, *params), 3.0, atol=1e-4)


def test_get_odds_routes_mixture_to_kernel():
    mix = {
        "w1": np.array([0.6, 0.4]),
        "loc1": np.array([10.0, 8.0]),
        "scale1": np.array([2.0, 1.5]),
        "loc2": np.array([16.0, 12.0]),
        "scale2": np.array([3.0, 2.0]),
    }
    got = get_odds(12.5, np.array([12.4, 9.6]), "Mixture", mix=mix)
    # Default step=1 puts (high, low) at (13.0, 12.0) for a 12.5 line.
    expected = _mixnorm_odds(
        13.0, 12.0, mix["w1"], mix["loc1"], mix["scale1"], mix["loc2"], mix["scale2"]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert np.all((got > 0.0) & (got < 1.0))


def test_get_odds_mixture_even_money_at_center():
    mix = {
        "w1": np.array([0.5]),
        "loc1": np.array([18.5]),
        "scale1": np.array([2.0]),
        "loc2": np.array([22.5]),
        "scale2": np.array([2.0]),
    }
    p_under = get_odds(20.5, np.array([20.5]), "Mixture", mix=mix)
    assert p_under[0] == pytest.approx(0.5, abs=0.02)


def test_push_prob_zero_for_mixture():
    assert get_push_prob(3.0, 3.4, "Mixture") == 0.0


def _mixture_sv(**kwargs):
    class _Model:
        start_values = None

    model = _Model()
    set_model_start_values(model, "Mixture", X_MOMENTS, **kwargs)
    return model.start_values


def test_mixture_start_values_raw_mode_and_no_gate_column():
    sv = _mixture_sv()
    # ZeroYr of 0.6 earns ZINB/ZAGamma a 7th gate column; Mixture must not.
    assert sv.shape == (2, 6)
    mu = X_MOMENTS["MeanYr"].to_numpy()
    std = X_MOMENTS["STDYr"].to_numpy()
    np.testing.assert_allclose(sv[:, 0], mu)
    np.testing.assert_allclose(sv[:, 1], mu + std)
    np.testing.assert_allclose(np.exp(sv[:, 2]), 0.8 * std, rtol=1e-9)
    np.testing.assert_allclose(np.exp(sv[:, 3]), 1.2 * std, rtol=1e-9)


def test_mixture_start_values_logits_softmax_to_dominant_base():
    logits = _mixture_sv()[:, 4:6]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, np.tile([0.75, 0.25], (2, 1)), rtol=1e-9)


def test_mixture_start_values_offset_mode():
    sv = _mixture_sv(offset_mode=True)
    std = X_MOMENTS["STDYr"].to_numpy()
    np.testing.assert_allclose(sv[:, 0], 0.0)
    np.testing.assert_allclose(sv[:, 1], std)
    np.testing.assert_allclose(np.exp(sv[:, 2]), 0.8 * std, rtol=1e-9)
    np.testing.assert_allclose(np.exp(sv[:, 3]), 1.2 * std, rtol=1e-9)


def test_mixture_start_values_normalized_mode():
    sv = _mixture_sv(normalized=True)
    mu = X_MOMENTS["MeanYr"].to_numpy()
    std = X_MOMENTS["STDYr"].to_numpy()
    cv = np.clip(std / mu, 0.01, 10)
    np.testing.assert_allclose(sv[:, 0], 1.0)
    np.testing.assert_allclose(sv[:, 1], 1.0 + cv)
    np.testing.assert_allclose(np.exp(sv[:, 2]), 0.8 * cv, rtol=1e-9)
    np.testing.assert_allclose(np.exp(sv[:, 3]), 1.2 * cv, rtol=1e-9)
