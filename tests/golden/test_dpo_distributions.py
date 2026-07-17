"""Golden pins for the Double Poisson serve-path math in helpers.distributions.

The numpy kernel here and the torch kernel in ``sportstradamus.double_poisson``
are deliberately separate implementations (the serve import graph stays
torch-free), so the first pin asserts they agree — any convention drift between
training and serving fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.distributions import (
    DecodedParams,
    _dp_cdf_pmf,
    _dp_log_pmf_grid,
    _dp_mean,
    _dp_mu_from_mean,
    _dp_ppf,
    decode_predictive_mean,
    dp_crps,
    fused_loc,
    get_ev,
    get_odds,
    get_push_prob,
    set_model_start_values,
)

PARAM_GRID = [(0.5, 0.5), (0.5, 2.0), (1.5, 1.0), (2.0, 2.0), (6.0, 0.5)]


@pytest.mark.parametrize("mu,phi", PARAM_GRID)
def test_numpy_matches_torch_kernel(mu, phi):
    torch = pytest.importorskip("torch")
    from sportstradamus.double_poisson import DoublePoissonTorch

    grid, log_pmf = _dp_log_pmf_grid(mu, phi)
    torch_dist = DoublePoissonTorch(torch.tensor([mu]), torch.tensor([phi]))
    torch_lp = torch_dist.log_prob(torch.tensor(grid, dtype=torch.float32).unsqueeze(-1))
    np.testing.assert_allclose(log_pmf[:, 0], torch_lp.squeeze(-1).numpy(), atol=1e-4)


def test_exact_mean_and_newton_inversion_low_mu():
    for mean in (0.3, 0.5, 1.1):
        for phi in (0.5, 1.0, 2.5):
            mu = _dp_mu_from_mean(np.array([mean]), np.array([phi]))
            assert _dp_mean(mu, np.array([phi]))[0] == pytest.approx(mean, rel=1e-5)


def test_get_odds_get_ev_round_trip():
    # Pairs stay inside get_ev's SN_MAX_MEAN_FACTOR·line implied-mean cap.
    for line, true_ev in [(0.5, 0.4), (0.5, 1.2), (1.5, 0.4), (1.5, 2.6), (2.5, 1.2), (2.5, 2.6)]:
        under = get_odds(line, true_ev, "DPO", cv=0.8)
        assert 0.0 < under < 1.0
        assert get_ev(line, under, cv=0.8, dist="DPO") == pytest.approx(true_ev, rel=1e-3)


def test_get_ev_clamps_beyond_mean_cap():
    under = get_odds(0.5, 2.6, "DPO", cv=0.8)  # implied mean past 5·line
    assert get_ev(0.5, under, cv=0.8, dist="DPO") == pytest.approx(2.5)


def test_get_odds_monotone_in_mean():
    evs = np.array([0.5, 1.0, 2.0, 4.0])
    unders = get_odds(1.5, evs, "DPO", cv=0.6, phi=np.full(4, 1.2))
    assert np.all(np.diff(unders) < 0)


def test_push_prob_integer_and_half_lines():
    push_int = get_push_prob(2.0, 1.8, "DPO", cv=0.5)
    push_half = get_push_prob(2.5, 1.8, "DPO", cv=0.5)
    assert 0.0 < push_int < 1.0
    assert push_half == 0.0
    _, log_pmf = _dp_log_pmf_grid(
        _dp_mu_from_mean(np.array([1.8]), np.array([1.0 / (1.0 + 0.5 * 1.8)])),
        np.array([1.0 / (1.0 + 0.5 * 1.8)]),
    )
    assert push_int == pytest.approx(np.exp(log_pmf[2, 0]), rel=1e-9)


def test_cdf_pmf_edges():
    cdf_neg, pmf_neg = _dp_cdf_pmf(-1.0, np.array([1.0]), np.array([1.0]))
    assert cdf_neg[0] == 0.0 and pmf_neg[0] == 0.0
    cdf_tail, _ = _dp_cdf_pmf(500.0, np.array([1.0]), np.array([1.0]))
    assert cdf_tail[0] == pytest.approx(1.0, abs=1e-9)


def test_ppf_inverts_cdf():
    mu = np.full(3, 1.7)
    phi = np.full(3, 1.3)
    for q in (0.05, 0.5, 0.95):
        k = _dp_ppf(q, mu, phi)
        cdf_at_k, _ = _dp_cdf_pmf(k, mu, phi)
        cdf_below, _ = _dp_cdf_pmf(k - 1, mu, phi)
        assert np.all(cdf_at_k >= q)
        assert np.all(cdf_below < q)


def test_fused_loc_dpo_endpoints():
    ev_model = np.array([2.0, 0.8])
    phi_model = np.array([2.0, 1.5])
    ev_book = np.array([1.6, 1.0])
    cv = 0.7
    mean_w1, phi_w1, gate = fused_loc(1.0, ev_model, ev_book, cv, "DPO", phi=phi_model)
    np.testing.assert_allclose(mean_w1, ev_model, rtol=1e-9)
    np.testing.assert_allclose(phi_w1, phi_model, rtol=1e-9)
    assert gate is None
    mean_w0, phi_w0, _ = fused_loc(0.0, ev_model, ev_book, cv, "DPO", phi=phi_model)
    np.testing.assert_allclose(mean_w0, ev_book, rtol=1e-9)
    np.testing.assert_allclose(phi_w0, 1.0 / (1.0 + cv * ev_book), rtol=1e-9)


def test_decode_predictive_mean_dpo():
    params = pd.DataFrame({"mu": [0.5, 2.0], "phi": [2.0, 0.5]})
    decoded = decode_predictive_mean(params, "DPO")
    assert isinstance(decoded, DecodedParams)
    np.testing.assert_allclose(decoded.ev, _dp_mean(params["mu"], params["phi"]), rtol=1e-9)
    np.testing.assert_allclose(decoded.phi, params["phi"], rtol=1e-9)
    assert decoded.gate is None and decoded.r is None


def test_dp_crps_prefers_calibrated_dispersion():
    rng = np.random.default_rng(11)
    y = rng.poisson(2.0, size=400).astype(float)
    mu = np.full(400, 2.0)
    crps_right = dp_crps(y, mu, np.full(400, 1.0)).mean()
    crps_too_wide = dp_crps(y, mu, np.full(400, 0.1)).mean()
    crps_too_narrow = dp_crps(y, mu, np.full(400, 20.0)).mean()
    assert crps_right < crps_too_wide
    assert crps_right < crps_too_narrow


def test_dpo_start_values_shape_and_space():
    class _Model:
        start_values = None

    x = pd.DataFrame({"MeanYr": [1.5, 0.6], "STDYr": [1.0, 1.2], "ZeroYr": [0.1, 0.4]})
    model = _Model()
    set_model_start_values(model, "DPO", x)
    assert model.start_values.shape == (2, 2)
    softplus = np.log1p(np.exp(model.start_values))
    np.testing.assert_allclose(softplus[:, 0], x["MeanYr"], rtol=1e-3)
    assert np.all(softplus[:, 1] >= 0.25) and np.all(softplus[:, 1] <= 4.0)
