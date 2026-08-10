"""Unit tests for the count-family screen (WS-3 Phase 0 entry bar).

Synthetic only — no network, no real training parquets. Pins the routing tree
against a known NB null, the Poisson-binomial tail math (exact vs binomial and
vs the normal approximation), and the randomized-quantile-residual variance on
well-calibrated vs under-dispersed samples.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sportstradamus.scripts.count_family_screen import (
    _RQR_VAR_BAR,
    poisson_binomial_pvalue,
    route_zero_mod,
    rqr_z,
)

pytestmark = pytest.mark.diagnostics


def _route_via_null(zero_probs: np.ndarray, k_obs: int) -> str:
    null_mean, pvalue = poisson_binomial_pvalue(zero_probs, k_obs)
    return route_zero_mod(k_obs, null_mean, pvalue)


def test_routing_clean_is_plain_nb():
    zero_probs = np.full(500, 0.20)
    assert _route_via_null(zero_probs, k_obs=100) == "plain-NB"


def test_routing_excess_zeros_is_hurdle():
    zero_probs = np.full(500, 0.20)
    assert _route_via_null(zero_probs, k_obs=160) == "hurdle-NB"


def test_routing_zero_deficit_is_dp_mandatory():
    zero_probs = np.full(500, 0.20)
    assert _route_via_null(zero_probs, k_obs=45) == "DP-mandatory"


def test_poisson_binomial_exact_matches_binomial_on_homogeneous_p():
    n, p, k = 20, 0.30, 6
    null_mean, pvalue = poisson_binomial_pvalue(np.full(n, p), k, exact=True)
    assert null_mean == pytest.approx(n * p)
    p_le = float(stats.binom.cdf(k, n, p))
    p_ge = float(stats.binom.sf(k - 1, n, p))
    assert pvalue == pytest.approx(min(1.0, 2.0 * min(p_le, p_ge)))


def test_poisson_binomial_exact_and_normal_approx_agree():
    rng = np.random.default_rng(0)
    zero_probs = rng.uniform(0.1, 0.5, size=1200)
    k = round(float(zero_probs.sum())) + 15
    _, p_exact = poisson_binomial_pvalue(zero_probs, k, exact=True)
    _, p_approx = poisson_binomial_pvalue(zero_probs, k, exact=False)
    assert p_exact == pytest.approx(p_approx, abs=0.03)


def test_rqr_variance_near_one_on_well_calibrated_nb():
    rng = np.random.default_rng(7)
    r, p = 5.0, 0.5
    y = stats.nbinom.rvs(r, 1.0 - p, size=6000, random_state=rng).astype(float)
    r_arr, p_arr = np.full(y.size, r), np.full(y.size, p)
    var = rqr_z(y, r_arr, p_arr, rng).var(ddof=1)
    assert var == pytest.approx(1.0, abs=0.1)
    assert var > _RQR_VAR_BAR


def test_rqr_variance_below_bar_on_underdispersed_data():
    rng = np.random.default_rng(11)
    mean = 5.0
    y = stats.poisson.rvs(mean, size=6000, random_state=rng).astype(float)
    # Evaluate against a deliberately over-dispersed NB (small r -> variance 30 vs
    # the data's 5): the fit over-disperses, so the residuals must compress.
    r = 1.0
    p = mean / (mean + r)
    r_arr, p_arr = np.full(y.size, r), np.full(y.size, p)
    assert rqr_z(y, r_arr, p_arr, rng).var(ddof=1) < _RQR_VAR_BAR
