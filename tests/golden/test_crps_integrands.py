"""Unit tests for the per-row CRPS integrands in ``helpers.distributions``.

These are the shared scoring kernels behind both the L1 dispersion calibration
(``_dispersion_crps_loss``) and the crps blend-weight objective
(``fit_model_weight_crps``). Each is checked against an independent oracle: the
SkewNormal grid against the Gaussian closed-form CRPS at zero skew (the skew-normal
has no closed-form CRPS), the NegBin sum against a brute-force support sum, the
Gamma closed-form against a fine-grid integral of the same CDF.
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import nbinom, norm

from sportstradamus.helpers.distributions import gamma_crps, negbin_crps, skewnorm_crps


def _gaussian_crps(y, mu, sigma):
    """Closed-form Gaussian CRPS (Gneiting & Raftery 2007)."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_skewnorm_crps_matches_gaussian_closed_form_at_zero_skew():
    rng = np.random.default_rng(0)
    n = 50
    loc = rng.uniform(5, 25, n)
    scale = rng.uniform(1, 5, n)
    alpha = np.zeros(n)
    y = rng.uniform(0, 30, n)
    got = skewnorm_crps(y, loc, scale, alpha)
    want = _gaussian_crps(y, loc, scale)
    # Grid resolution (500 pts, left-Riemann sum·dx matching _dispersion_crps_loss) bounds the
    # absolute accuracy at ~2-3%; the fit only needs the arg-min, which is grid-invariant.
    assert np.allclose(got, want, rtol=3e-2, atol=0.05)


def test_negbin_crps_matches_brute_force_support_sum():
    r = np.array([4.0, 8.0])
    p = np.array([0.4, 0.6])
    y = np.array([3.0, 5.0])
    got = negbin_crps(y, r, p)
    k = np.arange(0, 400)
    want = np.array(
        [np.sum((nbinom.cdf(k, r[i], p[i]) - (y[i] <= k)) ** 2) for i in range(2)]
    )
    assert np.allclose(got, want, rtol=1e-6)


def test_gamma_crps_closed_form_matches_fine_grid():
    alpha = np.array([2.0, 5.0])
    scale = np.array([3.0, 1.5])
    y = np.array([4.0, 8.0])
    got = gamma_crps(y, alpha, scale)
    x = np.linspace(0, 200, 60000)
    dx = x[1] - x[0]
    want = np.array(
        [np.sum((gamma_dist.cdf(x, alpha[i], scale=scale[i]) - (y[i] <= x)) ** 2) * dx for i in range(2)]
    )
    assert np.allclose(got, want, rtol=1e-2, atol=1e-2)


def test_skewnorm_crps_gate_shrinks_zero_outcome_crps():
    # A zero outcome under a zero-inflated predictive: more correctly-placed gate
    # mass at zero lowers its CRPS.
    y = np.array([0.0])
    loc, scale, alpha = np.array([8.0]), np.array([3.0]), np.array([2.0])
    hi = skewnorm_crps(y, loc, scale, alpha, gate=np.array([0.9]))
    lo = skewnorm_crps(y, loc, scale, alpha, gate=np.array([0.1]))
    assert hi[0] < lo[0]
