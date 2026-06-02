import numpy as np

from sportstradamus.helpers.combined_markets import (
    count_sum_over_prob,
    normal_sum_over_prob,
)


def test_normal_sum_mean_and_monotonicity():
    # Two component books N(220, 40^2) and N(25, 20^2), independent (rho=0).
    # Combined mean 245, sd sqrt(40^2+20^2). P(over line) decreases as the line rises.
    p_low = normal_sum_over_prob(line=200.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    p_mid = normal_sum_over_prob(line=245.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    p_high = normal_sum_over_prob(line=290.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    assert p_low > p_mid > p_high
    assert abs(p_mid - 0.5) < 1e-6  # line at the combined mean => 0.5


def test_negative_rho_tightens_variance():
    # Game-script substitution: negative rho shrinks combined variance, so an
    # over line above the mean is LESS likely (mass pulled toward the mean).
    p_indep = normal_sum_over_prob(line=300.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=0.0)
    p_negcorr = normal_sum_over_prob(line=300.0, mu1=220, sd1=40, mu2=25, sd2=20, rho=-0.4)
    assert p_negcorr < p_indep


def test_var_floor_handles_degenerate_negative_correlation():
    # rho=-1 with equal SDs drives combined variance to ~0; the _VAR_FLOOR guard
    # must keep norm.sf finite (no zero-scale divide).
    p = normal_sum_over_prob(line=0.0, mu1=10.0, sd1=10.0, mu2=5.0, sd2=10.0, rho=-1.0)
    assert 0.0 <= p <= 1.0


def test_count_sum_matches_independent_convolution():
    # Sum of two small Poisson-like count books; P(over 1.5) = P(total >= 2).
    p = count_sum_over_prob(line=1.5, pmf1=np.array([0.5, 0.3, 0.2]), pmf2=np.array([0.6, 0.4]))
    # total pmf via direct convolution
    conv = np.convolve([0.5, 0.3, 0.2], [0.6, 0.4])
    expected = conv[2:].sum()  # P(total >= 2)
    assert abs(p - expected) < 1e-9
