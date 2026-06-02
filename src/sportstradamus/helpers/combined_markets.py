"""Honest book references for combined QB markets with no direct sportsbook line.

``qb-yards = passing-yards + rushing-yards`` and ``qb-tds = passing-tds +
rushing-tds`` are offered on DFS sites but not quoted by sportsbooks, so the
archive carries a fabricated ``p_book = 0.5`` placeholder. These helpers build an
honest combined-market over/under probability by convolving the sharp component
books — a Normal sum for the (continuous) yardage market and a discrete PMF
convolution for the (count) TD market. The pass/rush correlation comes from the
per-league correlation matrix; it is negative (game-script substitution), so an
independence assumption overstates the combined variance.
"""
import numpy as np
from scipy.stats import norm

# Minimum allowed variance before taking the square root; prevents divide-by-zero
# when both components have near-zero standard deviation.
_VAR_FLOOR: float = 1e-12


def normal_sum_over_prob(
    line: float, mu1: float, sd1: float, mu2: float, sd2: float, rho: float
) -> float:
    """P(X1 + X2 > line) for jointly-Normal components with correlation ``rho``."""
    mu = mu1 + mu2
    var = sd1**2 + sd2**2 + 2.0 * rho * sd1 * sd2
    sd = np.sqrt(max(var, _VAR_FLOOR))
    return float(norm.sf(line, loc=mu, scale=sd))


def count_sum_over_prob(line: float, pmf1: np.ndarray, pmf2: np.ndarray) -> float:
    """P(N1 + N2 >= ceil(line)) for independent count components given their PMFs.

    The line may be a half-integer (1.5) or an integer (2.0); ``ceil`` maps both
    to the discrete "over" threshold, matching the scorecard's ``Result >= Line``.
    Independence is the documented approximation for the
    TD convolution; game-script dependence is second-order at these low counts.
    """
    total = np.convolve(np.asarray(pmf1, dtype=float), np.asarray(pmf2, dtype=float))
    threshold = int(np.ceil(line))
    return float(total[threshold:].sum())
