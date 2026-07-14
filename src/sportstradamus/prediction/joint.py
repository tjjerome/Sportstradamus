"""Joint-probability math for correlated slips — the Gaussian-copula / Σ seam.

Σ is assembled in ``correlation.py`` as ``GameArrays``; joint slip pricing is
consumed only through :func:`parlay_payout_prob`. Alternative dependence models
swap in behind a flag mirroring the existing ``legacy`` flag.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm

from sportstradamus.prediction.payouts import expected_payout_with_pushes

# PSD repair: floor on minimum eigenvalue. Sub-floor matrices are projected
# to the nearest PSD instead of dropped (audit §2.1, finding bullet 3).
_PSD_EIG_TOLERANCE: float = 1e-4

# Push-handling thresholds.
# Below this, a leg's push prob is treated as zero so the fast (analytical)
# mvn.cdf path runs. Set just above floating-point noise from ``get_push_prob``.
_PUSH_PROB_FLOOR: float = 1e-6


def _nearest_psd(sigma: np.ndarray, tol: float = _PSD_EIG_TOLERANCE) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD via eigenvalue clipping.

    Symmetrize, clip eigenvalues at ``tol``, then rescale the diagonal back to
    1 so single-leg variances stay unit (the inputs are correlation matrices).

    Args:
        sigma: Symmetric ``(n, n)`` matrix from ``C[bet_id, bet_id]``.
        tol: Eigenvalue floor; matches the PSD acceptance threshold elsewhere.

    Returns:
        np.ndarray: PSD ``(n, n)`` matrix with unit diagonal.
    """
    sigma = (sigma + sigma.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, tol, None)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    diag_scale = 1.0 / np.sqrt(np.diag(repaired))
    return repaired * diag_scale[:, None] * diag_scale[None, :]


def psd_or_none(SIG, legacy):
    """Gate a leg-correlation matrix on the PSD floor, repairing or rejecting it.

    Under ``legacy=True`` a sub-floor matrix is dropped (``None``, matching the
    pre-2026.05 behavior of skipping the parlay outright); otherwise it is
    projected to the nearest PSD via :func:`_nearest_psd` instead of dropped.
    """
    min_eig = np.min(np.linalg.eigvalsh(SIG))
    if legacy:
        return None if min_eig < _PSD_EIG_TOLERANCE else SIG
    if min_eig < _PSD_EIG_TOLERANCE:
        return _nearest_psd(SIG)
    return SIG


def parlay_payout_prob(
    p,
    push_legs,
    SIG,
    bet_size,
    boost,
    payout,
    full_payouts,
    payout_base,
    legacy,
    *,
    full_refund_below_size=None,
):
    """Expected payout for a parlay, routing to the analytical or push-aware path.

    The fast analytical ``mvn.cdf`` path only ever gives P(all hit), so any
    curve with a payout at more than one miss-count (Underdog flex/insurance)
    or any leg with a non-floor push probability must route to the Monte-Carlo
    :func:`expected_payout_with_pushes` path instead — under ``legacy=True``
    the analytical path always runs, matching pre-2026.05 scoring.

    ``boost`` may be a per-leg ``np.ndarray`` (push-repricing, see
    :func:`expected_payout_with_pushes`) as well as a scalar; ``full_refund_below_size``
    passes straight through to that function unchanged.
    """
    has_pushes = bool(np.any(push_legs > _PUSH_PROB_FLOOR))
    # Curves with payouts at multiple miss-counts (e.g. Underdog flex and
    # insurance) need the MC path even with zero pushes — the analytical
    # mvn.cdf only gives P(all hit), discarding the partial-hit tiers.
    curve = full_payouts.get(bet_size, [payout_base, 0.0])
    multi_tier = sum(1 for v in curve if v > 0) > 1
    if (has_pushes or multi_tier) and not legacy:
        return expected_payout_with_pushes(
            p,
            push_legs,
            SIG,
            bet_size,
            boost=boost,
            payout_curve=full_payouts,
            full_refund_below_size=full_refund_below_size,
        )
    return payout * multivariate_normal.cdf(norm.ppf(p), np.zeros(bet_size), SIG)
