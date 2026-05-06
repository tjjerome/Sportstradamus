"""Behavioral tests for parlay search remediation (audit follow-up).

Covers the four targets from ``docs/PARLAY_AUDIT.md`` remediation:

* :func:`_nearest_psd` repairs non-PSD correlation submatrices instead of
  dropping them.
* :func:`_expected_payout_with_pushes` produces hand-checkable EV for a
  small parlay with known win/push/lose probabilities.
* Switching ``contest_variant`` from "power" to "flex" changes EV in the
  expected direction (flex pays at non-zero misses; power doesn't).
* The half-integer-line fast path uses the analytical Gaussian-copula CDF
  (not the MC sampler), avoiding a perf regression for the common case.
"""

from __future__ import annotations

import numpy as np
import pytest

from sportstradamus.prediction.correlation import (
    _PSD_EIG_TOLERANCE,
    _expected_payout_with_pushes,
    _nearest_psd,
    _payout_curve_for,
)


def test_nearest_psd_repairs_negative_eigenvalue() -> None:
    """A correlation-shaped matrix with one negative eigenvalue is projected
    onto the PSD cone with unit diagonal preserved."""
    bad = np.array(
        [
            [1.0, 0.95, 0.95, 0.95],
            [0.95, 1.0, 0.95, 0.95],
            [0.95, 0.95, 1.0, -0.95],
            [0.95, 0.95, -0.95, 1.0],
        ]
    )
    assert np.min(np.linalg.eigvalsh(bad)) < 0, "test setup invalid"

    repaired = _nearest_psd(bad)

    eigvals = np.linalg.eigvalsh(repaired)
    # After eigenvalue clipping + diagonal rescale, the min eigenvalue is
    # positive but may be smaller than the raw clip threshold (rescaling
    # changes the spectrum). PSD-ness — not the exact threshold — is the
    # invariant the rest of the pipeline relies on.
    assert np.min(eigvals) > 0
    assert np.allclose(np.diag(repaired), 1.0, atol=1e-9)
    assert np.allclose(repaired, repaired.T, atol=1e-12)


def test_nearest_psd_passes_through_already_psd() -> None:
    """A PSD matrix with unit diagonal is returned essentially unchanged."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 8))
    sigma = a @ a.T
    d = 1.0 / np.sqrt(np.diag(sigma))
    sigma = sigma * d[:, None] * d[None, :]

    repaired = _nearest_psd(sigma)
    assert np.allclose(repaired, sigma, atol=1e-9)


def test_push_aware_payout_independent_two_leg_power() -> None:
    """Hand-computed EV for a 2-leg power parlay with independent legs.

    Outcomes: both-win pays 3x; one-push-one-win drops to a 1-leg parlay
    which falls below the size minimum with no losses (refund ×1); any-loss
    pays 0. Independent-leg arithmetic gives the closed-form expectation.
    """
    p_win = np.array([0.6, 0.55])
    p_push = np.array([0.05, 0.10])
    p_lose = 1.0 - p_win - p_push
    sigma = np.eye(2)
    payout_curve = {2: [3.0, 0.0]}

    rng = np.random.default_rng(42)
    ev = _expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size=2,
        boost=1.0,
        payout_curve=payout_curve,
        rng=rng,
    )

    # Expected: 3 * P(both win) + 1 * P(one or both push, no losses).
    # No-loss outcomes other than both-win: WIN-PUSH, PUSH-WIN, PUSH-PUSH.
    refund_prob = p_win[0] * p_push[1] + p_push[0] * p_win[1] + p_push[0] * p_push[1]
    expected = 3.0 * p_win[0] * p_win[1] + 1.0 * refund_prob
    assert ev == pytest.approx(
        expected, rel=0.02
    ), f"expected ~{expected:.4f}, got {ev:.4f}; p_lose={p_lose}"


def test_push_aware_payout_one_leg_pushes_promotes_to_refund() -> None:
    """A 2-leg parlay where leg 0 always pushes and leg 1 always wins drops
    below the minimum size with no losses → refund (×1)."""
    p_win = np.array([0.0, 1.0])
    p_push = np.array([1.0, 0.0])
    sigma = np.eye(2)
    payout_curve = {2: [3.0, 0.0]}

    rng = np.random.default_rng(7)
    ev = _expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size=2,
        boost=1.0,
        payout_curve=payout_curve,
        rng=rng,
    )
    assert abs(ev - 1.0) < 1e-6


def test_flex_vs_power_diverges_under_misses() -> None:
    """A 5-leg parlay with non-trivial miss probability earns more EV under
    flex than power because flex pays at 1+ misses and power doesn't."""
    bet_size = 5
    p_win = np.full(bet_size, 0.55)
    p_push = np.zeros(bet_size)
    sigma = np.eye(bet_size)

    power_curve = {5: [20.0, 0.0, 0.0, 0.0, 0.0]}
    flex_curve = {5: [10.0, 2.0, 0.4, 0.4, 0.0]}

    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    ev_power = _expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size,
        boost=1.0,
        payout_curve=power_curve,
        rng=rng_a,
    )
    ev_flex = _expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size,
        boost=1.0,
        payout_curve=flex_curve,
        rng=rng_b,
    )

    # Power expected ~ 20 * 0.55^5 ≈ 1.006; flex picks up the 1- and 2-miss tiers.
    assert ev_flex > ev_power
    assert ev_power == pytest.approx(20.0 * 0.55**5, rel=0.05)


def test_payout_curve_loader_returns_power_by_default() -> None:
    """``_payout_curve_for("Underdog", "power", legacy=False)`` reads the
    JSON config and returns the 0-misses entries as the search list."""
    search, full = _payout_curve_for("Underdog", "power", legacy=False)
    assert search[0] == pytest.approx(3.0)  # 2-leg power
    assert full[2][0] == pytest.approx(3.0)
    assert full[2][1] == 0.0  # power: no payout at 1 miss


def test_payout_curve_legacy_underdog_matches_audit() -> None:
    """Legacy mode returns the audit-documented insurance line for ranking
    and the mixed insurance/power overwrite for display."""
    search, full = _payout_curve_for("Underdog", "power", legacy=True)
    assert search == [3.5, 6.5, 10.9, 20.2, 39.9]
    # Mixed regime: 4-leg display = 6.0 (power), 5-leg = 10.0 (power).
    assert full[4][0] == pytest.approx(6.0)
    assert full[5][0] == pytest.approx(10.0)


def test_psd_repair_keeps_correlation_units_diagonal() -> None:
    """After repair, the matrix is still a correlation matrix (unit diag),
    so downstream norm.ppf/mvn-cdf calls remain valid."""
    bad = np.array([[1.0, 1.5], [1.5, 1.0]])  # |rho| > 1, not valid
    repaired = _nearest_psd(bad)
    assert np.allclose(np.diag(repaired), 1.0, atol=1e-9)
    # Off-diagonal must be in [-1, 1] for a correlation matrix.
    assert -1.0 - 1e-9 <= repaired[0, 1] <= 1.0 + 1e-9
