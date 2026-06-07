"""Pure-core tests for the Operation Ship 75 combination search.

Covers the two functions that carry the search's logic without touching the training
pipeline: ``recover_fused_predictive`` (invert post-hoc calibration back to the trained
predictive) and ``evaluate_trial`` (rank a trained model's calibration corner by min-gate
slack). The Optuna driver + ``meditate`` orchestration are validated by a live cell run,
not mocks.
"""

import numpy as np
import pytest
from scipy import stats as st

from sportstradamus.training.combo_search import (
    evaluate_trial,
    recover_fused_predictive,
)

# A gate row whose non-g4 gates all pass with room, so g4 is free to bind. g4_pit_ks is
# absent on purpose — evaluate_trial substitutes each calibration mode's own g4.
_BASE_ROW_PASSING = {
    "g1_brier_diff_ci_hi": -0.01,  # model beats book
    "g2_star_z": 0.1,
    "g3_bench_z": 0.1,
    "g5_ece_debiased": 0.03,
    "g4_pit_ks_max": 0.05,
}


def test_recover_fused_predictive_inverts_scale_and_skew():
    """The fused predictive is the served one with the scale divided by ``dispersion_cal`` and the
    skew shifted back by ``skew_cal``; the mean is calibration-invariant and passes through.
    """
    mean = np.array([10.0, 12.0])
    served_sigma = np.array([4.0, 6.0])  # = fused * dispersion_cal
    served_alpha = np.array([2.5, 1.5])  # = fused + skew_cal

    f_mean, f_sigma, f_skew = recover_fused_predictive(
        mean, served_sigma, served_alpha, dispersion_cal=2.0, skew_cal=1.5
    )

    assert np.allclose(f_mean, mean)
    assert np.allclose(f_sigma, [2.0, 3.0])
    assert np.allclose(f_skew, [1.0, 0.0])


def test_recover_fused_predictive_identity_when_uncalibrated():
    """``dispersion_cal=1`` / ``skew_cal=0`` (a cell that never calibrated) recovers the served
    predictive unchanged — the search can read an uncalibrated dump directly.
    """
    mean, sigma, alpha = np.array([5.0]), np.array([3.0]), np.array([0.4])
    f_mean, f_sigma, f_skew = recover_fused_predictive(mean, sigma, alpha, 1.0, 0.0)
    assert np.allclose(f_sigma, sigma) and np.allclose(f_skew, alpha) and np.allclose(f_mean, mean)


def test_evaluate_trial_picks_skew_joint_on_underskew():
    """On an ``alpha≈0`` collapse over right-skewed truth the trial's best calibration corner is
    ``skew-joint`` and the substituted Gate-4 clears, so the min-gate slack is positive.
    """
    rng = np.random.default_rng(0)
    y = st.skewnorm.rvs(4.0, size=4000, random_state=rng)
    mean, sigma, skew0 = np.full(4000, float(y.mean())), np.full(4000, float(y.std())), np.zeros(4000)

    result = evaluate_trial(_BASE_ROW_PASSING, mean, sigma, skew0, y)

    assert result.mode == "skew-joint"
    assert result.slack > 0  # joint clears Gate 4 → cell ships


def test_evaluate_trial_picks_dispersion_on_underdispersed_symmetric():
    """A symmetric under-dispersed cell needs width, not shape: the winning corner is ``dispersion``
    with ``c > 1`` and ``s == 0``.
    """
    rng = np.random.default_rng(4)
    mu, s_true, n = 7.0, 3.0, 4000
    y = rng.normal(mu, s_true, n)
    mean, sigma, skew0 = np.full(n, mu), np.full(n, s_true / 2.0), np.zeros(n)

    result = evaluate_trial(_BASE_ROW_PASSING, mean, sigma, skew0, y)

    assert result.mode == "dispersion"
    assert result.c > 1.5 and result.s == 0.0
    assert result.slack > 0


def test_evaluate_trial_slack_capped_by_non_g4_binding_gate():
    """Calibration only moves Gate 4. When another gate binds tighter than any Gate-4 corner can
    reach, the trial slack is that gate's headroom — the search cannot rank the cell as shipping on
    calibration alone.
    """
    rng = np.random.default_rng(4)
    y = rng.normal(7.0, 3.0, 4000)
    mean, sigma, skew0 = np.full(4000, 7.0), np.full(4000, 1.5), np.zeros(4000)  # under-dispersed
    base = {**_BASE_ROW_PASSING, "g2_star_z": 0.49}  # g2 slack = (0.5-0.49)/0.5 = 0.02, very tight

    result = evaluate_trial(base, mean, sigma, skew0, y)

    assert result.slack == pytest.approx(0.02, abs=1e-9)  # g2 binds, not g4
