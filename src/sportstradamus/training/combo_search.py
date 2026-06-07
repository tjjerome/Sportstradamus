"""Operation Ship 75 combination search — per-market Optuna over (normalization × loss).

The search ranks each market's axis corner cheaply, then confirms the winner under real HPO
before any ship. The expensive axis is ``normalization × loss`` (one deterministic train per
combo); calibration is free — a transform of the trained predictive's arrays (mean held fixed),
swept post-hoc by :func:`scorecard.sweep_calibration_modes`. This module carries the pure search
logic; the deterministic ``meditate`` trials + Optuna study live in ``search_market``.
"""

from dataclasses import dataclass

import numpy as np

from sportstradamus.training.scorecard import min_gate_slack, sweep_calibration_modes


@dataclass(frozen=True)
class TrialResult:
    """The best calibration corner for one trained ``(normalization, loss)`` model.

    ``mode`` is the winning calibration mode; ``c`` / ``s`` its fitted scale + additive skew
    shift; ``pit_ks`` the Gate-4 KS it reaches; ``slack`` the min-gate ship margin (``> 0`` ⇔
    ships) the Optuna objective maximizes.
    """

    mode: str
    slack: float
    c: float
    s: float
    pit_ks: float


def recover_fused_predictive(
    mean: np.ndarray,
    served_sigma: np.ndarray,
    served_alpha: np.ndarray,
    dispersion_cal: float,
    skew_cal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert post-hoc calibration back to the pre-calibration (fused) predictive.

    A trained cell dumps its *served* SkewNormal params (scale already multiplied by
    ``dispersion_cal``, skew already shifted by ``skew_cal``); the calibration sweep needs the
    fused predictive to re-fit every mode from scratch. The mean is held fixed by calibration
    (``skewnormal_loc_from_mean``) so it passes through. ``dispersion_cal=1`` / ``skew_cal=0``
    recovers an uncalibrated dump unchanged.
    """
    fused_sigma = np.asarray(served_sigma, dtype=float) / dispersion_cal
    fused_skew = np.asarray(served_alpha, dtype=float) - skew_cal
    return mean, fused_sigma, fused_skew


def evaluate_trial(
    base_row: dict[str, object],
    mean: np.ndarray,
    sigma: np.ndarray,
    skew: np.ndarray,
    y: np.ndarray,
) -> TrialResult:
    """Rank one trained model's calibration corner by the min-gate ship slack.

    Sweeps the four post-hoc calibration modes on the fused predictive (free, no retrain),
    substitutes each mode's Gate-4 KS into the trial's honest gate row, and returns the mode with
    the largest :func:`~sportstradamus.training.scorecard.min_gate_slack`. Gates 2/3 are
    mean-invariant and Gates 1/5 are read from ``base_row`` (the L4a brief established calibration
    barely moves them), so only Gate 4 varies across modes — no re-pricing.
    """
    fits = sweep_calibration_modes(mean, sigma, skew, y)
    slack, fit = max(
        ((min_gate_slack({**base_row, "g4_pit_ks": f.pit_ks}), f) for f in fits),
        key=lambda t: t[0],
    )
    return TrialResult(mode=fit.mode, slack=slack, c=fit.c, s=fit.s, pit_ks=fit.pit_ks)
