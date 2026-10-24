"""Contracts for the parlay calibration audit's joint-probability recovery."""

import math

import pandas as pd
import pytest

from sportstradamus.scripts.audit_parlay_calibration import (
    _recover_indep_joint_prob,
    _recover_joint_prob,
    _recovered_payout,
)


def _underdog_three_leg(boost: float) -> pd.Series:
    return pd.Series(
        {"Platform": "Underdog", "Bet Size": 3, "Boost": boost, "Model EV": 2.4, "Indep P": 2.0}
    )


def test_stored_boost_already_carries_the_payout_base():
    row = _underdog_three_leg(6.5)

    assert _recovered_payout(row) == 6.5
    assert _recover_joint_prob(row) == pytest.approx(0.36923, abs=1e-5)
    assert _recover_indep_joint_prob(row) == pytest.approx(0.30769, abs=1e-5)


def test_payout_clips_at_the_ceiling():
    row = _underdog_three_leg(150.0)

    assert _recovered_payout(row) == 100.0
    assert _recover_joint_prob(row) == pytest.approx(0.024)


def test_missing_boost_recovers_nothing():
    row = _underdog_three_leg(float("nan"))

    assert math.isnan(_recovered_payout(row))
    assert math.isnan(_recover_joint_prob(row))
    assert math.isnan(_recover_indep_joint_prob(row))
