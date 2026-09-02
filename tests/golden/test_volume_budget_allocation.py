"""Pins the shared team-volume budget allocator every league's volume path uses.

``apply_volume_budget`` redistributes one team's ``proj {market} mean`` /
``proj {market} std`` onto a target total, precision-weighted by variance;
``scale_team_volume_to_budget`` is the NBA/NHL minute-budget derivation layered on
top of it, and ``StatsNFL._rescale_team_volume`` is the play-share one. It is pure
DataFrame math in mean/std space, so no cached game logs or pickles are needed.

The predecessor of this file characterized the allocator against a verbatim copy
of the pre-consolidation SkewNormal loop, which worked in ``loc``/``scale``/
``alpha`` space. That space is gone -- the volume path now decodes to mean/std
before any rescale, which is what lets the DPO ``carries`` model through it -- so
these assertions are stated as closed forms rather than against a reference copy.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats.base import (
    _VOLUME_SCALE_RATIO_CAP,
    _VOLUME_SCALE_RATIO_FLOOR,
    apply_volume_budget,
    scale_team_volume_to_budget,
)

_BUDGET_PARAMS = {
    "budget_mean": 240.0,
    "typical_rotation": 9,
    "avg_unmodeled_min": 14.0,
    "per_player_floor": 12.0,
    "per_player_cap": 40.0,
}


def _profile(means, stds, market="PTS", teams=None):
    return pd.DataFrame(
        {
            "team": [1] * len(means) if teams is None else teams,
            f"proj {market} mean": [float(m) for m in means],
            f"proj {market} std": [float(s) for s in stds],
        }
    )


def test_deficit_is_split_by_variance_share():
    # vars = [1, 4, 9] -> shares [1/14, 4/14, 9/14] of a +10 deficit.
    df = _profile([10, 20, 30], [1, 2, 3])

    apply_volume_budget(df, "PTS", df.index, target=70.0, per_player_cap=1000.0)

    expected = np.array([10, 20, 30]) + np.array([1, 4, 9]) / 14 * 10
    np.testing.assert_allclose(df["proj PTS mean"], expected)
    assert df["proj PTS mean"].sum() == pytest.approx(70.0)


def test_std_follows_the_mean_ratio_so_cv_survives():
    df = _profile([10, 20, 30], [1, 2, 3])
    cv_before = df["proj PTS std"] / df["proj PTS mean"]

    apply_volume_budget(df, "PTS", df.index, target=70.0, per_player_cap=1000.0)

    cv_after = df["proj PTS std"] / df["proj PTS mean"]
    np.testing.assert_allclose(cv_after, cv_before)


def test_zero_variance_falls_back_to_proportional_shares():
    df = _profile([10, 20, 30], [0, 0, 0])

    apply_volume_budget(df, "PTS", df.index, target=70.0, per_player_cap=1000.0)

    np.testing.assert_allclose(df["proj PTS mean"], [70 / 6, 70 / 3, 35.0])
    np.testing.assert_allclose(df["proj PTS std"], 0.0)


def test_team_projecting_no_volume_is_left_alone():
    df = _profile([0, 0], [0, 0])
    before = df.copy()

    apply_volume_budget(df, "PTS", df.index, target=70.0, per_player_cap=40.0)

    pd.testing.assert_frame_equal(df, before)


def test_per_player_cap_clips_the_mean_and_shrinks_its_std():
    df = _profile([10, 20, 30], [1, 2, 3])

    apply_volume_budget(df, "PTS", df.index, target=70.0, per_player_cap=25.0)

    assert df.loc[2, "proj PTS mean"] == 25.0
    assert df.loc[2, "proj PTS std"] == pytest.approx(3 * 25 / 30)
    assert df["proj PTS mean"].sum() < 70.0  # the cap costs the team its target


def test_ratio_floor_bounds_a_collapsing_std():
    # A -91 deficit drives player B's mean to zero; without the floor its std
    # would collapse with it and the fused shape would go degenerate.
    df = _profile([100, 1], [1, 1])

    apply_volume_budget(df, "PTS", df.index, target=10.0, per_player_cap=1000.0)

    assert df.loc[1, "proj PTS mean"] == 0.0
    assert df.loc[1, "proj PTS std"] == pytest.approx(_VOLUME_SCALE_RATIO_FLOOR)


def test_ratio_cap_bounds_an_exploding_std():
    df = _profile([1, 100], [10, 0.001])

    apply_volume_budget(df, "PTS", df.index, target=1000.0, per_player_cap=10_000.0)

    assert df.loc[0, "proj PTS mean"] > 10 * 1
    assert df.loc[0, "proj PTS std"] == pytest.approx(10 * _VOLUME_SCALE_RATIO_CAP)


def test_projection_columns_stay_float64_without_a_dtype_warning():
    # The decode emits float64 (the raw model heads are float32); a rescale must
    # not silently narrow it back and trip pandas' incompatible-dtype warning.
    df = _profile([10, 20, 30], [1, 2, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        apply_volume_budget(df, "PTS", df.index, target=70.0, per_player_cap=1000.0)

    assert df["proj PTS mean"].dtype == np.float64
    assert df["proj PTS std"].dtype == np.float64


def test_budget_reserves_minutes_for_unmodeled_players():
    # 5 modeled of a 9-man rotation -> 4 * 14 = 56 minutes held back from 240.
    df = _profile([30] * 5, [3] * 5, market="MIN")

    scale_team_volume_to_budget(df, "MIN", **_BUDGET_PARAMS)

    assert df["proj MIN mean"].sum() == pytest.approx(184.0)


def test_budget_floor_wins_when_it_exceeds_the_reserved_budget():
    # 15 modeled players * a 20-minute floor = 300 > the 240 budget.
    df = _profile([5] * 15, [1] * 15, market="MIN")

    scale_team_volume_to_budget(df, "MIN", **{**_BUDGET_PARAMS, "per_player_floor": 20.0})

    assert df["proj MIN mean"].sum() == pytest.approx(300.0)


def test_unassigned_team_sentinel_rows_are_skipped():
    df = _profile([30] * 7, [3] * 7, market="MIN", teams=[0, 0, 1, 1, 1, 1, 1])
    before = df.loc[df["team"] == 0].copy()

    scale_team_volume_to_budget(df, "MIN", **_BUDGET_PARAMS)

    pd.testing.assert_frame_equal(df.loc[df["team"] == 0], before)
    # The sentinel rows do not count toward the rotation: 5 modeled, not 7.
    assert df.loc[df["team"] == 1, "proj MIN mean"].sum() == pytest.approx(184.0)
