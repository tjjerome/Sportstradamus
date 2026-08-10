"""Characterization test for the shared NBA/NHL team-volume budget scaler.

`scale_team_volume_to_budget` was extracted from a per-team loop that lived
byte-identical in StatsNBA.get_volume_stats and StatsNHL.get_volume_stats. This
test pins it against a verbatim reference copy of that loop across synthetic
playerProfile frames and every edge branch (team==0 skip, total<=0 skip,
zero-variance fallback, no-alpha Gaussian path, ratio/cap clipping). The scaler
is pure DataFrame math, so no cached game logs are needed.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats.base import (
    _VOLUME_SCALE_RATIO_CAP,
    _VOLUME_SCALE_RATIO_FLOOR,
    scale_team_volume_to_budget,
)

_PARAMS = {
    "budget_mean": 240.0,
    "typical_rotation": 9,
    "avg_unmodeled_min": 14.0,
    "per_player_floor": 12.0,
    "per_player_cap": 40.0,
}


def _reference(
    player_profile,
    market,
    *,
    budget_mean,
    typical_rotation,
    avg_unmodeled_min,
    per_player_floor,
    per_player_cap,
):
    """Verbatim copy of the pre-consolidation NBA/NHL per-team loop."""
    teams = player_profile.loc[player_profile["team"] != 0].groupby("team")
    for _team, team_df in teams:
        loc = team_df[f"proj {market} loc"].copy()
        scale = team_df[f"proj {market} scale"].copy()
        sn_alpha = (
            team_df[f"proj {market} alpha"].copy()
            if f"proj {market} alpha" in team_df.columns
            else pd.Series(0, index=loc.index)
        )
        N = len(team_df)

        delta = sn_alpha / np.sqrt(1 + sn_alpha**2)
        true_means = loc + scale * delta * np.sqrt(2 / np.pi)
        true_vars = scale**2

        total = true_means.sum()
        if total <= 0:
            continue

        unmodeled_count = max(0, typical_rotation - N)
        unmodeled_reserve = unmodeled_count * avg_unmodeled_min

        upper_target = budget_mean - unmodeled_reserve
        lower_target = N * per_player_floor
        target = max(upper_target, lower_target)

        deficit = target - total
        total_var = true_vars.sum()
        if total_var > 0:
            adjustments = true_vars / total_var * deficit
        else:
            adjustments = true_means / total * deficit

        new_means = (true_means + adjustments).clip(lower=0, upper=per_player_cap)

        ratio = new_means / true_means.replace(0, np.nan)
        ratio = ratio.fillna(1.0).clip(
            lower=_VOLUME_SCALE_RATIO_FLOOR, upper=_VOLUME_SCALE_RATIO_CAP
        )
        new_scale = scale * ratio
        new_loc = new_means - new_scale * delta * np.sqrt(2 / np.pi)

        player_profile.loc[loc.index, f"proj {market} loc"] = new_loc
        player_profile.loc[scale.index, f"proj {market} scale"] = new_scale
        player_profile.loc[loc.index, f"proj {market} mean"] = new_means
        player_profile.loc[scale.index, f"proj {market} std"] = new_scale


def _synthetic(seed, n=14, market="PTS", with_alpha=True):
    rng = np.random.default_rng(seed)
    cols = {
        "team": rng.integers(0, 4, n),  # 0 is the "no team" sentinel and is skipped
        f"proj {market} loc": rng.uniform(0, 30, n),
        f"proj {market} scale": rng.uniform(0.5, 8, n),
    }
    if with_alpha:
        cols[f"proj {market} alpha"] = rng.uniform(-2, 2, n)
    return pd.DataFrame(cols)


def _assert_match(df, market):
    expected = df.copy()
    actual = df.copy()
    _reference(expected, market, **_PARAMS)
    scale_team_volume_to_budget(actual, market, **_PARAMS)
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("seed", range(8))
def test_matches_reference_random(seed):
    _assert_match(_synthetic(seed), "PTS")


def test_no_alpha_column_gaussian_path():
    _assert_match(_synthetic(3, market="TOI", with_alpha=False), "TOI")


def test_zero_total_team_skipped():
    df = pd.DataFrame(
        {
            "team": [1, 1],
            "proj PTS loc": [0.0, 0.0],
            "proj PTS scale": [0.0, 0.0],
            "proj PTS alpha": [0.0, 0.0],
        }
    )
    actual = df.copy()
    scale_team_volume_to_budget(actual, "PTS", **_PARAMS)
    pd.testing.assert_frame_equal(actual, df)


def test_zero_variance_fallback():
    df = pd.DataFrame(
        {
            "team": [1, 1, 1],
            "proj PTS loc": [10.0, 20.0, 5.0],
            "proj PTS scale": [0.0, 0.0, 0.0],
            "proj PTS alpha": [0.0, 0.0, 0.0],
        }
    )
    _assert_match(df, "PTS")


def test_no_alpha_column_float32_preserves_dtype_no_warning():
    """Real model output is float32 (PyTorch); the no-alpha fallback must not
    silently promote it to float64 and trip pandas' incompatible-dtype warning."""
    df = _synthetic(3, market="TOI", with_alpha=False).astype(
        dict.fromkeys(["proj TOI loc", "proj TOI scale"], "float32")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        scale_team_volume_to_budget(df, "TOI", **_PARAMS)
    assert df["proj TOI loc"].dtype == np.float32
    assert df["proj TOI scale"].dtype == np.float32
    assert df["proj TOI mean"].dtype == np.float32
    assert df["proj TOI std"].dtype == np.float32
