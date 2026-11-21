"""The NBA profile's minutes history and per-48 rates pooled over its windows (study §7.1).

A pooled rate is a window's summed stat over its summed minutes, per 48. The per-game mean
beside it counts a game without minutes as a zero rate and weighs a short stint like a whole
game; the pooled one weighs each game by its minutes. Every value reads games before the date.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA, StatsWNBA

D = date(2025, 12, 10)
# One player's games, oldest first: (minutes, assists, opponent). The third has no minutes;
# the seventh is on D, so no profile at D may read it.
_GAMES = [
    (24, 4, "XXX"),
    (30, 6, "XXX"),
    (0, 0, "YYY"),
    (10, 4, "XXX"),
    (20, 2, "YYY"),
    (36, 12, "XXX"),
    (48, 30, "YYY"),
]


def _profile() -> pd.Series:
    stats = StatsNBA()
    days_back = len(_GAMES) - 1
    rows = [
        {
            "GAME_ID": f"g{i}",
            "GAME_DATE": (D - timedelta(days=days_back - i)).isoformat(),
            "PLAYER_NAME": "A",
            "TEAM_ABBREVIATION": "TTT",
            "OPP": opponent,
            "MIN": float(minutes),
            "AST": float(assists),
            "AST_48": assists / minutes * 48 if minutes else float("nan"),
        }
        for i, (minutes, assists, opponent) in enumerate(_GAMES)
    ]
    stats.gamelog = pd.DataFrame(rows).reindex(columns=stats.gamelog.columns)
    stats.base_profile(D)
    return stats.playerProfile.loc["A"]


def test_minutes_history_reads_the_last_games_before_the_date():
    profile = _profile()

    assert profile["MIN last game"] == 36
    assert profile["MIN last 3"] == pytest.approx((10 + 20 + 36) / 3)


def test_a_pooled_rate_weighs_games_by_minutes_and_skips_one_without():
    profile = _profile()

    assert profile["AST_48 pooled"] == pytest.approx((4 + 6 + 0 + 4 + 2 + 12) / 120 * 48)
    # The last five games: the minuteless one adds nothing to either sum, where the per-game
    # mean beside it counts that game as a rate of zero.
    assert profile["AST_48 pooled short"] == pytest.approx(
        (6 + 4 + 2 + 12) / (30 + 10 + 20 + 36) * 48
    )
    assert profile["AST_48 short"] == pytest.approx((9.6 + 0 + 19.2 + 4.8 + 16) / 5)
    rates = [stat for stat in StatsNBA().stat_types if stat.endswith("_48")]
    assert {f"{rate} pooled{w}" for rate in rates for w in ("", " short")} <= set(profile.index)


def test_opponent_adjusted_assists_divide_by_what_each_opponent_allows():
    # Per game, XXX allowed (4 + 6 + 4 + 12) / 4 = 6.5 assists and YYY (0 + 2) / 2 = 1.0, against
    # a league average of 3.75: each assist off XXX counts 3.75 / 6.5, each off YYY 3.75.
    profile = _profile()

    adjusted = 26 * 3.75 / 6.5 + 2 * 3.75
    assert profile["AST_48 opp adj pooled"] == pytest.approx(adjusted / 120 * 48)


def test_the_wnba_profile_keeps_the_base_columns():
    columns = StatsWNBA().get_stat_columns("PTS")

    assert not [c for c in columns if "pooled" in c or c.startswith("Player MIN last")]
