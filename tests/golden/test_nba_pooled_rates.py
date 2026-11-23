"""The NBA and WNBA profiles' minutes history and per-minute rates pooled over their windows.

A pooled rate is a window's summed stat over its summed minutes, per regulation game: 48 minutes
in the NBA, 40 in the WNBA (study §7.1). The per-game mean beside it counts a game without
minutes as a zero rate and weighs a short stint like a whole game; the pooled one weighs each
game by its minutes. Every value reads games before the date.
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
# Each league's assist rate column and the minutes it scales to.
_LEAGUES = [(StatsNBA, "AST_48", 48), (StatsWNBA, "AST_40", 40)]


def _profile(cls, rate, scale) -> pd.Series:
    stats = cls()
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
            rate: assists / minutes * scale if minutes else float("nan"),
        }
        for i, (minutes, assists, opponent) in enumerate(_GAMES)
    ]
    stats.gamelog = pd.DataFrame(rows).reindex(columns=stats.gamelog.columns)
    stats.base_profile(D)
    return stats.playerProfile.loc["A"]


@pytest.mark.parametrize(("cls", "rate", "scale"), _LEAGUES)
def test_minutes_history_reads_the_last_games_before_the_date(cls, rate, scale):
    profile = _profile(cls, rate, scale)

    assert profile["MIN last game"] == 36
    assert profile["MIN last 3"] == pytest.approx((10 + 20 + 36) / 3)


@pytest.mark.parametrize(("cls", "rate", "scale"), _LEAGUES)
def test_a_pooled_rate_weighs_games_by_minutes_and_skips_one_without(cls, rate, scale):
    profile = _profile(cls, rate, scale)

    assert profile[f"{rate} pooled"] == pytest.approx((4 + 6 + 0 + 4 + 2 + 12) / 120 * scale)
    # The last five games: the minuteless one adds nothing to either sum, where the per-game
    # mean beside it counts that game as a rate of zero.
    assert profile[f"{rate} pooled short"] == pytest.approx(
        (6 + 4 + 2 + 12) / (30 + 10 + 20 + 36) * scale
    )
    assert profile[f"{rate} short"] == pytest.approx(
        (6 / 30 + 0 + 4 / 10 + 2 / 20 + 12 / 36) * scale / 5
    )
    rates = [stat for stat in cls().stat_types if stat.endswith(f"_{scale}")]
    assert {f"{rate} pooled{w}" for rate in rates for w in ("", " short")} <= set(profile.index)


@pytest.mark.parametrize(("cls", "rate", "scale"), _LEAGUES)
def test_opponent_adjusted_assists_divide_by_what_each_opponent_allows(cls, rate, scale):
    # Per game, XXX allowed (4 + 6 + 4 + 12) / 4 = 6.5 assists and YYY (0 + 2) / 2 = 1.0, against
    # a league average of 3.75: each assist off XXX counts 3.75 / 6.5, each off YYY 3.75.
    profile = _profile(cls, rate, scale)

    adjusted = 26 * 3.75 / 6.5 + 2 * 3.75
    assert profile[f"{rate} opp adj pooled"] == pytest.approx(adjusted / 120 * scale)


def test_the_wnba_columns_pool_every_rate_per_40():
    stats = StatsWNBA()
    columns = set(stats.get_stat_columns("PTS"))

    rates = [stat for stat in stats.stat_types if stat.endswith("_40")] + ["AST_40 opp adj"]
    batch = {"Player MIN last game", "Player MIN last 3"}
    batch |= {f"Player {rate} pooled{w}" for rate in rates for w in ("", " short")}
    assert len(batch) == 22
    assert batch <= columns
    assert not [c for c in columns if "_48" in c]
