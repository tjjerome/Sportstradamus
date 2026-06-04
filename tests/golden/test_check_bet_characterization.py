"""Characterization of :func:`analysis.check_bet`.

``check_bet`` resolves one parlay row against game logs and returns
``(legs, misses)``. It reads only ``stats[league].gamelog`` /
``log_strings`` so it is pinned here with synthetic fakes (no real data),
decomposed from a single CC-24 function into an orchestrator plus leg
helpers. The fixtures exercise every branch: the league-absent and
empty-gameday short-circuits, the NHL / NBA market-name overrides, combo
(``" + "``) and matchup (``" vs. "``) legs, single legs, a push
(``result == line`` skips the leg), and unparseable / empty legs.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.analysis import check_bet


class _FakeStats:
    def __init__(self, gamelog: pd.DataFrame, log_strings: dict[str, str]) -> None:
        self.gamelog = gamelog
        self.log_strings = log_strings


_LS = {"date": "GAME_DATE", "player": "PLAYER_NAME", "team": "TEAM_ABBREVIATION"}
_STAT_MAP = {"PrizePicks": {"Points": "points", "Assists": "assists"}}


def _nba_stats() -> dict[str, _FakeStats]:
    gamelog = pd.DataFrame(
        {
            "GAME_DATE": ["2025-01-15"] * 3,
            "PLAYER_NAME": ["LeBron James", "Anthony Davis", "Stephen Curry"],
            "TEAM_ABBREVIATION": ["LAL", "LAL", "GSW"],
            "points": [30.0, 10.0, 28.0],
            "assists": [8.0, 3.0, 5.0],
        }
    )
    return {"NBA": _FakeStats(gamelog, _LS)}


def _nhl_stats() -> dict[str, _FakeStats]:
    gamelog = pd.DataFrame(
        {
            "GAME_DATE": ["2025-01-15"] * 2,
            "PLAYER_NAME": ["Connor McDavid", "Cale Makar"],
            "TEAM_ABBREVIATION": ["EDM", "COL"],
            "points": [2.0, 1.0],
            "blocked": [0.0, 4.0],
            "assists": [1.0, 1.0],
        }
    )
    return {"NHL": _FakeStats(gamelog, _LS)}


def _bet(**legs) -> pd.Series:
    row = {"League": "NBA", "Platform": "PrizePicks", "Date": "2025-01-15", "Game": "LAL/GSW"}
    row.update(legs)
    return pd.Series(row)


def test_mixed_legs_hit_miss_push_and_combo():
    bet = _bet(
        **{
            "Leg 1": "LeBron James Over 25.5 Points - 52.0%",  # 30 > 25.5 -> hit
            "Leg 2": "Stephen Curry Under 6.5 Assists - 48.0%",  # 5 < 6.5 -> hit
            "Leg 3": "Anthony Davis Over 12.5 Points - 50.0%",  # 10 < 12.5 -> miss
            "Leg 4": "LeBron James Over 30 Points - 50.0%",  # 30 == 30 -> push (skip)
            "Leg 5": "LeBron James + Anthony Davis Over 35.5 Points - 55.0%",  # 40 > 35.5 -> hit
        }
    )
    assert check_bet(bet, _nba_stats(), _STAT_MAP) == (4, 1)


def test_versus_combo_leg():
    bet = _bet(**{"Leg 1": "LeBron James vs. Stephen Curry Over 55.5 Points - 50.0%"})  # 58 > 55.5
    assert check_bet(bet, _nba_stats(), _STAT_MAP) == (1, 0)


def test_unparseable_and_empty_legs_are_skipped():
    bet = _bet(
        **{
            "Leg 1": "",
            "Leg 2": "garbage leg no pattern",
            "Leg 3": "LeBron James Over 25.5 Points - 52.0%",  # only this one counts
        }
    )
    assert check_bet(bet, _nba_stats(), _STAT_MAP) == (1, 0)


def test_nhl_market_overrides():
    bet = _bet(
        League="NHL",
        Game="EDM/COL",
        **{
            "Leg 1": "Connor McDavid Over 1.5 Points - 50.0%",  # 2 > 1.5 -> hit
            "Leg 2": "Cale Makar Under 2.5 Blocked Shots - 50.0%",  # 4 > 2.5 -> miss
        },
    )
    assert check_bet(bet, _nhl_stats(), _STAT_MAP) == (2, 1)


def test_league_absent_returns_nan():
    bet = _bet(League="MLB", **{"Leg 1": "X Over 1.5 Points - 50.0%"})
    legs, misses = check_bet(bet, _nba_stats(), _STAT_MAP)
    assert pd.isna(legs) and pd.isna(misses)


def test_no_gameday_rows_returns_nan():
    bet = _bet(Date="2025-02-01", **{"Leg 1": "LeBron James Over 1.5 Points - 50.0%"})
    legs, misses = check_bet(bet, _nba_stats(), _STAT_MAP)
    assert pd.isna(legs) and pd.isna(misses)
