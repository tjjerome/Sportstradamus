"""Characterization of :func:`analysis.check_bet`.

``check_bet`` resolves one parlay row against game logs and returns
``(legs, misses)``. It reads ``bet.legs`` (a list of structured leg dicts,
already resolved to gamelog column names by the parlay-write path) plus
``stats[league].gamelog`` / ``log_strings``, so it is pinned here with
synthetic fakes (no real data). The fixtures exercise every branch: the
league-absent and empty-gameday short-circuits, combo (``" + "``) and
matchup (``" vs. "``) legs, single legs, and a push (``result == line``
skips the leg).
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


def _bet(legs, **overrides) -> pd.Series:
    row = {
        "League": "NBA",
        "Platform": "PrizePicks",
        "Date": "2025-01-15",
        "Game": "LAL/GSW",
        "legs": legs,
    }
    row.update(overrides)
    return pd.Series(row)


def test_mixed_legs_hit_miss_push_and_combo():
    legs = [
        {"player": "LeBron James", "bet": "Over", "line": 25.5, "stat": "points"},  # 30>25.5 hit
        {"player": "Stephen Curry", "bet": "Under", "line": 6.5, "stat": "assists"},  # 5<6.5 hit
        {"player": "Anthony Davis", "bet": "Over", "line": 12.5, "stat": "points"},  # 10<12.5 miss
        {"player": "LeBron James", "bet": "Over", "line": 30, "stat": "points"},  # 30==30 push
        {
            "player": "LeBron James + Anthony Davis",
            "bet": "Over",
            "line": 35.5,
            "stat": "points",
        },  # 40>35.5 hit
    ]
    bet = _bet(legs)
    assert check_bet(bet, _nba_stats(), _STAT_MAP) == (4, 1)


def test_versus_combo_leg():
    legs = [
        {
            "player": "LeBron James vs. Stephen Curry",
            "bet": "Over",
            "line": 55.5,
            "stat": "points",
        }  # 58 > 55.5
    ]
    bet = _bet(legs)
    assert check_bet(bet, _nba_stats(), _STAT_MAP) == (1, 0)


def test_nhl_market_overrides():
    """End-to-end check that check_bet resolves NHL-shaped gamelog data.

    The market-name translation this test was originally named for (e.g.
    "Blocked Shots" -> "blocked") now happens once at parlay-write time
    (``_resolve_leg_stat`` / ``_leg_market_map``, see
    ``tests/golden/test_parlay_search.py``); by the time a leg reaches
    ``check_bet`` its ``stat`` is already the resolved gamelog column name.
    """
    legs = [
        {"player": "Connor McDavid", "bet": "Over", "line": 1.5, "stat": "points"},  # 2>1.5 hit
        {"player": "Cale Makar", "bet": "Under", "line": 2.5, "stat": "blocked"},  # 4>2.5 miss
    ]
    bet = _bet(legs, League="NHL", Game="EDM/COL")
    assert check_bet(bet, _nhl_stats(), _STAT_MAP) == (2, 1)


def test_league_absent_returns_nan():
    bet = _bet([{"player": "X", "bet": "Over", "line": 1.5, "stat": "points"}], League="MLB")
    legs, misses = check_bet(bet, _nba_stats(), _STAT_MAP)
    assert pd.isna(legs) and pd.isna(misses)


def test_no_gameday_rows_returns_nan():
    bet = _bet(
        [{"player": "LeBron James", "bet": "Over", "line": 1.5, "stat": "points"}],
        Date="2025-02-01",
    )
    legs, misses = check_bet(bet, _nba_stats(), _STAT_MAP)
    assert pd.isna(legs) and pd.isna(misses)
