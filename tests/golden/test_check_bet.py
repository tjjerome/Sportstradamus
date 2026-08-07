"""Settlement grading of :func:`analysis.check_bet`.

``check_bet`` resolves one parlay row against game logs and returns
``(legs, misses)`` — the grading kernel every resolved-parlay P&L number
downstream depends on. Pinned with synthetic fixtures: Over/Under hit and
miss grading, the push rule (``result == line`` skips the leg entirely),
combo (``" + "``) and matchup (``" vs. "``) legs summing both players, and
the unresolvable short-circuits returning NaN — never a graded loss.
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


def test_grades_over_under_hit_miss_and_skips_push():
    legs = [
        {"player": "LeBron James", "bet": "Over", "line": 25.5, "stat": "points"},  # 30>25.5 hit
        {"player": "Stephen Curry", "bet": "Under", "line": 6.5, "stat": "assists"},  # 5<6.5 hit
        {"player": "Anthony Davis", "bet": "Over", "line": 12.5, "stat": "points"},  # 10<12.5 miss
        {"player": "Stephen Curry", "bet": "Under", "line": 4.5, "stat": "assists"},  # 5>4.5 miss
        {"player": "LeBron James", "bet": "Over", "line": 30, "stat": "points"},  # 30==30 push
    ]
    assert check_bet(_bet(legs), _nba_stats(), _STAT_MAP) == (4, 2)


def test_plus_combo_leg_sums_both_players():
    legs = [
        {
            "player": "LeBron James + Anthony Davis",
            "bet": "Over",
            "line": 35.5,
            "stat": "points",
        }  # 30+10=40 > 35.5
    ]
    assert check_bet(_bet(legs), _nba_stats(), _STAT_MAP) == (1, 0)


def test_versus_combo_leg_sums_both_players():
    legs = [
        {
            "player": "LeBron James vs. Stephen Curry",
            "bet": "Over",
            "line": 55.5,
            "stat": "points",
        }  # 30+28=58 > 55.5
    ]
    assert check_bet(_bet(legs), _nba_stats(), _STAT_MAP) == (1, 0)


def test_league_absent_returns_nan_not_a_loss():
    bet = _bet([{"player": "X", "bet": "Over", "line": 1.5, "stat": "points"}], League="MLB")
    legs, misses = check_bet(bet, _nba_stats(), _STAT_MAP)
    assert pd.isna(legs) and pd.isna(misses)


def test_no_gameday_rows_returns_nan_not_a_loss():
    bet = _bet(
        [{"player": "LeBron James", "bet": "Over", "line": 1.5, "stat": "points"}],
        Date="2025-02-01",
    )
    legs, misses = check_bet(bet, _nba_stats(), _STAT_MAP)
    assert pd.isna(legs) and pd.isna(misses)
