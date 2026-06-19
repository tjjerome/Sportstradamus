"""Derivation + parity pins for the cross-league ``Playoff`` flag.

``Stats._playoff_flag`` labels postseason games. NBA/WNBA/NHL decode the native
game-ID season-type code (:data:`_PLAYOFF_GAMEID_CODE`) -- exact, and inherently
dropping preseason / All-Star / NBA Cup / Commissioner's Cup / 4 Nations, which is
the exclusion the flag wants. NFL reads the gamelog ``season type`` and MLB the
stamped ``game type``. Historical reads the realized row; the serve branch (a future
date) infers the phase from the upcoming schedule -- the game-ID leagues from the
season's latest completed game, NFL/MLB from a flag carried on ``upcoming_games``.

These pin the per-league decode (including the special-event exclusions a date or
game-count heuristic would mislabel), the games-based serve inference, and the
historical-vs-serve agreement that the NBA-only main parity harness can't reach.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

PAST = date(2025, 5, 1)
FUTURE = date.today() + timedelta(days=3)

# (game ID, expected Playoff) per league, exercising every season-type code seen in
# the cached gamelogs -- including the special events that must NOT flag as playoff.
_GAMEID_CASES = {
    "NBA": (
        StatsNBA,
        {
            "po": ("0042400001", 1),  # playoffs
            "reg": ("0022400001", 0),  # regular season
            "pi": ("0052400001", 1),  # play-in (postseason)
            "cup": ("0062400001", 0),  # NBA Cup final (regular-counted)
        },
    ),
    "WNBA": (
        StatsWNBA,
        {
            "po": ("1042400001", 1),  # playoffs
            "reg": ("1022400001", 0),  # regular season
            "cc": ("1052400001", 0),  # Commissioner's Cup (not a bracket game)
        },
    ),
    "NHL": (
        StatsNHL,
        {
            "po": ("2024030001", 1),  # playoffs (chars 5-6 == 03)
            "reg": ("2024020001", 0),  # regular season (02)
            "pre": ("2024010001", 0),  # preseason (01)
            "fn": ("2024190001", 0),  # 4 Nations Face-Off exhibition (19)
        },
    ),
}


@pytest.mark.parametrize("league", list(_GAMEID_CASES))
def test_gameid_historical_decode(league):
    cls, cases = _GAMEID_CASES[league]
    s = cls()
    ls = s.log_strings
    s.gamelog = pd.DataFrame(
        [
            {ls["player"]: p, ls["game"]: gid, ls["date"]: PAST.isoformat()}
            for p, (gid, _) in cases.items()
        ]
    )
    stats = pd.DataFrame(index=list(cases))
    s._playoff_flag(stats, PAST, {})
    assert stats["Playoff"].to_dict() == {p: exp for p, (_, exp) in cases.items()}


@pytest.mark.parametrize("league", list(_GAMEID_CASES))
def test_gameid_serve_uses_latest_completed_game(league):
    """The upcoming schedule has no game ID, so serve reads the phase from the
    season's latest completed game -- playoff-coded latest game => upcoming playoff."""
    cls, cases = _GAMEID_CASES[league]
    s = cls()
    ls = s.log_strings
    reg_id = cases["reg"][0]
    po_id = cases["po"][0]
    for latest_id, exp in ((po_id, 1), (reg_id, 0)):
        s.gamelog = pd.DataFrame(
            [
                {ls["player"]: "early", ls["game"]: reg_id, ls["date"]: "2025-01-01"},
                {ls["player"]: "latest", ls["game"]: latest_id, ls["date"]: "2025-06-15"},
            ]
        )
        stats = pd.DataFrame(index=["early", "latest"])
        s._playoff_flag(stats, FUTURE, {})
        assert (stats["Playoff"] == exp).all(), f"{league} latest={latest_id}"


@pytest.mark.parametrize("league", list(_GAMEID_CASES))
def test_gameid_historical_and_serve_agree_on_a_playoff_game(league):
    cls, cases = _GAMEID_CASES[league]
    s = cls()
    ls = s.log_strings
    po_id = cases["po"][0]
    s.gamelog = pd.DataFrame(
        [{ls["player"]: "X", ls["game"]: po_id, ls["date"]: PAST.isoformat()}]
    )
    hist = pd.DataFrame(index=["X"])
    s._playoff_flag(hist, PAST, {})
    serve = pd.DataFrame(index=["X"])
    s._playoff_flag(serve, FUTURE, {})
    assert hist.loc["X", "Playoff"] == 1 == serve.loc["X", "Playoff"]


def test_nfl_historical_from_season_type():
    s = StatsNFL()
    s.gamelog = pd.DataFrame(
        [
            {"player display name": p, "gameday": PAST.isoformat(), "season type": st}
            for p, st in {"P1": "POST", "P2": "REG"}.items()
        ]
    )
    stats = pd.DataFrame(index=["P1", "P2"])
    s._playoff_flag(stats, PAST, {})
    assert stats.loc["P1", "Playoff"] == 1
    assert stats.loc["P2", "Playoff"] == 0


def test_nfl_serve_from_upcoming_playoff_flag():
    s = StatsNFL()
    s.upcoming_games = {"AAA": {"playoff": True}, "BBB": {"playoff": False}}
    stats = pd.DataFrame(index=["P1", "P2"])
    s._playoff_flag(stats, FUTURE, {"P1": "AAA", "P2": "BBB"})
    assert stats.loc["P1", "Playoff"] == 1
    assert stats.loc["P2", "Playoff"] == 0


def test_mlb_historical_from_game_type():
    s = StatsMLB()
    s.gamelog = pd.DataFrame(
        [
            {"playerName": "P1", "gameDate": PAST.isoformat(), "gameId": 1, "game type": "F"},
            {"playerName": "P2", "gameDate": PAST.isoformat(), "gameId": 2, "game type": "R"},
        ]
    )
    stats = pd.DataFrame(index=["P1", "P2"])
    s._playoff_flag(stats, PAST, {})
    assert stats.loc["P1", "Playoff"] == 1  # F = Wild Card
    assert stats.loc["P2", "Playoff"] == 0  # R = regular


def test_mlb_historical_missing_game_type_reads_regular():
    """Rows logged before the column existed (pre-backfill) carry no type -> 0."""
    s = StatsMLB()
    s.gamelog = pd.DataFrame([{"playerName": "P1", "gameDate": PAST.isoformat(), "gameId": 1}])
    stats = pd.DataFrame(index=["P1"])
    s._playoff_flag(stats, PAST, {})
    assert stats.loc["P1", "Playoff"] == 0


def test_mlb_serve_from_upcoming_playoff_flag():
    s = StatsMLB()
    s.upcoming_games = {"NYY": {"Playoff": True}, "BOS": {"Playoff": False}}
    stats = pd.DataFrame(index=["P1", "P2"])
    s._playoff_flag(stats, FUTURE, {"P1": "NYY", "P2": "BOS"})
    assert stats.loc["P1", "Playoff"] == 1
    assert stats.loc["P2", "Playoff"] == 0
