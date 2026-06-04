"""Characterization pin for ``StatsNHL.parse_game``.

``parse_game`` is a CC-20, ~240-line per-game boxscore parser with ZERO coverage.
This pins the exact (gamelog, teamlog) it returns so the decomposition (extracting
the team-level and skater stat-dict blocks) is proven behavior-preserving.

The synthetic moneypuck game CSV exercises BOTH branches (Team Level aggregates +
skater rows), the power-play (5on4) points lookup, the opponent-goalie lookup, and
the team-row TimeShare/ShotShare denominators. Numeric columns default to 1.0 with
key denominators overridden so the ``if shotsAgainst else 0`` ternaries take their
truthy paths.
"""

from __future__ import annotations

import pandas as pd
import pytest

import sportstradamus.stats.nhl as nhl_mod
from sportstradamus.stats.nhl import StatsNHL

_TEAM_COLS = [
    "OffIce_F_shotAttempts", "OffIce_F_unblockedShotAttempts", "OffIce_F_hits",
    "OffIce_F_takeaways", "OffIce_F_penalityMinutes", "OffIce_shotAttempts_For_Percentage",
    "OffIce_unblockedShotAttempts_For_Percentage", "OffIce_hits_For_Percentage",
    "OffIce_takeaways_For_Percentage", "OffIce_penalityMinutes_For_Percentage",
    "OffIce_A_blockedShotAttempts", "OffIce_A_shotAttempts",
    "OffIce_F_flurryScoreVenueAdjustedxGoals", "OffIce_A_flurryScoreVenueAdjustedxGoals",
    "OffIce_A_goals", "OffIce_F_goals", "OffIce_A_shotsOnGoal", "OffIce_A_savedShotsOnGoal",
    "OffIce_A_freeze", "OffIce_A_xFreeze", "OffIce_A_rebounds", "OffIce_A_xRebounds",
    "OffIce_A_reboundGoals", "OffIce_A_reboundxGoals", "OffIce_F_iceTime", "OffIce_F_shotsOnGoal",
]
_SKATER_COLS = [
    "I_F_points", "I_F_shotsOnGoal", "I_A_blockedShotAttempts", "I_F_goals",
    "I_F_primaryAssists", "I_F_secondaryAssists", "I_F_hits", "I_F_faceOffsWon", "I_F_iceTime",
    "OnIce_A_savedShotsOnGoal", "OnIce_A_shotsOnGoal", "OnIce_A_goals", "I_F_shotAttempts",
    "I_F_flurryScoreVenueAdjustedxGoals", "OnIce_unblockedShotAttempts_For_Percentage",
    "OnIce_A_flurryScoreVenueAdjustedxGoals", "OnIce_A_freeze", "OnIce_A_xFreeze",
    "OnIce_A_rebounds", "OnIce_A_xRebounds", "OnIce_A_reboundGoals", "OnIce_A_reboundxGoals",
]
_ALL_NUM = _TEAM_COLS + _SKATER_COLS


def _row(situation, position, team, playerName, playerId, **over):
    r = dict.fromkeys(_ALL_NUM, 1.0)
    r.update({"situation": situation, "position": position, "team": team,
              "playerName": playerName, "playerId": playerId})
    r.update(over)
    return r


def _game_df():
    return pd.DataFrame([
        _row("all", "Team Level", "BOS", "BOS", 0, OffIce_F_iceTime=3600.0,
             OffIce_F_shotsOnGoal=30.0, OffIce_A_shotsOnGoal=28.0, OffIce_A_shotAttempts=50.0,
             OffIce_F_shotAttempts=55.0, OffIce_F_goals=3.0, OffIce_A_goals=2.0),
        _row("all", "Team Level", "MTL", "MTL", 0, OffIce_F_iceTime=3600.0,
             OffIce_F_shotsOnGoal=28.0, OffIce_A_shotsOnGoal=30.0, OffIce_A_shotAttempts=55.0,
             OffIce_F_shotAttempts=50.0, OffIce_F_goals=2.0, OffIce_A_goals=3.0),
        _row("all", "C", "BOS", "Skater One", 1001, I_F_iceTime=1200.0, I_F_shotAttempts=5.0,
             OnIce_A_shotsOnGoal=10.0, I_F_goals=2.0, I_F_points=4.0, I_F_shotsOnGoal=6.0,
             I_F_primaryAssists=1.0, I_F_secondaryAssists=1.0, I_F_hits=3.0),
        _row("all", "G", "BOS", "Bos Goalie", 1003, I_F_iceTime=3600.0),
        _row("all", "G", "MTL", "Mtl Goalie", 2003, I_F_iceTime=3600.0),
        _row("5on4", "C", "BOS", "Skater One", 1001, I_F_points=2.0),
    ])


class _Resp:
    def __init__(self, df):
        self.text = df.to_csv(index=False)
        self.status_code = 200


@pytest.fixture
def parsed(monkeypatch):
    monkeypatch.setattr(nhl_mod, "scraper", type("S", (), {"get": staticmethod(lambda url: {
        "season": 20242025,
        "awayTeam": {"abbrev": "MTL", "score": 2},
        "homeTeam": {"abbrev": "BOS", "score": 3},
    })})())
    monkeypatch.setattr(nhl_mod, "requests",
                        type("R", (), {"get": staticmethod(lambda url: _Resp(_game_df()))}))
    gamelog, teamlog = StatsNHL().parse_game("G1", "2024-10-31")
    return {"gamelog": gamelog, "teamlog": teamlog}


def test_parse_game_row_counts(parsed):
    assert len(parsed["teamlog"]) == 2
    assert len(parsed["gamelog"]) == 3


def test_parse_game_team_level_rows(parsed):
    by_team = {r["team"]: r for r in parsed["teamlog"]}
    bos = by_team["BOS"]
    assert bos["opponent"] == "MTL"
    assert bos["home"] is True
    assert bos["WL"] == "W"
    assert bos["Corsi"] == pytest.approx(55.0)
    assert bos["Block_Pct"] == pytest.approx(1.0 / 50.0)
    assert bos["GOE"] == pytest.approx((3.0 - 1.0) / 55.0)
    assert bos["SV"] == pytest.approx(1.0 / 28.0)
    mtl = by_team["MTL"]
    assert mtl["home"] is False
    assert mtl["WL"] == "L"


def test_parse_game_skater_row(parsed):
    skater = next(r for r in parsed["gamelog"] if r["playerName"] == "Skater One")
    assert skater["gameId"] == "G1"
    assert skater["team"] == "BOS"
    assert skater["opponent"] == "MTL"
    assert skater["opponent goalie"] == "Mtl Goalie"
    assert skater["home"] is True
    assert skater["playerId"] == 1001
    assert skater["points"] == pytest.approx(4.0)
    assert skater["shots"] == pytest.approx(6.0)
    assert skater["goals"] == pytest.approx(2.0)
    assert skater["assists"] == pytest.approx(2.0)
    assert skater["sogBS"] == pytest.approx(7.0)
    assert skater["hits"] == pytest.approx(3.0)
    assert skater["timeOnIce"] == pytest.approx(20.0)
    assert skater["powerPlayPoints"] == pytest.approx(2.0)
    assert skater["fantasy points prizepicks"] == pytest.approx(36.5)
    assert skater["skater fantasy points underdog"] == pytest.approx(29.5)
    assert skater["GOE"] == pytest.approx(0.2)
    assert skater["TimeShare"] == pytest.approx(20.0 / 60.0)
    assert skater["ShotShare"] == pytest.approx(0.2)
    assert skater["Shot60"] == pytest.approx(18.0)


def test_parse_game_empty_on_non_200(monkeypatch):
    monkeypatch.setattr(nhl_mod, "scraper", type("S", (), {"get": staticmethod(lambda url: {
        "season": 20242025, "awayTeam": {"abbrev": "MTL", "score": 2},
        "homeTeam": {"abbrev": "BOS", "score": 3}})})())
    monkeypatch.setattr(nhl_mod, "requests",
                        type("R", (), {"get": staticmethod(lambda url: type("E", (), {"status_code": 404, "text": ""})())}))
    gamelog, teamlog = StatsNHL().parse_game("G1", "2024-10-31")
    assert gamelog == []
    assert teamlog == []
