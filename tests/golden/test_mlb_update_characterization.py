"""Characterization pin for ``StatsMLB.update``.

CC-18 ingest with ZERO coverage (integration stubs ``update``; ``parse_game`` is its
own unit, tested separately, and is mocked here as a seam). It computes the schedule
window, builds the ``upcoming_games`` dict (probable pitchers + batting orders, with
the ``AZ`` -> ``ARI`` abbreviation fix and doubleheader ``game_num`` key-suffixing),
filters the Final game ids to parse, then trims old games and writes.

This pins the ``upcoming_games`` dict, the ids handed to ``parse_game`` (Final-only,
excluding exhibition ``E``/spring ``S``/all-star ``A`` types and already-seen ids),
and the retained gamelog/teamlog (4-year retention, ``AL``/``NL`` opponent drop,
keep-last dedup) so the decomposition is behavior-preserving. All ``statsapi`` calls
+ ``parse_game`` + ``write_gamelog`` are mocked; ``datetime.today`` is frozen.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

import sportstradamus.stats.mlb as mlb_mod
from sportstradamus.stats.mlb import StatsMLB


class _FakeDatetime(_dt.datetime):
    @classmethod
    def today(cls):
        return _dt.datetime(2025, 7, 15, 12, 0, 0)


def _games():
    return [
        {"status": "Scheduled", "game_type": "R", "away_id": 12, "home_id": 11, "game_id": 100,
         "game_num": 1, "away_probable_pitcher": "Pitcher X", "home_probable_pitcher": "Pitcher Y"},
        {"status": "Pre-Game", "game_type": "R", "away_id": 12, "home_id": 11, "game_id": 101,
         "game_num": 2, "away_probable_pitcher": "Pitcher Z", "home_probable_pitcher": "Pitcher W"},
        {"status": "Final", "game_type": "R", "away_id": 12, "home_id": 11, "game_id": 200,
         "game_num": 1, "away_probable_pitcher": "p", "home_probable_pitcher": "p"},
        {"status": "Final", "game_type": "E", "away_id": 12, "home_id": 11, "game_id": 201,
         "game_num": 1, "away_probable_pitcher": "p", "home_probable_pitcher": "p"},
        {"status": "Final", "game_type": "R", "away_id": 12, "home_id": 11, "game_id": "G1",
         "game_num": 1, "away_probable_pitcher": "p", "home_probable_pitcher": "p"},
        {"status": "Final", "game_type": "S", "away_id": 12, "home_id": 11, "game_id": 202,
         "game_num": 1, "away_probable_pitcher": "p", "home_probable_pitcher": "p"},
    ]


_TEAMS = {"teams": [{"id": 12, "abbreviation": "AZ"}, {"id": 11, "abbreviation": "BOS"},
                    {"id": 10, "abbreviation": "NYY"}]}
_BOX = {"playerInfo": {"a": {"id": 501, "fullName": "Bat One"},
                       "b": {"id": 502, "fullName": "Bat Two"}},
        "away": {"battingOrder": [501]}, "home": {"battingOrder": [502]}}


@pytest.fixture
def updated(monkeypatch):
    monkeypatch.setattr(mlb_mod, "datetime", _FakeDatetime)
    monkeypatch.setattr(mlb_mod, "clean_data", False)
    monkeypatch.setattr(mlb_mod, "mlb", type("M", (), {
        "schedule": staticmethod(lambda start_date, end_date: _games()),
        "get": staticmethod(lambda *a, **k: _TEAMS),
        "boxscore_data": staticmethod(lambda gid: _BOX),
    }))
    parsed: list = []
    monkeypatch.setattr(StatsMLB, "parse_game", lambda self, gid: parsed.append(gid))
    captured: dict = {}
    monkeypatch.setattr(mlb_mod, "write_gamelog", lambda key, g, t, p: captured.update(
        {"key": key, "gamelog": g.copy(), "teamlog": t.copy()}))

    s = StatsMLB()
    s.season_start = _dt.date(2025, 5, 26)
    s.gamelog = pd.DataFrame([
        {"gameId": "G1", "playerId": 1, "gameDate": "2025-07-05", "opponent": "NYY",
         "playerName": "Player A", "value": 1.0},
        {"gameId": "G2", "playerId": 2, "gameDate": "2020-01-23", "opponent": "BOS",
         "playerName": "Player B", "value": 2.0},
        {"gameId": "G3", "playerId": 3, "gameDate": "2025-07-10", "opponent": "AL",
         "playerName": "Player C", "value": 3.0},
        {"gameId": "G1", "playerId": 1, "gameDate": "2025-07-05", "opponent": "NYY",
         "playerName": "Player A", "value": 99.0},
    ])
    s.teamlog = pd.DataFrame([
        {"gameId": "G1", "team": "NYY", "gameDate": "2025-07-05", "tvalue": 10.0},
        {"gameId": "G2", "team": "BOS", "gameDate": "2020-01-23", "tvalue": 20.0},
    ])
    s.players = pd.DataFrame()
    s.update()
    return {"upcoming": s.upcoming_games, "parsed": parsed, **captured}


def test_update_upcoming_games(updated):
    assert updated["upcoming"] == {
        "ARI": {"Pitcher": "Pitcher X", "Home": False, "Opponent": "BOS",
                "Opponent Pitcher": "Pitcher Y", "Batting Order": ["Bat One"]},
        "BOS": {"Pitcher": "Pitcher Y", "Home": True, "Opponent": "ARI",
                "Opponent Pitcher": "Pitcher X", "Batting Order": ["Bat Two"]},
        "ARI2": {"Pitcher": "Pitcher Z", "Home": False, "Opponent": "BOS",
                 "Opponent Pitcher": "Pitcher W", "Batting Order": ["Bat One"]},
        "BOS2": {"Pitcher": "Pitcher W", "Home": True, "Opponent": "ARI",
                 "Opponent Pitcher": "Pitcher Z", "Batting Order": ["Bat Two"]},
    }


def test_update_parsed_final_ids(updated):
    assert updated["parsed"] == [200]


def test_update_retained_gamelog(updated):
    g = updated["gamelog"]
    assert g["gameId"].tolist() == ["G1"]
    assert g["value"].tolist() == [99.0]


def test_update_retained_teamlog(updated):
    t = updated["teamlog"]
    assert t["gameId"].tolist() == ["G1"]
    assert t["tvalue"].tolist() == [10.0]
