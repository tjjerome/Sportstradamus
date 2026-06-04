"""Characterization pin for ``StatsNHL.update``.

``update`` is a ~285-line, CC-22 orchestration of NHL schedule + moneypuck +
dobbersports fetches with ZERO test coverage (integration fake-mode never
instantiates NHL). This pins the exact ``write_gamelog`` payload + upcoming_games
so the CC decomposition is proven behavior-preserving.

``parse_game`` is mocked here (it is its own decomposition unit, pinned separately);
this test exercises ``update``'s orchestration: the schedule while-loop, the
game-log assembly + archive enrichment, the dobbersports upcoming-goalies logic,
and the moneypuck skater/goalie season-summary column math.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

import sportstradamus.stats.base as base_mod
import sportstradamus.stats.nhl as nhl_mod
from sportstradamus.stats.nhl import StatsNHL


class _FakeDatetime(_dt.datetime):
    @classmethod
    def today(cls):
        return _dt.datetime(2024, 11, 1, 12, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return _dt.datetime(2024, 11, 1, 12, 0, 0, tzinfo=tz)


def _fake_scraper_get(url):
    if "api-web.nhle.com/v1/schedule" in url:
        return {"nextStartDate": "2024-11-02",
                "gameWeek": [{"date": "2024-10-31", "games": [{"id": "G1"}]}]}
    if "weekly-schedule/weekly-games" in url:
        return {"data": {"content": {"weeklyGames": [
            {"teamName": "Boston Bruins", "games": [
                {"gameId": "D1", "opponentTeam": {"name": "Montreal Canadiens"},
                 "teamType": "HOME"}]}]}}}
    if "/v1/game/" in url:
        return {"data": {"gameDate": "2024-11-05T19:00:00+00:00",
                         "predictedGoalies": {"HOME": [{"goalie": {"fullName": "Goalie Two"}}]}}}
    raise AssertionError(f"unexpected scraper url {url}")


_BIOS = pd.DataFrame([
    {"name": "Skater One", "playerId": 1001, "height": "6' 2\"", "weight": 200,
     "birthDate": "1995-01-01", "position": "C", "nationality": "USA",
     "primaryNumber": "10", "primaryPosition": "C", "team": "BOS"},
    {"name": "Goalie Two", "playerId": 2002, "height": "6' 4\"", "weight": 210,
     "birthDate": "1993-05-05", "position": "G", "nationality": "CAN",
     "primaryNumber": "30", "primaryPosition": "G", "team": "BOS"},
])
_SKATERS = pd.DataFrame([{
    "name": "Skater One", "playerId": 1001, "situation": "all", "team": "BOS", "position": "C",
    "onIce_fenwickPercentage": 0.55, "offIce_fenwickPercentage": 0.50,
    "icetime": 3600.0, "games_played": 2, "I_F_shifts": 20,
    "I_F_flurryScoreVenueAdjustedxGoals": 2.0, "I_F_shotAttempts": 10,
    "I_F_shotsOnGoal": 6, "I_F_goals": 3, "I_F_rebounds": 1, "I_F_freeze": 2,
    "I_F_oZoneShiftStarts": 8, "I_F_dZoneShiftStarts": 2, "I_F_flyShiftStarts": 4,
    "I_F_hits": 5, "I_F_takeaways": 3, "I_F_giveaways": 2,
    "I_F_primaryAssists": 4, "I_F_secondaryAssists": 2,
    "penalityMinutes": 6, "penalityMinutesDrawn": 4, "shotsBlockedByPlayer": 3,
}])
_GOALIES = pd.DataFrame([{
    "name": "Goalie Two", "playerId": 2002, "situation": "all", "team": "BOS", "position": "G",
    "icetime": 3600.0, "games_played": 2, "ongoal": 30, "goals": 3,
    "freeze": 5, "xFreeze": 4, "rebounds": 2, "xRebounds": 1, "flurryAdjustedxGoals": 2.5,
}])


class _Resp:
    def __init__(self, df):
        self.text = df.to_csv(index=False)
        self.status_code = 200


class _FakeRequests:
    @staticmethod
    def get(url, *a, **k):
        if "allPlayersLookup" in url:
            return _Resp(_BIOS)
        if "skaters.csv" in url:
            return _Resp(_SKATERS)
        if "goalies.csv" in url:
            return _Resp(_GOALIES)
        raise AssertionError(f"unexpected requests url {url}")


class _FakeArchive:
    default_totals = {"NHL": 2.674}

    def get_team_market_map(self, league, market, dates=None):
        return {}


def _fake_parse_game(self, gameId, gameDate):
    return (
        [{"gameId": "G1", "playerName": "Skater One", "gameDate": "2024-10-31",
          "team": "BOS", "position": "C", "goals": 1, "assists": 2, "points": 3}],
        [{"gameId": "G1", "team": "BOS", "gameDate": "2024-10-31", "goals": 3}],
    )


@pytest.fixture
def nhl_update(monkeypatch):
    monkeypatch.setattr(nhl_mod, "datetime", _FakeDatetime)
    monkeypatch.setattr(nhl_mod, "scraper",
                        type("S", (), {"get": staticmethod(_fake_scraper_get)})())
    monkeypatch.setattr(nhl_mod, "requests", _FakeRequests)
    monkeypatch.setattr(nhl_mod, "clean_data", False)
    monkeypatch.setattr(base_mod, "archive", _FakeArchive())
    monkeypatch.setattr(StatsNHL, "parse_game", _fake_parse_game)

    captured: dict = {}
    monkeypatch.setattr(nhl_mod, "write_gamelog",
                        lambda key, g, t, p: captured.update(
                            {"key": key, "gamelog": g.copy(), "teamlog": t.copy(), "players": p}))

    s = StatsNHL()
    s.gamelog = pd.DataFrame()
    s.teamlog = pd.DataFrame()
    s.players = {}
    s.update()
    captured["upcoming"] = s.upcoming_games
    return captured


def test_nhl_update_writes_nhl_key(nhl_update):
    assert nhl_update["key"] == "nhl"


def test_nhl_update_gamelog_row(nhl_update):
    g = nhl_update["gamelog"]
    assert g.shape == (1, 10)
    row = g.iloc[0].to_dict()
    assert row["gameId"] == "G1"
    assert row["playerName"] == "Skater One"
    assert row["team"] == "BOS"
    assert row["goals"] == 1
    assert row["points"] == 3
    assert row["moneyline"] == pytest.approx(0.5)
    assert row["totals"] == pytest.approx(2.674)


def test_nhl_update_teamlog_row(nhl_update):
    t = nhl_update["teamlog"]
    assert t.shape == (1, 4)
    assert t.iloc[0].to_dict() == {"gameId": "G1", "team": "BOS", "gameDate": "2024-10-31", "goals": 3}


def test_nhl_update_upcoming_games(nhl_update):
    assert nhl_update["upcoming"] == {
        "BOS": {"Opponent": "MTL", "Home": True, "Goalie": "Goalie Two"}
    }


def test_nhl_update_skater_profile(nhl_update):
    skater = nhl_update["players"][2024][1001]
    assert skater["playerName"] == "Skater One"
    assert skater["team"] == "BOS"
    assert skater["position"] == "C"
    assert skater["height"] == 74
    assert skater["Fenwick"] == pytest.approx(0.05)
    assert skater["timePerGame"] == pytest.approx(30.0)
    assert skater["timePerShift"] == pytest.approx(180.0)
    assert skater["xGoals"] == pytest.approx(0.2)
    assert skater["shotsOnGoal"] == pytest.approx(0.6)
    assert skater["goals"] == pytest.approx(0.1)
    assert skater["assists"] == pytest.approx(6.0)
    assert skater["hits"] == pytest.approx(5.0)
    assert skater["oZoneStarts"] == pytest.approx(0.8)
    assert skater["bmi"] == pytest.approx(200 / 74 / 74)


def test_nhl_update_goalie_profile(nhl_update):
    goalie = nhl_update["players"][2024][2002]
    assert goalie["playerName"] == "Goalie Two"
    assert goalie["position"] == "G"
    assert goalie["height"] == 76
    assert goalie["savePct"] == pytest.approx(0.9)
    assert goalie["goalsAgainst"] == pytest.approx((3 - 2.5) / 30)
    assert goalie["freezeAgainst"] == pytest.approx((5 - 4) / 27)
    assert goalie["reboundsAgainst"] == pytest.approx((2 - 1) / 27)
