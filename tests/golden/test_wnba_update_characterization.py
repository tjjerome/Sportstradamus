"""Characterization pin for ``StatsWNBA.update``.

``update`` is a ~330-line, CC-33 orchestration of stats.wnba.com fetches that has
ZERO test coverage (integration fake-mode stubs ``update`` to a no-op, so the
end-to-end pipeline never exercises it). This test drives it through BOTH the
player-stat assembly and the team-log assembly with minimal synthetic API data,
mocking every I/O seam (``nba_client.fetch`` per-endpoint, ``fetch_upcoming_games``,
``write_gamelog``, the base ``archive``, frozen ``today``), and pins the exact
``write_gamelog`` payload so the CC decomposition is proven behavior-preserving.

The pinned values were captured from the pre-refactor code: derived columns
(PRA/BLST/fantasy-points/per-40), the bios+shot-zone player profile, the team
home/away OPP_ pairing, and the archive-default moneyline/totals enrichment.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

import sportstradamus.stats.base as base_mod
import sportstradamus.stats.wnba as wnba_mod
from sportstradamus.stats import nba_client
from sportstradamus.stats.wnba import StatsWNBA


class _FakeDatetime(_dt.datetime):
    @classmethod
    def today(cls):
        return cls(2026, 6, 4, 12, 0, 0)


class _Resp:
    def __init__(self, normalized=None, raw=None):
        self._n = normalized
        self._r = raw

    def get_normalized_dict(self):
        return self._n

    def get_dict(self):
        return self._r


_PLAYER_INDEX = [
    {"PERSON_ID": 1, "PLAYER_NAME": "Test Player", "TEAM_ABBREVIATION": "NY", "POSITION": "G-F"},
]
_BIOS = {
    "PLAYER_NAME": ["Test Player"],
    "PLAYER_ID": [1],
    "PLAYER_HEIGHT_INCHES": [72.0],
    "PLAYER_WEIGHT": ["180"],
    "TEAM_ABBREVIATION": ["NYL"],
    "AGE": [25.0],
    "USG_PCT": [0.25],
    "TS_PCT": [0.55],
}
# shot row: indices 7,10,13,16,19,22,25,28 are summed for fga; name=row[1], team=row[3]
_SHOT_ROW = [0, "Test Player", 0, "NYL"] + [0] * 31
for _i in (7, 10, 13, 16, 19, 22, 25, 28):
    _SHOT_ROW[_i] = 2.0
_SHOT_DATA = {"rowSet": [_SHOT_ROW]}


def _player_log_base():
    return [
        {
            "PLAYER_ID": 1,
            "GAME_ID": "G1",
            "PLAYER_NAME": "Test Player",
            "TEAM_ABBREVIATION": "NY",
            "MATCHUP": "NY vs. CON",
            "GAME_DATE": "2026-05-20",
            "MIN": 30.0,
            "PTS": 20,
            "REB": 8,
            "AST": 5,
            "BLK": 2,
            "STL": 1,
            "TOV": 3,
            "FTM": 4,
            "FGM": 8,
            "FGA": 15,
            "FG3M": 2,
            "FG3A": 6,
            "OREB": 2,
            "DREB": 6,
            "BLKA": 1,
            "FTA": 5,
            "PF": 2,
            "PFD": 3,
            "WL": "W",
        }
    ]


def _team_log_base():
    return [
        {
            "TEAM_ID": 10,
            "GAME_ID": "G1",
            "TEAM_ABBREVIATION": "NY",
            "MATCHUP": "NY vs. CON",
            "GAME_DATE": "2026-05-20",
            "PTS": 88,
            "FTM": 14,
            "FGA": 70,
            "BLK": 5,
            "BLKA": 4,
        },
        {
            "TEAM_ID": 20,
            "GAME_ID": "G1",
            "TEAM_ABBREVIATION": "CON",
            "MATCHUP": "CON @ NY",
            "GAME_DATE": "2026-05-20",
            "PTS": 80,
            "FTM": 10,
            "FGA": 65,
            "BLK": 3,
            "BLKA": 5,
        },
    ]


def _fake_fetch(endpoint, host=None, **params):
    name = endpoint.__name__
    measure = params.get("measure_type_player_game_logs_nullable")
    if name == "PlayerIndex":
        return _Resp(normalized={"PlayerIndex": _PLAYER_INDEX})
    if name == "LeagueDashPlayerBioStats":
        return _Resp(normalized={"LeagueDashPlayerBioStats": _BIOS})
    if name == "LeagueDashPlayerShotLocations":
        return _Resp(raw={"resultSets": _SHOT_DATA})
    if name == "PlayerGameLogs":
        if measure == "Advanced":
            return _Resp(
                normalized={
                    "PlayerGameLogs": [
                        {"PLAYER_ID": 1, "GAME_ID": "G1", "OFF_RATING": 110.0, "DEF_RATING": 100.0}
                    ]
                }
            )
        if measure == "Usage":
            return _Resp(
                normalized={"PlayerGameLogs": [{"PLAYER_ID": 1, "GAME_ID": "G1", "USG_PCT": 0.25}]}
            )
        return _Resp(normalized={"PlayerGameLogs": _player_log_base()})
    if name == "TeamGameLogs":
        if measure == "Advanced":
            return _Resp(
                normalized={
                    "TeamGameLogs": [
                        {"TEAM_ID": 10, "GAME_ID": "G1", "OFF_RATING": 112.0},
                        {"TEAM_ID": 20, "GAME_ID": "G1", "OFF_RATING": 104.0},
                    ]
                }
            )
        if measure == "Scoring":
            return _Resp(
                normalized={
                    "TeamGameLogs": [
                        {"TEAM_ID": 10, "GAME_ID": "G1", "PCT_PTS_2PT": 0.5},
                        {"TEAM_ID": 20, "GAME_ID": "G1", "PCT_PTS_2PT": 0.45},
                    ]
                }
            )
        return _Resp(normalized={"TeamGameLogs": _team_log_base()})
    raise AssertionError(f"unexpected endpoint {name}")


class _FakeArchive:
    default_totals = {"WNBA": 81.667}

    def get_team_market_map(self, league, market, dates=None):
        return {}


_GAMELOG_COLS = [
    "PLAYER_ID",
    "GAME_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "GAME_DATE",
    "POS",
    "HOME",
    "OPP",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "BLST",
    "PRA",
    "fantasy points prizepicks",
    "FGA_40",
    "OFF_RATING",
    "OREB",
    "moneyline",
    "totals",
]
_TEAMLOG_COLS = [
    "TEAM_ID",
    "GAME_ID",
    "TEAM_ABBREVIATION",
    "OPP",
    "HOME",
    "GAME_DATE",
    "FTM",
    "FGA",
    "BLK",
    "BLKA",
    "PTS",
    "FTR",
    "BLK_RATIO",
    "OPP_PTS",
]


def _seed_gamelog():
    row = dict.fromkeys(_GAMELOG_COLS, 0)
    row.update(
        {
            "PLAYER_ID": 99,
            "GAME_ID": "G0",
            "PLAYER_NAME": "Old Player",
            "TEAM_ABBREVIATION": "LAS",
            "GAME_DATE": "2026-05-18",
            "POS": "C",
            "OFF_RATING": 100.0,
        }
    )
    return pd.DataFrame([row], columns=_GAMELOG_COLS)


def _seed_teamlog():
    row = dict.fromkeys(_TEAMLOG_COLS, 0)
    row.update(
        {"TEAM_ID": 30, "GAME_ID": "G0", "TEAM_ABBREVIATION": "LAS", "GAME_DATE": "2026-05-18"}
    )
    return pd.DataFrame([row], columns=_TEAMLOG_COLS)


@pytest.fixture
def captured_update(monkeypatch):
    monkeypatch.setattr(wnba_mod, "datetime", _FakeDatetime)
    monkeypatch.setattr(wnba_mod, "fetch_upcoming_games", lambda *a, **k: {})
    monkeypatch.setattr(wnba_mod, "clean_data", False)
    monkeypatch.setattr(nba_client, "fetch", _fake_fetch)
    monkeypatch.setattr(base_mod, "archive", _FakeArchive())
    # Pin the season gate open so the pipeline runs regardless of the host's
    # real league_activity.json snapshot or the calendar.
    monkeypatch.setattr(
        base_mod.odds_budget, "update_window_open", lambda league, season_start: True
    )

    captured: dict = {}
    monkeypatch.setattr(
        wnba_mod,
        "write_gamelog",
        lambda key, gamelog, teamlog, players: captured.update(
            {"key": key, "gamelog": gamelog.copy(), "teamlog": teamlog.copy(), "players": players}
        ),
    )

    s = StatsWNBA()
    s.gamelog = _seed_gamelog()
    s.teamlog = _seed_teamlog()
    s.players = {}
    s.update()
    return captured


def test_wnba_update_writes_wnba_key(captured_update):
    assert captured_update["key"] == "wnba"


def test_wnba_update_player_row(captured_update):
    g = captured_update["gamelog"]
    assert g.shape == (2, 20)
    new = g[g.PLAYER_ID == 1].iloc[0].to_dict()
    assert new["GAME_ID"] == "G1"
    assert new["PLAYER_NAME"] == "Test Player"
    assert new["TEAM_ABBREVIATION"] == "NYL"
    assert new["GAME_DATE"] == "2026-05-20"
    assert new["POS"] == "G"
    assert new["HOME"] == 1
    assert new["OPP"] == "CON"
    assert new["MIN"] == pytest.approx(30.0)
    assert new["PTS"] == 20
    assert new["REB"] == 8
    assert new["AST"] == 5
    assert new["BLST"] == 3
    assert new["PRA"] == 33
    assert new["fantasy points prizepicks"] == pytest.approx(43.1)
    assert new["FGA_40"] == pytest.approx(20.0)
    assert new["OFF_RATING"] == pytest.approx(110.0)
    assert new["OREB"] == 2
    assert new["moneyline"] == pytest.approx(0.5)
    assert new["totals"] == pytest.approx(81.667)


def test_wnba_update_team_rows(captured_update):
    t = captured_update["teamlog"]
    assert t.shape == (3, 14)
    rows = {r["TEAM_ID"]: r.to_dict() for _, r in t[t.GAME_ID == "G1"].iterrows()}
    home = rows[10]
    assert home["TEAM_ABBREVIATION"] == "NYL"
    assert home["OPP"] == "CON"
    assert home["HOME"] == 1
    assert home["FTR"] == pytest.approx(0.2)
    assert home["BLK_RATIO"] == pytest.approx(1.25)
    assert home["OPP_PTS"] == 80
    away = rows[20]
    assert away["TEAM_ABBREVIATION"] == "CON"
    assert away["OPP"] == "NYL"
    assert away["HOME"] == 0
    assert away["FTR"] == pytest.approx(0.153846, rel=1e-5)
    assert away["BLK_RATIO"] == pytest.approx(0.6)
    assert away["OPP_PTS"] == 88


def test_wnba_update_player_profile(captured_update):
    profile = captured_update["players"][2026]["NYL"]["Test Player"]
    assert profile["POS"] == "G"
    assert profile["AGE"] == pytest.approx(25.0)
    assert profile["PLAYER_HEIGHT_INCHES"] == pytest.approx(72.0)
    assert profile["PLAYER_BMI"] == pytest.approx(180 / 72 / 72)
    assert profile["USG_PCT"] == pytest.approx(0.25)
    assert profile["TS_PCT"] == pytest.approx(0.55)
    for zone in ("RA_PCT", "ITP_PCT", "MR_PCT", "C3_PCT", "B3_PCT"):
        assert profile[zone] == pytest.approx(0.125)
