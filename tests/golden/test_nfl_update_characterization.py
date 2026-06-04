"""Characterization pin for ``StatsNFL.update``.

``update`` is a CC-20, ~370-line multi-source ingest with ZERO coverage: it pulls
nflreadpy player stats + snap counts, nfl_data_py schedules + id maps, derives the
combined/fantasy-point columns, and runs a per-row backfill loop that fills
opponent/home/game-id from the schedule and calls ``parse_pbp`` for missing player
and team rows.

This pins the exact ``write_gamelog`` payload (gamelog row, teamlog row, players
frame) plus ``upcoming_games`` so the decomposition is proven behavior-preserving.
All fetch seams + ``parse_pbp`` are mocked; ``datetime.today`` is frozen to
2024-09-10 so the week-1 game is past (drives the backfill) and the week-2 game is
future (drives ``upcoming_games``). The seeded gamelog carries only the backfill
target columns, so the new player row's opponent is NaN and the backfill loop runs.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

import sportstradamus.stats.base as base_mod
import sportstradamus.stats.nfl as nfl_mod
from sportstradamus.stats.nfl import StatsNFL

_COLS = [
    "player_id", "player_display_name", "position_group", "team", "season", "week", "season_type",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions", "sacks",
    "sack_fumbles", "sack_fumbles_lost", "passing_2pt_conversions", "carries", "rushing_yards",
    "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost", "rushing_2pt_conversions",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_fumbles_lost", "receiving_2pt_conversions", "target_share", "air_yards_share", "wopr",
]


class _FakeDatetime(_dt.datetime):
    @classmethod
    def today(cls):
        return _dt.datetime(2024, 9, 10, 12, 0, 0)


def _player_stats_df():
    row = dict.fromkeys(_COLS, 0)
    row.update({
        "player_id": "00-1", "player_display_name": "Test QB", "position_group": "QB", "team": "KC",
        "season": 2024, "week": 1, "season_type": "REG", "completions": 25, "attempts": 35,
        "passing_yards": 300, "passing_tds": 3, "carries": 4, "rushing_yards": 20, "rushing_tds": 1,
    })
    row["passing_interceptions"] = 1
    row["sacks_suffered"] = 2
    return pd.DataFrame([row])


def _snaps_df():
    return pd.DataFrame([{
        "game_id": "2024_01_BUF_KC", "pfr_game_id": "x", "season": 2024, "game_type": "REG",
        "week": 1, "player": "Test QB", "pfr_player_id": "y", "position": "QB", "team": "KC",
        "opponent": "BUF", "offense_snaps": 60, "offense_pct": 0.95, "defense_snaps": 0,
        "defense_pct": 0.0, "st_snaps": 0, "st_pct": 0.0,
    }])


def _sched_df():
    return pd.DataFrame([
        {"away_team": "BUF", "home_team": "KC", "gameday": "2024-09-08", "weekday": "Sunday",
         "gametime": "13:00", "week": 1, "game_id": "2024_01_BUF_KC"},
        {"away_team": "NYJ", "home_team": "KC", "gameday": "2024-09-15", "weekday": "Sunday",
         "gametime": "16:00", "week": 2, "game_id": "2024_02_NYJ_KC"},
    ])


def _ids_df():
    return pd.DataFrame([{
        "name": "Test QB", "gsis_id": "00-1", "position": "QB", "team": "KCC",
        "age": 28.0, "height": 75, "weight": 220,
    }])


class _Loadable:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


class _FakeArchive:
    default_totals = {"NFL": 41.5}

    def get_team_market_map(self, league, market, dates=None):
        return {}


def _fake_parse_pbp(self, week, team, season, player=None):
    if player is not None:
        return {"air_yards": 250.0}
    return {"def_pass_yards": 240.0}


@pytest.fixture
def updated(monkeypatch):
    monkeypatch.setattr(nfl_mod, "datetime", _FakeDatetime)
    monkeypatch.setattr(nfl_mod, "nflr", type("NFLR", (), {
        "load_player_stats": staticmethod(lambda: _Loadable(_player_stats_df())),
        "load_snap_counts": staticmethod(lambda: _Loadable(_snaps_df())),
    }))
    monkeypatch.setattr(nfl_mod, "nfl", type("NFL", (), {
        "import_schedules": staticmethod(lambda yrs: _sched_df()),
        "import_ids": staticmethod(lambda: _ids_df()),
    }))
    monkeypatch.setattr(nfl_mod, "clean_data", False)
    monkeypatch.setattr(base_mod, "archive", _FakeArchive())
    monkeypatch.setattr(StatsNFL, "parse_pbp", _fake_parse_pbp)
    captured = {}
    monkeypatch.setattr(nfl_mod, "write_gamelog", lambda key, g, t, p: captured.update(
        {"key": key, "gamelog": g.copy(), "teamlog": t.copy(), "players": p}))

    s = StatsNFL()
    s.gamelog = pd.DataFrame(columns=["season", "week", "player id", "opponent", "home",
                                      "game id", "gameday"])
    s.teamlog = pd.DataFrame(columns=["season", "week", "team", "gameday"])
    s.players = pd.DataFrame()
    s.update()
    captured["upcoming_games"] = s.upcoming_games
    return captured


def test_update_write_key(updated):
    assert updated["key"] == "nfl"


def test_update_gamelog_shape_and_columns(updated):
    g = updated["gamelog"]
    assert g.shape == (1, 50)
    assert set(g.columns) == {
        "season", "week", "player id", "opponent", "home", "game id", "gameday",
        "player display name", "position", "team", "season type", "completions", "attempts",
        "passing yards", "passing tds", "interceptions", "sacks", "sack fumbles",
        "sack fumbles lost", "passing 2pt conversions", "carries", "rushing yards", "rushing tds",
        "rushing fumbles", "rushing fumbles lost", "rushing 2pt conversions", "receptions",
        "targets", "receiving yards", "receiving tds", "receiving fumbles", "receiving fumbles lost",
        "receiving 2pt conversions", "target share", "air yards share", "wopr", "snap pct",
        "fumbles", "fumbles lost", "yards", "qb yards", "tds", "qb tds",
        "fantasy points prizepicks", "fantasy points underdog", "fantasy points parlayplay",
        "yards per target", "moneyline", "totals", "air yards",
    }


def test_update_backfilled_schedule_fields(updated):
    row = updated["gamelog"].iloc[0]
    assert row["opponent"] == "BUF"
    assert row["home"] is True
    assert row["game id"] == "2024_01_BUF_KC"
    assert row["gameday"] == "2024-09-08"
    assert row["player display name"] == "Test Qb"
    assert row["position"] == "QB"
    assert row["team"] == "KC"


def test_update_derived_columns(updated):
    row = updated["gamelog"].iloc[0]
    assert row["interceptions"] == pytest.approx(1.0)
    assert row["sacks"] == pytest.approx(2.0)
    assert row["yards"] == pytest.approx(20.0)
    assert row["qb yards"] == pytest.approx(320.0)
    assert row["tds"] == pytest.approx(1.0)
    assert row["qb tds"] == pytest.approx(4.0)
    assert row["fumbles"] == pytest.approx(0.0)
    assert row["fumbles lost"] == pytest.approx(0.0)
    assert row["yards per target"] == pytest.approx(0.0)
    assert row["fantasy points prizepicks"] == pytest.approx(31.0)
    assert row["fantasy points underdog"] == pytest.approx(31.0)
    assert row["fantasy points parlayplay"] == pytest.approx(34.0)
    assert row["snap pct"] == pytest.approx(0.95)
    assert row["moneyline"] == pytest.approx(0.5)
    assert row["totals"] == pytest.approx(41.5)


def test_update_parse_pbp_backfill(updated):
    assert updated["gamelog"].iloc[0]["air yards"] == pytest.approx(250.0)
    team = updated["teamlog"]
    assert team.shape == (1, 5)
    trow = team.iloc[0]
    assert trow["team"] == "KC"
    assert trow["def_pass_yards"] == pytest.approx(240.0)
    assert trow["gameday"] == "2024-09-08"


def test_update_players_frame(updated):
    players = updated["players"]
    assert list(players.index) == ["Test Qb"]
    prow = players.loc["Test Qb"]
    assert prow["age"] == pytest.approx(28.0)
    assert prow["height"] == 75
    assert prow["weight"] == 220
    assert prow["team"] == "KC"
    assert prow["position"] == "QB"


def test_update_upcoming_games(updated):
    assert updated["upcoming_games"] == {
        "KC": {"Home": True, "Opponent": "NYJ", "gameday": "2024-09-15", "gametime": "Sun 16:00"},
        "NYJ": {"Home": False, "Opponent": "KC", "gameday": "2024-09-15", "gametime": "Sun 16:00"},
    }
