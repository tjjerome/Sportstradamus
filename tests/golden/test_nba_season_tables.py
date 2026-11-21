"""The NBA comp pool reads the running season's bio and shot tables as of its date's eve.

The stored season holds the tables of its last update, after every training row's date, so a
rebuilt row's comps read later usage and shot zones than the served row's did (study §14 item 6).
The pool reads the tables through the day before its date instead, stored per day so a rebuild
fetches each day once, and serving reads yesterday's, the ones the update stores. The client is
faked: no test reaches stats.nba.com.
"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from sportstradamus.stats import StatsNBA, StatsWNBA, nba, nba_client, nba_season_tables
from sportstradamus.stats.nba_client import NBAStatsError

D = date(2025, 12, 10)
_ZONES = 8
_SHOT_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "NICKNAME"]
_SHOT_COLUMNS += ["FGM", "FGA", "FG_PCT"] * _ZONES
# A's attempts per zone: restricted area, paint, mid-range, two corners, above the break,
# backcourt and both corners; the profile counts the first three, the sixth and the last.
_A_ATTEMPTS = (3.0, 1.0, 2.0, 0.5, 0.5, 1.0, 0.0, 1.0)
_STORED = dict.fromkeys(nba._DATED_COMP_FIELDS, 0.99) | {"POS": "P", "POST_PCT": 0.3}
_PLAYERS = {
    "2024-25": {"TTT": {"A": _STORED | {"USG_PCT": 0.2}, "C": _STORED | {"POS": "C"}}},
    "2025-26": {"TTT": {"A": _STORED, "C": _STORED | {"POS": "C"}}},
}


class _FakeClient:
    """Answers the two dated tables with A alone (C has not played yet) and records each call."""

    def __init__(self, listed=("A",)):
        self.listed = listed
        self.calls = []

    def __call__(self, endpoint_cls, *, host=nba_client.NBA_HOST, **params):
        self.calls.append((endpoint_cls.__name__, params))
        bios = [{"PLAYER_NAME": name, "USG_PCT": 0.25, "TS_PCT": 0.55} for name in self.listed]
        zones = [v for fga in _A_ATTEMPTS for v in (fga / 2, fga, 0.5)]
        shots = {
            "headers": [{}, {"columnNames": _SHOT_COLUMNS}],
            "rowSet": [[1, name, 2, "TTT", 25.0, name, *zones] for name in self.listed],
        }
        synergy = {"resultSets": [{"headers": ["PLAYER_ID", "POSS_PCT", "PPP"], "rowSet": []}]}
        return SimpleNamespace(
            get_normalized_dict=lambda: {"LeagueDashPlayerBioStats": bios},
            get_dict=lambda: (
                synergy if "Synergy" in endpoint_cls.__name__ else {"resultSets": shots}
            ),
        )

    def dated(self):
        return [
            (name, params["season"], params["date_to_nullable"])
            for name, params in self.calls
            if "date_to_nullable" in params
        ]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(nba_season_tables, "SEASON_TABLES_DIR", tmp_path)
    fake = _FakeClient()
    monkeypatch.setattr(nba_client, "fetch", fake)
    return fake


def _stats(cls=StatsNBA, players=_PLAYERS):
    stats = cls()
    stats.players = {
        season: {t: dict(r) for t, r in teams.items()} for season, teams in players.items()
    }
    return stats


def _freeze_today(monkeypatch, today):
    class _Frozen(datetime):
        @classmethod
        def today(cls):
            return datetime(today.year, today.month, today.day)

    monkeypatch.setattr(nba, "datetime", _Frozen)


def _asked(day):
    return [
        ("LeagueDashPlayerBioStats", "2025-26", day.strftime("%m/%d/%Y")),
        ("LeagueDashPlayerShotLocations", "2025-26", day.strftime("%m/%d/%Y")),
    ]


def test_a_training_rows_pool_reads_the_running_season_through_the_day_before(client):
    stats = _stats()

    pool = stats._player_seasons_through(D)

    assert client.dated() == _asked(D - timedelta(days=1))
    a = pool["TTT"]["A"]
    assert (a["USG_PCT"], a["TS_PCT"]) == (0.25, 0.55)
    assert a["ITP_PCT"] == pytest.approx((3 + 1) / (3 + 1 + 2 + 1 + 1))
    assert (a["POS"], a["POST_PCT"]) == ("P", 0.3)
    assert pool["TTT"]["C"] == _PLAYERS["2024-25"]["TTT"]["C"]
    assert stats.players == _PLAYERS


def test_a_day_is_fetched_once(client):
    _stats()._player_seasons_through(D)
    _stats()._player_seasons_through(D + timedelta(days=1))
    _stats()._player_seasons_through(D)

    assert client.dated() == _asked(D - timedelta(days=1)) + _asked(D)


def test_serving_reads_the_tables_the_update_stored_yesterday(client, monkeypatch):
    _freeze_today(monkeypatch, D)
    stats = _stats()

    stats._fetch_league_endpoints(D - timedelta(days=1))
    stats._player_seasons_through(D)
    stats._player_seasons_through(D + timedelta(days=1))

    assert client.dated() == _asked(D - timedelta(days=1))


def test_before_the_seasons_first_game_the_pool_is_past_seasons(client):
    client.listed = ()

    assert _stats()._player_seasons_through(D) == _PLAYERS["2024-25"]


def test_a_failed_read_pools_the_stored_season(monkeypatch, tmp_path):
    monkeypatch.setattr(nba_season_tables, "SEASON_TABLES_DIR", tmp_path)

    def blocked(*_, **__):
        raise NBAStatsError("blocked")

    monkeypatch.setattr(nba_client, "fetch", blocked)

    assert _stats()._player_seasons_through(D) == _PLAYERS["2025-26"]
    assert not list(tmp_path.iterdir())


def test_the_wnba_pool_reads_its_season_as_stored(client):
    players = {2025: {"LVA": {"W": {"POS": "G", "USG_PCT": 0.3}}}}

    assert _stats(StatsWNBA, players)._player_seasons_through(date(2025, 7, 1)) == players[2025]
    assert not client.calls
