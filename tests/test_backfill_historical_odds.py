"""Pin the apikey shapes on both backfill fetch paths (``get_props`` takes the
key string itself, ``get_moneylines`` the whole creds dict — the probe once
passed the dict into ``get_props``, a guaranteed 401 on every paid dry run),
plus the alternate-market / close-layer modes."""

import datetime
import importlib.resources as pkg_resources
import json

from sportstradamus import data
from sportstradamus.moneylines import (
    _archive_event_props,
    _historical_observed_at,
)
from sportstradamus.scripts import backfill_historical_odds as bho

FAKE_KEYS = {bho.HISTORICAL_KEY_NAME: "key-string", "odds_api": "other-key"}
PROPS = {"NFL": {"player_pass_yds": "passing yards"}}
DATES = [datetime.datetime(2023, 10, 29)]


def test_probe_passes_key_string(monkeypatch):
    seen = []
    monkeypatch.setattr(bho, "get_props", lambda archive, apikey, *a, **k: seen.append(apikey))
    bho._probe(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6)
    assert seen == ["key-string"]


def test_backfill_passes_key_string_and_creds_dict(monkeypatch, tmp_path):
    class FakeArchive:
        def write(self):
            pass

    props_keys, moneyline_keys = [], []
    monkeypatch.setattr(bho, "Archive", FakeArchive)
    monkeypatch.setattr(bho, "PROGRESS_PATH", tmp_path / "progress.json")
    monkeypatch.setattr(
        bho, "get_props", lambda archive, apikey, *a, **k: props_keys.append(apikey)
    )
    monkeypatch.setattr(
        bho, "get_moneylines", lambda archive, apikey, **k: moneyline_keys.append(apikey)
    )
    bho._backfill(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6)
    assert props_keys == ["key-string"]
    assert moneyline_keys == [FAKE_KEYS]


def test_job_sig_modes_are_distinct_and_base_format_stable():
    assert bho._job_sig("NFL", PROPS) == "NFL|passing yards"
    assert bho._job_sig("NFL", PROPS, "alt") == "NFL|passing yards|alt"
    assert bho._job_sig("NFL", PROPS, "altonly", "close") == "NFL|passing yards|altonly|close"
    assert bho._job_sig("NFL", PROPS, None, "close") == "NFL|passing yards|close"


def test_alt_market_names_resolve_to_stat_map():
    with open(pkg_resources.files(data) / "config" / "stat_map.json") as f:
        stat_map = json.load(f)["Odds API"]
    for league, alts in bho.ALT_MARKET_KEYS.items():
        assert set(alts.values()) <= set(stat_map[league].values()), league
        assert all(key.endswith("_alternate") for key in alts), league


def test_backfill_close_layer_skips_lines_and_stamps_close_hour(monkeypatch, tmp_path):
    class FakeArchive:
        def write(self):
            pass

    calls, moneyline_calls = [], []
    monkeypatch.setattr(bho, "Archive", FakeArchive)
    monkeypatch.setattr(bho, "PROGRESS_PATH", tmp_path / "progress.json")
    monkeypatch.setattr(bho, "get_props", lambda archive, apikey, *a, **k: calls.append(k))
    monkeypatch.setattr(
        bho, "get_moneylines", lambda archive, apikey, **k: moneyline_calls.append(k)
    )
    bho._backfill(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 23, "alt", "close")
    assert moneyline_calls == []
    assert [c["observed_at_hour"] for c in calls] == [bho.CLOSE_LAYER_HOUR]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert list(progress) == ["NFL|passing yards|alt|close"]


def test_historical_observed_at_hour_override():
    d = datetime.datetime(2025, 6, 10)
    assert _historical_observed_at(True, d).hour == 1
    assert _historical_observed_at(True, d, bho.CLOSE_LAYER_HOUR).hour == 23
    assert _historical_observed_at(False, d, 23) is None


def _alt_game(market_key, extra_markets=()):
    outcomes = [
        {"description": "Aaron Judge", "name": "Over", "point": p, "price": pr}
        for p, pr in ((0.5, 1.4), (1.5, 2.6), (2.5, 6.0))
    ]
    markets = [{"key": market_key, "outcomes": outcomes}, *extra_markets]
    return {
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "bookmakers": [{"key": "draftkings", "markets": markets}],
    }


class _RecordingArchive:
    def __init__(self):
        self.merged, self.ladders = [], []

    def merge_player_books(self, league, market, date, player, book_evs, lines, **kwargs):
        self.merged.append((market, player, dict(book_evs)))

    def add_ladder(self, league, market, date, entity, book, rungs, observed_at=None):
        self.ladders.append((market, entity, book, list(rungs)))

    def set_team_books(self, *args, **kwargs):
        pass


def test_alternate_market_feeds_ladder_only():
    props = {"MLB": {"batter_hits": "hits", "batter_hits_alternate": "hits"}}
    archive = _RecordingArchive()
    _archive_event_props(
        archive, _alt_game("batter_hits_alternate"), "MLB", props, "2025-06-10", None
    )
    assert archive.merged == []
    assert len(archive.ladders) == 1
    market, entity, book, rungs = archive.ladders[0]
    assert (market, entity, book) == ("hits", "Aaron Judge", "draftkings")
    assert len(rungs) == 3


def test_standard_market_still_prices_ev():
    props = {"MLB": {"batter_hits": "hits", "batter_hits_alternate": "hits"}}
    game = _alt_game(
        "batter_hits",
        extra_markets=[
            {
                "key": "batter_hits_alternate",
                "outcomes": [
                    {"description": "Aaron Judge", "name": "Over", "point": 3.5, "price": 12.0}
                ],
            }
        ],
    )
    archive = _RecordingArchive()
    _archive_event_props(archive, game, "MLB", props, "2025-06-10", None)
    assert [(m, p) for m, p, _ in archive.merged] == [("hits", "Aaron Judge")]
    (_market, _entity, _book, rungs) = archive.ladders[0]
    assert len(rungs) == 4
