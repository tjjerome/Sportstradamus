"""Pin the apikey shapes on both backfill fetch paths (``get_props`` takes the
key string itself, ``get_moneylines`` the whole creds dict — the probe once
passed the dict into ``get_props``, a guaranteed 401 on every paid dry run),
plus the alternate-market / close-layer modes."""

import datetime
import importlib.resources as pkg_resources
import json

import pytest
from click.testing import CliRunner

from sportstradamus import data
from sportstradamus.moneylines import (
    _archive_event_props,
    _historical_observed_at,
)
from sportstradamus.scripts import backfill_historical_odds as bho

FAKE_KEYS = {bho.HISTORICAL_KEY_NAME: "key-string", "odds_api": "other-key"}
PROPS = {"NFL": {"player_pass_yds": "passing yards"}}
DATES = [datetime.datetime(2023, 10, 29)]


class _FakeArchive:
    def write(self):
        pass


@pytest.fixture
def fake_fetches(monkeypatch, tmp_path):
    """Stub both paid fetch paths, recording ``(apikey, kwargs)`` per call, and
    point the resume log at ``tmp_path / "progress.json"``."""
    prop_calls, moneyline_calls = [], []
    monkeypatch.setattr(bho, "Archive", _FakeArchive)
    monkeypatch.setattr(bho, "PROGRESS_PATH", tmp_path / "progress.json")
    monkeypatch.setattr(
        bho, "get_props", lambda archive, apikey, *a, **k: prop_calls.append((apikey, k))
    )
    monkeypatch.setattr(
        bho, "get_moneylines", lambda archive, apikey, **k: moneyline_calls.append((apikey, k))
    )
    return prop_calls, moneyline_calls


def test_probe_passes_key_string(monkeypatch):
    seen = []
    monkeypatch.setattr(bho, "get_props", lambda archive, apikey, *a, **k: seen.append(apikey))
    bho._probe(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6)
    assert seen == ["key-string"]


def test_backfill_passes_key_string_and_creds_dict(fake_fetches):
    prop_calls, moneyline_calls = fake_fetches
    bho._backfill(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6)
    props_keys = [apikey for apikey, _ in prop_calls]
    moneyline_keys = [apikey for apikey, _ in moneyline_calls]
    assert props_keys == ["key-string"]
    assert moneyline_keys == [FAKE_KEYS]


def test_job_sig_modes_are_distinct_and_base_format_stable():
    assert bho._job_sig("NFL", PROPS) == "NFL|passing yards"
    assert bho._job_sig("NFL", PROPS, "alt") == "NFL|passing yards|alt"
    assert bho._job_sig("NFL", PROPS, "altonly", "close") == "NFL|passing yards|altonly|close"
    assert bho._job_sig("NFL", PROPS, None, "close") == "NFL|passing yards|close"


def test_backfill_game_lines_only_skips_the_paid_prop_calls(fake_fetches, tmp_path):
    """Repairing a window where game lines went unfetched must not re-buy props:
    the per-event prop calls are the whole cost, the h2h/totals/spreads call is one."""
    prop_calls, moneyline_calls = fake_fetches
    bho._backfill(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6, game_lines_only=True)
    assert prop_calls == []
    assert len(moneyline_calls) == 1
    # Its own resume key, so a game-lines run can't mark dates done for the full job.
    assert list(json.loads((tmp_path / "progress.json").read_text())) == [
        "NFL|passing yards|gamelines"
    ]


def test_alt_market_names_resolve_to_stat_map():
    with open(pkg_resources.files(data) / "config" / "stat_map.json") as f:
        stat_map = json.load(f)["Odds API"]
    for league, alts in bho.ALT_MARKET_KEYS.items():
        assert set(alts.values()) <= set(stat_map[league].values()), league
        assert all(key.endswith("_alternate") for key in alts), league


def test_backfill_close_layer_skips_lines_and_stamps_close_hour(fake_fetches, tmp_path):
    prop_calls, moneyline_calls = fake_fetches
    bho._backfill(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 23, "alt", "close")
    calls = [kwargs for _, kwargs in prop_calls]
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


def test_backfill_props_only_skips_the_game_line_call(fake_fetches, tmp_path):
    prop_calls, moneyline_calls = fake_fetches
    bho._backfill(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6, props_only=True)
    assert moneyline_calls == []
    assert len(prop_calls) == 1
    assert list(json.loads((tmp_path / "progress.json").read_text())) == [
        "NFL|passing yards|propsonly"
    ]


def test_job_sig_props_only_is_its_own_resume_key():
    assert bho._job_sig("NFL", PROPS, props_only=True) == "NFL|passing yards|propsonly"


def _invoke_main(monkeypatch, fake_fetches, args):
    prop_calls, moneyline_calls = fake_fetches
    seeds = []
    keys = {**FAKE_KEYS, "odds_api_alt": "alt-key-string"}
    monkeypatch.setattr(
        bho, "_load_keys_and_props", lambda league, markets, alt_mode: (dict(keys), PROPS)
    )
    monkeypatch.setattr(
        bho, "_game_dates", lambda league, markets, start, end: seeds.append(list(markets)) or DATES
    )
    result = CliRunner().invoke(bho.main, ["--start", "2024-09-05", "--end", "2024-09-05", *args])
    props_keys = [apikey for apikey, _ in prop_calls]
    moneyline_keys = [apikey for apikey, _ in moneyline_calls]
    return result, props_keys, moneyline_keys, seeds


def test_main_key_name_dates_from_and_props_only(monkeypatch, fake_fetches):
    result, props_keys, moneyline_keys, seeds = _invoke_main(
        monkeypatch,
        fake_fetches,
        ["--key-name", "odds_api_alt", "--dates-from", "Moneyline, Totals", "--props-only"],
    )
    assert result.exit_code == 0, result.output
    assert seeds == [["Moneyline", "Totals"]]
    assert props_keys == ["alt-key-string"]
    assert moneyline_keys == []


def test_main_key_name_funds_the_game_line_call_too(monkeypatch, fake_fetches):
    result, _, moneyline_keys, seeds = _invoke_main(
        monkeypatch, fake_fetches, ["--key-name", "odds_api_alt"]
    )
    assert result.exit_code == 0, result.output
    assert seeds == [["passing yards"]]
    assert [k[bho.HISTORICAL_KEY_NAME] for k in moneyline_keys] == ["alt-key-string"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--props-only", "--game-lines-only"], "opposites"),
        (["--props-only", "--layer", "close", "--snapshot-hour", "23"], "already props-only"),
        (["--dates-from", "Moneyline", "--check-all"], "walks the calendar"),
    ],
)
def test_main_rejects_contradictory_fetch_flags(monkeypatch, fake_fetches, args, message):
    result, *_ = _invoke_main(monkeypatch, fake_fetches, args)
    assert result.exit_code != 0
    assert message in result.output


def test_combo_keys_are_mapped_for_nfl():
    with open(pkg_resources.files(data) / "config" / "stat_map.json") as f:
        nfl = json.load(f)["Odds API"]["NFL"]
    assert nfl["player_rush_reception_yds"] == "yards"
    assert nfl["player_pass_rush_yds"] == "qb yards"
