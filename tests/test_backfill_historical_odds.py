"""Pin the apikey shapes on both backfill fetch paths (``get_props`` takes the
key string itself, ``get_moneylines`` the whole creds dict — the probe once
passed the dict into ``get_props``, a guaranteed 401 on every paid dry run),
plus the alternate-market / close-layer modes."""

import datetime
import importlib.resources as pkg_resources
import json
from types import SimpleNamespace

import duckdb
import pytest
from click.testing import CliRunner

from sportstradamus import data
from sportstradamus.moneylines import (
    OddsAPIAuthError,
    _archive_event_props,
    _historical_observed_at,
)
from sportstradamus.scripts import backfill_historical_odds as bho

FAKE_KEYS = {bho.HISTORICAL_KEY_NAME: "key-string", "odds_api": "other-key"}
PROPS = {"NFL": {"player_pass_yds": "passing yards"}}
DATES = [datetime.datetime(2023, 10, 29)]
# Opening week of 2024: a one-game Thursday, a fourteen-game Sunday, a one-game Monday.
THURSDAY, SUNDAY, MONDAY = (datetime.date(2024, 9, day) for day in (5, 8, 9))
WEEK_ONE = [THURSDAY, SUNDAY, MONDAY]
# Monday's game was the 49ers', and the archive holds no Moneyline rows for their games.
ARCHIVED_GAMES = {THURSDAY: 1, SUNDAY: 14}


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


def _invoke_main(monkeypatch, fake_fetches, args, dates=DATES):
    prop_calls, moneyline_calls = fake_fetches
    seeds = []
    keys = {**FAKE_KEYS, "odds_api_alt": "alt-key-string"}
    monkeypatch.setattr(
        bho, "_load_keys_and_props", lambda league, markets, alt_mode: (dict(keys), PROPS)
    )
    monkeypatch.setattr(
        bho, "_game_dates", lambda league, markets, start, end: seeds.append(list(markets)) or dates
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
        (["--surplus-only", "--props-only", "--dry-run"], "sizes a real run"),
        (["--surplus-only"], "pass --props-only"),
        (["--surplus-only", "--props-only", "--check-all"], "prices archived game-dates"),
    ],
)
def test_main_rejects_contradictory_fetch_flags(monkeypatch, fake_fetches, args, message):
    # A flag set that slips past its guard must fail here, not reach the quota probe.
    _stub_surplus(monkeypatch, 0)
    result, *_ = _invoke_main(monkeypatch, fake_fetches, args)
    assert result.exit_code != 0
    assert message in result.output


def test_combo_keys_are_mapped_for_nfl():
    with open(pkg_resources.files(data) / "config" / "stat_map.json") as f:
        nfl = json.load(f)["Odds API"]["NFL"]
    assert nfl["player_rush_reception_yds"] == "yards"
    assert nfl["player_pass_rush_yds"] == "qb yards"


def _fetched_dates(fake_fetches):
    prop_calls, _ = fake_fetches
    return [kwargs["date"].date() for _, kwargs in prop_calls]


def test_main_newest_first_fetches_the_latest_dates_first(monkeypatch, fake_fetches):
    """A pull that stops early leaves one unbroken block ending at ``--end``."""
    result, *_ = _invoke_main(
        monkeypatch, fake_fetches, ["--newest-first", "--max-dates", "2"], dates=WEEK_ONE
    )
    assert result.exit_code == 0, result.output
    assert _fetched_dates(fake_fetches) == [MONDAY, SUNDAY]


def _stub_surplus(monkeypatch, *allowances):
    """Stub the free quota probe, the archived slate sizes and the governor's allowance:
    one value per plan, in order."""
    probes, asked = [], []
    granted = iter(allowances)
    quota = SimpleNamespace(headers={"X-Requests-Remaining": "5000"})
    monkeypatch.setattr(
        bho, "_get_with_retry", lambda url, params=None: probes.append(params) or quota
    )
    monkeypatch.setattr(
        bho.odds_budget,
        "backfill_allowance",
        lambda now, remaining, cfg: asked.append(remaining) or next(granted),
    )
    monkeypatch.setattr(bho, "_games_per_date", lambda league, dates: dict(ARCHIVED_GAMES))
    return probes, asked


@pytest.mark.parametrize(
    ("allowances", "bought"),
    [
        # 11 + 151, and Thursday's 21 does not fit what that run left.
        ((162, 20), [MONDAY, SUNDAY]),
        # A date is priced a game high, so a run leaves credits over; the next plan spends them.
        ((162, 21), [MONDAY, SUNDAY, THURSDAY]),
        # Sunday does not fit, and Thursday is not bought around it: no holes.
        ((161, 150), [MONDAY]),
        ((0,), []),
    ],
)
def test_main_surplus_only_buys_the_newest_runs_the_leftover_covers(
    monkeypatch, fake_fetches, allowances, bought
):
    """One market: a date is priced at 1 for the events call plus 10 a game, for one
    game more than the archive shows. That covers Monday, which it shows none for."""
    probes, asked = _stub_surplus(monkeypatch, *allowances)
    result, *_ = _invoke_main(
        monkeypatch,
        fake_fetches,
        ["--surplus-only", "--props-only", "--newest-first", "--key-name", "odds_api_alt"],
        dates=WEEK_ONE,
    )
    assert result.exit_code == 0, result.output
    assert _fetched_dates(fake_fetches) == bought
    # Each plan reads the leftover off the key that pays, from its live remaining-credits header.
    assert probes == [{"apiKey": "alt-key-string"}] * len(allowances)
    assert asked == [5000] * len(allowances)


def test_main_surplus_only_resumes_past_the_dates_already_held(monkeypatch, fake_fetches, tmp_path):
    _stub_surplus(monkeypatch, 172)
    (tmp_path / "progress.json").write_text(
        json.dumps({"NFL|passing yards|propsonly": ["2024-09-09"]})
    )
    result, *_ = _invoke_main(
        monkeypatch,
        fake_fetches,
        ["--surplus-only", "--props-only", "--newest-first"],
        dates=WEEK_ONE,
    )
    assert result.exit_code == 0, result.output
    assert _fetched_dates(fake_fetches) == [SUNDAY, THURSDAY]


def test_main_surplus_only_finished_history_never_touches_the_api(
    monkeypatch, fake_fetches, tmp_path
):
    probes, _ = _stub_surplus(monkeypatch, 10_000)
    held = [d.isoformat() for d in WEEK_ONE]
    (tmp_path / "progress.json").write_text(json.dumps({"NFL|passing yards|propsonly": held}))
    result, *_ = _invoke_main(
        monkeypatch,
        fake_fetches,
        ["--surplus-only", "--props-only", "--newest-first"],
        dates=WEEK_ONE,
    )
    assert result.exit_code == 0, result.output
    assert _fetched_dates(fake_fetches) == []
    assert probes == []


def test_main_surplus_only_stops_at_a_date_with_a_failed_call(monkeypatch, fake_fetches, tmp_path):
    """The fetch swallows a failed call; marking its date done would leave a hole."""
    _stub_surplus(monkeypatch, 10_000)
    prop_calls, _ = fake_fetches
    monkeypatch.setitem(bho.odds_budget._run, "failed", 0)

    def one_sunday_call_fails(archive, apikey, *args, **kwargs):
        prop_calls.append((apikey, kwargs))
        if kwargs["date"].date() == SUNDAY:
            bho.odds_budget._run["failed"] += 1

    monkeypatch.setattr(bho, "get_props", one_sunday_call_fails)
    result, *_ = _invoke_main(
        monkeypatch,
        fake_fetches,
        ["--surplus-only", "--props-only", "--newest-first"],
        dates=WEEK_ONE,
    )
    assert result.exit_code != 0
    assert "2024-09-08: an Odds API call failed" in result.output
    # Thursday sits behind the failed Sunday and is never bought around it.
    assert _fetched_dates(fake_fetches) == [MONDAY, SUNDAY]
    assert json.loads((tmp_path / "progress.json").read_text()) == {
        "NFL|passing yards|propsonly": ["2024-09-09"]
    }


def test_main_out_of_credits_exits_nonzero_with_the_bought_dates_saved(
    monkeypatch, fake_fetches, tmp_path
):
    """An unattended run that drains or loses its key must not report success."""
    prop_calls, _ = fake_fetches

    def key_dies_on_sunday(archive, apikey, *args, **kwargs):
        if kwargs["date"].date() == SUNDAY:
            raise OddsAPIAuthError("out of credits")
        prop_calls.append((apikey, kwargs))

    monkeypatch.setattr(bho, "get_props", key_dies_on_sunday)
    result, *_ = _invoke_main(
        monkeypatch, fake_fetches, ["--props-only", "--newest-first"], dates=WEEK_ONE
    )
    assert result.exit_code != 0
    assert "STOP (401) at 2024-09-08" in result.output
    assert _fetched_dates(fake_fetches) == [MONDAY]
    assert json.loads((tmp_path / "progress.json").read_text()) == {
        "NFL|passing yards|propsonly": ["2024-09-09"]
    }


def test_games_per_date_counts_two_moneyline_teams_a_game(monkeypatch, tmp_path):
    db = tmp_path / "archive.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE odds (league TEXT, market TEXT, game_date DATE, entity TEXT)")
    con.executemany(
        "INSERT INTO odds VALUES (?, ?, ?, ?)",
        [
            *[("NFL", "Moneyline", SUNDAY, team) for team in ("Bal", "Kc", "Phi", "Gb", "Bal")],
            ("NFL", "Moneyline", MONDAY, "Sf"),  # one side of the game never archived
            ("NFL", "passing yards", SUNDAY, "Some Quarterback"),
            ("NBA", "Moneyline", SUNDAY, "Bos"),
        ],
    )
    con.close()
    monkeypatch.setattr(bho, "ARCHIVE_DB", db)
    assert bho._games_per_date("NFL", [SUNDAY, MONDAY]) == {SUNDAY: 2, MONDAY: 1}
