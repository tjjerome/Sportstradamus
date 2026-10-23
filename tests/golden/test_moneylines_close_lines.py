"""Characterization of ``moneylines._close_lines_pass`` — the targeted
close-line scrape behind ``confer --close-lines``.

Pins the closing-window event selection against the ``upcoming_events`` ledger
(past-commence pruning, and the no-op tick that calls no API, takes no lock and
writes nothing), the per-event fetch on every region and inning market, the
plus-key routing with its one-shot margin-key fallback, and the capture spool:
every fetch is spooled before it is archived, a busy archive lock leaves it there,
and a later tick archives it at its stored fetch time and its game's Chicago date
without refetching it. No test lets the pass write the ledger, which broad confer
owns.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytz

from sportstradamus import moneylines
from sportstradamus.helpers import odds_budget
from sportstradamus.helpers.locks import _LOCK_DIR_ENV, ARCHIVE_LOCK_FILE, file_lock

_PROPS = {"NBA": {"player_points": "PTS"}}
_KEYS = {"odds_api": "fake-margin", "odds_api_plus": "fake-plus"}


def _event(minutes, event_id, league="NBA", sport_key="basketball_nba"):
    commence = datetime.now(pytz.utc) + timedelta(minutes=minutes)
    return {
        "sport_key": sport_key,
        "event_id": event_id,
        "league": league,
        "commence_time": commence.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _event_url(event_id, sport="basketball_nba"):
    return moneylines.ODDS_API_EVENT_ODDS_URL.format(sport=sport, eventId=event_id)


class _FakeResponse:
    def __init__(self, payload, status_code=HTTPStatus.OK):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeArchive:
    def __init__(self, write_error):
        self.write_calls = 0
        self.write_error = write_error

    def write(self):
        self.write_calls += 1
        if self.write_error is not None:
            raise self.write_error


def _ok_get(url, params=None):
    return _FakeResponse({"url": url})


def _no_api(url, params=None):
    raise AssertionError(f"unexpected API call {url}")


def _wire(monkeypatch, tmp_path, events, fake_get, spooled=(), write_error=None):
    """Wire the pass's collaborators in memory; returns the trackers the tests read.

    The spool round-trips through JSON on every read and write, as the file does.
    """
    monkeypatch.setattr(odds_budget, "ODDS_USAGE_LEDGER_PATH", tmp_path / "ledger.jsonl")
    odds_budget.set_run_context("close_lines")
    monkeypatch.setenv(_LOCK_DIR_ENV, str(tmp_path))
    wired = SimpleNamespace(
        spool=json.loads(json.dumps(list(spooled))),
        spool_writes=0,
        archives=[],
        archived=[],
        lock_path=tmp_path / Path(ARCHIVE_LOCK_FILE).name,
    )

    def _write_spool(captures):
        wired.spool = json.loads(json.dumps(captures))
        wired.spool_writes += 1

    def _archive():
        wired.archives.append(_FakeArchive(write_error))
        return wired.archives[-1]

    def _write_ledger(events):
        raise AssertionError("close-lines wrote upcoming_events.json")

    def _archive_event_props(archive, game, league, props, game_date, observed_at):
        wired.archived.append((game["url"], league, game_date, observed_at))

    monkeypatch.setattr(moneylines, "read_upcoming_events", lambda: [dict(e) for e in events])
    monkeypatch.setattr(moneylines, "write_upcoming_events", _write_ledger)
    monkeypatch.setattr(
        moneylines, "read_close_line_captures", lambda: json.loads(json.dumps(wired.spool))
    )
    monkeypatch.setattr(moneylines, "write_close_line_captures", _write_spool)
    monkeypatch.setattr(moneylines, "Archive", _archive)
    monkeypatch.setattr(moneylines, "_archive_event_props", _archive_event_props)
    monkeypatch.setattr(moneylines, "_get_with_retry", fake_get)
    return wired


def test_close_lines_fetches_the_window_and_archives_it(monkeypatch, tmp_path) -> None:
    events = [
        _event(-10, "past"),
        _event(2, "soon"),
        _event(10, "due1"),
        _event(20, "due2"),
        _event(40, "later"),
    ]
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        return _FakeResponse({"url": url})

    wired = _wire(monkeypatch, tmp_path, events, fake_get)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert calls == [_event_url("due1"), _event_url("due2")]
    assert [url for url, *_ in wired.archived] == calls
    assert [archive.write_calls for archive in wired.archives] == [1]
    # Archived captures stay spooled without their payload until their games start.
    assert [(c["event_id"], c["game"]) for c in wired.spool] == [("due1", None), ("due2", None)]


def test_close_lines_tick_with_nothing_due_touches_nothing(monkeypatch, tmp_path) -> None:
    archived = _event(15, "archived")
    events = [_event(-10, "past"), _event(2, "soon"), archived, _event(40, "later")]
    spooled = [{**archived, "observed_at": "2026-10-22T23:40:00", "game": None}]
    wired = _wire(monkeypatch, tmp_path, events, _no_api, spooled=spooled)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert wired.archives == []
    assert wired.spool_writes == 0
    assert not wired.lock_path.exists()


def test_close_lines_spools_the_capture_while_the_archive_lock_is_busy(
    monkeypatch, tmp_path
) -> None:
    wired = _wire(monkeypatch, tmp_path, [_event(10, "due1")], _ok_get)
    monkeypatch.setattr(moneylines, "CLOSE_LINES_LOCK_WAIT_S", 0)

    with file_lock(ARCHIVE_LOCK_FILE, timeout_s=0, label="prophecize", contention_hint=""):
        moneylines._close_lines_pass(_KEYS, _PROPS)

    assert wired.archives == []
    [capture] = wired.spool
    assert (capture["event_id"], capture["game"]["url"]) == ("due1", _event_url("due1"))
    # Naive UTC, the stamp the archive stages.
    fetched = datetime.fromisoformat(capture["observed_at"])
    assert abs(fetched - datetime.now(pytz.utc).replace(tzinfo=None)) < timedelta(minutes=1)


def test_close_lines_archives_a_spooled_capture_at_its_fetch_time(monkeypatch, tmp_path) -> None:
    # Still inside its window: only the spool keeps it from a second paid fetch.
    event = _event(10, "due1")
    fetched = datetime(2026, 10, 22, 23, 41, 7)
    spooled = [{**event, "observed_at": fetched.isoformat(), "game": {"url": "spooled"}}]
    wired = _wire(monkeypatch, tmp_path, [event], _no_api, spooled=spooled)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert [(url, observed_at) for url, _, _, observed_at in wired.archived] == [
        ("spooled", fetched)
    ]
    assert [c["game"] for c in wired.spool] == [None]


def test_close_lines_does_not_refetch_a_spooled_event(monkeypatch, tmp_path) -> None:
    done, due = _event(10, "done"), _event(20, "due2")
    spooled = [{**done, "observed_at": "2026-10-22T23:40:00", "game": None}]
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        return _FakeResponse({"url": url})

    wired = _wire(monkeypatch, tmp_path, [done, due], fake_get, spooled=spooled)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert calls == [_event_url("due2")]
    assert [(c["event_id"], c["game"]) for c in wired.spool] == [("done", None), ("due2", None)]


def test_close_lines_archives_a_started_game_under_its_chicago_date(monkeypatch, tmp_path) -> None:
    started = {
        "sport_key": "basketball_nba",
        "event_id": "started",
        "league": "NBA",
        "commence_time": "2026-10-22T00:15:00Z",
        "observed_at": "2026-10-21T23:55:00",
        "game": {"url": "spooled"},
    }
    wired = _wire(monkeypatch, tmp_path, [], _no_api, spooled=[started])

    moneylines._close_lines_pass(_KEYS, _PROPS)

    # 00:15 UTC tips off at 7:15 pm the evening before in Chicago.
    assert wired.archived == [("spooled", "NBA", "2026-10-21", datetime(2026, 10, 21, 23, 55))]
    assert wired.spool == []


def test_close_lines_requests_every_region_and_inning_market(monkeypatch, tmp_path) -> None:
    props = {
        "NBA": {"player_points": "PTS", "player_turnovers": "TOV"},
        "MLB": {"batter_hits": "hits"},
    }
    events = [_event(10, "nba1"), _event(20, "mlb1", league="MLB", sport_key="baseball_mlb")]
    calls = []

    def fake_get(url, params=None):
        calls.append((url, params))
        return _FakeResponse({"url": url})

    _wire(monkeypatch, tmp_path, events, fake_get)

    moneylines._close_lines_pass(_KEYS, props)

    mlb_markets = "batter_hits,totals_1st_1_innings,spreads_1st_1_innings"
    assert calls == [
        (_event_url("nba1"), {"apiKey": "fake-plus", "markets": "player_points", "regions": "us"}),
        (
            _event_url("nba1"),
            {"apiKey": "fake-plus", "markets": "player_turnovers", "regions": "us2"},
        ),
        (
            _event_url("mlb1", sport="baseball_mlb"),
            {"apiKey": "fake-plus", "markets": mlb_markets, "regions": "us"},
        ),
    ]


def test_close_lines_falls_back_to_margin_key_once(monkeypatch, tmp_path) -> None:
    events = [_event(10, "due1"), _event(20, "due2")]
    calls = []

    def fake_get(url, params=None):
        calls.append((url, params["apiKey"]))
        if params["apiKey"] == _KEYS["odds_api_plus"]:
            raise moneylines.OddsAPIAuthError("plus key exhausted")
        return _FakeResponse({"url": url})

    wired = _wire(monkeypatch, tmp_path, events, fake_get)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert calls == [
        (_event_url("due1"), "fake-plus"),
        (_event_url("due1"), "fake-margin"),
        (_event_url("due2"), "fake-margin"),
    ]
    assert [url for url, *_ in wired.archived] == [_event_url("due1"), _event_url("due2")]


def test_close_lines_margin_key_exhaustion_raises_with_earlier_fetches_spooled(
    monkeypatch, tmp_path
) -> None:
    events = [_event(10, "due1"), _event(20, "due2")]

    def fake_get(url, params=None):
        if url == _event_url("due2"):
            raise moneylines.OddsAPIAuthError("out of credits")
        return _FakeResponse({"url": url})

    wired = _wire(monkeypatch, tmp_path, events, fake_get)

    with pytest.raises(moneylines.OddsAPIAuthError):
        moneylines._close_lines_pass(_KEYS, _PROPS)

    assert wired.archives == []
    assert [(c["event_id"], c["game"]["url"]) for c in wired.spool] == [
        ("due1", _event_url("due1"))
    ]


def test_close_lines_archive_failure_raises_and_keeps_the_spool(monkeypatch, tmp_path) -> None:
    wired = _wire(
        monkeypatch,
        tmp_path,
        [_event(10, "due1")],
        _ok_get,
        write_error=RuntimeError("duckdb write failed"),
    )

    with pytest.raises(RuntimeError, match="duckdb write failed"):
        moneylines._close_lines_pass(_KEYS, _PROPS)

    assert [c["game"]["url"] for c in wired.spool] == [_event_url("due1")]


@pytest.mark.parametrize("status", [HTTPStatus.NOT_FOUND, HTTPStatus.INTERNAL_SERVER_ERROR])
def test_close_lines_failed_fetch_stays_due(monkeypatch, tmp_path, status, caplog) -> None:
    events = [_event(10, "due1"), _event(20, "due2")]
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        failed = calls == [_event_url("due1")]
        return _FakeResponse({"url": url}, status_code=status if failed else HTTPStatus.OK)

    wired = _wire(monkeypatch, tmp_path, events, fake_get)

    with caplog.at_level(logging.WARNING, logger="log"):
        moneylines._close_lines_pass(_KEYS, _PROPS)
    assert [c["event_id"] for c in wired.spool] == ["due2"]
    assert f"due1 us returned status {int(status)}" in caplog.text

    moneylines._close_lines_pass(_KEYS, _PROPS)
    assert calls == [_event_url("due1"), _event_url("due2"), _event_url("due1")]
    assert [c["event_id"] for c in wired.spool] == ["due2", "due1"]
