"""Characterization of ``moneylines._close_lines_pass`` — the targeted
close-line scrape behind ``confer --close-lines``.

Pins the closing-window event selection against the ``upcoming_events``
ledger (including past-commence pruning and the no-op empty-window exit),
the plus-key routing on every per-event fetch, and the one-shot margin-key
fallback when the plus key exhausts mid-run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from http import HTTPStatus

import pytest
import pytz

from sportstradamus import moneylines
from sportstradamus.helpers import odds_budget

_PROPS = {"NBA": {"player_points": "PTS"}}
_KEYS = {"odds_api": "fake-margin", "odds_api_plus": "fake-plus"}


def _event(minutes, event_id):
    commence = datetime.now(pytz.utc) + timedelta(minutes=minutes)
    return {
        "sport_key": "basketball_nba",
        "event_id": event_id,
        "league": "NBA",
        "commence_time": commence.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "markets": ["player_points"],
    }


def _event_url(event_id):
    return moneylines.ODDS_API_EVENT_ODDS_URL.format(sport="basketball_nba", eventId=event_id)


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = HTTPStatus.OK
        self._payload = payload

    def json(self):
        return self._payload


class _FakeArchive:
    def __init__(self):
        self.write_calls = 0

    def write(self):
        self.write_calls += 1


def _wire(monkeypatch, tmp_path, events, fake_get):
    """Wire the pass's collaborators; returns (written, archives, archived) trackers."""
    monkeypatch.setattr(odds_budget, "ODDS_USAGE_LEDGER_PATH", tmp_path / "ledger.jsonl")
    odds_budget.set_run_context("close_lines")
    written, archives, archived = [], [], []

    def _archive():
        archives.append(_FakeArchive())
        return archives[-1]

    monkeypatch.setattr(moneylines, "read_upcoming_events", lambda: events)
    monkeypatch.setattr(moneylines, "write_upcoming_events", written.append)
    monkeypatch.setattr(moneylines, "Archive", _archive)
    monkeypatch.setattr(
        moneylines,
        "_archive_event_props",
        lambda archive, game, league, props, gameDate: archived.append((game, league, gameDate)),
    )
    monkeypatch.setattr(moneylines, "_get_with_retry", fake_get)
    return written, archives, archived


def test_close_lines_window_selection_and_prune(monkeypatch, tmp_path) -> None:
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

    written, _, archived = _wire(monkeypatch, tmp_path, events, fake_get)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert calls == [_event_url("due1"), _event_url("due2")]
    assert [game["url"] for game, _, _ in archived] == calls
    assert written == [events[1:]]  # past-commence entry pruned from the ledger


def test_close_lines_empty_window_is_a_noop(monkeypatch, tmp_path) -> None:
    events = [_event(-10, "past"), _event(2, "soon"), _event(40, "later")]

    def fake_get(url, params=None):
        raise AssertionError(f"unexpected API call {url}")

    written, archives, archived = _wire(monkeypatch, tmp_path, events, fake_get)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert archives == []
    assert archived == []
    assert written == [events[1:]]


def test_close_lines_fetches_on_the_plus_key(monkeypatch, tmp_path) -> None:
    events = [_event(10, "due1"), _event(20, "due2")]
    calls = []

    def fake_get(url, params=None):
        calls.append(dict(params))
        return _FakeResponse({})

    _wire(monkeypatch, tmp_path, events, fake_get)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert calls == [{"apiKey": "fake-plus", "regions": "us", "markets": "player_points"}] * 2


def test_close_lines_falls_back_to_margin_key_once(monkeypatch, tmp_path) -> None:
    events = [_event(10, "due1"), _event(20, "due2")]
    calls = []

    def fake_get(url, params=None):
        calls.append((url, params["apiKey"]))
        if params["apiKey"] == _KEYS["odds_api_plus"]:
            raise moneylines.OddsAPIAuthError("plus key exhausted")
        return _FakeResponse({"url": url})

    _, archives, archived = _wire(monkeypatch, tmp_path, events, fake_get)

    moneylines._close_lines_pass(_KEYS, _PROPS)

    assert calls == [
        (_event_url("due1"), "fake-plus"),
        (_event_url("due1"), "fake-margin"),
        (_event_url("due2"), "fake-margin"),
    ]
    assert [game["url"] for game, _, _ in archived] == [_event_url("due1"), _event_url("due2")]
    assert len(archives) == 1
    assert archives[0].write_calls == 1


def test_close_lines_margin_key_exhaustion_propagates(monkeypatch, tmp_path) -> None:
    events = [_event(10, "due1")]

    def fake_get(url, params=None):
        raise moneylines.OddsAPIAuthError("out of credits")

    _, archives, _ = _wire(monkeypatch, tmp_path, events, fake_get)

    with pytest.raises(moneylines.OddsAPIAuthError):
        moneylines._close_lines_pass(_KEYS, _PROPS)

    assert archives[0].write_calls == 0
