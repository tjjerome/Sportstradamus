"""Characterization of ``moneylines._classify_league_activity`` — the
feed-derived league-activity classifier feeding the budget governor and the
prophecize/meditate/pickem-build update gates.

Pins the tier assignment from events-feed commence times (live / preseason /
idle-omitted), the sticky-live carry across mid-season gaps, the
previous-record carry-forward on a failed events fetch, and the persisted
snapshot shape at ``helpers.io.LEAGUE_ACTIVITY_PATH``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from http import HTTPStatus

import pytz

from sportstradamus import moneylines
from sportstradamus.helpers import io

_NOW = datetime.now(pytz.utc)


def _sport(key, title):
    return {"key": key, "title": title}


def _events_url(sport_key):
    return moneylines.ODDS_API_EVENTS_URL.format(sport=sport_key)


def _commence(days):
    return (_NOW + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


class _FakeResponse:
    def __init__(self, payload, status_code=HTTPStatus.OK):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _wire(monkeypatch, tmp_path, events_by_sport, previous=None):
    path = tmp_path / "league_activity.json"
    monkeypatch.setattr(io, "LEAGUE_ACTIVITY_PATH", path)
    if previous is not None:
        path.write_text(json.dumps(previous))

    def fake_get(url, params=None):
        return events_by_sport[url]

    monkeypatch.setattr(moneylines, "_get_with_retry", fake_get)
    return path


def test_classify_tiers_and_persist_snapshot(monkeypatch, tmp_path) -> None:
    events = {
        _events_url("baseball_mlb"): _FakeResponse(
            [{"commence_time": _commence(0.2)}, {"commence_time": _commence(1.2)}]
        ),
        _events_url("americanfootball_nfl"): _FakeResponse([{"commence_time": _commence(10)}]),
        _events_url("icehockey_nhl"): _FakeResponse([{"commence_time": _commence(60)}]),
        _events_url("basketball_wnba"): _FakeResponse([]),
    }
    path = _wire(monkeypatch, tmp_path, events)
    sports = [
        _sport("baseball_mlb", "MLB"),
        _sport("americanfootball_nfl", "NFL"),
        _sport("icehockey_nhl", "NHL"),
        _sport("basketball_wnba", "WNBA"),
    ]

    tiers = moneylines._classify_league_activity("k", sports)

    # Beyond-horizon NHL and empty-slate WNBA are idle: absent entirely.
    assert tiers == {"MLB": "live", "NFL": "preseason"}
    snapshot = json.loads(path.read_text())
    assert set(snapshot) == {"updated", "leagues"}
    assert snapshot["leagues"] == {
        # Mid-season MLB never saw its opener; preseason NFL's earliest
        # upcoming game IS opening night.
        "MLB": {"tier": "live", "next_game": _commence(0.2)},
        "NFL": {"tier": "preseason", "next_game": _commence(10), "opener": _commence(10)},
    }


def test_classify_live_is_sticky_through_midweek_gap(monkeypatch, tmp_path) -> None:
    events = {
        _events_url("americanfootball_nfl"): _FakeResponse([{"commence_time": _commence(4)}])
    }
    previous = {
        "updated": _NOW.isoformat(),
        "leagues": {"NFL": {"tier": "live", "next_game": _commence(-3)}},
    }
    _wire(monkeypatch, tmp_path, events, previous=previous)

    tiers = moneylines._classify_league_activity("k", [_sport("americanfootball_nfl", "NFL")])

    assert tiers == {"NFL": "live"}


def test_classify_opener_freezes_at_promotion_to_live(monkeypatch, tmp_path) -> None:
    events = {
        _events_url("americanfootball_nfl"): _FakeResponse([{"commence_time": _commence(0.5)}])
    }
    previous = {
        "updated": _NOW.isoformat(),
        "leagues": {
            "NFL": {"tier": "preseason", "next_game": _commence(0.5), "opener": _commence(0.5)}
        },
    }
    path = _wire(monkeypatch, tmp_path, events, previous=previous)

    moneylines._classify_league_activity("k", [_sport("americanfootball_nfl", "NFL")])

    # Live records carry the preseason-observed opener instead of tracking
    # the (now mid-season) earliest upcoming game.
    assert json.loads(path.read_text())["leagues"]["NFL"] == {
        "tier": "live",
        "next_game": _commence(0.5),
        "opener": _commence(0.5),
    }


def test_classify_season_end_leaves_idle_record_with_last_game(monkeypatch, tmp_path) -> None:
    events = {_events_url("basketball_wnba"): _FakeResponse([])}
    previous = {
        "updated": _NOW.isoformat(),
        "leagues": {
            "WNBA": {
                "tier": "live",
                "next_game": _commence(-0.5),
                "opener": _commence(-60),
            }
        },
    }
    path = _wire(monkeypatch, tmp_path, events, previous=previous)

    tiers = moneylines._classify_league_activity("k", [_sport("basketball_wnba", "WNBA")])

    # Feed emptied: the finals game rolls into last_game so Stats.update keeps
    # its post-season ingest tail; the governor stops polling immediately.
    assert tiers == {}
    assert json.loads(path.read_text())["leagues"] == {
        "WNBA": {"tier": "idle", "last_game": _commence(-0.5)}
    }


def test_classify_index_absent_league_demotes_to_idle(monkeypatch, tmp_path) -> None:
    events = {}
    previous = {
        "updated": _NOW.isoformat(),
        "leagues": {
            "NBA": {"tier": "live", "next_game": _commence(-2)},
            "NHL": {"tier": "idle", "last_game": _commence(-20)},
        },
    }
    path = _wire(monkeypatch, tmp_path, events, previous=previous)

    tiers = moneylines._classify_league_activity("k", [])

    assert tiers == {}
    assert json.loads(path.read_text())["leagues"] == {
        "NBA": {"tier": "idle", "last_game": _commence(-2)},
        "NHL": {"tier": "idle", "last_game": _commence(-20)},
    }


def test_classify_failed_fetch_carries_previous_record(monkeypatch, tmp_path) -> None:
    events = {
        _events_url("baseball_mlb"): _FakeResponse(None, HTTPStatus.INTERNAL_SERVER_ERROR),
        _events_url("basketball_nba"): _FakeResponse(None, HTTPStatus.INTERNAL_SERVER_ERROR),
    }
    previous = {
        "updated": _NOW.isoformat(),
        "leagues": {"MLB": {"tier": "live", "next_game": _commence(0.5)}},
    }
    path = _wire(monkeypatch, tmp_path, events, previous=previous)
    sports = [_sport("baseball_mlb", "MLB"), _sport("basketball_nba", "NBA")]

    tiers = moneylines._classify_league_activity("k", sports)

    # MLB rides its previous record; NBA had none, so it stays absent.
    assert tiers == {"MLB": "live"}
    assert json.loads(path.read_text())["leagues"] == previous["leagues"]
