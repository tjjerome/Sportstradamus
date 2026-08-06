"""Characterization of ``moneylines.get_props`` — the player-prop ingest's live
(non-fixture) path.

The integration suite always passes ``fixture_dir``, so it exercises
``_get_props_from_fixtures`` and never the live ``get_props`` body (sports-index
resolution, request building, the per-sport/per-event fetch loop, the ledger
build, and the ``write_upcoming_events`` call). This pins that path on a
synthetic Odds API payload so the CC-20 decomposition can be proven behavior
preserving. The per-event parse itself is pinned separately by
``test_moneylines_archive_event_props``; here the focus is the orchestration and
the upcoming-events ledger.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from http import HTTPStatus

import pytz

from sportstradamus import moneylines

_CHICAGO = pytz.timezone("America/Chicago")
_TODAY = datetime.today()
_DATE = _CHICAGO.localize(datetime(_TODAY.year, _TODAY.month, _TODAY.day, 12, 0))
_COMMENCE = (_DATE + timedelta(days=1)).astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_GAME_DATE = (_DATE + timedelta(days=1)).astimezone(_CHICAGO).strftime("%Y-%m-%d")

_PROPS = {"NBA": {"player_points": "PTS"}}
_SPORTS = [{"key": "basketball_nba", "title": "NBA", "active": True}]
_SPORTS_TWO_LEAGUES = [
    {"key": "basketball_nba", "title": "NBA", "active": True},
    {"key": "basketball_wnba", "title": "WNBA", "active": True},
]
_EVENTS = [{"id": "ev1", "commence_time": _COMMENCE}]
_GAME = {
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "bookmakers": [
        {
            "key": "draftkings",
            "markets": [
                {
                    "key": "player_points",
                    "outcomes": [
                        {
                            "description": "LeBron James",
                            "name": "Over",
                            "point": 25.5,
                            "price": 1.9,
                        },
                        {
                            "description": "LeBron James",
                            "name": "Under",
                            "point": 25.5,
                            "price": 1.9,
                        },
                    ],
                },
            ],
        }
    ],
}


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = HTTPStatus.OK
        self._payload = payload

    def json(self):
        return self._payload


def _fake_get_with_retry(url, params=None):
    routes = {
        moneylines.ODDS_API_SPORTS_URL: _SPORTS,
        moneylines.ODDS_API_EVENTS_URL.format(sport="basketball_nba"): _EVENTS,
        moneylines.ODDS_API_EVENT_ODDS_URL.format(sport="basketball_nba", eventId="ev1"): _GAME,
    }
    return _FakeResponse(routes[url])


class _FakeArchive:
    def __init__(self):
        self.player_books = []
        self.ladder_calls = []

    def merge_player_books(
        self, league, market, gameDate, player, ev, lines, observed_at=None, book_quotes=None
    ):
        self.player_books.append((league, market, gameDate, player, ev, lines, observed_at))

    def add_ladder(self, league, market, gameDate, player, book, rungs, observed_at=None):
        self.ladder_calls.append((league, market, gameDate, player, book, rungs, observed_at))

    def set_team_books(self, league, market, gameDate, team, books):  # pragma: no cover
        raise AssertionError("synthetic payload has no totals market")


_EXPECTED_PLAYER = [
    ("NBA", "PTS", _GAME_DATE, "Lebron James", {"draftkings": 25.50000000098013}, [25.5], None),
]

_EXPECTED_LEDGER = [
    {
        "sport_key": "basketball_nba",
        "event_id": "ev1",
        "league": "NBA",
        "commence_time": _COMMENCE,
        "markets": ["player_points"],
    },
]


def test_get_props_live_path(monkeypatch, assert_player_books_close) -> None:
    written = []
    monkeypatch.setattr(moneylines, "_get_with_retry", _fake_get_with_retry)
    monkeypatch.setattr(moneylines, "read_upcoming_events", list)
    monkeypatch.setattr(moneylines, "write_upcoming_events", written.append)
    archive = _FakeArchive()

    returned = moneylines.get_props(archive, "kp", _PROPS, date=_DATE)

    assert returned is archive
    assert_player_books_close(archive.player_books, _EXPECTED_PLAYER)
    assert written == [_EXPECTED_LEDGER]


def test_get_props_leagues_filter(monkeypatch) -> None:
    routes = {
        moneylines.ODDS_API_SPORTS_URL: _SPORTS_TWO_LEAGUES,
        moneylines.ODDS_API_EVENTS_URL.format(sport="basketball_nba"): _EVENTS,
        moneylines.ODDS_API_EVENT_ODDS_URL.format(sport="basketball_nba", eventId="ev1"): _GAME,
    }
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        return _FakeResponse(routes[url])

    monkeypatch.setattr(moneylines, "_get_with_retry", fake_get)
    monkeypatch.setattr(moneylines, "read_upcoming_events", list)
    monkeypatch.setattr(moneylines, "write_upcoming_events", lambda events: None)
    archive = _FakeArchive()

    moneylines.get_props(archive, "kp", _PROPS, date=_DATE, leagues=("NBA",))

    # WNBA is active in the index but excluded by the governor's allowance:
    # its events endpoint is never hit.
    assert calls == [
        moneylines.ODDS_API_SPORTS_URL,
        moneylines.ODDS_API_EVENTS_URL.format(sport="basketball_nba"),
        moneylines.ODDS_API_EVENT_ODDS_URL.format(sport="basketball_nba", eventId="ev1"),
    ]


def test_region_split_is_credit_neutral() -> None:
    """A us2-only market moves regions without adding to the bill.

    The Odds API charges markets x regions, so the guard against cron spend rising is
    that every market appears in exactly one region's request. NBA TOV went dark for
    months because its last book (fliff) sits in us2 and we only ever asked us.
    """
    props = {"NBA": {"player_points": "PTS", "player_turnovers": "TOV"}}

    calls = moneylines._region_markets("NBA", props)

    assert dict(calls) == {"us": ["player_points"], "us2": ["player_turnovers"]}
    billed = sum(len(markets) * len(regions.split(",")) for regions, markets in calls)
    assert billed == len(props["NBA"])


def test_mlb_inning_markets_ride_the_us_request() -> None:
    props = {"MLB": {"batter_hits": "hits"}}

    us, us2 = moneylines._region_markets("MLB", props)

    assert us == ("us", ["batter_hits", "totals_1st_1_innings", "spreads_1st_1_innings"])
    assert us2 == ("us2", [])
