"""Characterization of ``books.underdog.get_ud`` — the Underdog scraper.

``get_ud`` is network-bound and stubbed (``-> dict``) in the integration suite,
so this pins its output and its request plan on hand-built payloads. The feed
(list containers) carries a two-sided line with alternates, an MLB fantasy line,
a combo player, a suspended line, a season future with no game, a one-sided
line whose other side is suspended, and a solo-game line in an unmodeled
league. The NBA lobby payload (dict containers) carries a team total, a spread,
a moneyline, a yes/no line, a three-way period moneyline, a suspended line and
a live line. NFL is absent from the feed, so no NFL pill is requested; only the
NBA player line qualifies for an alternate-line fetch. The feed's NBA game title
is listed away-first so the lobby ``teams`` dict, not the title, must set the
sides; the MLB lobby answers empty so its sides come from the title.
"""

from __future__ import annotations

import time
import types
from http import HTTPStatus

import pytest

from sportstradamus.books import underdog

_SCHEDULED = "2026-06-03T23:00:00Z"


def _line(line_id, appearance_id, stat, display_stat, stat_value, options, **flags):
    return {
        "id": line_id,
        "over_under_id": f"ou-{line_id}",
        "status": flags.get("status", "active"),
        "live_event": flags.get("live", False),
        "stat_value": stat_value,
        "over_under": {
            "has_alternates": flags.get("alternates", False),
            "appearance_stat": {
                "appearance_id": appearance_id,
                "stat": stat,
                "display_stat": display_stat,
            },
        },
        "options": [
            {"choice": choice, "payout_multiplier": multiplier, "status": status}
            for choice, multiplier, status in options
        ],
    }


_EVEN = [("higher", "1.0", "active"), ("lower", "1.0", "active")]

_FEED = {
    "players": [
        {"id": "p1", "first_name": "LeBron", "last_name": "James", "sport_id": "NBA"},
        {"id": "p2", "first_name": "Jayson", "last_name": "Tatum", "sport_id": "NBA"},
        {"id": "p3", "first_name": "Aaron", "last_name": "Judge", "sport_id": "MLB"},
        {
            "id": "p4",
            "first_name": "LeBron James",
            "last_name": "+ Jayson Tatum",
            "sport_id": "NBACOMBOS",
        },
        {"id": "p5", "first_name": "Connor", "last_name": "McDavid", "sport_id": "NHL"},
        {"id": "p6", "first_name": "Novak", "last_name": "Djokovic", "sport_id": "TENNIS"},
    ],
    "games": [
        # Listed away-first, as esports titles are: the NBA lobby's teams dict,
        # read before the feed is parsed, decides the sides, not the title.
        {
            "id": 1,
            "home_team_id": "t1",
            "away_team_id": "t2",
            "sport_id": "NBA",
            "abbreviated_title": "BOS vs LAL",
            "scheduled_at": _SCHEDULED,
        },
        {
            "id": 2,
            "home_team_id": "t3",
            "away_team_id": "t4",
            "sport_id": "MLB",
            "abbreviated_title": "HOU @ NYY",
            "scheduled_at": _SCHEDULED,
        },
    ],
    "solo_games": [
        {"id": 3, "title": "Djokovic vs Alcaraz", "sport_id": "TENNIS", "scheduled_at": _SCHEDULED},
    ],
    "appearances": [
        {"id": "a1", "type": "Player", "player_id": "p1", "team_id": "t1", "match_id": 1},
        {"id": "a2", "type": "Player", "player_id": "p2", "team_id": "t2", "match_id": 1},
        {"id": "a3", "type": "Player", "player_id": "p3", "team_id": "t3", "match_id": 2},
        {"id": "a4", "type": "Player", "player_id": "p4", "team_id": "t1", "match_id": 1},
        {"id": "a5", "type": "Player", "player_id": "p5", "team_id": "t5", "match_id": 99},
        {"id": "a6", "type": "Player", "player_id": "p6", "team_id": None, "match_id": 3},
    ],
    "over_under_lines": [
        _line(
            "f1",
            "a1",
            "points",
            "Points",
            "25.5",
            [("higher", "1.0", "active"), ("lower", "1.2", "active")],
            alternates=True,
        ),
        _line("f2", "a3", "fantasy_points", "Fantasy Points", "7.5", _EVEN),
        _line("f3", "a4", "points", "Points", "50.5", _EVEN, alternates=True),
        _line("f4", "a5", "points", "Points", "120.5", _EVEN, alternates=True),
        _line("f5", "a2", "rebounds", "Rebounds", "8.5", _EVEN, status="suspended"),
        _line(
            "f6",
            "a2",
            "technical_fouls",
            "Technical Fouls",
            "0.5",
            [("higher", "1.9", "active"), ("lower", "1.5", "suspended")],
            alternates=True,
        ),
        _line("f7", "a6", "aces", "Aces", "8.5", _EVEN, alternates=True),
    ],
}

_NBA_LOBBY = {
    "teams": {"t1": {"id": "t1", "abbr": "LAL"}, "t2": {"id": "t2", "abbr": "BOS"}},
    "players": {},
    "games": {
        "1": {
            "id": 1,
            "home_team_id": "t1",
            "away_team_id": "t2",
            "sport_id": "NBA",
            "abbreviated_title": "BOS @ LAL",
            "scheduled_at": _SCHEDULED,
        }
    },
    "appearances": {
        "m1": {"id": "m1", "type": "Match", "player_id": None, "team_id": None, "match_id": 1},
        "h1": {"id": "h1", "type": "Team", "player_id": None, "team_id": "t1", "match_id": 1},
        "v1": {"id": "v1", "type": "Team", "player_id": None, "team_id": "t2", "match_id": 1},
    },
    "over_under_lines": {
        "c1": _line(
            "c1",
            "h1",
            "team_total_points",
            "Team Total Points",
            "112.5",
            [("higher", "0.9", "active"), ("lower", "1.0", "active")],
        ),
        "c2": _line(
            "c2",
            "m1",
            "spread",
            "Spread",
            "-3.5",
            [("away", "0.95", "active"), ("home", "0.95", "active")],
        ),
        "c3": _line(
            "c3",
            "m1",
            "moneyline",
            "Moneyline",
            None,
            [("away", "1.3", "active"), ("home", "0.7", "active")],
        ),
        "c4": _line(
            "c4",
            "v1",
            "team_first_basket",
            "First Basket",
            None,
            [("yes", "1.5", "active"), ("no", "0.5", "active")],
        ),
        "c5": _line(
            "c5",
            "m1",
            "period_1_moneyline",
            "1Q Moneyline",
            None,
            [("away", "1.2", "active"), ("draw", "4.0", "active"), ("home", "0.8", "active")],
        ),
        "c6": _line(
            "c6", "h1", "team_total_points", "Team Total Points", "110.5", _EVEN, status="suspended"
        ),
        "c7": _line("c7", "m1", "points", "Total Points", "225.5", _EVEN, live=True),
    },
}

_ALT_F1 = {
    "projections": [
        {
            "id": "main",
            "is_main": True,
            "stat_value": "25.5",
            "options": [
                {"choice": "higher", "payout_multiplier": "1.0", "status": "active"},
                {"choice": "lower", "payout_multiplier": "1.2", "status": "active"},
            ],
        },
        {
            "id": "rung-high",
            "is_main": False,
            "stat_value": "30.5",
            "options": [{"choice": "higher", "payout_multiplier": "2.5", "status": "active"}],
        },
        {
            "id": "rung-low",
            "is_main": False,
            "stat_value": "20.5",
            "options": [
                {"choice": "higher", "payout_multiplier": "0.6", "status": "active"},
                {"choice": "lower", "payout_multiplier": "2.0", "status": "suspended"},
            ],
        },
    ]
}


def _core_url(sport: str) -> str:
    return (
        f"{underdog.UD_LOBBY_URL}/match_grouped_lines?sport_id={sport}&include_live=true"
        f"&{underdog._UD_QUERY}"
    )


_ALT_F1_URL = underdog.UD_ALT_LINES_URL.format(over_under_id="ou-f1")


class _FakeScraper:
    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url):
        self.calls.append(url)
        return self.routes[url]


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = HTTPStatus.OK
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, timeout=None):
        _FakeSession.calls.append(url)
        return _FakeResponse({_ALT_F1_URL: _ALT_F1}[url])


class _FakeRequests:
    RequestException = Exception
    Session = _FakeSession


def _offer(player, league, team, opponent, market, line, over, under):
    return {
        "Player": player,
        "League": league,
        "Team": team,
        "Opponent": opponent,
        "Date": "2026-06-03",
        "Commence": _SCHEDULED,
        "Market": market,
        "Line": line,
        "Boost_Over": over,
        "Boost_Under": under,
    }


_LEBRON = _offer("Lebron James", "NBA", "LAL", "BOS", "Points", 25.5, 1.0, 1.2)

_EXPECTED = {
    "MLB": {
        "Hitter Fantasy Points": [
            _offer("Aaron Judge", "MLB", "NYY", "HOU", "Hitter Fantasy Points", 7.5, 1.0, 1.0)
        ],
    },
    "NBA": {
        "Points": [
            _LEBRON,
            {**_LEBRON, "Line": 30.5, "Boost_Over": 2.5, "Boost_Under": 0.0},
            {**_LEBRON, "Line": 20.5, "Boost_Over": 0.6, "Boost_Under": 0.0},
        ],
        "Technical Fouls": [
            _offer("Jayson Tatum", "NBA", "BOS", "LAL", "Technical Fouls", 0.5, 1.9, 0.0)
        ],
        "team_total_points": [
            _offer("LAL", "NBA", "LAL", "BOS", "team_total_points", 112.5, 0.9, 1.0)
        ],
        "spread": [_offer("LAL", "NBA", "LAL", "BOS", "spread", -3.5, 0.95, 0.95)],
        "moneyline": [_offer("LAL", "NBA", "LAL", "BOS", "moneyline", 0.5, 0.7, 1.3)],
        "team_first_basket": [
            _offer("BOS", "NBA", "BOS", "LAL", "team_first_basket", 0.5, 1.5, 0.5)
        ],
        "period_1_moneyline": [
            _offer("LAL", "NBA", "LAL", "BOS", "period_1_moneyline", 0.5, 0.8, 0.0)
        ],
    },
    # Solo games carry no team id: the legacy quirk keeps Team empty and names
    # the home side as the opponent.
    "TENNIS": {"Aces": [_offer("Novak Djokovic", "TENNIS", "", "Djokovic", "Aces", 8.5, 1.0, 1.0)]},
}


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(underdog, "requests", _FakeRequests)
    monkeypatch.setattr(underdog, "get_mlb_pitchers", dict)
    monkeypatch.setattr(underdog, "nhl_goalies", set())
    _FakeSession.calls.clear()


def test_get_ud_parses_feed_and_lobby(monkeypatch: pytest.MonkeyPatch, offline) -> None:
    scraper = _FakeScraper(
        {underdog.UD_LINES_URL: _FEED, _core_url("MLB"): {}, _core_url("NBA"): _NBA_LOBBY}
    )
    monkeypatch.setattr(underdog, "_get_scraper", lambda: scraper)

    assert underdog.get_ud() == _EXPECTED
    assert scraper.calls == [underdog.UD_LINES_URL, _core_url("MLB"), _core_url("NBA")]
    assert _FakeSession.calls == [_ALT_F1_URL]


def test_get_ud_empty_feed_stops(monkeypatch: pytest.MonkeyPatch, offline) -> None:
    scraper = _FakeScraper({underdog.UD_LINES_URL: {}})
    monkeypatch.setattr(underdog, "_get_scraper", lambda: scraper)

    assert underdog.get_ud() == {}
    assert scraper.calls == [underdog.UD_LINES_URL]
    assert _FakeSession.calls == []


def test_alt_lines_stop_on_empty_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rate-limited origin answers 200 with no projections; the pass must not burn the board."""
    calls: list[str] = []

    class _EmptySession(_FakeSession):
        def get(self, url, timeout=None):
            calls.append(url)
            time.sleep(0.005)
            return _FakeResponse({"projections": []})

    monkeypatch.setattr(
        underdog,
        "requests",
        types.SimpleNamespace(Session=_EmptySession, RequestException=Exception),
    )
    candidates = [({**_LEBRON, "Line": float(line)}, f"ou-{line}") for line in range(60)]

    assert underdog._ud_alt_lines(candidates) == []
    assert underdog.UD_ALT_LINES_EMPTY_STREAK <= len(calls) < len(candidates)
