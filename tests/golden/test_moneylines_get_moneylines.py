"""Characterization of ``moneylines.get_moneylines`` — the Odds API game-odds
ingest.

``get_moneylines`` is network-bound and is skipped in the integration suite
(``confer`` only calls it when ``fixture_dir is None``), so it has no behavioral
coverage. This pins its archive writes on a synthetic Odds API payload that
exercises the sports-index filter (active + league-of-interest), the
non-historical request path, the per-game date-window + team-abbreviation
resolution, and the h2h / totals / spreads book parse (including the
totals+spread blend written to the ``Totals`` bucket), plus the plus-key
routing on every request and the budget governor's ``leagues`` narrowing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from http import HTTPStatus

import pytz

from sportstradamus import moneylines

_CHICAGO = pytz.timezone("America/Chicago")

# Anchor to today's calendar date at Chicago noon so ``get_moneylines``'s
# ``date.date() != datetime.today().date()`` historical check is reliably False
# regardless of the test host's wall-clock timezone.
_TODAY = datetime.today()
_DATE = _CHICAGO.localize(datetime(_TODAY.year, _TODAY.month, _TODAY.day, 12, 0))
_COMMENCE = (_DATE + timedelta(days=1)).astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_GAME_DATE = (_DATE + timedelta(days=1)).astimezone(_CHICAGO).strftime("%Y-%m-%d")

_KEYS = {"odds_api": "k", "odds_api_plus": "kp"}

_SPORTS = [
    {"key": "basketball_nba", "title": "NBA", "active": True},
    {"key": "americanfootball_nfl", "title": "NFL", "active": False},
    {"key": "icehockey_nhl", "title": "NHL", "active": True},
    {"key": "basketball_wnba", "title": "WNBA", "active": True},
]

_GAMES = [
    {
        "commence_time": _COMMENCE,
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "price": -150},
                            {"name": "Boston Celtics", "price": 130},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 220.5, "price": -110},
                            {"name": "Under", "point": 220.5, "price": -110},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "point": -3.5, "price": -110},
                            {"name": "Boston Celtics", "point": 3.5, "price": -110},
                        ],
                    },
                ],
            },
        ],
    },
]


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = HTTPStatus.OK
        self._payload = payload

    def json(self):
        return self._payload


def _fake_get_with_retry(calls):
    routes = {
        moneylines.ODDS_API_SPORTS_URL: _SPORTS,
        moneylines.ODDS_API_ODDS_URL.format(sport="basketball_nba"): _GAMES,
        moneylines.ODDS_API_ODDS_URL.format(sport="icehockey_nhl"): [],
        moneylines.ODDS_API_ODDS_URL.format(sport="basketball_wnba"): [],
    }

    def _get(url, params=None):
        calls.append((url, dict(params)))
        return _FakeResponse(routes[url])

    return _get


class _FakeArchive:
    def __init__(self):
        self.calls = []

    def set_team_books(self, league, market, date, team, books):
        self.calls.append((league, market, date, team, books))


_EXPECTED = [
    ("NBA", "Moneyline", _GAME_DATE, "BOS", {"draftkings": 0.42016806722689076}),
    ("NBA", "Moneyline", _GAME_DATE, "LAL", {"draftkings": 0.5798319327731092}),
    ("NBA", "Totals", _GAME_DATE, "BOS", {"draftkings": 106.30129478600526}),
    ("NBA", "Totals", _GAME_DATE, "LAL", {"draftkings": 109.72961408582509}),
]


def test_get_moneylines_writes_team_books(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(moneylines, "_get_with_retry", _fake_get_with_retry(calls))
    archive = _FakeArchive()

    returned = moneylines.get_moneylines(archive, _KEYS, date=_DATE)

    assert returned is archive
    assert archive.calls == _EXPECTED
    # The sports-index probe and every odds request spend from the plus key.
    # NHL rides the 5-league LEAGUES_OF_INTEREST even with an empty slate.
    assert [(url, params["apiKey"]) for url, params in calls] == [
        (moneylines.ODDS_API_SPORTS_URL, "kp"),
        (moneylines.ODDS_API_ODDS_URL.format(sport="basketball_nba"), "kp"),
        (moneylines.ODDS_API_ODDS_URL.format(sport="icehockey_nhl"), "kp"),
        (moneylines.ODDS_API_ODDS_URL.format(sport="basketball_wnba"), "kp"),
    ]


def test_get_moneylines_leagues_filter(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(moneylines, "_get_with_retry", _fake_get_with_retry(calls))
    archive = _FakeArchive()

    moneylines.get_moneylines(archive, _KEYS, date=_DATE, leagues=("NBA",))

    # WNBA is active in the index but excluded by the governor's allowance.
    assert [url for url, _ in calls] == [
        moneylines.ODDS_API_SPORTS_URL,
        moneylines.ODDS_API_ODDS_URL.format(sport="basketball_nba"),
    ]
    assert archive.calls == _EXPECTED
