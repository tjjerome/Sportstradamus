"""Characterization of ``moneylines.get_moneylines`` — the Odds API game-odds
ingest.

``get_moneylines`` is network-bound and is skipped in the integration suite
(``confer`` only calls it when ``fixture_dir is None``), so it has no behavioral
coverage. This pins its archive writes on a synthetic Odds API payload that
exercises the sports-index filter (active + league-of-interest), the
non-historical request path, the per-game date-window + team-abbreviation
resolution, and the h2h / totals / spreads book parse (including the
totals+spread blend written to the ``Totals`` bucket) so the CC-24
decomposition can be proven behavior preserving.
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

_SPORTS = [
    {"key": "basketball_nba", "title": "NBA", "active": True},
    {"key": "americanfootball_nfl", "title": "NFL", "active": False},
    {"key": "icehockey_nhl", "title": "NHL", "active": True},
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
    def __init__(self, payload, headers=None):
        self.status_code = HTTPStatus.OK
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        return self._payload


def _fake_get_with_retry(url, params=None):
    if url == moneylines.ODDS_API_SPORTS_URL:
        return _FakeResponse(_SPORTS, {"X-Requests-Remaining": "1000"})
    if url == moneylines.ODDS_API_ODDS_URL.format(sport="basketball_nba"):
        return _FakeResponse(_GAMES)
    raise AssertionError(f"unexpected url {url}")


class _FakeArchive:
    def __init__(self):
        self.calls = []

    def set_team_books(self, league, market, date, team, books):
        self.calls.append((league, market, date, team, books))


_EXPECTED = [
    ("NBA", "Moneyline", _GAME_DATE, "BOS", {"draftkings": 0.42016806722689076}),
    ("NBA", "Moneyline", _GAME_DATE, "LAL", {"draftkings": 0.5798319327731092}),
    ("NBA", "Totals", _GAME_DATE, "BOS", {"draftkings": 106.30092918426874}),
    ("NBA", "Totals", _GAME_DATE, "LAL", {"draftkings": 109.72999141502845}),
]


def test_get_moneylines_writes_team_books(monkeypatch) -> None:
    monkeypatch.setattr(moneylines, "_get_with_retry", _fake_get_with_retry)
    archive = _FakeArchive()

    returned = moneylines.get_moneylines(
        archive, {"odds_api": "k", "odds_api_plus": "kp"}, date=_DATE
    )

    assert returned is archive
    assert archive.calls == _EXPECTED
