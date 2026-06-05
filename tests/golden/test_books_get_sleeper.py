"""Characterization of ``books.get_sleeper`` — the Sleeper payload parser.

``get_sleeper`` is network-bound and stubbed (``-> dict``) in the integration
suite, so it has no behavioral coverage. This pins its output on a hand-built
payload that exercises every parse seam — the games map (dict-form vs
string-form team metadata), the ``available`` + ``alt`` line merge, per-line
outcome padding (over-only and under-only), opponent resolution, and the
unknown-player skip — so the CC-19 decomposition can be proven behavior
preserving.
"""

from __future__ import annotations

from http import HTTPStatus

import pytest

from sportstradamus import books

_PLAYERS_NBA = {
    "p1": {"first_name": "LeBron", "last_name": "James", "team": "LAL"},
    "p2": {"first_name": "Jayson", "last_name": "Tatum", "team": "BOS"},
}

_GAMES = [
    {
        "sport": "NBA",
        "game_id": "g1",
        "metadata": {"home_team": {"team": "LAL"}, "away_team": "BOS"},
        "date": "2026-06-03",
    },
]

_AVAILABLE = [
    {
        "sport": "NBA",
        "subject_id": "p1",
        "game_id": "g1",
        "wager_type": "Points",
        "options": [
            {"outcome": "over", "outcome_value": 25.5, "payout_multiplier": 1.0},
            {"outcome": "under", "outcome_value": 25.5, "payout_multiplier": 1.2},
        ],
    },
    {
        "sport": "NBA",
        "subject_id": "p2",
        "game_id": "g1",
        "wager_type": "Rebounds",
        "options": [
            {"outcome": "over", "outcome_value": 8.5, "payout_multiplier": 0.9},
        ],
    },
    {
        "sport": "NBA",
        "subject_id": "ZZZ",
        "game_id": "g1",
        "wager_type": "Points",
        "options": [
            {"outcome": "over", "outcome_value": 1.5, "payout_multiplier": 1.0},
        ],
    },
]

_ALT = [
    {
        "sport": "NBA",
        "subject_id": "p1",
        "game_id": "g1",
        "wager_type": "Assists",
        "options": [
            {"outcome": "under", "outcome_value": 5.5, "payout_multiplier": 1.1},
        ],
    },
]

_ROUTES = {
    books.SLEEPER_AVAILABLE_URL: _AVAILABLE,
    books.SLEEPER_ALT_URL: _ALT,
    books.SLEEPER_GAMES_URL: _GAMES,
    books.SLEEPER_PLAYERS_URL.format(league="NBA"): _PLAYERS_NBA,
}


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = HTTPStatus.OK
        self._payload = payload

    def json(self):
        return self._payload


class _FakeRequests:
    @staticmethod
    def get(url):
        return _FakeResponse(_ROUTES[url])


_EXPECTED = {
    "NBA": {
        "Points": [
            {
                "Player": "Lebron James",
                "League": "NBA",
                "Team": "LAL",
                "Opponent": "BOS",
                "Date": "2026-06-03",
                "Market": "Points",
                "Line": 25.5,
                "Boost_Over": 1.0,
                "Boost_Under": 1.2,
            },
        ],
        "Rebounds": [
            {
                "Player": "Jayson Tatum",
                "League": "NBA",
                "Team": "BOS",
                "Opponent": "LAL",
                "Date": "2026-06-03",
                "Market": "Rebounds",
                "Line": 8.5,
                "Boost_Over": 0.9,
                "Boost_Under": 0.0,
            },
        ],
        "Assists": [
            {
                "Player": "Lebron James",
                "League": "NBA",
                "Team": "LAL",
                "Opponent": "BOS",
                "Date": "2026-06-03",
                "Market": "Assists",
                "Line": 5.5,
                "Boost_Over": 0.0,
                "Boost_Under": 1.1,
            },
        ],
    },
}


def test_get_sleeper_parses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(books, "requests", _FakeRequests)

    assert books.get_sleeper() == _EXPECTED
