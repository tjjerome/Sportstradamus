"""Characterization of ``books.get_ud`` — the Underdog payload parser.

``get_ud`` is network-bound and stubbed (``-> dict``) in the integration suite,
so it has no behavioral coverage. This pins its output on a hand-built payload
that exercises every parse seam — team / player / match / appearance maps, the
``games`` + ``solo_games`` title parse, over/under parsing (boosts, opponent
resolution, MLB-fantasy reclassification), and a rival (vs.) line — so the
CC-45 decomposition can be proven behavior-preserving.
"""

from __future__ import annotations

import pytest

from sportstradamus import books

_TEAMS = {
    "teams": [
        {"id": "t1", "abbr": "LAL"},
        {"id": "t2", "abbr": "BOS"},
        {"id": "t3", "abbr": "NYY"},
        {"id": "t4", "abbr": "HOU"},
    ]
}

_LINES = {
    "players": [
        {"id": "p1", "first_name": "LeBron", "last_name": "James", "sport_id": "NBA"},
        {"id": "p2", "first_name": "Jayson", "last_name": "Tatum", "sport_id": "NBA"},
        {"id": "p3", "first_name": "Aaron", "last_name": "Judge", "sport_id": "MLB"},
    ],
    "games": [
        {
            "id": "g1",
            "home_team_id": "t1",
            "away_team_id": "t2",
            "sport_id": "NBA",
            "scheduled_at": "2026-06-03T23:00:00Z",
        },
        {
            "id": "g3",
            "home_team_id": "t3",
            "away_team_id": "t4",
            "sport_id": "MLB",
            "scheduled_at": "2026-06-03T23:00:00Z",
        },
    ],
    "solo_games": [
        {
            "id": "g2",
            "title": "LAL @ BOS",
            "sport_id": "NBA",
            "scheduled_at": "2026-06-03T23:00:00Z",
        },
        {
            "id": "g4",
            "title": "Lakers vs Celtics",
            "sport_id": "NBA",
            "scheduled_at": "2026-06-03T23:00:00Z",
        },
    ],
    "appearances": [
        {"id": "a1", "player_id": "p1", "team_id": "t1", "match_id": "g1"},
        {"id": "a2", "player_id": "p2", "team_id": "t2", "match_id": "g1"},
        {"id": "a3", "player_id": "p3", "team_id": "t3", "match_id": "g3"},
    ],
    "over_under_lines": [
        {
            "over_under": {"appearance_stat": {"appearance_id": "a1", "display_stat": "Points"}},
            "options": [
                {"choice": "higher", "payout_multiplier": "1.0"},
                {"choice": "lower", "payout_multiplier": "1.2"},
            ],
            "stat_value": "25.5",
        },
        {
            "over_under": {"appearance_stat": {"appearance_id": "a2", "display_stat": "Rebounds"}},
            "options": [
                {"choice": "higher", "payout_multiplier": "0.9"},
                {"choice": "lower", "payout_multiplier": "1.1"},
            ],
            "stat_value": "8.5",
        },
        {
            "over_under": {
                "appearance_stat": {"appearance_id": "a3", "display_stat": "Fantasy Points"}
            },
            "options": [
                {"choice": "higher", "payout_multiplier": "1.0"},
                {"choice": "lower", "payout_multiplier": "1.0"},
            ],
            "stat_value": "7.5",
        },
    ],
}

_RIVALS = {
    "players": [],
    "games": [],
    "appearances": [],
    "rival_lines": [
        {
            "options": [
                {
                    "appearance_stat": {"appearance_id": "a1", "display_stat": "Fewer Points"},
                    "spread": "2.0",
                },
                {
                    "appearance_stat": {"appearance_id": "a2", "display_stat": "Fewer Points"},
                    "spread": "1.0",
                },
            ]
        },
    ],
}


class _FakeScraper:
    def get(self, url):
        return {
            books.UD_TEAMS_URL: _TEAMS,
            books.UD_LINES_URL: _LINES,
            books.UD_RIVALS_URL: _RIVALS,
        }[url]


_EXPECTED = {
    "MLB": {
        "Hitter Fantasy Points": [
            {
                "Player": "Aaron Judge",
                "League": "MLB",
                "Team": "NYY",
                "Opponent": "HOU",
                "Date": "2026-06-03",
                "Commence": "2026-06-03T23:00:00Z",
                "Market": "Hitter Fantasy Points",
                "Line": 7.5,
                "Boost_Over": 1.0,
                "Boost_Under": 1.0,
            },
        ],
    },
    "NBA": {
        "Points": [
            {
                "Player": "Lebron James",
                "League": "NBA",
                "Team": "LAL",
                "Opponent": "BOS",
                "Date": "2026-06-03",
                "Commence": "2026-06-03T23:00:00Z",
                "Market": "Points",
                "Line": 25.5,
                "Boost_Over": 1.0,
                "Boost_Under": 1.2,
            },
            {
                "Player": "Lebron James vs. Jayson Tatum",
                "League": "NBA",
                "Team": "LAL/BOS",
                "Opponent": "BOS/LAL",
                "Date": "2026-06-03",
                "Commence": "2026-06-03T23:00:00Z",
                "Market": "H2H Points",
                "Line": 1.0,
                "Boost": 1,
            },
        ],
        "Rebounds": [
            {
                "Player": "Jayson Tatum",
                "League": "NBA",
                "Team": "BOS",
                "Opponent": "LAL",
                "Date": "2026-06-03",
                "Commence": "2026-06-03T23:00:00Z",
                "Market": "Rebounds",
                "Line": 8.5,
                "Boost_Over": 0.9,
                "Boost_Under": 1.1,
            },
        ],
    },
}


def test_get_ud_parses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(books, "_get_scraper", _FakeScraper)
    monkeypatch.setattr(books, "get_mlb_pitchers", dict)
    monkeypatch.setattr(books, "nhl_goalies", set())

    assert books.get_ud() == _EXPECTED
