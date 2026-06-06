"""Characterization of ``moneylines._archive_event_props`` — the per-event Odds
API prop parser.

This is the gnarliest parser in the module (CC 21): it splits an event's
bookmaker markets into player props, game totals, and spreads, with several
subtle player-prop normalizations. It IS exercised by the integration suite via
``_get_props_from_fixtures``, but the fixtures don't necessarily hit every
branch, so this pins the tricky ones directly on a synthetic event:

* multi-point dedup to the "true" line (price closest to even),
* the ``Yes``/``No`` -> ``Over``/``Under`` rename with the 0.5 point default,
* the single-``Under`` -> ``Over`` flip with its price inversion,
* a normal two-sided line,
* a ``totals`` market (whose bucket name parses to ``""``), and
* a ``spreads`` market.

It also pins a latent property preserved by the CC-21 decomposition: the
totals/spreads bucket blend looks up ``spread_*`` (keyed by the parsed
spread-name) with a *book* key, so the spread adjustment always resolves to 0
and the written value is simply ``total / 2``.
"""

from __future__ import annotations

from sportstradamus import moneylines

_PROPS = {"NBA": {"player_points": "PTS"}}

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
                        {
                            "description": "Anthony Davis",
                            "name": "Over",
                            "point": 20.5,
                            "price": 1.5,
                        },
                        {
                            "description": "Anthony Davis",
                            "name": "Under",
                            "point": 20.5,
                            "price": 2.5,
                        },
                        {
                            "description": "Anthony Davis",
                            "name": "Over",
                            "point": 22.5,
                            "price": 2.0,
                        },
                        {
                            "description": "Anthony Davis",
                            "name": "Under",
                            "point": 22.5,
                            "price": 2.0,
                        },
                        {"description": "Stephen Curry", "name": "Yes", "price": 1.8},
                        {"description": "Stephen Curry", "name": "No", "price": 2.0},
                        {
                            "description": "Russell Westbrook",
                            "name": "Under",
                            "point": 5.5,
                            "price": 2.5,
                        },
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 220.5, "price": 1.9},
                        {"name": "Under", "point": 220.5, "price": 1.9},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "point": -3.5, "price": 1.9},
                        {"name": "Boston Celtics", "point": 3.5, "price": 1.9},
                    ],
                },
            ],
        },
    ],
}


class _FakeArchive:
    def __init__(self):
        self.player_books = []
        self.team_books = []

    def merge_player_books(self, league, market, gameDate, player, ev, lines, observed_at=None):
        self.player_books.append((league, market, gameDate, player, ev, lines, observed_at))

    def set_team_books(self, league, market, gameDate, team, books):
        self.team_books.append((league, market, gameDate, team, books))


_EXPECTED_PLAYER = [
    ("NBA", "PTS", "2026-06-04", "Anthony Davis", {"draftkings": 22.50000000000054}, [22.5], None),
    ("NBA", "PTS", "2026-06-04", "Lebron James", {"draftkings": 25.500000000000618}, [25.5], None),
    (
        "NBA",
        "PTS",
        "2026-06-04",
        "Russell Westbrook",
        {"draftkings": 5.999344441953691},
        [5.5],
        None,
    ),
    ("NBA", "PTS", "2026-06-04", "Stephen Curry", {"draftkings": 0.5785114444497812}, [0.5], None),
]

_EXPECTED_TEAM = [
    ("NBA", "", "2026-06-04", "LAL", {"draftkings": 110.24999999999997}),
    ("NBA", "", "2026-06-04", "BOS", {"draftkings": 110.24999999999997}),
]


def test_archive_event_props_parses_event(assert_player_books_close) -> None:
    archive = _FakeArchive()

    moneylines._archive_event_props(archive, _GAME, "NBA", _PROPS, "2026-06-04")

    assert_player_books_close(archive.player_books, _EXPECTED_PLAYER)
    assert archive.team_books == _EXPECTED_TEAM
