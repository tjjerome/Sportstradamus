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

import pytest

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
        self.ladder_calls = []

    def merge_player_books(self, league, market, gameDate, player, ev, lines, observed_at=None):
        self.player_books.append((league, market, gameDate, player, ev, lines, observed_at))

    def set_team_books(self, league, market, gameDate, team, books):
        self.team_books.append((league, market, gameDate, team, books))

    def add_ladder(self, league, market, gameDate, player, book, rungs, observed_at=None):
        self.ladder_calls.append((league, market, gameDate, player, book, rungs, observed_at))


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


def test_archive_event_props_parses_event() -> None:
    archive = _FakeArchive()

    moneylines._archive_event_props(archive, _GAME, "NBA", _PROPS, "2026-06-04")

    # This synthetic event is built mostly from *degenerate* single-sided lines
    # (Westbrook Under-only, Curry Yes/No). get_ev extrapolates those one-sided
    # prices, and the extrapolation moves a few percent with the scipy version
    # and the (CI-stubbed) stat_cv calibration -- too much for the shared
    # assert_player_books_close 1e-4 band. Pin the parser structure exactly and
    # give the per-book EV a tolerant band; the tight symmetric-line EV pin lives
    # in test_moneylines_get_props.
    assert len(archive.player_books) == len(_EXPECTED_PLAYER)
    for got, exp in zip(archive.player_books, _EXPECTED_PLAYER, strict=True):
        *got_head, got_ev, got_line, got_obs = got
        *exp_head, exp_ev, exp_line, exp_obs = exp
        assert (got_head, got_line, got_obs) == (exp_head, exp_line, exp_obs)
        assert got_ev == pytest.approx(exp_ev, rel=0.15)
    assert archive.team_books == _EXPECTED_TEAM


def test_archive_event_props_captures_full_ladder() -> None:
    """Every offered alt-line rung is persisted, not just the even-money one the
    consensus collapses to (Anthony Davis offers 20.5 and 22.5)."""
    archive = _FakeArchive()

    moneylines._archive_event_props(archive, _GAME, "NBA", _PROPS, "2026-06-04")

    ladder = {(player, book): rungs for _, _, _, player, book, rungs, _ in archive.ladder_calls}
    davis = ladder[("Anthony Davis", "draftkings")]
    assert [r[0] for r in davis] == [20.5, 22.5]
    assert davis[0][1] == pytest.approx(0.625, abs=1e-6)
    assert davis[1][1] == pytest.approx(0.5, abs=1e-6)
