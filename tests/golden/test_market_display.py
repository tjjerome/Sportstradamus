"""Golden pins for the render-only market display translation layer."""

import pytest

from sportstradamus.helpers import market_display_name, stat_meta
from sportstradamus.helpers.market_display import _market_display
from sportstradamus.leg_schema import build_leg, leg_label

_FANTASY_CELLS = [
    (league, slug)
    for league, markets in _market_display().items()
    for slug in markets
    if "fantasy" in slug
]


def test_every_stat_meta_cell_has_a_display_mapping():
    for league, markets in stat_meta.items():
        for market in markets:
            display = market_display_name(league, market)
            assert display != market, f"{league}/{market} has no display mapping"


@pytest.mark.parametrize(("league", "slug"), _FANTASY_CELLS)
def test_every_fantasy_market_reads_as_plain_fantasy_points(league, slug):
    # The platform and the player's role are already on screen beside the name, so
    # "(Underdog)", "Hitter", "Pitcher", "Goalie" and "Skater" are noise in it.
    assert market_display_name(league, slug) == "Fantasy Points"


@pytest.mark.parametrize(
    ("league", "board_name", "label"),
    [
        ("NHL", "AST", "Assists"),
        ("NHL", "PTS", "Points"),
        ("NHL", "BLK", "Blocked Shots"),
        ("NBA", "fantasy points underdog", "Fantasy Points"),
        ("WNBA", "fantasy points underdog", "Fantasy Points"),
    ],
)
def test_board_name_reads_as_the_leagues_own_label(league, board_name, label):
    # The board names these markets by their platform's map; the display table names
    # them by the league's own key.
    assert market_display_name(league, board_name) == label


def test_unmapped_league_falls_back_to_slug():
    assert market_display_name("XFL", "PTS") == "PTS"


def test_unmapped_slug_in_known_league_falls_back_to_slug():
    assert market_display_name("NBA", "not_a_real_market") == "not_a_real_market"


def test_unmapped_board_name_falls_back_to_the_leagues_own_key():
    assert market_display_name("NBA", "not a market underdog") == "not a market prizepicks"


def test_leg_label_renders_display_name():
    leg = build_leg(
        {
            "Player": "Luka Doncic",
            "Team": "LAL",
            "Market": "PTS",
            "Stat": "PTS",
            "Bet": "Over",
            "Line": 26.5,
            "League": "NBA",
            "Game": "LAL/GSW",
            "Date": "2026-07-03",
            "Platform": "Underdog",
            "Win Prob": 0.623,
        }
    )
    label = leg_label(leg)
    assert "Points" in label
    assert "PTS" not in label


def test_leg_label_tolerates_leg_with_no_league_key():
    narrow_leg = {"player": "Luka Doncic", "bet": "Over", "line": 26.5, "market": "PTS"}
    label = leg_label(narrow_leg)
    assert label == "Luka Doncic Over 26.5 PTS"


def test_leg_stored_under_the_board_name_reads_as_the_leagues_own_label():
    # A leg saved while the board's Stat copied its Market holds the code in both.
    stored = {
        "player": "Connor McDavid",
        "bet": "Over",
        "line": 1.5,
        "market": "AST",
        "stat": "AST",
        "league": "NHL",
    }
    assert leg_label(stored) == "Connor McDavid Over 1.5 Assists"
