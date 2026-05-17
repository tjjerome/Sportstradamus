"""Unit tests for the dashboard parlay-leg helpers and family naming."""

import pandas as pd

from sportstradamus.dashboard_detail import (
    family_labels_for_game,
    find_offer_idx,
    parse_leg,
)


def test_parse_leg_well_formed():
    assert parse_leg("Ayo Dosunmu Over 9.5 Points - 78.1%, 1.0x") == {
        "Player": "Ayo Dosunmu",
        "Bet": "Over",
        "Line": 9.5,
        "Market": "Points",
    }


def test_parse_leg_multiword_market_and_under():
    assert parse_leg("Cade Cunningham Under 6.5 Pass + Rush Yds - 55%, 1.04x") == {
        "Player": "Cade Cunningham",
        "Bet": "Under",
        "Line": 6.5,
        "Market": "Pass + Rush Yds",
    }


def test_parse_leg_malformed_returns_none():
    assert parse_leg("") is None
    assert parse_leg(None) is None
    assert parse_leg("just some text with no side") is None
    assert parse_leg("Player Over notanumber Points - 1%") is None


def test_find_offer_idx_hit_and_miss():
    offers = pd.DataFrame(
        {
            "Player": ["Ayo Dosunmu", "Nikola Jokic"],
            "Bet": ["Over", "Over"],
            "Market": ["Points", "Rebounds"],
            "Line": [9.5, 10.5],
        }
    )
    hit = find_offer_idx(parse_leg("Ayo Dosunmu Over 9.5 Points - 78%, 1.0x"), offers)
    assert hit == 0
    miss = find_offer_idx(
        parse_leg("Ayo Dosunmu Over 12.5 Points - 40%, 1.0x"), offers
    )
    assert miss is None
    assert find_offer_idx(None, offers) is None


def test_family_labels_distinct_and_deterministic():
    # "Common Star" appears in BOTH families and must never be the headliner.
    # Each family has its own distinctive player.
    game = pd.DataFrame(
        {
            "Family": [1.0, 1.0, 2.0, 2.0],
            "Leg 1": [
                "Common Star Over 25.5 Points - 60%, 1.0x",
                "Common Star Over 25.5 Points - 60%, 1.0x",
                "Common Star Over 25.5 Points - 60%, 1.0x",
                "Common Star Over 25.5 Points - 60%, 1.0x",
            ],
            "Leg 2": [
                "Alpha Guard Over 8.5 Assists - 58%, 1.0x",
                "Alpha Guard Over 8.5 Assists - 58%, 1.0x",
                "Beta Big Under 7.5 Rebounds - 57%, 1.0x",
                "Beta Big Under 7.5 Rebounds - 57%, 1.0x",
            ],
        }
    )
    labels = family_labels_for_game(game)
    assert set(labels) == {1.0, 2.0}
    assert "Common Star" not in labels[1.0]
    assert "Common Star" not in labels[2.0]
    assert "Alpha Guard" in labels[1.0]
    assert "Beta Big" in labels[2.0]
    assert labels[1.0] != labels[2.0]
    # Deterministic across calls.
    assert family_labels_for_game(game) == labels
