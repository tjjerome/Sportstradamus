"""Unit tests for the dashboard parlay-leg helpers (offer lookup + re-exported parse)."""

import pandas as pd

from sportstradamus.dashboard.legs import find_offer_idx, parse_leg


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
    miss = find_offer_idx(parse_leg("Ayo Dosunmu Over 12.5 Points - 40%, 1.0x"), offers)
    assert miss is None
    assert find_offer_idx(None, offers) is None


def test_find_offer_idx_resolves_platform_market_codes():
    # Real data: parlay legs carry the platform's display label, but
    # current_offers.parquet stores the canonical code. find_offer_idx must
    # translate via stat_map[platform], including the spaced-name fallback.
    offers = pd.DataFrame(
        {
            "Player": ["Ayo Dosunmu", "Victor Wembanyama", "Jokic"],
            "Bet": ["Over", "Under", "Over"],
            "Market": ["PRA", "FGA", "FG3M"],
            "Line": [16.5, 17.5, 2.5],
        }
    )
    # Underdog: spaced display name -> "Pts+Rebs+Asts" -> PRA
    assert (
        find_offer_idx(
            parse_leg("Ayo Dosunmu Over 16.5 Pts + Rebs + Asts - 75%, 1.0x"),
            offers,
            "Underdog",
        )
        == 0
    )
    # Underdog: "FG Attempted" -> FGA
    assert (
        find_offer_idx(
            parse_leg("Victor Wembanyama Under 17.5 FG Attempted - 83%, 1.0x"),
            offers,
            "Underdog",
        )
        == 1
    )
    # Sleeper: snake key "threes_made" -> FG3M
    assert (
        find_offer_idx(
            parse_leg("Jokic Over 2.5 threes_made - 50%, 1.0x"),
            offers,
            "Sleeper",
        )
        == 2
    )
