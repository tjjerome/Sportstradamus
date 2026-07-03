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
    hit = find_offer_idx({"player": "Ayo Dosunmu", "bet": "Over", "market": "Points", "line": 9.5}, offers)
    assert hit == 0
    miss = find_offer_idx({"player": "Ayo Dosunmu", "bet": "Over", "market": "Points", "line": 12.5}, offers)
    assert miss is None
    assert find_offer_idx(None, offers) is None


def test_find_offer_idx_pins_platform_when_same_leg_on_both_books():
    """The same (player, market, line) trades on both books with different
    boosts; the match must follow the requested platform, not the first row —
    otherwise a slip snapshots the wrong book's Boost and misprices (the 1.78x
    Sleeper multiplier double-counted under the Underdog payout curve)."""
    offers = pd.DataFrame(
        {
            "Player": ["Aaliyah Edwards", "Aaliyah Edwards"],
            "Bet": ["Over", "Over"],
            "Market": ["PTS", "PTS"],
            "Line": [10.5, 10.5],
            "Platform": ["Sleeper", "Underdog"],
            "Boost": [1.78, 1.00],
        }
    )
    leg = {"player": "Aaliyah Edwards", "bet": "Over", "market": "PTS", "line": 10.5}
    assert find_offer_idx(leg, offers, "Underdog") == 1  # not row 0 (Sleeper)
    assert find_offer_idx(leg, offers, "Sleeper") == 0
