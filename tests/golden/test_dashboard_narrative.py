"""Pins for ``dashboard.narrative`` — per-game thesis headline + context strip.

Pure lookups keyed on the canonical ``Game`` slash key plus ``Date`` (so a
doubleheader's two games stay distinct); no Streamlit, no I/O.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.narrative import context_strip, home_away, top_thesis


def test_home_away_orders_home_first_from_home_flag():
    # The Game key is alphabetical (TOR/WAS); the Home flag puts the host first.
    group = pd.DataFrame(
        [
            {"Game": "TOR/WAS", "Team": "WAS", "Opponent": "TOR", "Home": True},
            {"Game": "TOR/WAS", "Team": "TOR", "Opponent": "WAS", "Home": False},
        ]
    )
    assert home_away(group) == ("WAS", "TOR")


def test_home_away_falls_back_to_canonical_without_home_column():
    group = pd.DataFrame(
        [
            {"Game": "TOR/WAS", "Team": "TOR", "Opponent": "WAS"},
            {"Game": "TOR/WAS", "Team": "WAS", "Opponent": "TOR"},
        ]
    )
    assert home_away(group) == ("TOR", "WAS")  # canonical (alphabetical) order


def test_top_thesis_picks_highest_ev():
    parlays = pd.DataFrame(
        [
            {"Game": "NYK/SAS", "Date": "2026-06-13", "Model EV": 1.2, "Thesis": "low"},
            {"Game": "NYK/SAS", "Date": "2026-06-13", "Model EV": 1.9, "Thesis": "high"},
            {"Game": "BOS/PHI", "Date": "2026-06-13", "Model EV": 5.0, "Thesis": "other game"},
        ]
    )
    assert top_thesis(parlays, game="NYK/SAS", date="2026-06-13") == "high"


def test_top_thesis_doubleheader_splits_on_date():
    parlays = pd.DataFrame(
        [
            {"Game": "NYL/WAS", "Date": "2026-06-13", "Model EV": 1.5, "Thesis": "game one"},
            {"Game": "NYL/WAS", "Date": "2026-06-14", "Model EV": 9.0, "Thesis": "game two"},
        ]
    )
    assert top_thesis(parlays, game="NYL/WAS", date="2026-06-13") == "game one"


def test_top_thesis_empty_when_no_match():
    parlays = pd.DataFrame(
        [{"Game": "NYK/SAS", "Date": "2026-06-13", "Model EV": 1.2, "Thesis": "x"}]
    )
    assert top_thesis(parlays, game="ZZZ/YYY", date="2026-06-13") == ""
    assert top_thesis(pd.DataFrame(), game="NYK/SAS", date="2026-06-13") == ""


def test_context_strip_returns_fields():
    ctx = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "NYK/SAS",
                "Date": "2026-06-13",
                "game_total": 216.1,
                "spread": 5.3,
                "fav_team": "SAS",
                "shape": "coinflip",
                "baseline_total": 220.0,
            }
        ]
    )
    assert context_strip(ctx, game="NYK/SAS", date="2026-06-13") == {
        "game_total": 216.1,
        "spread": 5.3,
        "fav_team": "SAS",
        "shape": "coinflip",
        "baseline_total": 220.0,
    }


def test_context_strip_none_when_missing():
    ctx = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "NYK/SAS",
                "Date": "2026-06-13",
                "game_total": 216.1,
                "spread": 5.3,
                "fav_team": "SAS",
                "shape": "coinflip",
            }
        ]
    )
    assert context_strip(ctx, game="BOS/PHI", date="2026-06-13") is None
    assert context_strip(pd.DataFrame(), game="NYK/SAS", date="2026-06-13") is None


def test_context_strip_doubleheader_splits_on_date():
    ctx = pd.DataFrame(
        [
            {
                "League": "WNBA",
                "Game": "DAL/POR",
                "Date": "2026-06-13",
                "game_total": 163.3,
                "spread": 0.0,
                "fav_team": None,
                "shape": "even",
                "baseline_total": 165.6,
            },
            {
                "League": "WNBA",
                "Game": "DAL/POR",
                "Date": "2026-06-14",
                "game_total": 170.0,
                "spread": 4.0,
                "fav_team": "DAL",
                "shape": "coinflip",
                "baseline_total": 165.6,
            },
        ]
    )
    strip = context_strip(ctx, game="DAL/POR", date="2026-06-14")
    assert strip["fav_team"] == "DAL"
    assert strip["shape"] == "coinflip"
