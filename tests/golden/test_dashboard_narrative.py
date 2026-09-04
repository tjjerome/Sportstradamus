"""Pins for ``dashboard.narrative`` — per-game thesis headline + context strip.

Pure lookups keyed on the canonical ``Game`` slash key plus ``Date`` (so a
doubleheader's two games stay distinct); no Streamlit, no I/O.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.narrative import (
    SHAPE_CAPTION,
    SHAPE_DISPLAY,
    SHAPE_HELP,
    bet_arrow,
    context_strip,
    game_headline,
    home_away,
    match_label,
    storyless_prophecy,
    top_thesis,
)


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


def test_game_headline_story_wins_over_parlay():
    stories = pd.DataFrame(
        [
            {"Game": "NYK/SAS", "headline": "low", "model_ev": 1.2},
            {"Game": "NYK/SAS", "headline": "The Knicks pull away", "model_ev": 1.9},
        ]
    )
    parlays = pd.DataFrame(
        [{"Game": "NYK/SAS", "Date": "2026-06-13", "Model EV": 5.0, "Thesis": "parlay says"}]
    )
    assert (
        game_headline(stories, parlays, game="NYK/SAS", date="2026-06-13") == "The Knicks pull away"
    )


def test_game_headline_falls_back_to_parlay_when_no_story_row():
    stories = pd.DataFrame([{"Game": "BOS/PHI", "headline": "other game", "model_ev": 2.0}])
    parlays = pd.DataFrame(
        [{"Game": "NYK/SAS", "Date": "2026-06-13", "Model EV": 1.2, "Thesis": "parlay thesis"}]
    )
    assert game_headline(stories, parlays, game="NYK/SAS", date="2026-06-13") == "parlay thesis"


def test_game_headline_blank_story_headline_falls_back():
    stories = pd.DataFrame([{"Game": "NYK/SAS", "headline": "", "model_ev": 3.0}])
    parlays = pd.DataFrame(
        [{"Game": "NYK/SAS", "Date": "2026-06-13", "Model EV": 1.2, "Thesis": "parlay backup"}]
    )
    assert game_headline(stories, parlays, game="NYK/SAS", date="2026-06-13") == "parlay backup"


def test_game_headline_empty_when_both_sources_empty():
    assert game_headline(pd.DataFrame(), pd.DataFrame(), game="NYK/SAS", date="2026-06-13") == ""


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
                "ml_fav_prob": 0.64,
            }
        ]
    )
    assert context_strip(ctx, game="NYK/SAS", date="2026-06-13") == {
        "game_total": 216.1,
        "spread": 5.3,
        "fav_team": "SAS",
        "shape": "coinflip",
        "baseline_total": 220.0,
        "ml_fav_prob": 0.64,
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
                "ml_fav_prob": float("nan"),
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
                "ml_fav_prob": 0.58,
            },
        ]
    )
    strip = context_strip(ctx, game="DAL/POR", date="2026-06-14")
    assert strip["fav_team"] == "DAL"
    assert strip["shape"] == "coinflip"
    assert strip["ml_fav_prob"] == 0.58


def test_bet_arrow_over_is_up_and_on_token_green():
    arrow = bet_arrow("Over")
    assert arrow == bet_arrow("Over")  # stable constant, not regenerated per call
    assert theme.GREEN in arrow
    assert "M4.5 0 9 10 0 10Z" in arrow


def test_bet_arrow_under_is_down_and_on_token_red():
    arrow = bet_arrow("Under")
    assert theme.RED in arrow
    assert "M4.5 10 9 0 0 0Z" in arrow


def test_match_label_away_uses_at_sign():
    assert match_label("LVA", "IND", home=False) == "LVA @ IND"


def test_match_label_home_uses_v():
    assert match_label("LVA", "IND", home=True) == "LVA v IND"


def test_shape_display_renames_the_even_sentinel_to_clouded():
    assert SHAPE_DISPLAY["even"] == "Clouded"
    assert SHAPE_CAPTION["even"] == "unquoted"
    # Same five pipeline shapes, in the Tonight legend's left-to-right order.
    assert list(SHAPE_DISPLAY) == list(SHAPE_CAPTION)
    assert set(SHAPE_DISPLAY) == {"shootout", "blowout", "coinflip", "even", "grind"}


def test_shape_help_glosses_the_clouded_shape():
    assert "clouded (no line quoted yet)" in SHAPE_HELP


def test_storyless_prophecy_names_the_favored_legs():
    assert storyless_prophecy(20) == "No prophecy yet — 20 favored legs, no story binds them"
    assert storyless_prophecy(2).startswith("No prophecy yet — 2 favored legs")


def test_storyless_prophecy_stays_thin_edges_below_two_legs():
    assert storyless_prophecy(1) == "No prophecy tonight — thin edges"
    assert storyless_prophecy(0) == "No prophecy tonight — thin edges"
