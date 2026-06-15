"""Pure helpers behind the deep-dive dialog (market-trust read, sidecar decode,
detail-row lookup, edge badge) and the Model-Lab nav handoff.

The Streamlit render itself is exercised by the dashboard AppTest smoke; these pin the
unit logic the render depends on — bet-side / window selection, JSON decode, the
five-key join, and the league+market preselect a "View in Model Lab" click hands off.
"""

from __future__ import annotations

import json

import pandas as pd

from sportstradamus.dashboard.components.deep_dive import (
    _decode,
    _detail_row,
    _edge_badge,
    _live_read,
)
from sportstradamus.dashboard.narrative import lab_filters_for_nav_cell


def _live_df() -> pd.DataFrame:
    # Same cell at both windows; the 7-day row carries decoy precisions so a test
    # failure to filter on window_days==30 is visible.
    return pd.DataFrame(
        [
            {"league": "NBA", "market": "PTS", "window_days": 7,
             "n_settled": 200, "precision_over_live": 0.10, "precision_under_live": 0.20},
            {"league": "NBA", "market": "PTS", "window_days": 30,
             "n_settled": 120, "precision_over_live": 0.63, "precision_under_live": 0.58},
            {"league": "NBA", "market": "AST", "window_days": 30,
             "n_settled": 5, "precision_over_live": 0.80, "precision_under_live": float("nan")},
        ]
    )


def test_live_read_bet_side_and_window():
    live = _live_df()
    assert _live_read(live, "NBA", "PTS", "Over") == (0.63, 120)
    assert _live_read(live, "NBA", "PTS", "Under") == (0.58, 120)


def test_live_read_sparse_returns_count():
    # AST has only 5 settled (below the panel's threshold) — the read still returns
    # the count so the caller can show the honest "needs more" caption.
    assert _live_read(_live_df(), "NBA", "AST", "Over") == (0.80, 5)


def test_live_read_nan_precision_is_none():
    prec, n = _live_read(_live_df(), "NBA", "AST", "Under")
    assert prec is None and n == 5


def test_live_read_absent_cell_and_empty():
    assert _live_read(_live_df(), "NBA", "REB", "Over") == (None, 0)
    assert _live_read(pd.DataFrame(), "NBA", "PTS", "Over") == (None, 0)


def test_decode_json_and_degenerate():
    payload = json.dumps([{"comp": "X", "n_games": 2}])
    row = pd.Series({"comps_vs_opp": payload, "volume_trend": "", "other_stats": None})
    assert _decode(row, "comps_vs_opp") == [{"comp": "X", "n_games": 2}]
    assert _decode(row, "volume_trend") == []
    assert _decode(row, "other_stats") == []
    assert _decode(None, "comps_vs_opp") == []


def test_detail_row_five_key_join():
    details = pd.DataFrame(
        [
            {"League": "NBA", "Date": "2026-06-15", "Player": "A. One",
             "Market": "PTS", "Opponent": "BOS", "volume_trend": "[1]"},
            {"League": "NBA", "Date": "2026-06-15", "Player": "A. One",
             "Market": "AST", "Opponent": "BOS", "volume_trend": "[2]"},
        ]
    )
    row = pd.Series(
        {"League": "NBA", "Date": "2026-06-15", "Player": "A. One", "Market": "AST", "Opponent": "BOS"}
    )
    hit = _detail_row(row, details)
    assert hit is not None and hit["volume_trend"] == "[2]"

    miss = pd.Series(
        {"League": "NBA", "Date": "2026-06-15", "Player": "A. One", "Market": "REB", "Opponent": "BOS"}
    )
    assert _detail_row(miss, details) is None
    assert _detail_row(row, pd.DataFrame()) is None


def test_edge_badge_sign_color():
    assert ":green[" in _edge_badge(1.12) and "+12%" in _edge_badge(1.12)
    assert ":red[" in _edge_badge(0.95) and "-5%" in _edge_badge(0.95)


def test_lab_filters_for_nav_cell():
    assert lab_filters_for_nav_cell(("NBA", "PTS")) == {"lab_leagues": ["NBA"], "lab_markets": ["PTS"]}
    assert lab_filters_for_nav_cell(None) == {}
    assert lab_filters_for_nav_cell(()) == {}
