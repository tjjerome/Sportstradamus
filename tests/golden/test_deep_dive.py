"""Pure helpers behind the deep-dive dialog (sidecar decode, detail-row lookup, edge
badge) and the page-change dialog reset.

The Streamlit render itself is exercised by the dashboard AppTest smoke; these pin the
unit logic the render depends on — JSON decode, the five-key join, and the cross-page
detail-stack reset.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

import pandas as pd

from sportstradamus import data as data_pkg
from sportstradamus.dashboard.components.deep_dive import (
    _decode,
    _detail_row,
    _edge_badge,
    drop_detail_on_page_change,
)


def test_stat_tooltips_config_well_formed():
    # League-keyed {stat: gloss}; the deep-dive Other-stats tab looks tooltips up by
    # (League, stat), so the NFL per-position volume stats must each have a gloss.
    with open(pkg_resources.files(data_pkg) / "config" / "stat_tooltips.json") as f:
        tips = json.load(f)
    assert {"NBA", "WNBA", "NFL", "MLB", "NHL"} <= set(tips)
    assert {"attempts", "carries", "targets"} <= set(tips["NFL"])
    assert "MIN" in tips["NBA"]
    assert all(isinstance(v, str) and v for league in tips.values() for v in league.values())


def test_drop_detail_on_page_change():
    # Same page across reruns → the open dialog + its corr-nav history survive.
    s = {"_active_page": "board", "detail_stack": [5], "last_grid_key": "k"}
    drop_detail_on_page_change(s, "board")
    assert s["detail_stack"] == [5]

    # Navigating to another surface drops the stale dialog so it can't re-open.
    drop_detail_on_page_change(s, "receipts")
    assert s["detail_stack"] == [] and s["last_grid_key"] is None
    assert s["_active_page"] == "receipts"

    # Fresh session (no marker yet) initializes the marker and leaves nothing open.
    s2: dict = {}
    drop_detail_on_page_change(s2, "games")
    assert s2["detail_stack"] == [] and s2["_active_page"] == "games"


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
            {
                "League": "NBA",
                "Date": "2026-06-15",
                "Player": "A. One",
                "Market": "PTS",
                "Opponent": "BOS",
                "volume_trend": "[1]",
            },
            {
                "League": "NBA",
                "Date": "2026-06-15",
                "Player": "A. One",
                "Market": "AST",
                "Opponent": "BOS",
                "volume_trend": "[2]",
            },
        ]
    )
    row = pd.Series(
        {
            "League": "NBA",
            "Date": "2026-06-15",
            "Player": "A. One",
            "Market": "AST",
            "Opponent": "BOS",
        }
    )
    hit = _detail_row(row, details)
    assert hit is not None and hit["volume_trend"] == "[2]"

    miss = pd.Series(
        {
            "League": "NBA",
            "Date": "2026-06-15",
            "Player": "A. One",
            "Market": "REB",
            "Opponent": "BOS",
        }
    )
    assert _detail_row(miss, details) is None
    assert _detail_row(row, pd.DataFrame()) is None


def test_edge_badge_sign_color():
    assert ":green[" in _edge_badge(1.12) and "+12%" in _edge_badge(1.12)
    assert ":red[" in _edge_badge(0.95) and "-5%" in _edge_badge(0.95)
