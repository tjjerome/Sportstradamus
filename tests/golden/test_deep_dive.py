"""Pure helpers behind the deep-dive dialog shell (detail-row lookup, edge badge, the
context-strip text helpers) and the page-change dialog reset.

Render paths (``show_detail`` and the tab renderers in ``deep_dive_tabs``) call ``st.*``
directly and are Streamlit-runtime — verified manually, per the dashboard-wide precedent
(see ``test_satellite_picker.py``, ``test_constellation.py``); these pin the unit logic
the shell depends on — the five-key join, the cross-page detail-stack reset, and the
context-strip formatting. The tab-body unit logic lives in ``test_deep_dive_tabs.py``.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

import pandas as pd

from sportstradamus import data as data_pkg
from sportstradamus.dashboard.components.deep_dive import (
    _detail_row,
    _edge_badge,
    _team_total_text,
    _win_prob_text,
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


def test_team_total_text_with_delta():
    # Delta vs season avg is ou - baseline_total / 2 (88.2 - 170.2/2 = +3.1).
    strip = {"baseline_total": 170.2}
    assert _team_total_text(88.2, strip) == "88.2 team total (+3.1)"
    # No baseline → total alone, no delta.
    assert _team_total_text(88.2, {"baseline_total": float("nan")}) == "88.2 team total"
    assert _team_total_text(88.2, None) == "88.2 team total"
    # Missing O/U → the N/A form, never a crash.
    assert _team_total_text(float("nan"), strip) == "team total N/A"
    assert _team_total_text(None, strip) == "team total N/A"


def test_win_prob_text_from_ml_fav_prob():
    assert _win_prob_text({"ml_fav_prob": 0.64}) == "64% win prob"
    # NaN prob (no moneyline) and a missing strip both degrade to N/A, not a crash.
    assert _win_prob_text({"ml_fav_prob": float("nan")}) == "win prob N/A"
    assert _win_prob_text(None) == "win prob N/A"
