"""Pure helpers behind the deep-dive dialog shell (detail-row lookup, EV badge, the
context-strip text helpers) and the page-change dialog reset.

Render paths (``show_detail`` and the tab renderers in ``deep_dive_tabs``) call ``st.*``
directly and are Streamlit-runtime — verified manually, per the dashboard-wide precedent
(see ``test_satellite_picker.py``, ``test_constellation.py``); these pin the unit logic
the shell depends on — the detail-row joins (the sidecar's five keys, the movement
snapshot's Platform-split keys), the cross-page detail-stack reset, and the
context-strip formatting. The tab-body unit logic lives in ``test_deep_dive_tabs.py``.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

import pandas as pd

from sportstradamus import data as data_pkg
from sportstradamus.dashboard.components.deep_dive import (
    _detail_row,
    _ev_badge,
    _team_total_text,
    _win_prob_text,
    drop_detail_on_page_change,
)
from sportstradamus.helpers.io import LINE_MOVEMENT_KEYS


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


def test_detail_row_movement_keys_split_platforms():
    # Underdog and Sleeper post and move their own lines for the same prop, so the
    # Movement tab's lookup keys on Platform: each offer gets its own app's history, and
    # the Sleeper row sits second so a Platform-blind join would hand it Underdog's.
    prop = {
        "League": "NFL",
        "Market": "rushing yards",
        "Player": "B. Robinson",
        "Date": "2026-09-13",
    }
    movement = pd.DataFrame(
        [
            {**prop, "Platform": "Underdog", "changes": "[underdog]"},
            {**prop, "Platform": "Sleeper", "changes": "[sleeper]"},
        ]
    )
    for platform, changes in (("Sleeper", "[sleeper]"), ("Underdog", "[underdog]")):
        offer = pd.Series({**prop, "Platform": platform, "Opponent": "TB"})
        hit = _detail_row(offer, movement, LINE_MOVEMENT_KEYS)
        assert hit is not None and hit["changes"] == changes


def _offer(**fit) -> pd.Series:
    """An Underdog even pick (its 1.0 multiplier pays 1.83) as the dialog receives it: one row
    of the offers frame, so its numbers ride in an object Series."""
    row = {"Boost": 1.0, "Platform": "Underdog", "Win Prob": 0.60, "Model EV": 1.10, **fit}
    return pd.DataFrame([row]).iloc[0]


def test_badge_leads_with_the_fitted_ev_and_shows_the_models_claim():
    badge = _ev_badge(_offer(**{"Fit Prob": 0.52, "Fit Prob Low": 0.47, "EV": 0.95}))
    assert badge == "**EV 0.95** · Win 52%  \n:gray[Model EV 1.10 · Model win 60%]"


def test_badge_wears_no_mark_when_the_lower_bound_clears_the_payout():
    """The lower bound (0.56 at a 1.83 payout) once earned a "+EV" mark; the badge is the
    fitted number alone."""
    badge = _ev_badge(_offer(**{"Fit Prob": 0.60, "Fit Prob Low": 0.56, "EV": 1.098}))
    assert badge == "**EV 1.10** · Win 60%  \n:gray[Model EV 1.10 · Model win 60%]"


def test_badge_on_a_leg_the_fit_did_not_price_shows_only_the_models_claim():
    badge = _ev_badge(_offer(**{"Fit Prob": None, "Fit Prob Low": None, "EV": None}))
    assert badge == "**No EV**  \n:gray[Model EV 1.10 · Model win 60%]"


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
