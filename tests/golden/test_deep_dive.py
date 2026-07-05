"""Pure helpers behind the deep-dive dialog (sidecar decode, detail-row lookup, edge
badge, correlated-partner resolution) and the page-change dialog reset.

Render paths (``show_detail`` and its tab renderers) call ``st.*`` directly and are
Streamlit-runtime — verified manually, per the dashboard-wide precedent (see
``test_satellite_picker.py``, ``test_constellation.py``); these pin the unit logic
the render depends on — JSON decode, the five-key join, the cross-page detail-stack
reset, and the ``Corr Same``/``Corr Opp`` → ``find_offer_idx`` platform-pinned lookup.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data as data_pkg
from sportstradamus.dashboard.components.deep_dive import (
    _decode,
    _detail_row,
    _edge_badge,
    _p_over,
    _team_total_text,
    _win_prob_text,
    drop_detail_on_page_change,
)
from sportstradamus.dashboard.legs import find_offer_idx


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


def test_p_over_flips_on_bet_side():
    # Win Prob is max(Model Over, Model Under); on an Over pick it *is* P(over)...
    assert _p_over(pd.Series({"Win Prob": 0.61, "Bet": "Over"})) == pytest.approx(0.61)
    # ...and on an Under pick P(over) is its complement.
    assert _p_over(pd.Series({"Win Prob": 0.61, "Bet": "Under"})) == pytest.approx(0.39)
    # Book-fallback rows with no Win Prob yield None (caller hides the line).
    assert _p_over(pd.Series({"Bet": "Over"})) is None


def _extract_corr_items(raw):
    """Mirrors ``_render_corr_tab``'s ``Corr Same``/``Corr Opp`` extraction."""
    return [] if raw is None else list(raw)


def test_corr_same_opp_none_safe_extraction():
    """A ``None`` cell (an offer with no correlated partners on that side) must
    degrade to an empty list, not crash on ``_render_corr_cards``'s
    ``if not items: return`` guard."""
    row = pd.Series({"Corr Same": None, "Corr Opp": None, "Team": "NYK", "Opponent": "SAS"})
    assert _extract_corr_items(row.get("Corr Same")) == []
    assert _extract_corr_items(row.get("Corr Opp")) == []

    populated = pd.Series({"Corr Same": [{"player": "X"}], "Corr Opp": None})
    assert _extract_corr_items(populated.get("Corr Same")) == [{"player": "X"}]
    assert _extract_corr_items(populated.get("Corr Opp")) == []


def test_corr_items_survive_parquet_ndarray_roundtrip(tmp_path):
    """Regression: a list<struct> cell decodes to a numpy ndarray after a parquet
    round-trip, not a Python list — every real dashboard read of ``current_offers``
    goes through exactly this round-trip. ``x or []`` raises ValueError ("truth value
    of an array with more than one element is ambiguous") the moment an offer has 2+
    correlated partners on one side, which is the common case, not an edge case —
    caught here by actually writing+reading parquet rather than constructing an
    in-memory list that would mask the bug."""
    partners = [{"player": "A", "mult": 1.1}, {"player": "B", "mult": 1.2}]
    path = tmp_path / "corr_roundtrip.parquet"
    pd.DataFrame({"Corr Same": [partners], "Corr Opp": [None]}).to_parquet(path)
    row = pd.read_parquet(path).iloc[0]

    assert isinstance(row.get("Corr Same"), np.ndarray)  # confirms the round-trip shape
    assert _extract_corr_items(row.get("Corr Same")) == partners
    assert _extract_corr_items(row.get("Corr Opp")) == []


def test_corr_partner_lookup_pins_platform_across_markets_and_lines():
    """Regression for the retired ``_find_corr_row_idx``: it matched a correlated
    partner back to an offer row by player-name substring alone, with no market/line
    check — two different props on the same player would collide (and, since
    ``filtered`` can hold both platforms at once on Board, so could two books' rows).
    The structured partner record now carries player/market/bet/line, and
    ``_render_corr_cards`` resolves it via ``find_offer_idx`` pinned to the
    *displayed offer's own* platform — the fix threaded through in this task."""
    filtered = pd.DataFrame(
        {
            "Player": ["Mitchell Robinson", "Mitchell Robinson", "Mitchell Robinson"],
            "Market": ["PRA", "REB", "PRA"],
            "Bet": ["Over", "Over", "Over"],
            "Line": [9.5, 5.5, 9.5],
            "Platform": ["Underdog", "Underdog", "Sleeper"],
            "Boost": [1.03, 1.06, 1.85],
        }
    )
    partner = {"player": "Mitchell Robinson", "market": "PRA", "bet": "Over", "line": 9.5}

    # Same player, same market/line, but two different books at different Boosts —
    # the platform-blind old code would grab whichever row's index came first.
    assert find_offer_idx(partner, filtered, platform="Underdog") == 0
    assert find_offer_idx(partner, filtered, platform="Sleeper") == 2

    # Same player, different market (REB vs PRA) — a substring match on "Mitchell
    # Robinson Over ... " alone (the old _find_corr_row_idx logic) couldn't tell
    # these apart; the structured lookup must not resolve the REB row for a PRA partner.
    reb_partner = {"player": "Mitchell Robinson", "market": "REB", "bet": "Over", "line": 5.5}
    assert find_offer_idx(reb_partner, filtered, platform="Underdog") == 1
    assert find_offer_idx(partner, filtered, platform="Underdog") != 1
