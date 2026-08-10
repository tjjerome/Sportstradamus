"""Pure helpers behind the deep-dive tab bodies (``deep_dive_tabs``): the sidecar JSON
decode, the derived P(over), the comps-tab diverging heatmap buckets, the Other-stats
percentile gauge geometry, the Correlated tab's positive-EV-lift survivor filter, and
the ``Corr Same``/``Corr Opp`` → ``find_offer_idx`` platform-pinned lookup.

The render paths (``render_*_tab`` and their private ``_render_*`` helpers) call ``st.*``
directly and are Streamlit-runtime — verified manually, per the dashboard-wide precedent.
These pin the unit logic those renders depend on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sportstradamus.dashboard.components.deep_dive_tabs import (
    _comps_table_html,
    _decode,
    _gauge_svg,
    _heat_css,
    _lift_survivors,
    _p_over,
)
from sportstradamus.dashboard.legs import find_offer_idx
from sportstradamus.leg_schema import build_leg


def test_decode_json_and_degenerate():
    payload = json.dumps([{"comp": "X", "n_games": 2}])
    row = pd.Series({"comps_vs_opp": payload, "volume_trend": "", "other_stats": None})
    assert _decode(row, "comps_vs_opp") == [{"comp": "X", "n_games": 2}]
    assert _decode(row, "volume_trend") == []
    assert _decode(row, "other_stats") == []
    assert _decode(None, "comps_vs_opp") == []


def test_p_over_flips_on_bet_side():
    # Win Prob is max(Model Over, Model Under); on an Over pick it *is* P(over)...
    assert _p_over(pd.Series({"Win Prob": 0.61, "Bet": "Over"})) == pytest.approx(0.61)
    # ...and on an Under pick P(over) is its complement.
    assert _p_over(pd.Series({"Win Prob": 0.61, "Bet": "Under"})) == pytest.approx(0.39)
    # Book-fallback rows with no Win Prob yield None (caller hides the line).
    assert _p_over(pd.Series({"Bet": "Over"})) is None


def test_heat_css_diverging_buckets_center_zero():
    # span = max abs deviation across the column; center 0 (pct_diff is already signed).
    # Mockup comps span 0.14: +14% strong-pos blue, +6% mild-pos blue, +1% unpainted,
    # -9% strong-neg red — same buckets grid.py paints.
    span = 0.14
    assert "background-color: #2351B4" in _heat_css(0.14, span)  # >= 0.6*span -> POS_STRONG
    assert "background-color: #3D72D6" in _heat_css(0.06, span)  # >= 0.2*span -> POS
    assert _heat_css(0.01, span) == ""  # < 0.2*span -> neutral band, unpainted
    assert "background-color: #9A2B2B" in _heat_css(-0.09, span)  # <= -0.6*span -> NEG_STRONG
    assert "background-color: #C0453E" in _heat_css(-0.04, span)  # <= -0.2*span -> NEG
    # Light text on every painted cell keeps contrast.
    assert "color: #E6E9EF" in _heat_css(0.14, span)


def test_heat_css_degenerate_span_and_nan():
    # A column with no spread (span 0) paints nothing, and a NaN cell stays neutral.
    assert _heat_css(0.5, 0.0) == ""
    assert _heat_css(float("nan"), 0.14) == ""


_MOCKUP_COMPS = [
    {"comp": "N. Ogwumike", "n_games": 4, "avg_vs_opp": 21.8, "pct_diff": 0.14},
    {"comp": "B. Griner", "n_games": 6, "avg_vs_opp": 19.2, "pct_diff": 0.06},
    {"comp": "A. Thomas", "n_games": 3, "avg_vs_opp": 17.5, "pct_diff": 0.01},
    {"comp": "J. Jones", "n_games": 5, "avg_vs_opp": 14.9, "pct_diff": -0.09},
]


def test_comps_table_html_preserves_knn_order():
    # comps_vs_opponent already returns closest-comp-first; the table must not re-sort
    # by pct_diff (that would silently swap "most similar" for "most different").
    span = float(max(abs(c["pct_diff"]) for c in _MOCKUP_COMPS))
    html = _comps_table_html(_MOCKUP_COMPS, "Avg vs IND", span)
    names = ["Ogwumike", "Griner", "Thomas", "Jones"]
    positions = [html.index(n) for n in names]
    assert positions == sorted(positions)


def test_comps_table_html_zebra_mono_right_align_and_heatmap():
    # DESIGN §4: zebra or 1px separators, right-aligned Plex Mono numerals; the last
    # column heatmapped on the diverging ramp. st.dataframe can't reach any of these on
    # a canvas-rendered grid, hence the plain <table> — this pins the rendered markup.
    span = float(max(abs(c["pct_diff"]) for c in _MOCKUP_COMPS))
    html = _comps_table_html(_MOCKUP_COMPS, "Avg vs IND", span)
    assert "<table" in html
    assert "background:rgba(255,255,255,.015)" in html  # zebra on alternating rows
    assert "font-family:'IBM Plex Mono',monospace;text-align:right" in html
    assert "Avg vs IND" in html and "vs their avg" in html  # header labels
    assert "+14%" in html and "-9%" in html
    assert "background-color: #2351B4" in html  # strong-positive heat on +14%
    assert "background-color: #9A2B2B" in html  # strong-negative heat on -9%


def test_gauge_svg_fill_fraction_is_percentile():
    # Bottom-anchored fill: height = 26 * pct, y = 2 + 26 - height. Mockup pins:
    # pct 0.81 -> h 21.06, y 6.94; pct 0.29 -> h 7.54, y 20.46.
    g81 = _gauge_svg(0.81)
    assert 'height="21.06"' in g81 and 'y="6.94"' in g81
    g29 = _gauge_svg(0.29)
    assert 'height="7.54"' in g29 and 'y="20.46"' in g29
    # A full-height track sits behind the fill in the border token.
    assert 'height="26.0"' in g81 and 'fill="#2A2E37"' in g81


def test_gauge_svg_fill_is_sequential_not_gold():
    # DESIGN §2: gold is never a data mark. The gauge fill (a magnitude encoding) is a
    # SEQUENTIAL_COLORS step, never the gold token, and never the primary-button blue.
    g = _gauge_svg(0.5)
    assert "#C9A227" not in g  # not gold
    assert 'fill="#7FAAE8"' in g  # SEQUENTIAL_COLORS[3]


def _corr_fixture() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """A focus offer (idx 10) + two candidates (11, 12) on a non-0-based index, plus a
    positive-rho correlation slice — the shape ``_lift_survivors`` resolves against."""
    filtered = pd.DataFrame(
        {
            "Player": ["A. Wilson", "J. Young", "A. Boston"],
            "Team": ["LVA", "LVA", "IND"],
            "Market": ["PTS", "3PM", "PTS"],
            "Stat": ["PTS", "3PM", "PTS"],
            "Bet": ["Over", "Over", "Over"],
            "Line": [23.5, 2.5, 18.5],
            "League": ["WNBA", "WNBA", "WNBA"],
            "Game": ["LVA/IND", "LVA/IND", "LVA/IND"],
            "Date": ["2026-07-05", "2026-07-05", "2026-07-05"],
            "Platform": ["Underdog", "Underdog", "Underdog"],
            "Win Prob": [0.61, 0.58, 0.55],
            "Boost": [1.0, 1.0, 1.0],
            "Push Prob": [0.0, 0.0, 0.0],
            "Kelly": [0.02, 0.03, 0.01],
        },
        index=[10, 11, 12],
    )
    focus_leg = build_leg(filtered.loc[10])
    corr = pd.DataFrame(
        {
            "Game": ["LVA/IND", "LVA/IND"],
            "leg_a": ["A. Wilson|PTS|Over", "A. Wilson|PTS|Over"],
            "leg_b": ["J. Young|3PM|Over", "A. Boston|PTS|Over"],
            "rho": [0.4, 0.3],
        }
    )
    return filtered, focus_leg, corr


def test_lift_survivors_resolves_idx_and_returns_multiplier():
    filtered, focus_leg, corr = _corr_fixture()
    items = [{"player": "J. Young", "market": "3PM", "bet": "Over", "line": 2.5, "mult": 1.15}]
    survivors = _lift_survivors(items, filtered, focus_leg, corr, "Underdog")
    assert len(survivors) == 1
    item, idx, lift = survivors[0]
    assert item["player"] == "J. Young"
    assert idx == 11  # resolved against the non-0-based grid index
    # ev_lift returns a raw EV multiplier (near/above 1.0), not a zero-centered delta —
    # the display is `lift - 1.0`, mirroring _edge_badge.
    assert lift > 1.0


def test_lift_survivors_drops_missing_and_nonpositive():
    filtered, focus_leg, corr = _corr_fixture()
    # A partner not in the current grid (idx None) has no computable lift — dropped.
    missing = [{"player": "Nobody", "market": "AST", "bet": "Over", "line": 5.5, "mult": 1.2}]
    assert _lift_survivors(missing, filtered, focus_leg, corr, "Underdog") == []

    # A low-probability pair prices below the solo baseline (lift <= 1.0) — dropped, so
    # the "positive-lift rows only" filter isn't silently defeated.
    cold = filtered.assign(**{"Win Prob": [0.25, 0.25, 0.25], "Kelly": [-0.5, -0.5, -0.5]})
    cold_focus = build_leg(cold.loc[10])
    items = [{"player": "J. Young", "market": "3PM", "bet": "Over", "line": 2.5, "mult": 1.15}]
    assert _lift_survivors(items, cold, cold_focus, corr, "Underdog") == []


def _extract_corr_items(raw):
    """Mirrors ``render_corr_tab``'s ``Corr Same``/``Corr Opp`` extraction."""
    return [] if raw is None else list(raw)


def test_corr_same_opp_none_safe_extraction():
    """A ``None`` cell (an offer with no correlated partners on that side) must
    degrade to an empty list, not crash on ``_render_corr_cards``'s
    ``if not survivors: return`` guard."""
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
    ``_lift_survivors`` resolves it via ``find_offer_idx`` pinned to the *displayed
    offer's own* platform — the fix threaded through in this task."""
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
