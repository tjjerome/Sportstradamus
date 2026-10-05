"""Receipts realized-by-side panel (``dashboard/components/by_side.py``).

``by_side_grid`` shapes a ``realized.by_split`` result for the themed grid: one split, keys
ascending with Over before Under, rates in percentage points. ``render_by_side`` is
smoke-run in Streamlit bare mode, where widgets return their defaults and the grid renders
nothing, so the whole function body executes; with the grid call captured it shows the
panel draws the sport-filtered legs the page hands it, never a source of its own.
"""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st

from sportstradamus.dashboard.components import by_side
from sportstradamus.dashboard.components.by_side import by_side_grid, render_by_side
from sportstradamus.dashboard.data import sport_filtered
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.realized import by_split, settled_offers
from tests.golden.test_receipts_reconciles_ledger import HISTORY

_GRID_COLS = [
    "Side",
    "Bets",
    "Hit%",
    "Breakeven%",
    "Model%",
    "Book%",
    "Edge captured",
    "Units",
    "ROI",
]


def _recommended_stats(history: pd.DataFrame = HISTORY) -> pd.DataFrame:
    offers = settled_offers(history)
    return by_split(offers[offers["Recommended"]])


def test_grid_keeps_only_the_chosen_split():
    out = by_side_grid(_recommended_stats(), "side")
    assert list(out.columns) == ["All", *_GRID_COLS]
    assert out["All"].tolist() == ["all", "all"]
    # Over: both A offers, D, I, K. Under: B, C, J.
    assert list(zip(out["Side"], out["Bets"], strict=True)) == [("Over", 5), ("Under", 3)]


def test_grid_sorts_keys_ascending_then_over_before_under():
    out = by_side_grid(_recommended_stats(), "league")
    assert list(zip(out["League"], out["Side"], strict=True)) == [
        ("NBA", "Over"),
        ("NBA", "Under"),
        ("WNBA", "Over"),
        ("WNBA", "Under"),
    ]


def test_grid_splits_on_the_quote_class():
    out = by_side_grid(_recommended_stats(), "quote")
    assert list(zip(out["Quote"], out["Side"], strict=True)) == [
        ("fallback", "Over"),
        ("quoted", "Over"),
        ("quoted", "Under"),
    ]


def test_grid_scales_rates_to_percentage_points():
    row = by_side_grid(_recommended_stats(), "side").iloc[0]
    # The five Over legs: I and K hit; reads 0.62, 0.62, 0.61, 0.72, 0.72 over books
    # 0.55, 0.55, 0.60, 0.60, 0.60; the two Underdog legs pay the per-pick baseline and the
    # Sleeper legs 1.75, 1.5, 1.5.
    assert row["Hit%"] == pytest.approx(40.0)
    assert row["Breakeven%"] == pytest.approx(100 * 5 / (2 * UNDERDOG_BOOST_BASELINE + 4.75))
    assert row["Model%"] == pytest.approx(65.8)
    assert row["Book%"] == pytest.approx(58.0)
    assert row["Units"] == pytest.approx(-2.0)
    assert row["ROI"] == pytest.approx(-40.0)


def test_grid_empty_input_keeps_the_display_columns():
    out = by_side_grid(_recommended_stats(HISTORY.iloc[:0]), "league")
    assert out.empty
    assert list(out.columns) == ["League", *_GRID_COLS]


def test_render_runs_in_bare_mode():
    # The empty-cohort caption, a one-sided cohort (Under reads "—"), and both sides
    # through the control, the metric pair and the themed grid.
    render_by_side(_recommended_stats(HISTORY.iloc[:0]))
    render_by_side(_recommended_stats(HISTORY[HISTORY["Bet"] == "Over"]))
    render_by_side(_recommended_stats())


def test_sport_filter_reaches_the_panel(monkeypatch):
    drawn = []
    monkeypatch.setattr(by_side, "render_themed_grid", lambda grid, **_: drawn.append(grid))
    monkeypatch.setitem(st.session_state, "sport", "NBA")
    render_by_side(_recommended_stats(sport_filtered(HISTORY)))
    # Only the NBA legs: both A offers Over, B and C Under.
    assert drawn[0]["Bets"].tolist() == [2, 2]
