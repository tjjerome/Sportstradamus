"""Receipts realized-by-side panel (``dashboard/components/by_side.py``).

``by_side_grid`` shapes the nightly ledger for the themed grid: the recommended cohort at
one window and one split, keys ascending with Over before Under, rates in percentage
points. ``render_by_side`` is smoke-run in Streamlit bare mode, where widgets return their
defaults and the grid renders nothing, so the whole function body executes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.dashboard.components.by_side import by_side_grid, render_by_side
from sportstradamus.realized import REALIZED_BY_SIDE_COLS

_GRID_COLS = ["Side", "Bets", "Hit%", "Model%", "Book%", "Units", "ROI"]
_ROW = {
    "computed_at": pd.Timestamp("2026-10-03 12:00"),
    "window_days": 30,
    "cohort": "recommended",
    "split": "side",
    "key": "all",
    "side": "Over",
    "n": 40,
    "hit_rate": 0.6,
    "pred_rate": 0.58,
    "book_rate": 0.5,
    "units": 3.14159,
    "roi": 0.0785,
}


def _ledger(*rows: dict) -> pd.DataFrame:
    # Pinned to the ledger's own column list so a schema change there fails here, loudly.
    assert set(_ROW) == set(REALIZED_BY_SIDE_COLS)
    return pd.DataFrame([_ROW | row for row in rows], columns=REALIZED_BY_SIDE_COLS)


def test_grid_keeps_only_the_window_cohort_and_split():
    out = by_side_grid(
        _ledger(
            {},
            {"window_days": 90, "n": 99},
            {"cohort": "bettable", "n": 98},
            {"split": "league", "key": "NBA", "n": 97},
        ),
        30,
        "side",
    )
    assert list(out.columns) == ["All", *_GRID_COLS]
    assert out["All"].tolist() == ["all"]
    assert out["Bets"].tolist() == [40]


def test_grid_sorts_keys_ascending_then_over_before_under():
    out = by_side_grid(
        _ledger(
            {"split": "league", "key": "NFL", "side": "Under"},
            {"split": "league", "key": "NFL", "side": "Over"},
            {"split": "league", "key": "NBA", "side": "Under"},
        ),
        30,
        "league",
    )
    assert list(zip(out["League"], out["Side"], strict=True)) == [
        ("NBA", "Under"),
        ("NFL", "Over"),
        ("NFL", "Under"),
    ]


def test_grid_scales_rates_to_percentage_points():
    row = by_side_grid(_ledger({"roi": -0.05}), 30, "side").iloc[0]
    assert row["Hit%"] == pytest.approx(60.0)
    assert row["Model%"] == pytest.approx(58.0)
    assert row["Book%"] == pytest.approx(50.0)
    assert row["ROI"] == pytest.approx(-5.0)
    assert row["Units"] == pytest.approx(3.1)


def test_grid_empty_input_keeps_the_display_columns():
    out = by_side_grid(_ledger(), 30, "league")
    assert out.empty
    assert list(out.columns) == ["League", *_GRID_COLS]


def test_render_runs_in_bare_mode():
    # The empty-ledger caption, a one-sided ledger (Under reads "—"), and a two-sided
    # ledger through the controls, the metric pair and the themed grid.
    render_by_side(pd.DataFrame())
    render_by_side(_ledger({}))
    render_by_side(_ledger({}, {"side": "Under", "roi": -0.02, "units": -0.8}))
