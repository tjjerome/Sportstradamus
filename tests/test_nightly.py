"""Unit tests for the live-metrics aggregator added by Stage 0 deliverable 0.2."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sportstradamus.nightly import (
    LIVE_METRICS_COLUMNS,
    LIVE_METRICS_WINDOWS,
    _compute_live_metrics,
    _profit_sim_kelly_yield,
)

NOW = datetime(2026, 5, 20, 12, 0, 0)


def _build_offer(line: float, bet: str, model_p: float, books_p: float) -> tuple:
    """Return a 9-field Offers tuple matching helpers/io.py's _OFFER_FIELDS layout."""
    return (line, 1.0, "Underdog", bet, model_p, books_p, float("nan"), float("nan"), float("nan"))


def _build_history_fixture(now: datetime = NOW, n_per_cell: int = 18) -> pd.DataFrame:
    """3 leagues x 2 markets, ~n_per_cell rows spread over 40 days."""
    rng = np.random.default_rng(2026)
    rows = []
    leagues_markets = [
        ("NBA", "PTS"),
        ("NBA", "REB"),
        ("WNBA", "PTS"),
        ("WNBA", "REB"),
        ("NFL", "rushing_yards"),
        ("NFL", "passing_yards"),
    ]
    for idx in range(n_per_cell):
        for league, market in leagues_markets:
            day_offset = rng.integers(0, 40)
            date = (now - timedelta(days=int(day_offset))).strftime("%Y-%m-%d")
            line = float(rng.uniform(5.0, 25.0))
            bet = rng.choice(["Over", "Under"])
            model_p = float(rng.uniform(0.40, 0.70))
            books_p = float(rng.uniform(0.45, 0.55))
            actual = float(rng.normal(line, line * 0.2))
            rows.append(
                {
                    "Player": f"Player_{league}_{idx}",
                    "League": league,
                    "Team": "HOME",
                    "Date": date,
                    "Market": market,
                    "Model EV": line,
                    "Books EV": line,
                    "Dist": "SkewNormal",
                    "CV": 0.3,
                    "Model Param": line,
                    "Gate": np.nan,
                    "Temperature": 1.0,
                    "Disp Cal": 1.0,
                    "Step": "test",
                    "Offers": [_build_offer(line, bet, model_p, books_p)],
                    "Actual": actual,
                }
            )
    return pd.DataFrame(rows)


def test_compute_live_metrics_schema_and_key_uniqueness():
    history = _build_history_fixture()
    metrics = _compute_live_metrics(history, now=NOW)

    assert list(metrics.columns) == list(LIVE_METRICS_COLUMNS)
    assert set(metrics["window_days"].unique()) == set(LIVE_METRICS_WINDOWS)
    key_cols = ["league", "market", "computed_at", "window_days"]
    assert not metrics.duplicated(subset=key_cols).any()
    assert metrics["window_days"].dtype.kind == "i"
    assert metrics["n_settled"].dtype.kind == "i"
    assert metrics["book_bss"].dtype.kind == "f"
    assert metrics["empirical_over_rate"].dtype.kind == "f"
    assert metrics["predicted_over_rate"].dtype.kind == "f"
    assert metrics["top_decile_mae"].dtype.kind == "f"
    assert metrics["profit_sim_yield"].dtype.kind == "f"
    assert (metrics["computed_at"] == pd.Timestamp(NOW)).all()


def test_compute_live_metrics_includes_all_six_cells_in_30d_window():
    history = _build_history_fixture()
    metrics = _compute_live_metrics(history, now=NOW)
    cells_30d = metrics.loc[metrics["window_days"] == 30, ["league", "market"]]
    assert len(cells_30d.drop_duplicates()) == 6


def test_compute_live_metrics_handles_empty_window():
    """Cells with no settled rows in the 7d window still appear with n_settled=0."""
    rng = np.random.default_rng(42)
    rows = []
    # Place every offer 20–30 days back so the 7d window is empty but 30d isn't.
    for idx in range(6):
        date = (NOW - timedelta(days=20 + idx)).strftime("%Y-%m-%d")
        line = 10.0
        rows.append(
            {
                "Player": f"Player_{idx}",
                "League": "NBA",
                "Team": "HOME",
                "Date": date,
                "Market": "PTS",
                "Model EV": line,
                "Books EV": line,
                "Dist": "SkewNormal",
                "CV": 0.3,
                "Model Param": line,
                "Gate": np.nan,
                "Temperature": 1.0,
                "Disp Cal": 1.0,
                "Step": "test",
                "Offers": [_build_offer(line, "Over", 0.55, 0.50)],
                "Actual": float(rng.normal(line, 2.0)),
            }
        )
    history = pd.DataFrame(rows)
    metrics = _compute_live_metrics(history, now=NOW)
    row_7d = metrics[(metrics["window_days"] == 7) & (metrics["league"] == "NBA")].iloc[0]
    assert row_7d["n_settled"] == 0
    assert math.isnan(row_7d["book_bss"])
    assert math.isnan(row_7d["profit_sim_yield"])
    assert math.isnan(row_7d["top_decile_mae"])
    row_30d = metrics[(metrics["window_days"] == 30) & (metrics["league"] == "NBA")].iloc[0]
    assert row_30d["n_settled"] == 6


def test_compute_live_metrics_empty_history_returns_empty_schema():
    metrics = _compute_live_metrics(pd.DataFrame(), now=NOW)
    assert metrics.empty
    assert list(metrics.columns) == list(LIVE_METRICS_COLUMNS)


def test_compute_live_metrics_no_settled_returns_empty():
    """Every row unresolved (Actual=NaN) → no cells eligible for either window."""
    rows = []
    for idx in range(5):
        rows.append(
            {
                "Player": f"P_{idx}",
                "League": "NBA",
                "Date": NOW.strftime("%Y-%m-%d"),
                "Market": "PTS",
                "Offers": [_build_offer(10.0, "Over", 0.55, 0.50)],
                "Actual": np.nan,
            }
        )
    metrics = _compute_live_metrics(pd.DataFrame(rows), now=NOW)
    assert metrics.empty
    assert list(metrics.columns) == list(LIVE_METRICS_COLUMNS)


def test_compute_live_metrics_profit_sim_yield_signs():
    """All-winning bets at 50% odds → ~100% yield; all-losing → -100%."""
    rng = np.random.default_rng(7)
    win_rows, lose_rows = [], []
    for idx in range(15):
        date = (NOW - timedelta(days=int(rng.integers(0, 5)))).strftime("%Y-%m-%d")
        line = 10.0
        win_rows.append(
            {
                "Player": f"W_{idx}",
                "League": "NBA",
                "Date": date,
                "Market": "PTS",
                "Model EV": line,
                "Offers": [_build_offer(line, "Over", 0.60, 0.50)],
                "Actual": line + 2.0,
            }
        )
        lose_rows.append(
            {
                "Player": f"L_{idx}",
                "League": "WNBA",
                "Date": date,
                "Market": "PTS",
                "Model EV": line,
                "Offers": [_build_offer(line, "Over", 0.60, 0.50)],
                "Actual": line - 2.0,
            }
        )
    history = pd.DataFrame(win_rows + lose_rows)
    metrics = _compute_live_metrics(history, now=NOW)
    nba_30d = metrics[(metrics["league"] == "NBA") & (metrics["window_days"] == 30)].iloc[0]
    wnba_30d = metrics[(metrics["league"] == "WNBA") & (metrics["window_days"] == 30)].iloc[0]
    assert nba_30d["profit_sim_yield"] == pytest.approx(1.0, abs=1e-6)
    assert wnba_30d["profit_sim_yield"] == pytest.approx(-1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Phase 4 — Kelly-sized live ROI (set-baseline -> main gate)


def _kelly_group(bet_side: str, model_p: float, books_p: float, hit: int) -> pd.DataFrame:
    """One-offer fixture for ``_profit_sim_kelly_yield`` exercises."""
    return pd.DataFrame(
        {
            "Bet": [bet_side],
            "Model P": [model_p],
            "Books P": [books_p],
            "Boost": [1.0],
            "Hit": [hit],
        }
    )


def test_profit_sim_kelly_yield_empty_group_is_nan():
    empty = pd.DataFrame(columns=["Bet", "Model P", "Books P", "Boost", "Hit"])
    assert math.isnan(_profit_sim_kelly_yield(empty))


def test_profit_sim_kelly_yield_no_positive_kelly_is_nan():
    # Model P (0.40) below book p (0.50) on the Over side ⇒ Kelly = 0 ⇒ NaN
    # because no dollars were staked.
    group = _kelly_group("Over", model_p=0.40, books_p=0.50, hit=1)
    assert math.isnan(_profit_sim_kelly_yield(group))


def test_profit_sim_kelly_yield_winning_bet_is_positive():
    # Model P (0.60) > book p (0.50); decimal odds = 1 / 0.50 = 2.0 ⇒ b = 1
    # ⇒ raw Kelly = (0.60 * 2.0 - 1) / 1 = 0.20 ⇒ cap 0.05 ⇒ stake 5%.
    # Win pays b = 1 per stake ⇒ ROI per staked dollar = 1.0.
    group = _kelly_group("Over", model_p=0.60, books_p=0.50, hit=1)
    assert _profit_sim_kelly_yield(group) == pytest.approx(1.0)


def test_profit_sim_kelly_yield_losing_bet_is_negative_one():
    # Same Kelly stake (0.60 vs 0.50) but the bet lost ⇒ ROI per staked dollar
    # = -1.0 (the cap-clipped stake is forfeited).
    group = _kelly_group("Over", model_p=0.60, books_p=0.50, hit=0)
    assert _profit_sim_kelly_yield(group) == pytest.approx(-1.0)


def test_profit_sim_kelly_yield_aggregates_dollar_weighted():
    # Two bets: equal stakes (both cap at 5%), one win + one loss ⇒ ROI = 0.
    win = _kelly_group("Over", model_p=0.60, books_p=0.50, hit=1)
    loss = _kelly_group("Over", model_p=0.60, books_p=0.50, hit=0)
    group = pd.concat([win, loss], ignore_index=True)
    assert _profit_sim_kelly_yield(group) == pytest.approx(0.0)
