"""Unit tests for the live-metrics aggregator added by Stage 0 deliverable 0.2."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sportstradamus.nightly import (
    _MIN_BETS_FOR_PRECISION,
    LIVE_METRICS_COLUMNS,
    LIVE_METRICS_WINDOWS,
    _compute_live_metrics,
    _settled_offers,
    _side_precision,
)

NOW = datetime(2026, 5, 20, 12, 0, 0)


def _build_offer(line: float, bet: str, model_p: float, books_p: float) -> dict:
    """Return the offer-level columns for one flat history row (closing trio unfilled)."""
    return {
        "Line": line,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": bet,
        "Win Prob": model_p,
        "Market Prob": books_p,
        "Close Market Prob": float("nan"),
        "Market CLV": float("nan"),
        "Model CLV": float("nan"),
    }


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
                    "Projection": line,
                    "Market Projection": line,
                    "Dist": "SkewNormal",
                    "CV": 0.3,
                    "Model Param": line,
                    "Gate": np.nan,
                    "Temperature": 1.0,
                    "Disp Cal": 1.0,
                    "Step": "test",
                    **_build_offer(line, bet, model_p, books_p),
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
                "Projection": line,
                "Market Projection": line,
                "Dist": "SkewNormal",
                "CV": 0.3,
                "Model Param": line,
                "Gate": np.nan,
                "Temperature": 1.0,
                "Disp Cal": 1.0,
                "Step": "test",
                **_build_offer(line, "Over", 0.55, 0.50),
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
                **_build_offer(10.0, "Over", 0.55, 0.50),
                "Actual": np.nan,
            }
        )
    metrics = _compute_live_metrics(pd.DataFrame(rows), now=NOW)
    assert metrics.empty
    assert list(metrics.columns) == list(LIVE_METRICS_COLUMNS)


def test_settled_offers_excludes_legacy_and_push_rows():
    """Legacy NaN-Line rows and pushes never reach the live metrics.

    ~13k pre-flat-schema history rows have Actual filled but Line/Bet/Win Prob
    NaN — their Over/Under outcome is underivable, so counting them deflates
    both over rates and NaN-poisons the Kelly sim. Pushes (Actual == Line) are
    voided by the platforms, so counting them as unders/losses is the same bias.
    """
    date = NOW.strftime("%Y-%m-%d")

    def _row(player, actual, line, bet, model_p, books_p):
        return {
            "Player": player,
            "League": "NBA",
            "Date": date,
            "Market": "PTS",
            "Projection": 10.0,
            **_build_offer(line, bet, model_p, books_p),
            "Actual": actual,
        }

    history = pd.DataFrame(
        [
            _row("Hit_1", 12.0, 10.0, "Over", 0.60, 0.50),
            _row("Hit_2", 12.0, 10.0, "Over", 0.60, 0.50),
            _row("Miss", 8.0, 10.0, "Over", 0.60, 0.50),
            _row("Push", 10.0, 10.0, "Over", 0.60, 0.50),
            _row("Legacy", 22.0, np.nan, np.nan, np.nan, np.nan),
        ]
    )

    settled = _settled_offers(history)
    assert sorted(settled["Player"]) == ["Hit_1", "Hit_2", "Miss"]

    metrics = _compute_live_metrics(history, now=NOW)
    row_30d = metrics[metrics["window_days"] == 30].iloc[0]
    assert row_30d["n_settled"] == 3
    # 2 Overs of 3 decided; a leaked push or legacy row would give 2/4 or 2/5.
    assert row_30d["empirical_over_rate"] == pytest.approx(2 / 3)
    # All 3 decided bets are Over; a leaked legacy row (Bet NaN) would give 3/4.
    assert row_30d["predicted_over_rate"] == pytest.approx(1.0)
    # Fair-odds 2.0 payout, 2 wins on 3 flat stakes; a push-as-loss would give 0.
    assert row_30d["profit_sim_yield"] == pytest.approx(1 / 3)
    # A leaked NaN Win Prob row would NaN the whole cell's Kelly yield.
    assert row_30d["profit_sim_kelly_yield"] == pytest.approx(1 / 3)


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
                "Projection": line,
                **_build_offer(line, "Over", 0.60, 0.50),
                "Actual": line + 2.0,
            }
        )
        lose_rows.append(
            {
                "Player": f"L_{idx}",
                "League": "WNBA",
                "Date": date,
                "Market": "PTS",
                "Projection": line,
                **_build_offer(line, "Over", 0.60, 0.50),
                "Actual": line - 2.0,
            }
        )
    history = pd.DataFrame(win_rows + lose_rows)
    metrics = _compute_live_metrics(history, now=NOW)
    nba_30d = metrics[(metrics["league"] == "NBA") & (metrics["window_days"] == 30)].iloc[0]
    wnba_30d = metrics[(metrics["league"] == "WNBA") & (metrics["window_days"] == 30)].iloc[0]
    assert nba_30d["profit_sim_yield"] == pytest.approx(1.0, abs=1e-6)
    assert wnba_30d["profit_sim_yield"] == pytest.approx(-1.0, abs=1e-6)


def test_side_precision_reports_hit_rate_per_side():
    """`_side_precision` returns the side-conditional hit rate, NaN when too few bets."""
    # 40 Over bets, 24 hit -> precision_over = 0.60.
    # 40 Under bets, 16 hit -> precision_under = 0.40.
    over_bets = pd.DataFrame(
        {
            "Bet": ["Over"] * 40,
            "Result": (["Over"] * 24) + (["Under"] * 16),
        }
    )
    under_bets = pd.DataFrame(
        {
            "Bet": ["Under"] * 40,
            "Result": (["Under"] * 16) + (["Over"] * 24),
        }
    )
    group = pd.concat([over_bets, under_bets], ignore_index=True)
    assert _side_precision(group, "Over") == pytest.approx(0.60)
    assert _side_precision(group, "Under") == pytest.approx(0.40)


def test_side_precision_returns_nan_when_n_below_threshold():
    """Below `_MIN_BETS_FOR_PRECISION`, return NaN — small-n hit rates are noise."""
    n_small = _MIN_BETS_FOR_PRECISION - 1
    group = pd.DataFrame(
        {
            "Bet": ["Over"] * n_small + ["Under"] * 50,
            "Result": ["Over"] * n_small + ["Under"] * 25 + ["Over"] * 25,
        }
    )
    assert math.isnan(_side_precision(group, "Over"))
    assert _side_precision(group, "Under") == pytest.approx(0.50)


def test_compute_live_metrics_populates_precision_columns():
    """`_compute_live_metrics` plumbs precision_{over,under}_live through to the parquet schema."""
    # 120 rows/cell with random 50/50 sides yields ~60 per side — comfortably
    # above _MIN_BETS_FOR_PRECISION (30) so both sides report a finite value.
    history = _build_history_fixture(n_per_cell=120)
    metrics = _compute_live_metrics(history, now=NOW)
    # 30d window covers the full fixture (rng spans 0-40 days back).
    nba_pts_30d = metrics[
        (metrics["league"] == "NBA") & (metrics["market"] == "PTS") & (metrics["window_days"] == 30)
    ].iloc[0]
    # Both sides should report a finite precision in [0, 1].
    assert 0.0 <= nba_pts_30d["precision_over_live"] <= 1.0
    assert 0.0 <= nba_pts_30d["precision_under_live"] <= 1.0
