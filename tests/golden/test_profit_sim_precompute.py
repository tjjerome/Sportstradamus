"""Pure profit-sim precompute behind the Receipts strategy panel.

``filter_bet_universe`` (the Model-Edge / Consensus-Edge bet selection) and
``precompute_profit_sim_summary`` (the strategy x horizon grid nightly writes for
Receipts to read instantly) are pure over the exploded per-offer frame. No
Streamlit, no I/O.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from sportstradamus.analysis import (
    PROFIT_SIM_STRATEGIES,
    filter_bet_universe,
    precompute_profit_sim_summary,
)


def _exploded(n_days: int = 40, today: date = date(2026, 6, 1)) -> pd.DataFrame:
    # One resolved offer per day, alternating hit/miss, all inside the universe
    # (Model EV > 1 and Market EV > 1). Win Prob drives Kelly sizing.
    rows = []
    for i in range(n_days):
        d = today - timedelta(days=i)
        rows.append(
            {
                "Date": d.isoformat(),
                "Player": f"P{i % 7}",
                "Market": "PTS",
                "League": "NBA",
                "Platform": "Underdog",
                "Boost": 1.9,
                "Win Prob": 0.60,
                "Market Prob": 0.55,
                "Model EV": 1.14,  # 0.60 * 1.9
                "Market EV": 1.045,  # 0.55 * 1.9
                "Kelly": 0.15,
                "Bet": "Over",
                "Result": "Over" if i % 2 == 0 else "Under",
                "Hit": 1 if i % 2 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_filter_bet_universe_keeps_positive_edges_strictly():
    df = pd.DataFrame(
        {
            "Model EV": [1.20, 1.00, 0.95, 1.30],
            "Market EV": [1.05, 1.05, 1.05, 0.99],
        }
    )
    out = filter_bet_universe(df)
    # row0 keeps (both > 1); row1 drops (Model EV == 1, not strictly >);
    # row2 drops (model edge < 0); row3 drops (consensus edge < 0).
    assert list(out.index) == [0]


def test_filter_bet_universe_min_consensus_edge_raises_floor():
    df = pd.DataFrame({"Model EV": [1.20, 1.20], "Market EV": [1.04, 1.06]})
    out = filter_bet_universe(df, min_consensus_edge=0.05)
    # only the row whose consensus edge (Market EV - 1) exceeds 0.05 survives.
    assert list(out.index) == [1]


def test_filter_bet_universe_empty():
    assert filter_bet_universe(pd.DataFrame()).empty


def test_precompute_grid_one_row_per_strategy_and_horizon():
    grid = precompute_profit_sim_summary(_exploded(), today=date(2026, 6, 1), n_mc=2)
    # Every strategy appears; rows are (strategy, horizon) with the metric columns.
    assert set(grid["Strategy"]) == set(PROFIT_SIM_STRATEGIES)
    for col in ("Strategy", "Horizon", "ROI", "Sharpe", "Max Drawdown", "Win%"):
        assert col in grid.columns
    # 40 days of data → the 7d and 30d horizons populate; one row per strategy each.
    per_strategy = grid.groupby("Strategy").size()
    assert (per_strategy >= 2).all()
    assert np.isfinite(grid["ROI"]).all()


def test_precompute_respects_horizon_window():
    # All data older than 7 days → the 7d horizon yields no rows for any strategy.
    old = _exploded(n_days=40, today=date(2026, 6, 1))
    grid = precompute_profit_sim_summary(old, today=date(2026, 9, 1), n_mc=2)
    assert "7d" not in set(grid["Horizon"])
    assert "1y" in set(grid["Horizon"])  # 1y window still covers the data


def test_precompute_empty_universe_returns_empty_with_columns():
    # No row clears the universe filter (all model edge <= 0).
    df = _exploded()
    df["Model EV"] = 0.9
    grid = precompute_profit_sim_summary(df, today=date(2026, 6, 1), n_mc=2)
    assert grid.empty
    assert list(grid.columns) == ["Strategy", "Horizon", "ROI", "Sharpe", "Max Drawdown", "Win%"]


def test_precompute_empty_input():
    grid = precompute_profit_sim_summary(pd.DataFrame(), n_mc=2)
    assert grid.empty


def test_strategies_carry_the_owner_risk_knobs():
    # Conservative caps daily exposure at 5%, Moderate at 10%; Kelly variants are
    # fractional (< full Kelly). These are the knobs the owner specified.
    assert PROFIT_SIM_STRATEGIES["Flat · Conservative"]["daily_exposure_cap"] == 0.05
    assert PROFIT_SIM_STRATEGIES["Flat · Moderate"]["daily_exposure_cap"] == 0.10
    assert PROFIT_SIM_STRATEGIES["Flat · Conservative"]["flat_off_initial"] is True
    assert PROFIT_SIM_STRATEGIES["Kelly · Conservative"]["kelly_fraction"] < 1.0
    assert PROFIT_SIM_STRATEGIES["Kelly · Moderate"]["use_kelly"] is True
