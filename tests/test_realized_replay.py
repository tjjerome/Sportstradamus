"""The offline realized-money replay: row selection, the live rule at 1.83, scope rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.scripts import realized_replay as rr


def _test_set() -> pd.DataFrame:
    # a: pick'em Over hit; b: pick'em Under hit; c: near-even sportsbook row, P above the cap;
    # d: pick'em push (dropped); e: lopsided sportsbook row (excluded); f: synthetic (excluded).
    return pd.DataFrame(
        {
            "Player": ["a", "b", "c", "d", "e", "f"],
            "Date": [
                "2026-09-01",
                "2026-09-01",
                "2026-09-02",
                "2026-09-02",
                "2026-09-03",
                "2026-09-03",
            ],
            "Result": [3.0, 1.0, 2.0, 2.5, 9.0, 0.0],
            "Line": [2.5, 2.5, 1.5, 2.5, 5.5, 0.5],
            "P": [0.62, 0.40, 0.97, 0.58, 0.70, 0.55],
            "Odds": [0.50, 0.52, 0.47, 0.50, 0.30, 0.50],
            "QuoteAuthenticity": [
                "pickem",
                "pickem",
                "authentic",
                "pickem",
                "authentic",
                "synthetic",
            ],
            "Book_EV": [2.4, 2.6, 1.6, 2.4, 4.0, 0.6],
        }
    )


def test_replay_offers_keeps_market_lines_at_the_even_pick():
    offers = rr.replay_offers(_test_set(), "NBA", "PTS", "NBA_PTS")
    assert list(offers["Player"]) == ["a", "b", "c"]
    assert (offers["Payout"] == 1.83).all()
    assert list(offers["Bet"]) == ["Over", "Under", "Over"]
    assert list(offers["Hit"]) == [True, True, True]
    assert offers["Recommended"].all()
    assert offers.loc[offers["Player"] == "c", "Win Prob"].item() == 0.90
    assert list(offers["line_type"]) == ["pickem", "pickem", "near_even"]
    assert list(offers["Quote"]) == ["unquoted", "unquoted", "quoted"]


def test_cohort_gap_is_read_minus_hit_with_date_blocks():
    offers = pd.DataFrame(
        {
            "Recommended": [True, True, False],
            "Win Prob": [0.60, 0.60, 0.80],
            "Hit": [True, False, True],
            "Unit": [0.83, -1.0, 0.83],
            "Date": ["2026-09-01", "2026-09-02", "2026-09-02"],
        }
    )
    row = rr.cohort(offers, np.random.default_rng(0))
    assert row["n"] == 2 and row["dates"] == 2
    assert row["read"] == 0.60 and row["hit"] == 0.5
    assert abs(row["gap"] - 0.10) < 1e-12
    assert row["gap_lo"] <= row["gap"] <= row["gap_hi"]
    assert row["hit_lb"] <= row["hit"]


def test_live_recommended_keeps_the_even_pick_band_only():
    history = pd.DataFrame(
        {
            "Date": ["2026-10-10"] * 3,
            "Player": ["x", "y", "z"],
            "League": ["NBA"] * 3,
            "Market": ["fantasy points underdog", "PTS", "PTS"],
            "Line": [30.5, 20.5, 20.5],
            "Bet": ["Over"] * 3,
            "Win Prob": [0.60, 0.60, 0.60],
            "Boost": [1.0, 1.78, 1.5],
            "Platform": ["Underdog", "Sleeper", "Underdog"],
            "Model Version": ["20261018.x"] * 3,
            "Market Projection": [np.nan, 21.0, 21.0],
            "Market Prob": [0.5, 0.5, 0.5],
            "Alt Line": [False] * 3,
            "Result": ["Over", "Over", "Under"],
        }
    )
    live = rr.live_recommended(history, pd.Timestamp("2026-10-09"), None)
    assert list(live["Player"]) == ["x"]
    assert list(live["cell"]) == ["fantasy points prizepicks"]
