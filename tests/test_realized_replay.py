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


def _offline(n_dates: int, line_type: str, over_rate: float) -> pd.DataFrame:
    dates = [f"2026-09-{d:02d}" for d in range(1, n_dates + 1)]
    rows = n_dates * 20
    return pd.DataFrame(
        {
            "League": "NBA",
            "cell": "PTS",
            "Date": np.repeat(dates, 20),
            "Recommended": True,
            "Win Prob": 0.60,
            "Hit": np.tile([True, False], rows // 2),
            "Unit": np.tile([0.83, -1.0], rows // 2),
            "line_type": line_type,
            "over_rate": over_rate,
        }
    )


def test_l2_rows_drop_pickem_rows_of_lopsided_cells():
    offline = pd.concat([_offline(2, "pickem", 0.70), _offline(2, "near_even", 0.70)])
    assert set(rr.l2_rows(offline)["line_type"]) == {"near_even"}
    assert len(rr.l2_rows(_offline(2, "pickem", 0.55))) == 40


def test_l2_flag_needs_the_floors_and_reads_the_gap_form():
    live = pd.DataFrame(
        {
            "League": ["NBA"] * 4,
            "cell": ["PTS"] * 4,
            "Date": ["2026-10-10", "2026-10-11"] * 2,
            "Recommended": True,
            "Win Prob": [0.62, 0.62, 0.62, 0.62],
            "Hit": [True, True, False, True],
            "Unit": [0.83, 0.83, -1.0, 0.83],
        }
    )
    thin = rr.replay_table(_offline(2, "near_even", 0.5), live)
    assert thin.loc[thin["scope"] == "l2", "l2_flag"].item() is None
    full = rr.replay_table(_offline(25, "near_even", 0.5), live)
    l2 = full[full["scope"] == "l2"].iloc[0]
    assert l2["n"] == 500 and l2["dates"] == 25
    assert abs(l2["implied_live_hit_lb"] - (0.62 - l2["gap_hi"])) < 1e-12
    assert l2["l2_flag"] == bool(l2["implied_live_hit_lb"] > rr.BREAK_EVEN)


def test_tolerance_rows_need_the_live_floor():
    table = pd.DataFrame(
        {
            "scope": ["league", "l2", "l2", "l2"],
            "league": ["NBA", "NBA", "WNBA", "NHL"],
            "live_n": [500, 500, 26, rr.LIVE_MIN_LEGS],
            "gap_delta": [0.1, 0.2, 0.3, 0.4],
        }
    )
    assert list(rr.tolerance_rows(table)["league"]) == ["NBA", "NHL"]
