"""The offline realized-money replay: row selection, the 5 %-edge rule at 1.83, scope rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.scripts import realized_replay as rr


def _test_set() -> pd.DataFrame:
    # a: pick'em Over hit; b: pick'em Under hit; c: near-even sportsbook row, P above the cap;
    # d: pick'em push (dropped); e: lopsided sportsbook row (excluded); f: synthetic (excluded);
    # g: pick'em row the board's unquoted-disagreement gate drops. b and g carry decoded Odds away
    # from .5 that would flip both verdicts if the gate read the decode instead of the .5 price.
    return pd.DataFrame(
        {
            "Player": ["a", "b", "c", "d", "e", "f", "g"],
            "Date": [
                "2026-09-01",
                "2026-09-01",
                "2026-09-02",
                "2026-09-02",
                "2026-09-03",
                "2026-09-03",
                "2026-09-04",
            ],
            "Result": [3.0, 1.0, 2.0, 2.5, 9.0, 0.0, 4.0],
            "Line": [2.5, 2.5, 1.5, 2.5, 5.5, 0.5, 2.5],
            "P": [0.62, 0.40, 0.97, 0.58, 0.70, 0.55, 0.70],
            "Odds": [0.50, 0.40, 0.47, 0.50, 0.30, 0.50, 0.30],
            "QuoteAuthenticity": [
                "pickem",
                "pickem",
                "authentic",
                "pickem",
                "authentic",
                "synthetic",
                "pickem",
            ],
            "Book_EV": [2.4, 2.6, 1.6, 2.4, 4.0, 0.6, 2.4],
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
    assert offers["Market Prob"].tolist() == pytest.approx([0.5, 0.5, 0.53])
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


def test_live_recommended_keeps_the_even_pick_band_in_the_window():
    history = pd.DataFrame(
        {
            "Date": ["2026-10-10"] * 3 + ["2026-10-20"],
            "Player": ["x", "y", "z", "w"],
            "League": ["NBA"] * 4,
            "Market": ["fantasy points underdog", "PTS", "PTS", "PTS"],
            "Line": [30.5, 20.5, 20.5, 20.5],
            "Bet": ["Over"] * 4,
            "Win Prob": [0.60, 0.60, 0.60, 0.60],
            "Boost": [1.0, 1.78, 1.5, 1.0],
            "Platform": ["Underdog", "Sleeper", "Underdog", "Underdog"],
            "Model Version": ["20261018.x"] * 4,
            "Market Projection": [np.nan, 21.0, 21.0, np.nan],
            "Market Prob": [0.5, 0.5, 0.5, 0.5],
            "Alt Line": [False] * 4,
            "Result": ["Over", "Over", "Under", "Over"],
        }
    )
    live = rr.live_recommended(history, pd.Timestamp("2026-10-09"), pd.Timestamp("2026-10-15"))
    assert list(live["Player"]) == ["x"]
    assert list(live["cell"]) == ["fantasy points prizepicks"]


def _offline(n_dates: int, line_type: str, over_rate: float) -> pd.DataFrame:
    # Hits alternate within each date, so every date block reads alike and gap_hi == gap (.10).
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
    legs = rr.LIVE_MIN_LEGS
    live = pd.DataFrame(
        {
            "League": "NBA",
            "cell": "PTS",
            "Date": np.tile(["2026-10-10", "2026-10-11"], legs // 2),
            "Recommended": True,
            "Win Prob": 0.62,
            "Hit": np.tile([True, True, False, True], legs // 4),
            "Unit": np.tile([0.83, 0.83, -1.0, 0.83], legs // 4),
        }
    )
    thin = rr.replay_table(_offline(2, "near_even", 0.5), live)
    assert thin.loc[thin["scope"] == "l2", "l2_flag"].item() is None
    under_floor = rr.replay_table(_offline(25, "near_even", 0.5), live.head(legs - 1))
    assert under_floor.loc[under_floor["scope"] == "l2", "l2_flag"].item() is None
    full = rr.replay_table(_offline(25, "near_even", 0.5), live)
    l2 = full[full["scope"] == "l2"].iloc[0]
    assert l2["n"] == 500 and l2["dates"] == 25 and l2["live_n"] == legs
    assert l2["implied_live_hit_lb"] == pytest.approx(0.52)
    assert l2["l2_flag"] is False


def test_tolerance_rows_are_the_posted_league_rows_over_the_live_floor():
    table = pd.DataFrame(
        {
            "scope": ["l2", "posted", "posted", "posted", "posted"],
            "league": ["NBA", "NBA", "NBA", "WNBA", "NHL"],
            "market": ["", "", "PTS", "", ""],
            "live_n": [500, 500, 500, 26, rr.LIVE_MIN_LEGS],
            "gap_delta": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    tolerance = rr.tolerance_rows(table)
    assert list(tolerance["league"]) == ["NBA", "NHL"]
    assert list(tolerance["gap_delta"]) == [0.2, 0.5]


def test_still_posted_keeps_each_props_rungs_from_its_own_last_poll():
    # Rows out of time order, two rungs at prop x's last poll, and prop y (Sleeper, same player
    # and date) polled hours earlier: a global latest stamp would drop y, and taking each prop's
    # last row would keep one of x's two live rungs and could keep the stale one.
    last = pd.Timestamp("2026-11-01 23:30")
    rungs = pd.DataFrame(
        {
            "League": "NBA",
            "Market": "PTS",
            "Player": "x",
            "Date": "2026-11-01",
            "Platform": ["Underdog", "Sleeper", "Underdog", "Underdog", "Underdog"],
            "Line": [21.5, 20.5, 19.5, 20.5, 22.5],
            "last_poll": [
                last,
                pd.Timestamp("2026-11-01 17:30"),
                pd.Timestamp("2026-11-01 20:30"),
                last - pd.Timedelta(minutes=2),
                pd.Timestamp("2026-11-01 14:30"),
            ],
        }
    )
    assert rr.still_posted(rungs).tolist() == [True, True, False, True, False]


def _posted_leg(player: str, market: str, date, version: str, hit: bool) -> dict:
    return {
        "League": "NBA",
        "Market": market,
        "Player": player,
        "Date": date,
        "Model Version": version,
        "Recommended": True,
        "Win Prob": 0.60,
        "Hit": hit,
        "Unit": 0.83 if hit else -1.0,
    }


def test_posted_table_reads_live_legs_on_the_priced_rows_only():
    rows = pd.DataFrame(
        {
            "League": "NBA",
            "cell": "PTS",
            "Player": ["a", "b"],
            "Date": "2026-11-01",
            "version": "v2",
        }
    )
    legs = pd.DataFrame([_posted_leg("a", "PTS", "2026-11-01", "v2", True)]).assign(cell="PTS")
    # a: priced, read by v2. b: priced, read by the older v1. c: never priced. a's REB leg sits on
    # a row the dumps do not hold (a join on player and date alone would count it). Live dates
    # arrive as timestamps, the priced rows as day strings.
    live = pd.DataFrame(
        [
            _posted_leg("a", "PTS", pd.Timestamp("2026-11-01"), "v2", True),
            _posted_leg("b", "PTS", pd.Timestamp("2026-11-01"), "v1", False),
            _posted_leg("c", "PTS", pd.Timestamp("2026-11-01"), "v2", True),
            _posted_leg("a", "REB", pd.Timestamp("2026-11-01"), "v2", True),
        ]
    )
    table = rr.posted_table(legs, rows, live).set_index("market")
    assert list(table.index) == ["", "PTS"]
    assert (table["scope"] == "posted").all()
    assert table["n"].tolist() == [1, 1]
    assert table["live_n"].tolist() == [2, 2]
    assert table["live_same_version"].tolist() == [0.5, 0.5]
    assert table["live_hit"].tolist() == [0.5, 0.5]
