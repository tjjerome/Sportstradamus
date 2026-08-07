"""Money-math kernels of the reflect metrics in ``sportstradamus.analysis``.

Pins the payout/settlement behavior behind the dashboard reflect surfaces:

* ``compute_parlay_metrics`` — Underdog parlay P&L: payout-table lookup by
  (size, misses), the perfect-boost rule (a boost of 2x or more pays only on
  a clean entry), the 100x per-entry payout cap, stake rounding to the
  nearest half unit, and same-(Game, Date) entries averaged so one game day
  never double-counts P&L.
* ``compute_individual_metrics`` — flat -110 accounting (a hit pays 100/110,
  a miss loses the unit) and the push rule (a push is not a graded bet).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from sportstradamus.analysis import compute_individual_metrics, compute_parlay_metrics

REF = datetime.today().date()


def _d(days_ago: int) -> str:
    return str(REF - timedelta(days=days_ago))


def _ud_parlay(league: str, game: str, legs: int, misses: int, boost: float, rec_bet: float):
    return {
        "Date": _d(3),
        "League": league,
        "Platform": "Underdog",
        "Game": game,
        "Legs Resolved": legs,
        "Misses": misses,
        "Boost": boost,
        "Rec Bet": rec_bet,
        "Model EV": 0.10,
        "Bet Size": legs,
    }


def _profit_7d(rows: list[dict]) -> pd.DataFrame:
    profit_df, _daily, _size, _corr = compute_parlay_metrics(pd.DataFrame(rows), {}, {})
    return profit_df.loc[profit_df["Period"] == "7d"].set_index("League")


def test_underdog_payout_table_boost_and_cap():
    rows = [
        _ud_parlay("A", "A1", 2, 0, 1.0, 1.0),  # 2-leg clean: 3.5x table -> +2.5
        _ud_parlay("B", "B1", 2, 0, 2.5, 1.0),  # big boost on clean entry: 3.5*2.5 -> +7.75
        _ud_parlay("C", "C1", 4, 1, 2.5, 1.0),  # big boost + a miss reverts to table 1.5 -> +0.5
        _ud_parlay("D", "D1", 4, 1, 1.5, 1.0),  # small boost still applies: 1.5*1.5 -> +1.25
        _ud_parlay("E", "E1", 6, 0, 5.0, 1.0),  # 25*5=125 capped at 100x stake -> +99
    ]
    profit = _profit_7d(rows)["Profit"]
    assert profit["A"] == pytest.approx(2.5)
    assert profit["B"] == pytest.approx(7.75)
    assert profit["C"] == pytest.approx(0.5)
    assert profit["D"] == pytest.approx(1.25)
    assert profit["E"] == pytest.approx(99.0)


def test_underdog_stake_rounding_and_gameday_mean():
    rows = [
        _ud_parlay("F", "F1", 2, 0, 1.0, 1.3),  # rec bet 1.3 stakes 1.5 units: 2.5*1.5 -> +3.75
        _ud_parlay("G", "G1", 2, 0, 1.0, 1.0),  # same (Game, Date) pair below:
        _ud_parlay("G", "G1", 2, 0, 1.0, 1.0),  # averaged to +2.5, not summed to +5
    ]
    seven_d = _profit_7d(rows)
    assert seven_d.loc["F", "Profit"] == pytest.approx(3.75)
    assert seven_d.loc["G", "Profit"] == pytest.approx(2.5)
    assert seven_d.loc["G", "Parlays"] == 2
    assert seven_d.loc["G", "Hit Rate"] == pytest.approx(1.0)
    assert seven_d.loc["All", "Profit"] == pytest.approx(3.75 + 2.5)


def _history(results: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "League": "NBA",
                "Market": "points",
                "Date": _d(3),
                "Bet": "Over",
                "Result": result,
                "Model EV": 0.60,
                "Win Prob": 0.60,
                "Market EV": 0.55,
            }
            for result in results
        ]
    )


def test_individual_flat_110_profit_and_roi():
    _hist, daily, _cal, roi = compute_individual_metrics(_history(["Over", "Over", "Under"]))

    two_wins_one_loss = 2 * (100 / 110) - 1
    assert daily.loc[0, "Bets"] == 3
    assert daily.loc[0, "Hits"] == 2
    assert daily.loc[0, "Profit"] == pytest.approx(two_wins_one_loss)

    row = roi.loc[(roi["Period"] == "7d") & (roi["Threshold"] == 0.55) & (roi["Filter"] == "All")]
    assert row["Bets"].item() == 3
    assert row["Win%"].item() == pytest.approx(round(2 / 3, 4))
    assert row["ROI"].item() == pytest.approx(round(two_wins_one_loss / 3, 4))


def test_pushes_never_graded():
    _hist, _daily, _cal, roi = compute_individual_metrics(_history(["Over", "Over", "Push"]))
    row = roi.loc[(roi["Period"] == "7d") & (roi["Threshold"] == 0.55) & (roi["Filter"] == "All")]
    assert row["Bets"].item() == 2

    all_push = compute_individual_metrics(_history(["Push", "Push"]))
    assert all(df.empty for df in all_push)
