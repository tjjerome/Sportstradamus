"""Pure skeptic aggregations behind the Receipts surface.

Hero (``tailed_record``), the EV>5% skeptic record (``ev_threshold_record``),
``worst_month``, and the by-dimension ``record_grid`` — all over the exploded,
Push-dropped per-offer frame that ``get_filtered_history`` already returns. The
accounting is the flat -110 juice the surface's "Profit Units" KPI used, so the
hero number is provably the same as before. No Streamlit, no I/O.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.analysis import (
    _FLAT_DECIMAL_ODDS,
    JUICE_PAYOUT,
    ev_threshold_record,
    record_grid,
    tailed_record,
    worst_month,
)


def _exploded() -> pd.DataFrame:
    # Six resolved offers, no precomputed Hit (the aggregations derive it).
    # Win Prob spans the 0.55 EV>5% boundary (0.55 included; 0.52/0.50 excluded).
    return pd.DataFrame(
        [
            {"Bet": "Over", "Result": "Over", "Win Prob": 0.60,
             "Date": "2026-01-15", "League": "NBA", "Market": "PTS", "Platform": "Underdog"},
            {"Bet": "Over", "Result": "Under", "Win Prob": 0.58,
             "Date": "2026-01-20", "League": "NBA", "Market": "PTS", "Platform": "Underdog"},
            {"Bet": "Under", "Result": "Under", "Win Prob": 0.70,
             "Date": "2026-02-10", "League": "NBA", "Market": "AST", "Platform": "Sleeper"},
            {"Bet": "Over", "Result": "Under", "Win Prob": 0.52,
             "Date": "2026-02-15", "League": "WNBA", "Market": "PTS", "Platform": "Underdog"},
            {"Bet": "Under", "Result": "Under", "Win Prob": 0.55,
             "Date": "2026-03-05", "League": "WNBA", "Market": "REB", "Platform": "Sleeper"},
            {"Bet": "Over", "Result": "Over", "Win Prob": 0.50,
             "Date": "2026-03-10", "League": "NBA", "Market": "PTS", "Platform": "Underdog"},
        ]
    )


def test_tailed_record_hero_parity():
    df = _exploded()
    rec = tailed_record(df)
    # The exact inline accounting receipts.py used for its "Profit Units" KPI.
    hit = (df["Bet"] == df["Result"]).astype(int)
    legacy_units = (hit * JUICE_PAYOUT - (1 - hit)).sum()
    assert rec["units"] == pytest.approx(legacy_units)
    assert rec["n"] == 6
    assert rec["wins"] == 4
    assert rec["losses"] == 2
    assert rec["wins"] + rec["losses"] == rec["n"]
    assert rec["roi"] == pytest.approx(rec["units"] / rec["n"])
    assert rec["win_pct"] == pytest.approx(4 / 6)


def test_tailed_record_empty():
    rec = tailed_record(pd.DataFrame())
    assert rec["n"] == 0
    assert rec["units"] == 0
    assert rec["win_pct"] == 0
    assert rec["roi"] == 0


def test_ev_threshold_record_flat_edge_boundary():
    df = _exploded()
    rec = ev_threshold_record(df, edge_min=0.05)
    # 0.55 is the exact boundary: 0.55 * (1 + 100/110) - 1 == 0.05.
    assert 0.55 * _FLAT_DECIMAL_ODDS - 1 == pytest.approx(0.05)
    # Qualifiers: 0.60 (hit), 0.58 (miss), 0.70 (hit), 0.55 (hit). 0.52/0.50 drop out.
    assert rec["n"] == 4
    assert rec["wins"] == 3
    assert rec["losses"] == 1
    assert rec["win_pct"] == pytest.approx(0.75)
    units = 3 * JUICE_PAYOUT - 1
    assert rec["units"] == pytest.approx(units)
    assert rec["roi"] == pytest.approx(units / 4)


def test_ev_threshold_record_none_qualify():
    df = _exploded().assign(**{"Win Prob": 0.40})
    rec = ev_threshold_record(df, edge_min=0.05)
    assert rec["n"] == 0
    assert rec["units"] == 0
    assert rec["win_pct"] == 0


def test_worst_month_min_units_ties_earliest():
    wm = worst_month(_exploded())
    # 2026-01 and 2026-02 each net one hit + one miss (== JUICE_PAYOUT - 1);
    # the tie breaks to the earlier month.
    assert wm["month"] == "2026-01"
    assert wm["units"] == pytest.approx(JUICE_PAYOUT - 1)
    assert wm["n"] == 2
    assert wm["win_pct"] == pytest.approx(0.5)


def test_worst_month_empty():
    assert worst_month(pd.DataFrame()) == {}


def test_record_grid_by_league_sorted_by_units_desc():
    grid = record_grid(_exploded(), by="League")
    assert list(grid["League"]) == ["NBA", "WNBA"]  # +1.727u > -0.091u
    nba = grid.loc[grid["League"] == "NBA"].iloc[0]
    assert nba["Bets"] == 4
    assert nba["Win%"] == pytest.approx(0.75)
    assert nba["Units"] == pytest.approx(3 * JUICE_PAYOUT - 1)
    wnba = grid.loc[grid["League"] == "WNBA"].iloc[0]
    assert wnba["Bets"] == 2
    assert wnba["Win%"] == pytest.approx(0.5)
    assert wnba["Units"] == pytest.approx(JUICE_PAYOUT - 1)


def test_record_grid_by_platform():
    grid = record_grid(_exploded(), by="Platform")
    assert list(grid["Platform"]) == ["Sleeper", "Underdog"]  # +1.818u > -0.182u
    sleeper = grid.loc[grid["Platform"] == "Sleeper"].iloc[0]
    assert sleeper["Bets"] == 2
    assert sleeper["Win%"] == pytest.approx(1.0)
    assert sleeper["Units"] == pytest.approx(2 * JUICE_PAYOUT)
    assert sleeper["ROI"] == pytest.approx(JUICE_PAYOUT)


def test_record_grid_empty():
    grid = record_grid(pd.DataFrame(), by="League")
    assert grid.empty
    assert list(grid.columns) == ["League", "Bets", "Win%", "Units", "ROI"]
