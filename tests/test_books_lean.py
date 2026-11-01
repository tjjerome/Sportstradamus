"""X2: which legs qualify, the paper side, the confidence sequence's width, the verdict rule."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from sportstradamus.scripts import books_lean as bl


def _history() -> pd.DataFrame:
    # a: main + alt line of one NBA prop (keep the main); b: MLB home runs (rare, out);
    # c: before COUNT_ONCE_FROM (out); d: NHL leg never scored (out).
    return pd.DataFrame(
        {
            "Date": ["2026-10-10", "2026-10-10", "2026-10-10", "2026-10-01", "2026-10-11"],
            "Player": ["a", "a", "b", "c", "d"],
            "League": ["NBA", "NBA", "MLB", "NBA", "NHL"],
            "Market": ["PTS", "PTS", "home runs", "PTS", "shots"],
            "Line": [20.5, 24.5, 0.5, 20.5, 2.5],
            "Bet": ["Over", "Over", "Over", "Over", "Under"],
            "Win Prob": [0.60, 0.70, 0.55, 0.60, 0.58],
            "Boost": [1.0, 2.2, 1.0, 1.0, 1.0],
            "Platform": ["Underdog"] * 5,
            "Model Version": ["20261018.x"] * 5,
            "Market Projection": [21.0, 21.0, np.nan, 21.0, 2.4],
            "Market Prob": [0.5] * 5,
            "Alt Line": [False, True, False, False, False],
            "Result": ["Over", "Over", "Under", "Over", "Under"],
            "Scored At": [pd.Timestamp("2026-10-10 20:00")] * 4 + [pd.NaT],
        }
    )


def test_qualified_legs_one_per_prop_at_its_main_line_outside_rare_cells():
    legs = bl.qualified_legs(_history())
    assert legs[["Player", "Line"]].to_dict("records") == [{"Player": "a", "Line": 20.5}]
    assert list(legs["cell"]) == ["PTS"]


def test_paper_side_needs_the_lean_band_and_a_priced_side():
    legs = pd.DataFrame(
        {
            "Bet": ["Over", "Over", "Under", "Over", "Over", "Over", "Over"],
            "Payout Over": [1.83, 2.93, 1.83, 1.83, 1.83, 1.83, 2.93],
            "Payout Under": [1.83, 0.0, 1.83, 1.83, 1.83, 2.2, np.nan],
            "Result": ["Over", "Under", "Under", "Over", "Over", "Under", "Under"],
            "p_books": [0.58, 0.40, 0.42, 0.70, 0.52, 0.40, 0.40],
            "n_books": [2, 3, 2, 2, 2, 3, 2],
            "age_h": [0.5, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            "League": ["NBA", "NBA", "NHL", "NBA", "NBA", "NBA", "NBA"],
            "Date": ["2026-10-10"] * 7,
            "Platform": ["Underdog"] * 7,
        }
    )
    picks = bl.paper_sides(legs)
    # row 1: lean Under against a boosted Over whose Under the record holds unposted (0) -> out;
    # row 3: lean .70 past LEAN_MAX -> out; row 4: no lean -> out;
    # row 5: lean Under against the board's Over, priced at the Under's own posted payout;
    # row 6: the lean's side without a payout -> out.
    assert list(picks.index) == [0, 2, 5]
    assert list(picks["side"]) == ["Over", "Under", "Under"]
    assert list(picks["hit"]) == [True, True, True]
    assert list(picks["payout"]) == [1.83, 1.83, 2.2]
    assert picks["excess"].iloc[0] == pytest.approx(1 - 1 / 1.83)
    assert picks["excess"].iloc[2] == pytest.approx(1 - 1 / 2.2)
    assert list(picks["model_agrees"]) == [True, True, False]
    assert list(picks["league_group"]) == ["other", "NHL", "other"]
    assert list(picks["books3"]) == [False, False, True]
    assert list(picks["fresh"]) == [True, True, True]


def test_sequence_radius_at_the_design_sample_and_its_growth():
    rho = bl.mixture_rho(1000, 0.10)
    assert abs(rho - 37.7) < 1.0
    lo, hi = bl.sequence_bounds(np.zeros(1000), 0.10)
    assert hi == pytest.approx(0.0437, abs=0.001) and lo == pytest.approx(-hi)
    assert bl.sequence_bounds(np.zeros(400), 0.10)[1] > hi
    assert bl.sequence_bounds(np.zeros(1000), 0.10 / 3)[1] > hi


def test_verdict_rule():
    assert bl.verdict(n=500, dates=30, hit=0.60, excess_lb=0.01, cs_lo=-0.02, cs_hi=0.05) == "open"
    assert (
        bl.verdict(n=1000, dates=60, hit=0.60, excess_lb=0.01, cs_lo=-0.02, cs_hi=0.05) == "accept"
    )
    assert (
        bl.verdict(n=300, dates=20, hit=0.60, excess_lb=-0.01, cs_lo=0.001, cs_hi=0.05) == "accept"
    )
    assert bl.verdict(n=400, dates=30, hit=0.53, excess_lb=-0.05, cs_lo=-0.08, cs_hi=0.01) == "kill"
    assert (
        bl.verdict(n=200, dates=10, hit=0.40, excess_lb=-0.20, cs_lo=-0.30, cs_hi=-0.01) == "kill"
    )


def test_ladder_leans_reads_each_books_latest_poll_before_scoring():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE ladder(league VARCHAR, market VARCHAR, game_date DATE, entity VARCHAR, "
        "book VARCHAR, line DOUBLE, p_over DOUBLE, observed_at TIMESTAMP)"
    )
    # a: two sportsbooks and Underdog (a DFS book, out); an old poll at .70 superseded by the
    # book's 19:00 poll; a fanduel poll after the 20:00 scoring, ignored. b: no rung at all.
    rows = []
    for book, p_old, p_new in (
        ("fanduel", 0.70, 0.58),
        ("draftkings", 0.70, 0.60),
        ("Underdog", 0.5, 0.5),
    ):
        for line, dp in ((19.5, 0.08), (20.5, 0.0), (21.5, -0.08)):
            rows.append(
                ("NBA", "PTS", "2026-10-10", "a", book, line, p_old + dp, "2026-10-10 15:00:00")
            )
            rows.append(
                ("NBA", "PTS", "2026-10-10", "a", book, line, p_new + dp, "2026-10-10 19:00:00")
            )
    rows.append(("NBA", "PTS", "2026-10-10", "a", "fanduel", 20.5, 0.30, "2026-10-10 21:00:00"))
    con.executemany("INSERT INTO ladder VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    legs = pd.DataFrame(
        {
            "League": ["NBA", "NBA"],
            "cell": ["PTS", "PTS"],
            "Date": ["2026-10-10", "2026-10-10"],
            "Player": ["a", "b"],
            "Line": [20.5, 10.5],
            "Scored At": [pd.Timestamp("2026-10-10 20:00")] * 2,
        },
        index=[7, 3],
    )
    leans = bl.ladder_leans(con, legs)
    assert list(leans.index) == [7, 3]
    assert leans.loc[7, "p_books"] == pytest.approx(0.59)
    assert leans.loc[7, "n_books"] == 2 and leans.loc[7, "age_h"] == 1.0
    assert np.isnan(leans.loc[3, "p_books"]) and leans.loc[3, "n_books"] == 0
    assert np.isnan(leans.loc[3, "age_h"])


def test_look_rows_strata_and_the_split_level():
    rng = np.random.default_rng(7)
    n = 48
    picks = pd.DataFrame(
        {
            "Date": [f"2026-10-{10 + i % 12:02d}" for i in range(n)],
            "Platform": ["Underdog", "Sleeper"] * (n // 2),
            "league_group": ["NHL"] * (n // 2) + ["other"] * (n // 2),
            "hit": rng.random(n) < 0.6,
            "breakeven": 1 / 1.83,
            "books3": True,
            "fresh": False,
            "model_agrees": [True, False] * (n // 2),
        }
    )
    picks["excess"] = picks["hit"].astype(float) - picks["breakeven"]
    table = bl.look_rows(picks, pd.Timestamp("2026-11-01"))
    assert list(table.columns) == list(bl.LOOK_COLUMNS)
    strata = list(zip(table["arm"], table["platform"], table["league_group"], strict=True))
    assert strata == [
        ("primary", "all", "all"),
        ("primary", "Sleeper", "NHL"),
        ("primary", "Sleeper", "other"),
        ("primary", "Underdog", "NHL"),
        ("primary", "Underdog", "other"),
        ("books3", "all", "all"),
        ("model_agrees", "all", "all"),
    ]
    assert list(table["alpha"]) == [bl.CS_ALPHA] * 5 + [bl.CS_ALPHA / 3] * 2
    assert list(table["n"]) == [48, 12, 12, 12, 12, 48, 24]
    assert table["verdict"].isin(["open", "accept", "kill"]).all()
    # books3 holds every pick: its bound at alpha / 3 sits at or below the same draws' bound at alpha.
    _, lb_at_alpha, _ = bl._bootstrap_mean_ci_clustered(
        picks["excess"].to_numpy(float),
        picks["Date"].to_numpy(),
        bl.key_rng(bl.LEAN_SEED, "books3/all/all"),
        percentiles=(100 * bl.CS_ALPHA, 100 * (1 - bl.CS_ALPHA)),
    )
    books3 = table[table["arm"] == "books3"].iloc[0]
    assert books3["excess_lb"] <= lb_at_alpha
    assert books3["cs_lo"] < table.iloc[0]["cs_lo"]
