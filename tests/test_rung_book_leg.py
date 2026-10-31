"""X3: the book leg from sportsbook ladder rungs, the reader behind it, and the prices they round-trip to."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from sportstradamus.helpers import get_ev, get_odds
from sportstradamus.helpers.archive import Archive
from sportstradamus.helpers.training_quotes import DFS_PLATFORM_BOOKS
from sportstradamus.prediction import book_quotes as bq
from sportstradamus.prediction.model_prob import _zi_kwargs


def _rungs(*rows):
    return pd.DataFrame(rows, columns=["entity", "book", "line", "p_over", "observed_at"])


def test_exact_rungs_average_over_books():
    rungs = _rungs(("a", "fanduel", 2.5, 0.60, None), ("a", "draftkings", 2.5, 0.50, None))
    assert bq.rung_book_leg(rungs, 2.5) == (pytest.approx(0.55), 2)


def test_bracketing_rungs_interpolate_on_the_logit_scale():
    rungs = _rungs(
        ("a", "fanduel", 1.5, 0.70, None),
        ("a", "fanduel", 3.5, 0.30, None),
        ("a", "draftkings", 2.5, 0.52, None),
    )
    p, n = bq.rung_book_leg(rungs, 2.5)
    assert n == 2
    assert p == pytest.approx((expit((logit(0.70) + logit(0.30)) / 2) + 0.52) / 2)


def test_one_book_a_one_sided_ladder_or_a_rising_ladder_is_no_leg():
    assert np.isnan(bq.rung_book_leg(_rungs(("a", "fanduel", 2.5, 0.6, None)), 2.5)[0])
    one_sided = _rungs(("a", "fanduel", 0.5, 0.8, None), ("a", "draftkings", 0.5, 0.7, None))
    assert np.isnan(bq.rung_book_leg(one_sided, 1.5)[0])
    rising = _rungs(
        ("a", "fanduel", 1.5, 0.4, None),
        ("a", "fanduel", 3.5, 0.6, None),
        ("a", "draftkings", 2.5, 0.5, None),
    )
    assert np.isnan(bq.rung_book_leg(rising, 2.5)[0])


def test_get_ladder_rungs_keeps_each_books_latest_poll_before_at():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE ladder(league VARCHAR, market VARCHAR, game_date DATE, entity VARCHAR, "
        "book VARCHAR, line DOUBLE, p_over DOUBLE, observed_at TIMESTAMP)"
    )
    con.executemany(
        "INSERT INTO ladder VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "NHL",
                "shots",
                date(2026, 10, 20),
                "a",
                "fanduel",
                2.5,
                0.60,
                datetime(2026, 10, 20, 15, 0, 0),
            ),
            (
                "NHL",
                "shots",
                date(2026, 10, 20),
                "a",
                "fanduel",
                3.5,
                0.35,
                datetime(2026, 10, 20, 15, 0, 4),
            ),
            (
                "NHL",
                "shots",
                date(2026, 10, 20),
                "a",
                "fanduel",
                2.5,
                0.55,
                datetime(2026, 10, 20, 12, 0, 0),
            ),
            (
                "NHL",
                "shots",
                date(2026, 10, 20),
                "a",
                "draftkings",
                2.5,
                0.50,
                datetime(2026, 10, 20, 15, 1, 0),
            ),
            (
                "NHL",
                "shots",
                date(2026, 10, 20),
                "a",
                "draftkings",
                2.5,
                0.53,
                datetime(2026, 10, 20, 15, 2, 0),
            ),
            (
                "NHL",
                "shots",
                date(2026, 10, 20),
                "a",
                "Underdog",
                2.5,
                0.52,
                datetime(2026, 10, 20, 15, 0, 0),
            ),
        ],
    )
    stub = SimpleNamespace(_connection=con)
    rungs = Archive.get_ladder_rungs(
        stub, "NHL", "shots", "2026-10-20", ["a"], exclude_books=["Underdog"]
    )
    assert rungs[["book", "line", "p_over"]].to_dict("records") == [
        {"book": "draftkings", "line": 2.5, "p_over": 0.53},
        {"book": "fanduel", "line": 2.5, "p_over": 0.60},
        {"book": "fanduel", "line": 3.5, "p_over": 0.35},
    ]
    earlier = Archive.get_ladder_rungs(
        stub,
        "NHL",
        "shots",
        "2026-10-20",
        ["a"],
        exclude_books=["Underdog"],
        at=datetime(2026, 10, 20, 13, 0),
    )
    assert earlier["p_over"].tolist() == [0.55]
    unfiltered = Archive.get_ladder_rungs(stub, "NHL", "shots", "2026-10-20", ["a"])
    assert set(unfiltered["book"]) == {"draftkings", "fanduel", "Underdog"}
    assert Archive.get_ladder_rungs(stub, "NHL", "shots", "2026-10-21", ["a"]).empty


class _Ladder:
    def __init__(self, rungs):
        self.rungs = rungs
        self.calls = []

    def get_ladder_rungs(self, league, market, date, entities, *, exclude_books=(), at=None):
        self.calls.append((date, list(entities), exclude_books))
        return self.rungs[self.rungs["entity"].isin(entities)]


def test_apply_rung_book_leg_relegs_only_the_priced_rows(monkeypatch):
    rungs = _rungs(("a", "fanduel", 2.5, 0.60, None), ("a", "draftkings", 2.5, 0.50, None))
    ladder = _Ladder(rungs)
    monkeypatch.setattr(bq, "archive", ladder)
    offers = pd.DataFrame(
        {
            "Player": ["a", "a", "b"],
            "Line": [2.5, 3.5, 2.5],
            "Projection": [2.4, 2.4, 1.0],
            "Market Projection": [2.0, 2.0, np.nan],
            "Market EV": [0.4, 0.3, np.nan],
            "Quote Source": ["book", "book", None],
        }
    )
    dates = {"a": "2026-10-20", "b": "2026-10-21"}
    assert bq.apply_rung_book_leg(offers, "NHL", "shots", "DPO", 0.6, 0.0, dates) == 1
    assert ladder.calls == [
        ("2026-10-20", ["a"], DFS_PLATFORM_BOOKS),
        ("2026-10-21", ["b"], DFS_PLATFORM_BOOKS),
    ]
    assert offers.loc[0, "Market EV"] == pytest.approx(0.55)
    assert offers.loc[0, "Quote Books"] == 2
    decoded = 1 - get_odds(2.5, offers.loc[0, "Market Projection"], "DPO", 0.6, step=1)
    assert decoded == pytest.approx(0.55, abs=1e-6)
    assert offers.loc[0, "Quote Source"] == bq.RUNG_QUOTE_SOURCE
    assert offers.loc[1, "Market EV"] == 0.3 and np.isnan(offers.loc[2, "Market EV"])
    assert "Book Gate" not in offers.columns


def test_apply_rung_book_leg_hurdle_solves_the_gate_at_the_half_line_only(monkeypatch):
    rungs = _rungs(("a", "fanduel", 0.5, 0.42, None), ("a", "draftkings", 0.5, 0.42, None))
    monkeypatch.setattr(bq, "archive", _Ladder(rungs))
    offers = pd.DataFrame(
        {
            "Player": ["a", "a"],
            "Line": [0.5, 1.5],
            "Projection": [0.9, 0.9],
            "Market Projection": [0.8, 0.8],
            "Market EV": [0.5, 0.2],
            "Quote Source": ["book", "book"],
        }
    )
    assert (
        bq.apply_rung_book_leg(offers, "NHL", "goals", "ZINB", 0.8, 0.3, {"a": "2026-10-20"}) == 1
    )
    assert offers.loc[1, "Book Gate"] == 0.3 and offers.loc[0, "Market Projection"] == 0.9
    reproduced = 1 - get_odds(0.5, 0.9, "ZINB", 0.8, gate=offers.loc[0, "Book Gate"])
    assert reproduced == pytest.approx(0.42)
    assert offers.loc[0, "Market EV"] == pytest.approx(0.42) and offers.loc[1, "Market EV"] == 0.2


def test_zi_kwargs_pools_the_per_row_book_gate_when_a_cell_carries_one():
    frame = pd.DataFrame({"Model Gate": [0.2, 0.4], "Book Gate": [0.1, 0.5]})
    assert _zi_kwargs(frame, "ZINB", 0.3)["gate_book"].tolist() == [0.1, 0.5]
    assert _zi_kwargs(frame.drop(columns="Book Gate"), "ZINB", 0.3)["gate_book"] == 0.3


@pytest.mark.parametrize("dist", ["DPO", "NegBin"])
def test_count_family_leg_round_trips_through_the_default_decode(dist):
    line, under, cv = 2.5, 0.58, 0.7
    mean = get_ev(line, under, cv, dist)
    assert 1 - get_odds(line, mean, dist, cv, step=1) == pytest.approx(1 - under, abs=1e-6)
