"""Realized-performance building blocks (``realized.settled_offers`` and its readers).

Pins the per-offer frame every realized view prices from: posted, settled sides once per
platform at the platform payout, the quote classes, an already-annotated frame taken as
given; and the readers over it: the trailing window, ``by_split`` and ``cohort_summary``
agreeing, the worst month on platform units, and the two-cohort reliability frame.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, platform_payout
from sportstradamus.history_schema import HISTORY_COLS
from sportstradamus.realized import (
    by_split,
    calibration_summary,
    cohort_summary,
    settled_offers,
    window,
    worst_month,
)

_NOW = datetime(2026, 10, 3, 12)
# The chosen Under at 20.5 hits on 18 and misses on 25.
_HIT, _MISS = 18.0, 25.0
# A settled Underdog Under that hit, posted at the flat payout (Boost 1.0 -> 1.78), with a
# servable sportsbook quote behind it.
_OFFER = {
    "Player": "A",
    "League": "NBA",
    "Date": "2026-09-30",
    "Market": "PTS",
    "Line": 20.5,
    "Boost": 1.0,
    "Platform": "Underdog",
    "Bet": "Under",
    "Win Prob": 0.6,
    "Market Prob": 0.55,
    "Alt Line": False,
    "Model Version": "v1",
    "Market Projection": 19.5,
    "Actual": _HIT,
}
_PRICED_COLS = {"Payout", "Breakeven", "Hit", "Unit", "Recommended", "Quote"}
_CAL_COLS = ["Cohort", "Alt Line", "Bin", "Predicted", "Actual", "N", "ECE", "ROI"]


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=HISTORY_COLS)


def test_only_settled_posted_sides_count_once_per_platform():
    out = settled_offers(
        _frame(
            _OFFER,
            _OFFER,  # re-persisted: still one offer
            _OFFER | {"Platform": "Sleeper"},  # the same prop on Sleeper is a second bet
            _OFFER | {"Player": "B", "Boost": 0.0},  # side never posted
            _OFFER | {"Player": "C", "Actual": 20.5},  # push
            _OFFER | {"Player": "D", "Actual": np.nan},  # unsettled
        )
    )
    assert sorted(zip(out["Platform"], out["Player"], strict=True)) == [
        ("Sleeper", "A"),
        ("Underdog", "A"),
    ]
    assert set(HISTORY_COLS) | _PRICED_COLS <= set(out.columns)


def test_legs_price_at_the_platform_payout():
    out = settled_offers(
        _frame(
            _OFFER, _OFFER | {"Player": "B", "Platform": "Sleeper", "Boost": 2.0, "Actual": _MISS}
        )
    )
    np.testing.assert_allclose(out["Payout"], [UNDERDOG_BOOST_BASELINE, 2.0])
    np.testing.assert_allclose(out["Payout"], platform_payout(out["Boost"], out["Platform"]))
    np.testing.assert_allclose(out["Breakeven"], 1 / out["Payout"])
    assert out["Hit"].tolist() == [True, False]
    np.testing.assert_allclose(out["Unit"], [UNDERDOG_BOOST_BASELINE - 1, -1.0])


def test_quote_classes():
    out = settled_offers(
        _frame(
            _OFFER,
            _OFFER | {"Player": "B", "Market Projection": np.nan},
            # The book-priced path reads fallback even where a projection was stamped.
            _OFFER | {"Player": "C", "Model Version": "book_fallback"},
        )
    )
    assert out["Quote"].tolist() == ["quoted", "unquoted", "fallback"]


def test_a_frame_already_carrying_result_is_used_as_it_stands():
    # Re-annotating would grade from the absent Actual and drop the row; the dashboard
    # hands over frames it already annotated.
    out = settled_offers(_frame(_OFFER).drop(columns="Actual").assign(Result="Over"))
    assert out["Hit"].tolist() == [False]


def test_empty_input_comes_back_empty_with_the_priced_columns():
    for history in (pd.DataFrame(), _frame()):
        out = settled_offers(history)
        assert out.empty
        assert set(history.columns) | _PRICED_COLS == set(out.columns)
    assert out["Recommended"].dtype == bool


def test_window_keeps_the_trailing_days_from_a_utc_anchor():
    offers = settled_offers(_frame(_OFFER, _OFFER | {"Player": "B", "Date": "2026-08-20"}))
    for now in (_NOW, _NOW.replace(tzinfo=UTC)):
        assert window(offers, 30, now)["Player"].tolist() == ["A"]
    assert len(window(offers, 90, _NOW)) == 2


def test_cohort_summary_combines_the_side_split_rows():
    offers = settled_offers(
        _frame(
            _OFFER,
            _OFFER | {"Player": "B", "Actual": _MISS, "Market Prob": 0.5},
            _OFFER | {"Player": "C", "Bet": "Over", "Actual": _MISS, "Win Prob": 0.7},
            _OFFER | {"Player": "D", "Bet": "Over", "Platform": "Sleeper", "Boost": 2.2},
        )
    )
    summary = cohort_summary(offers)
    sides = by_split(offers).query("split == 'side'")
    n = sides["n"].sum()
    assert summary["n"] == n == 4
    for rate in ("hit_rate", "pred_rate", "book_rate", "payout"):
        assert summary[rate] == pytest.approx((sides[rate] * sides["n"]).sum() / n)
    assert summary["breakeven_rate"] == pytest.approx(n / (sides["payout"] * sides["n"]).sum())
    assert summary["units"] == pytest.approx(sides["units"].sum())
    assert summary["roi"] == pytest.approx(sides["units"].sum() / n)


def test_cohort_summary_of_an_empty_cohort():
    summary = cohort_summary(settled_offers(_frame(_OFFER | {"Actual": np.nan})))
    assert summary["n"] == 0
    assert all(np.isnan(value) for key, value in summary.items() if key != "n")


def test_worst_month_sums_platform_units_and_ties_to_the_earliest():
    # Flat -110 would call August worst (1-2); its 3.0 hit makes it break even at the
    # platform payout. July and September tie at 0.2 - 1 = -0.8; July is earlier.
    hit = _OFFER | {"Platform": "Sleeper"}
    miss = _OFFER | {"Actual": _MISS}
    offers = settled_offers(
        _frame(
            hit | {"Date": "2026-07-10", "Boost": 1.2},
            miss | {"Date": "2026-07-11"},
            hit | {"Date": "2026-08-10", "Boost": 3.0},
            miss | {"Date": "2026-08-11"},
            miss | {"Date": "2026-08-12"},
            hit | {"Date": "2026-09-10", "Boost": 1.2},
            miss | {"Date": "2026-09-11"},
        )
    )
    assert worst_month(offers) == {
        "month": "2026-07",
        "units": pytest.approx(-0.8),
        "n": 2,
        "win_pct": 0.5,
    }
    assert worst_month(offers.iloc[:0]) == {}


def test_calibration_summary_per_cohort_and_alt_split():
    # Main: two recommended 0.62 legs at 2.0 (one hit), an unrecommended 0.52 hit at 1.8 and
    # a 0.35 miss under the first bin edge (in ROI, not ECE). Alt: two 0.82 hits, at 1.5
    # (recommended) and at 3.0 (past the payout cap, so posted only).
    sleeper = _OFFER | {"Platform": "Sleeper"}
    offers = settled_offers(
        _frame(
            sleeper | {"Player": "A", "Win Prob": 0.62, "Boost": 2.0},
            sleeper | {"Player": "B", "Win Prob": 0.62, "Boost": 2.0, "Actual": _MISS},
            sleeper | {"Player": "C", "Win Prob": 0.52, "Boost": 1.8},
            sleeper | {"Player": "D", "Win Prob": 0.35, "Boost": 1.5, "Actual": _MISS},
            sleeper | {"Player": "E", "Win Prob": 0.82, "Boost": 1.5, "Alt Line": True},
            sleeper | {"Player": "F", "Win Prob": 0.82, "Boost": 3.0, "Alt Line": True},
        )
    )
    out = calibration_summary(offers)
    assert list(out.columns) == _CAL_COLS
    splits = out.groupby(["Cohort", "Alt Line"]).agg(
        N=("N", "sum"), ECE=("ECE", "first"), ROI=("ROI", "first")
    )
    expected = {  # (binned N, ECE, ROI)
        ("posted", False): (3, (2 * 0.12 + 0.48) / 3, (1.0 - 1.0 + 0.8 - 1.0) / 4),
        ("posted", True): (2, 0.18, (0.5 + 2.0) / 2),
        ("recommended", False): (2, 0.12, 0.0),
        ("recommended", True): (1, 0.18, 0.5),
    }
    assert set(splits.index) == set(expected)
    for split, (n, ece, roi) in expected.items():
        assert splits.loc[split, "N"] == n
        assert splits.loc[split, "ECE"] == pytest.approx(ece)
        assert splits.loc[split, "ROI"] == pytest.approx(roi)


def test_calibration_summary_drops_a_split_with_no_binned_row():
    # The alt leg settled, but its 0.35 sits under the first bin edge, so the alt split has
    # no reliability row at all rather than an empty or NaN one.
    sleeper = _OFFER | {"Platform": "Sleeper", "Boost": 2.0}
    offers = settled_offers(
        _frame(sleeper, sleeper | {"Player": "B", "Win Prob": 0.35, "Alt Line": True})
    )
    out = calibration_summary(offers)
    assert set(zip(out["Cohort"], out["Alt Line"], strict=True)) == {
        ("posted", False),
        ("recommended", False),
    }


def test_calibration_summary_of_nothing_settled_is_empty_with_columns():
    out = calibration_summary(settled_offers(_frame(_OFFER | {"Actual": np.nan})))
    assert out.empty
    assert list(out.columns) == _CAL_COLS
