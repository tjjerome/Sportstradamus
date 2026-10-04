"""Realized ledger by bet side (``realized.compute_realized_by_side``).

Pins the contract the Receipts panel reads: a fixed long schema, units priced at the
platform payout (not fair odds), unposted sides (``Boost == 0``) dropped, the recommended
edge floor and payout cap, right-closed payout bands, the breakeven rate, the quote split,
one row per real offer, and the trailing windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.prediction.stories import menu
from sportstradamus.realized import (
    REALIZED_BY_SIDE_COLS,
    RECOMMENDED_EDGE_MIN,
    compute_realized_by_side,
)

_NOW = datetime(2026, 10, 3, 12)
# A settled Underdog Under that hit (Actual below Line), posted at the flat payout
# (Boost 1.0 -> 1.78) with a 0.6 x 1.78 - 1 = 0.068 edge, so it sits in both cohorts.
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
    "Actual": 18.0,
}
_EXPECTED_DTYPES = {
    "computed_at": "datetime64[ns]",
    "window_days": "int16",
    "cohort": "object",
    "split": "object",
    "key": "object",
    "side": "object",
    "n": "int64",
    "hit_rate": "float64",
    "pred_rate": "float64",
    "book_rate": "float64",
    "breakeven_rate": "float64",
    "units": "float64",
    "roi": "float64",
}


def _ledger(*rows: dict) -> pd.DataFrame:
    return compute_realized_by_side(pd.DataFrame(list(rows)), now=_NOW)


def _one(out, *, split="side", key="all", side="Under", cohort="bettable", window_days=30):
    rows = out[
        (out["window_days"] == window_days)
        & (out["cohort"] == cohort)
        & (out["split"] == split)
        & (out["key"] == key)
        & (out["side"] == side)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def _keys(out, split, *, cohort="bettable"):
    in_split = (out["window_days"] == 30) & (out["cohort"] == cohort) & (out["split"] == split)
    return set(out.loc[in_split, "key"])


@pytest.mark.parametrize(
    ("history", "settles"),
    [
        (pd.DataFrame(), False),
        (pd.DataFrame([_OFFER | {"Actual": np.nan}]), False),
        (pd.DataFrame([_OFFER]), True),
    ],
    ids=["empty", "unsettled", "settled"],
)
def test_schema_and_dtypes_are_fixed(history, settles):
    # Nightly rewrites the parquet every run; an empty run must carry the full run's schema.
    out = compute_realized_by_side(history, now=_NOW)
    assert list(out.columns) == REALIZED_BY_SIDE_COLS
    assert out.dtypes.astype(str).to_dict() == _EXPECTED_DTYPES
    assert out.empty is not settles
    assert (out["computed_at"] == pd.Timestamp(_NOW)).all()


def test_units_price_at_the_platform_payout_not_fair_odds():
    # Fair odds on an 0.88 book favourite's Under would pay 1 / 0.12; Underdog pays Boost x
    # its flat baseline, and Sleeper's Boost is the whole decimal payout.
    out = _ledger(
        _OFFER | {"Market Prob": 0.88},
        _OFFER | {"Player": "B", "Platform": "Sleeper", "Boost": 1.5},
    )
    underdog = _one(out, split="platform", key="Underdog")
    assert underdog["units"] == pytest.approx(UNDERDOG_BOOST_BASELINE - 1)
    assert underdog["book_rate"] == pytest.approx(0.88)
    assert _one(out, split="platform", key="Sleeper")["units"] == pytest.approx(0.5)
    assert _one(out)["roi"] == pytest.approx((UNDERDOG_BOOST_BASELINE - 1 + 0.5) / 2)


def test_unposted_side_is_in_neither_cohort():
    # Boost 0 = the platform never posted the chosen side, so it was never a bet.
    out = _ledger(_OFFER, _OFFER | {"Player": "B", "Boost": 0.0, "Win Prob": 0.95})
    for cohort in ("bettable", "recommended"):
        row = _one(out, cohort=cohort)
        assert row["n"] == 1
        assert row["pred_rate"] == pytest.approx(0.6)


def test_recommended_needs_the_edge_floor_at_the_platform_payout():
    # 0.52 x a 2.0 Sleeper payout - 1 = 0.04, under the 0.05 floor: bettable, not recommended.
    out = _ledger(
        _OFFER, _OFFER | {"Player": "B", "Platform": "Sleeper", "Boost": 2.0, "Win Prob": 0.52}
    )
    assert _keys(out, "platform") == {"Underdog", "Sleeper"}
    assert _keys(out, "platform", cohort="recommended") == {"Underdog"}


@pytest.mark.parametrize(
    ("payout", "band"),
    [(0.9, "<=1.00"), (1.2, "1.00-1.50"), (2.5, "2.00-2.50"), (3.5, ">3.00")],
)
def test_payout_band_labels(payout, band):
    # Sleeper's Boost is the payout itself; 2.5 sits on a right-closed edge, inside 2.00-2.50.
    out = _ledger(_OFFER | {"Platform": "Sleeper", "Boost": payout})
    assert _keys(out, "payout_band") == {band}


def test_one_row_per_offer_per_platform():
    # The same offer persisted twice is one bet; the same prop on Sleeper is a second bet.
    out = _ledger(_OFFER, _OFFER, _OFFER | {"Platform": "Sleeper"})
    assert _one(out, split="platform", key="Underdog")["n"] == 1
    assert _one(out)["n"] == 2


def test_split_keys_fill_legacy_rows():
    # Pre-stamp rows carry no Alt Line / Model Version: they read as main and "legacy".
    out = _ledger(
        _OFFER,
        _OFFER | {"Player": "B", "Alt Line": None, "Model Version": np.nan},
        _OFFER | {"Player": "C", "Alt Line": True},
    )
    assert _one(out, split="alt_line", key="main")["n"] == 2
    assert _one(out, split="alt_line", key="alt")["n"] == 1
    assert _one(out, split="model_version", key="legacy")["n"] == 1
    assert _keys(out, "cell") == {"NBA/PTS"}


def test_recommended_stops_at_the_menu_payout_cap():
    # Both Sleeper legs clear the edge floor; 2.5 sits on the cap, 3.0 pays past it, where
    # the menu zeroes Kelly and never shows the leg.
    out = _ledger(
        _OFFER | {"Platform": "Sleeper", "Boost": 2.5},
        _OFFER | {"Player": "B", "Platform": "Sleeper", "Boost": 3.0},
    )
    assert _keys(out, "payout_band") == {"2.00-2.50", "2.50-3.00"}
    assert _keys(out, "payout_band", cohort="recommended") == {"2.00-2.50"}


def test_breakeven_rate_is_n_over_the_summed_payout():
    # Legs paying 1.78 and 2.0 break even at 2 / 3.78, not at the mean of 1 / payout.
    out = _ledger(_OFFER, _OFFER | {"Player": "B", "Platform": "Sleeper", "Boost": 2.0})
    assert _one(out)["breakeven_rate"] == pytest.approx(2 / (UNDERDOG_BOOST_BASELINE + 2.0))


def test_quote_split_reads_fallback_before_the_sportsbook_quote():
    out = _ledger(
        _OFFER,
        _OFFER | {"Player": "B", "Market Projection": np.nan},
        _OFFER | {"Player": "C", "Model Version": "book_fallback"},
    )
    for key in ("quoted", "unquoted", "fallback"):
        assert _one(out, split="quote", key=key)["n"] == 1


def test_recommended_edge_matches_the_story_menu_floor():
    assert RECOMMENDED_EDGE_MIN == menu._MENU_EDGE_FLOOR


def test_30_day_window_drops_older_offers():
    old = (_NOW - timedelta(days=40)).date().isoformat()
    out = _ledger(_OFFER, _OFFER | {"Player": "B", "Date": old})
    assert _one(out, window_days=30)["n"] == 1
    assert _one(out, window_days=90)["n"] == 2
