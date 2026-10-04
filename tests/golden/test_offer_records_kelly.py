"""Golden tests for the honest-Kelly / favored-payout-cap rule in
:func:`sportstradamus.prediction.offer_records.finalize_records`.

Kelly is zeroed rather than computed outside ``(1, MAX_FAVORED_PAYOUT]``: a payout at
or below 1x can never carry edge, and the realized ledger doesn't trust the model's
edge claim above the cap (see ``MAX_FAVORED_PAYOUT``'s module comment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.prediction import offer_records

_LEAGUE = "NBA"
_MARKET = "points"
_DIST = "NegBin"
_CV = 0.5
_STEP = 1.0
_MODEL_VERSION = "test-version"


class _StubArchive:
    """Minimal archive surface finalize_records reads: default_totals only."""

    default_totals = {_LEAGUE: 220.0}


@pytest.fixture(autouse=True)
def _stub_record_archive(monkeypatch):
    # finalize_records reads default_totals off offer_records' own LazyArchive, so a
    # test that leaves it unpatched opens the real archive and, under xdist, waits on
    # whichever worker already holds its lock.
    monkeypatch.setattr(offer_records, "archive", _StubArchive())


def _row(player: str, model_over: float, model_under: float, boost_over: float) -> dict:
    """One minimal offer row: ``Model Over`` always wins so ``Bet`` resolves to "Over"
    and ``boost_over`` prices the chosen side."""
    return {
        "League": _LEAGUE,
        "Date": "2026-06-03",
        "Team": "LAL",
        "Opponent": "BOS",
        "Player": player,
        "Market": _MARKET,
        "Line": 25.5,
        "Model Over": model_over,
        "Model Under": model_under,
        "Boost_Over": boost_over,
        "Boost_Under": np.nan,
        "Market EV": 0.5,
        "Projection": 25.5,
        "Push Prob": 0.0,
        "Market Projection": np.nan,
        "Books STD": np.nan,
        "Model Weight": 1.0,
        "Quote Source": None,
        "Quote Authenticity": None,
        "Quote Books": np.nan,
        "Quote Line": np.nan,
        "Quote Observed At": pd.NaT,
    }


def _finalize(row: dict, platform: str) -> dict:
    df = pd.DataFrame([row])
    records = offer_records.finalize_records(
        df, _LEAGUE, platform, _DIST, _CV, _STEP, None, 1.0, _MODEL_VERSION
    )
    assert len(records) == 1
    return records[0]


def test_underdog_payout_at_or_below_one_zeroes_kelly():
    # Raw Underdog multiplier 0.56 x the 1.78 baseline is a ~0.997x full payout.
    rec = _finalize(_row("Case A", 0.90, 0.05, 0.56), "Underdog")
    assert rec["Model EV"] < 1
    assert rec["Kelly"] == 0.0


def test_payout_above_favored_cap_zeroes_kelly():
    # 3.0 is a full payout already (Sleeper), and sits above MAX_FAVORED_PAYOUT (2.5).
    rec = _finalize(_row("Case B", 0.5, 0.1, 3.0), "Sleeper")
    assert rec["Model EV"] == pytest.approx(1.5)
    assert rec["Kelly"] == 0.0


def test_payout_within_range_computes_kelly():
    rec = _finalize(_row("Case C", 0.8, 0.1, 1.5), "Sleeper")
    assert rec["Model EV"] == pytest.approx(1.2)
    assert rec["Kelly"] == pytest.approx((1.2 - 1) / 0.5)
