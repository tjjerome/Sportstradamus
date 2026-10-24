"""Golden tests for the honest-Kelly / favored-payout-cap rule and the per-player rung trim
in :func:`sportstradamus.prediction.offer_records.finalize_records`.

Kelly is zeroed rather than computed outside ``(1, MAX_FAVORED_PAYOUT]``: a payout at
or below 1x can never carry edge, and the realized ledger doesn't trust the model's
edge claim above the cap (see ``MAX_FAVORED_PAYOUT``'s module comment). The trim ranks
and caps a player's rungs against the platform's even pick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.helpers.archive import _dfs_offer_probs
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


def _finalize_all(rows: list[dict], platform: str) -> list[dict]:
    return offer_records.finalize_records(
        pd.DataFrame(rows), _LEAGUE, platform, _DIST, _CV, _STEP, None, 1.0, _MODEL_VERSION
    )


def _finalize(row: dict, platform: str) -> dict:
    records = _finalize_all([row], platform)
    assert len(records) == 1
    return records[0]


def test_underdog_payout_at_or_below_one_zeroes_kelly():
    # A raw Underdog multiplier whose full payout is 0.99x.
    rec = _finalize(_row("Case A", 0.90, 0.05, 0.99 / UNDERDOG_BOOST_BASELINE), "Underdog")
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


@pytest.mark.parametrize(
    ("platform", "at_cap", "past_cap"),
    [
        # Underdog's boost is the raw multiplier: 2.05 x 1.78 is a 3.649x payout.
        ("Underdog", 2.05, 2.06),
        ("Sleeper", 3.65, 3.66),
    ],
)
def test_boost_cap_passes_what_a_3_65x_payout_passed_at_the_1_78_even_pick(
    platform, at_cap, past_cap
):
    # The cap was set as a 3.65x payout when an even pick paid 1.78x on both platforms; what
    # a 1.00x Underdog pick is worth since must not move which legs clear it.
    rows = [_row("at cap", 0.6, 0.1, at_cap), _row("past cap", 0.6, 0.1, past_cap)]
    assert [rec["Player"] for rec in _finalize_all(rows, platform)] == ["at cap"]


@pytest.mark.parametrize(
    ("platform", "boosts", "farthest"),
    [
        # Underdog ranks on the raw multiplier, whatever a 1.00x pick is worth.
        ("Underdog", [0.95, 0.97, 1.0, 1.03], 0.95),
        # Sleeper ranks its posted payout against its own even pick, 1.78x.
        ("Sleeper", [1.70, 1.78, 1.87, 1.95], 1.95),
    ],
)
def test_a_players_rungs_trim_to_the_three_nearest_the_even_pick(platform, boosts, farthest):
    rows = [_row("Case D", 0.6, 0.1, boost) | {"Line": boost} for boost in boosts]
    kept = {rec["Line"] for rec in _finalize_all(rows, platform)}
    assert kept == set(boosts) - {farthest}


def test_unquoted_one_sided_market_prob_is_the_price_the_archive_stores():
    # clv subtracts a leg's Market Prob from the close the archive hands back. With no book
    # on the leg both are the platform's own price, so an unmoved rung has to read the same
    # in the two places or it shows a closing-line move that never happened.
    rec = _finalize(_row("Case E", 0.45, 0.1, 1.4) | {"Market EV": np.nan}, "Underdog")
    stored_over, _ = _dfs_offer_probs({"Boost_Over": 1.4, "Boost_Under": 0}, "Underdog")
    assert rec["Market Prob"] == pytest.approx(stored_over)
