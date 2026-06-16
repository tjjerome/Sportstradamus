"""Tests for Phase 3 Step 5: fractional-Kelly stake sizing."""

from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from sportstradamus.strategies import kelly
from sportstradamus.strategies.kelly import (
    DEFAULT_KELLY_FRACTION,
    LIVE_BLEND_FLOOR,
    LIVE_BLEND_FULL,
    MAX_FRACTION_OF_BANKROLL,
    KellyCandidate,
    fractional_kelly_stake,
    joint_kelly_portfolio,
    resolve_shrinkage,
)

# --------------------------------------------------------------------------- #
# fractional_kelly_stake


def test_negative_ev_returns_zero():
    # win_prob 0.4, payout 2x → b=1, bp - q = 0.4 - 0.6 < 0
    assert fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.4,
        payout_multiplier=Decimal("2"),
    ) == Decimal("0")


def test_positive_ev_matches_closed_form():
    # b=2, p=0.5, fraction=0.25, bankroll=$1000 → f* = (2*0.5 - 0.5)/2 = 0.25
    # quarter-Kelly = 0.0625; cap at 0.005 → expected stake $5.00.
    stake = fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.5,
        payout_multiplier=Decimal("3"),
        fraction=DEFAULT_KELLY_FRACTION,
    )
    assert stake == Decimal("5.00")


def test_hard_cap_active():
    # Big edge → uncapped stake exceeds 0.5 % of bankroll → cap should pin it.
    stake = fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.9,
        payout_multiplier=Decimal("3"),
    )
    cap = Decimal("1000") * Decimal(repr(MAX_FRACTION_OF_BANKROLL))
    assert stake == cap.quantize(Decimal("0.01"))


# --------------------------------------------------------------------------- #
# resolve_shrinkage — blending ramp


def test_resolve_explicit_overrides():
    assert resolve_shrinkage(
        explicit=0.42, training_bss=0.9, live_bss=0.1, live_n=200
    ) == pytest.approx(0.42)


def test_resolve_midramp_blends_evenly():
    # n = (25+100)/2 = 62.5 → w_live = (62-25)/75 ≈ 0.4933 (using n=62)
    n = 62
    w_live = (n - LIVE_BLEND_FLOOR) / (LIVE_BLEND_FULL - LIVE_BLEND_FLOOR)
    expected = w_live * 0.2 + (1 - w_live) * 0.8
    out = resolve_shrinkage(training_bss=0.8, live_bss=0.2, live_n=n)
    assert out == pytest.approx(expected, abs=1e-9)


def test_resolve_neither_logs_debug_and_returns_one(caplog, monkeypatch):
    # The structured logger silences DEBUG and disables propagation; swap in
    # a plain stdlib logger for the duration of this test so caplog can see
    # the DEBUG record.
    plain = logging.getLogger("kelly-test")
    plain.setLevel(logging.DEBUG)
    monkeypatch.setattr(kelly, "_logger", plain)
    with caplog.at_level(logging.DEBUG, logger="kelly-test"):
        out = resolve_shrinkage(training_bss=None, live_bss=None, live_n=0)
    assert out == 1.0
    assert any("fallback" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# joint_kelly_portfolio


def test_portfolio_respects_total_budget():
    cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
    bankroll = Decimal("1000")
    out = joint_kelly_portfolio(
        bankroll=bankroll,
        candidates=[
            KellyCandidate("a", win_prob=0.6, payout_multiplier=Decimal("3")),
            KellyCandidate("b", win_prob=0.65, payout_multiplier=Decimal("3")),
        ],
        fraction=DEFAULT_KELLY_FRACTION,
    )
    total = sum(out.values(), Decimal("0"))
    assert total <= bankroll * Decimal(repr(DEFAULT_KELLY_FRACTION))


def test_portfolio_per_leg_cap_holds():
    cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
    bankroll = Decimal("1000")
    out = joint_kelly_portfolio(
        bankroll=bankroll,
        candidates=[
            KellyCandidate("a", win_prob=0.95, payout_multiplier=Decimal("3")),
        ],
    )
    cap = bankroll * Decimal(repr(MAX_FRACTION_OF_BANKROLL))
    assert out["a"] <= cap.quantize(Decimal("0.01"))


def test_size_candidates_shrinks_filters_and_pairs():
    # Direct (cvxpy-free) pin of the sizing step: shrinkage-adjusted p, net odds
    # b, and the floor/EV drops. Expected values derived by hand from the
    # effective_p = 0.5 + (win_prob - 0.5) * shrinkage rule.
    sized_p, sized_b, bet_ids = kelly._size_candidates(
        [
            KellyCandidate("a", win_prob=0.6, payout_multiplier=Decimal("3")),
            KellyCandidate("bad", win_prob=0.2, payout_multiplier=Decimal("2")),
            KellyCandidate(
                "floor", win_prob=0.6, payout_multiplier=Decimal("3"), model_shrinkage=0.0
            ),
            KellyCandidate("b", win_prob=0.65, payout_multiplier=Decimal("3"), model_shrinkage=0.5),
        ]
    )
    assert bet_ids == ["a", "b"]
    assert sized_p == pytest.approx([0.6, 0.575])
    assert sized_b == pytest.approx([2.0, 2.0])


def test_size_candidates_shrinks_filters_and_pairs():
    # Direct (cvxpy-free) pin of the sizing step: shrinkage-adjusted p, net odds
    # b, and the floor/EV drops. Expected values derived by hand from the
    # effective_p = 0.5 + (win_prob - 0.5) * shrinkage rule.
    sized_p, sized_b, bet_ids = kelly._size_candidates(
        [
            KellyCandidate("a", win_prob=0.6, payout_multiplier=Decimal("3")),
            KellyCandidate("bad", win_prob=0.2, payout_multiplier=Decimal("2")),
            KellyCandidate(
                "floor", win_prob=0.6, payout_multiplier=Decimal("3"), model_shrinkage=0.0
            ),
            KellyCandidate("b", win_prob=0.65, payout_multiplier=Decimal("3"), model_shrinkage=0.5),
        ]
    )
    assert bet_ids == ["a", "b"]
    assert sized_p == pytest.approx([0.6, 0.575])
    assert sized_b == pytest.approx([2.0, 2.0])


def test_portfolio_drops_floor_shrinkage_candidate():
    # A candidate whose shrinkage is at/below SHRINKAGE_FLOOR is dropped during
    # sizing even when its raw odds are +EV.
    cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
    out = joint_kelly_portfolio(
        bankroll=Decimal("1000"),
        candidates=[
            KellyCandidate("good", win_prob=0.6, payout_multiplier=Decimal("3")),
            KellyCandidate(
                "floored", win_prob=0.6, payout_multiplier=Decimal("3"), model_shrinkage=0.0
            ),
        ],
    )
    assert "floored" not in out
    assert "good" in out


def test_portfolio_all_filtered_returns_empty():
    # Every candidate is -EV → no surviving bet ids → empty mapping.
    cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
    out = joint_kelly_portfolio(
        bankroll=Decimal("1000"),
        candidates=[
            KellyCandidate("bad1", win_prob=0.2, payout_multiplier=Decimal("2")),
            KellyCandidate("bad2", win_prob=0.1, payout_multiplier=Decimal("2")),
        ],
    )
    assert out == {}


# --------------------------------------------------------------------------- #
# Resolution chain integration: explicit > CLV > training > fallback


def test_resolution_chain_via_fractional_kelly_stake():
    """The single-bet entrypoint accepts already-resolved shrinkage; this
    test exercises ``resolve_shrinkage`` then plumbs through to the stake
    function to confirm sources land where the docs say.
    """
    # Explicit beats both.
    s = resolve_shrinkage(explicit=0.0, training_bss=0.9, live_bss=0.9, live_n=200)
    assert s == 0.0
    stake = fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.7,
        payout_multiplier=Decimal("3"),
        model_shrinkage=s,
    )
    assert stake == Decimal("0")  # zero-shrinkage forces zero stake.

    # Live wins past the ramp full.
    s = resolve_shrinkage(training_bss=1.0, live_bss=0.0, live_n=LIVE_BLEND_FULL + 50)
    assert s == 0.0

    # Training only path.
    s = resolve_shrinkage(training_bss=0.8, live_bss=None, live_n=0)
    assert s == pytest.approx(0.8)

    # Final fallback.
    s = resolve_shrinkage(training_bss=None, live_bss=None, live_n=0)
    assert s == 1.0
