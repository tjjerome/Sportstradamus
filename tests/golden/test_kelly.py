"""Tests for Phase 3 Step 5: fractional-Kelly stake sizing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from sportstradamus.strategies import kelly
from sportstradamus.strategies.kelly import (
    DEFAULT_KELLY_FRACTION,
    MAX_FRACTION_OF_BANKROLL,
    NO_EVIDENCE_SHRINKAGE,
    KellyCandidate,
    fractional_kelly_stake,
    joint_kelly_portfolio,
    kelly_edge,
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
    # effective_p = p_book + (win_prob - p_book) * shrinkage rule, p_book = 1/payout:
    # "a" is full trust so p is its own 0.6; "b" is half trust toward the book's
    # 1/3, giving 1/3 + (0.65 - 1/3) * 0.5 = 0.491667.
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
    assert sized_p == pytest.approx([0.6, 0.4916666666666667])
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


def test_no_evidence_prior_stakes_a_long_entry_nothing():
    # The manufactured-edge scenario NO_EVIDENCE_SHRINKAGE exists for: a 5-leg
    # entry's tiny joint probability against a 20x payout. The no-evidence prior
    # must stake zero through the SHRINKAGE_FLOOR check...
    stake = fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.03,
        payout_multiplier=Decimal("20"),
        model_shrinkage=NO_EVIDENCE_SHRINKAGE,
    )
    assert stake == Decimal("0")

    # ...and no shrinkage value manufactures one, because the anchor is the book's
    # implied 1/20 = 0.05 and the model's 0.03 is below it. Under the old coin-flip
    # anchor every one of these staked, which is the bug the reject was hiding.
    for s_any in (0.01, 0.05, 0.25, 1.0):
        assert fractional_kelly_stake(
            bankroll=Decimal("1000"),
            win_prob=0.03,
            payout_multiplier=Decimal("20"),
            model_shrinkage=s_any,
        ) == Decimal("0")

    # A model that genuinely beats the book's line still stakes, scaled by trust,
    # and nothing at zero trust.
    assert fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.10,
        payout_multiplier=Decimal("20"),
        model_shrinkage=1.0,
    ) > Decimal("0")
    assert fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.10,
        payout_multiplier=Decimal("20"),
        model_shrinkage=0.0,
    ) == Decimal("0")


# --------------------------------------------------------------------------- #
# kelly_edge (sleeper-parity §5 extraction) + parlay.py's units formula


def test_kelly_edge_matches_fractional_kelly_stake_core():
    # Same (win_prob, payout_multiplier, model_shrinkage) through both: kelly_edge's
    # raw fraction, scaled by fraction and bankroll (cap not binding here), must
    # reproduce fractional_kelly_stake's stake -- proves the §5 extraction didn't
    # change fractional_kelly_stake's math. b=1, p=0.505 -> raw_kelly=0.01.
    raw = kelly_edge(win_prob=0.505, payout_multiplier=2.0, model_shrinkage=1.0)
    assert raw == pytest.approx(0.01)

    stake = fractional_kelly_stake(
        bankroll=Decimal("1000"),
        win_prob=0.505,
        payout_multiplier=Decimal("2"),
        fraction=DEFAULT_KELLY_FRACTION,
    )
    assert stake == Decimal("2.50")


def test_kelly_units_uses_shrinkage_and_max_fraction_cap():
    # The pure-math half of the units formula the entry search once scored Rec Bet by
    # (src/deprecated/parlay_search_orphans.py): shrinkage genuinely discounts the edge
    # vs full trust, and units scale by DEFAULT_KELLY_FRACTION / MAX_FRACTION_OF_BANKROLL.
    # Half trust sits halfway between the book's implied 1/3 and the model's 0.7:
    # p = 1/3 + (0.7 - 1/3) * 0.5 = 0.516667, raw = (2p - (1-p)) / 2 = 0.275.
    raw_full_trust = kelly_edge(win_prob=0.7, payout_multiplier=3.0, model_shrinkage=1.0)
    raw_half_trust = kelly_edge(win_prob=0.7, payout_multiplier=3.0, model_shrinkage=0.5)
    assert raw_half_trust < raw_full_trust
    assert raw_full_trust == pytest.approx(0.55)
    assert raw_half_trust == pytest.approx(0.275)

    units = raw_half_trust * DEFAULT_KELLY_FRACTION / MAX_FRACTION_OF_BANKROLL
    assert units == pytest.approx(13.75)


def test_zero_shrinkage_is_exactly_zero_edge_at_every_payout():
    # The property the book anchor exists for. Under the old coin-flip anchor
    # zero trust scored +0.25 at 3x and +0.46 at 15x, so a no-evidence leg looked
    # recommendable at any payout above 2x and had to be special-cased away.
    for payout in (1.5, 2.0, 3.0, 6.0, 15.0, 100.0):
        assert kelly_edge(
            win_prob=0.9, payout_multiplier=payout, model_shrinkage=0.0
        ) == pytest.approx(0.0, abs=1e-12)

    # Trust scales the edge away from the book's line rather than toward 0.5.
    assert kelly_edge(win_prob=0.9, payout_multiplier=3.0, model_shrinkage=1.0) > 0
    assert kelly_edge(win_prob=0.1, payout_multiplier=3.0, model_shrinkage=1.0) < 0
