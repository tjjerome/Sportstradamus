"""Behavior pins for the live slip scorer (``dashboard/slip_engine.py``).

These prove ``score_slip`` reuses the prediction layer's copula scorer over a
**block-diagonal** correlation matrix, prices Underdog vs Sleeper per the lane
contract, and sizes a ``Decimal`` fractional-Kelly stake — the one sanctioned
live calc. Exact equality is asserted only at two legs, where
``multivariate_normal.cdf`` is the closed-form bivariate normal (≥3 legs is
randomized QMC, ~1e-4 jitter, so those assertions stay structural).

``ev_lift`` and ``astrolabe_payload`` are thin reads on top of the same
``score_slip`` path (spec §4.1, §4.4c), pinned below.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sportstradamus.dashboard.slip_engine import (
    _block_diagonal_sig,
    astrolabe_payload,
    ev_lift,
    score_slip,
)
from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.payouts import (
    PAYOUT_CLIP_LO,
    SLEEPER_FULL_REFUND_MAX_SIZE,
    expected_payout_with_pushes,
    payout_curve_for,
)


def _leg(player, market, bet, line, p, boost, game, push=0.0):
    return {
        "player": player,
        "market": market,
        "bet": bet,
        "line": line,
        "win_prob": p,
        "push_prob": push,
        "boost": boost,
        "game": game,
    }


def _corr(rows):
    return pd.DataFrame(rows, columns=["Game", "leg_a", "leg_b", "rho"])


def test_block_diagonal_zeroes_cross_game():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.6, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.6, 1.0, "Z/W"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    sig = _block_diagonal_sig(legs, corr)
    assert sig[0, 1] == sig[1, 0] == 0.4  # within-game rho from the slice
    assert sig[0, 2] == sig[1, 2] == 0.0  # cross-game pairs independent
    assert np.allclose(np.diag(sig), 1.0)


def test_two_leg_same_game_matches_direct_parlay_call():
    """The reuse pin: ``score_slip`` joint/EV equals a hand-built copula call."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, platform="Underdog", bankroll=Decimal("1000"))

    p = np.array([0.6, 0.55])
    push = np.zeros(2)
    sig = psd_or_none(np.array([[1.0, 0.4], [0.4, 1.0]]), legacy=False)
    search, full = payout_curve_for("Underdog", "pooled", legacy=False)
    base = float(search[0])
    payout = float(np.clip(1.0 * base, 1.0, 100.0))
    expected = float(parlay_payout_prob(p, push, sig, 2, 1.0, payout, full, base, False))

    assert score.model_ev == pytest.approx(expected, rel=1e-9)
    assert score.joint_p > score.indep_p  # positive within-game rho lifts the joint
    assert score.play_type == "Power"
    assert isinstance(score.stake, Decimal)
    assert not score.payout_approximate


def test_cross_game_pair_scores_as_independent():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "Z/W"),
    ]
    score = score_slip(legs, _corr([]), platform="Underdog", bankroll=Decimal("1000"))
    assert score.joint_p == pytest.approx(score.indep_p, rel=1e-9)


def test_sleeper_payout_is_product_of_boosts():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.8, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 2.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, platform="Sleeper", bankroll=Decimal("1000"))
    assert score.payout == pytest.approx(1.8 * 2.0)  # size-2 Max carries no bonus
    assert not score.payout_approximate
    assert score.play_type == "Max"


def test_sleeper_max_size3_applies_real_bonus():
    """Stage-1's verified 3-leg bonus (sleeper_payouts.json) now flows through
    the live rail instead of the pre-Stage-1 flat-1.0 stub."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "X/Y"),
    ]
    score = score_slip(legs, _corr([]), platform="Sleeper", bankroll=Decimal("1000"))
    search, _ = payout_curve_for("Sleeper", "pooled", legacy=False)
    assert score.payout == pytest.approx(float(search[1]))  # size-3 index, boost=1.0
    assert score.play_type == "Max"
    assert not score.payout_approximate


def test_sleeper_flex_size5_wiring():
    """Pins play_type/payout_approximate wiring at Flex sizes. Flex's own
    per-candidate pricing (sleeper_flex_payout_curve) is exercised in
    test_parlay_search.py, not re-derived here — this cross-checks the same
    coarse ``base`` payout_curve_for returns, matching the Max test above."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "X/Y"),
        _leg("D", "STL", "Over", 1.5, 0.5, 1.0, "X/Y"),
        _leg("E", "BLK", "Over", 0.5, 0.5, 1.0, "X/Y"),
    ]
    score = score_slip(legs, _corr([]), platform="Sleeper", bankroll=Decimal("1000"))
    search, _ = payout_curve_for("Sleeper", "pooled", legacy=False)
    # base (search[3], size-5 index) is 0.751 < PAYOUT_CLIP_LO, so boost(1.0) * base
    # clips up to the floor -- score_slip's existing clip, not this test's math.
    assert score.payout == pytest.approx(max(float(search[3]), PAYOUT_CLIP_LO))
    assert score.play_type == "Flex"
    assert not score.payout_approximate


def test_underdog_play_type_by_size():
    two = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.6, 1.0, "X/Y"),
    ]
    four = [
        *two,
        _leg("C", "AST", "Over", 5.5, 0.6, 1.0, "X/Y"),
        _leg("D", "STL", "Over", 1.5, 0.6, 1.0, "X/Y"),
    ]
    assert (
        score_slip(two, _corr([]), platform="Underdog", bankroll=Decimal("1000")).play_type
        == "Power"
    )
    assert (
        score_slip(four, _corr([]), platform="Underdog", bankroll=Decimal("1000")).play_type
        == "Flex"
    )


def test_negative_edge_yields_zero_stake():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.2, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.2, 1.0, "X/Y"),
    ]
    score = score_slip(legs, _corr([]), platform="Underdog", bankroll=Decimal("1000"))
    assert score.stake == Decimal("0")


def test_push_leg_routes_finite_ev():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y", push=0.05),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.3)])
    score = score_slip(legs, corr, platform="Underdog", bankroll=Decimal("1000"))
    assert np.isfinite(score.model_ev)
    assert score.bet_size == 3


def test_sleeper_two_leg_slip_push_refunds_in_full_via_score_slip():
    """Regression pin: score_slip must thread full_refund_below_size for
    Sleeper (matching the beam-search path's rule at correlation.py:664), or a
    2-leg slip with a push-eligible leg silently busts instead of refunding
    (docs/handoffs/sleeper-parity.md §3 item 3)."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.0, 1.0, "X/Y", push=1.0),  # guaranteed push
        _leg("B", "REB", "Over", 8.5, 0.0, 1.0, "X/Y"),  # guaranteed loss
    ]
    score = score_slip(legs, _corr([]), platform="Sleeper", bankroll=Decimal("1000"))

    # Reference values for the two branches (mirrors test_parlay_search.py's
    # test_sleeper_two_leg_push_refunds_in_full_even_with_a_loss): Sleeper's
    # rule refunds in full; the generic Underdog-style rule busts instead.
    p_win, p_push, sigma, curve = np.zeros(2), np.array([1.0, 0.0]), np.eye(2), {2: [1.0, 0.0]}
    refunded = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        2,
        boost=1.0,
        payout_curve=curve,
        rng=np.random.default_rng(13),
        full_refund_below_size=SLEEPER_FULL_REFUND_MAX_SIZE,
    )
    busted = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        2,
        boost=1.0,
        payout_curve=curve,
        rng=np.random.default_rng(13),
        full_refund_below_size=None,
    )
    assert refunded == pytest.approx(1.0, abs=1e-6)
    assert busted == pytest.approx(0.0, abs=1e-6)
    assert score.model_ev == pytest.approx(refunded, abs=1e-4)


def test_ev_lift_matches_hand_built_pair_minus_solo():
    """The reuse pin: ``ev_lift`` equals a hand-built pair/solo ``score_slip`` diff."""
    focus = _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y")
    candidate = _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y")
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])

    lift = ev_lift(focus, candidate, corr, platform="Underdog", bankroll=0.0)

    pair = score_slip([focus, candidate], corr, platform="Underdog", bankroll=Decimal("0"))
    solo = score_slip([focus], corr, platform="Underdog", bankroll=Decimal("0"))
    assert lift == pytest.approx(pair.model_ev - solo.model_ev, rel=1e-9)
    # score_slip's single-leg early return hardcodes model_ev=0.0 (it's a parlay
    # scorer), so the solo term is always zero and the lift equals the pair EV.
    assert solo.model_ev == 0.0
    assert lift == pytest.approx(pair.model_ev, rel=1e-9)


def test_ev_lift_positive_rho_exceeds_zero_rho():
    focus = _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y")
    candidate = _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y")

    positive_rho = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    zero_rho = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.0)])

    lift_positive = ev_lift(focus, candidate, positive_rho, platform="Underdog", bankroll=0.0)
    lift_zero = ev_lift(focus, candidate, zero_rho, platform="Underdog", bankroll=0.0)

    assert lift_positive > lift_zero
    # Zero-rho collapses to the independent-joint case: same lift as an empty
    # corr slice and as a cross-game candidate (both default to rho=0).
    lift_empty_corr = ev_lift(focus, candidate, _corr([]), platform="Underdog", bankroll=0.0)
    assert lift_zero == pytest.approx(lift_empty_corr, rel=1e-9)


def test_astrolabe_payload_shape_and_crowns():
    focus = _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y")
    candidate = _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y")
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip([focus, candidate], corr, platform="Underdog", bankroll=Decimal("1000"))

    payload = astrolabe_payload(score, nonce=3)

    assert payload == {
        "legs": 2,
        "play_type": "Power",
        "payout": pytest.approx(3.0),
        "payout_approximate": False,
        "win_corr": pytest.approx(score.joint_p),
        "win_indep": pytest.approx(score.indep_p),
        "ev": pytest.approx(score.model_ev - 1),
        "kelly": pytest.approx((score.model_ev - 1) / (score.payout - 1)),
        "crowns": {"win": 0.30, "ev": 0.12, "kelly": 0.03},
        "nonce": 3,
    }


def test_astrolabe_payload_clamps_negative_kelly():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.2, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.2, 1.0, "X/Y"),
    ]
    score = score_slip(legs, _corr([]), platform="Underdog", bankroll=Decimal("1000"))
    raw_kelly = (score.model_ev - 1) / (score.payout - 1)
    assert raw_kelly < 0.0

    payload = astrolabe_payload(score, nonce=1)
    assert payload["kelly"] == 0.0


def test_astrolabe_payload_sleeper_play_type_and_payout_passes_through():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.8, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 2.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, platform="Sleeper", bankroll=Decimal("1000"))
    assert not score.payout_approximate

    payload = astrolabe_payload(score, nonce=9)
    assert payload["payout"] == pytest.approx(1.8 * 2.0)
    assert payload["payout_approximate"] is False
    assert payload["play_type"] == "Max"
