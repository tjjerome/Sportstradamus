"""Behavior pins for the live slip scorer (``dashboard/slip_engine.py``).

These prove ``score_slip`` reuses the prediction layer's copula scorer over a
**block-diagonal** correlation matrix, prices Underdog vs Sleeper per the lane
contract, and sizes a ``Decimal`` fractional-Kelly stake — the one sanctioned
live calc. Exact equality is asserted only at two legs, where
``multivariate_normal.cdf`` is the closed-form bivariate normal (≥3 legs is
randomized QMC, ~1e-4 jitter, so those assertions stay structural).
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sportstradamus.dashboard.slip_engine import _block_diagonal_sig, score_slip
from sportstradamus.prediction.parlay import (
    _parlay_payout_prob,
    _payout_curve_for,
    _psd_or_none,
)


def _leg(player, market, bet, line, p, boost, game, push=0.0):
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": line,
        "Win Prob": p,
        "Push Prob": push,
        "Boost": boost,
        "Game": game,
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
    sig = _psd_or_none(np.array([[1.0, 0.4], [0.4, 1.0]]), legacy=False)
    search, full = _payout_curve_for("Underdog", "pooled", legacy=False)
    base = float(search[0])
    payout = float(np.clip(1.0 * base, 1.0, 100.0))
    expected = float(_parlay_payout_prob(p, push, sig, 2, 1.0, payout, full, base, False))

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
    assert score.payout == pytest.approx(1.8 * 2.0)
    assert score.payout_approximate
    assert score.play_type == "Sleeper"


def test_underdog_play_type_by_size():
    two = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.6, 1.0, "X/Y"),
    ]
    four = two + [
        _leg("C", "AST", "Over", 5.5, 0.6, 1.0, "X/Y"),
        _leg("D", "STL", "Over", 1.5, 0.6, 1.0, "X/Y"),
    ]
    assert score_slip(two, _corr([]), platform="Underdog", bankroll=Decimal("1000")).play_type == "Power"
    assert score_slip(four, _corr([]), platform="Underdog", bankroll=Decimal("1000")).play_type == "Flex"


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
