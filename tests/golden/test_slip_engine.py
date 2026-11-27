"""Behavior pins for the live slip scorer (``dashboard/slip_engine.py``).

These prove ``score_slip`` reuses the prediction layer's copula scorer over a
**block-diagonal** correlation matrix, prices Underdog vs Sleeper per the lane
contract, folds the platform's same-game pair modifiers into the boost, prices the legs'
fitted chances as ``ev`` beside the model's claim as ``model_ev``, and sizes a ``Decimal``
fractional-Kelly stake on ``ev`` — the one sanctioned live calc. Exact equality is
asserted only at two legs, where ``multivariate_normal.cdf`` is the closed-form
bivariate normal (≥3 legs is randomized QMC, ~1e-4 jitter, so those assertions stay
structural).

``astrolabe_payload`` is a thin read on top of the same ``score_slip`` path
(spec §4.4c); ``banned_partners`` reads the same
pair-modifier slice for the star map; ``payout_check`` is the "pays at least" rule the
rail and the story preloader print. All pinned below.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sportstradamus.dashboard.slip_engine import (
    _block_diagonal_sig,
    astrolabe_payload,
    banned_partners,
    modifier_map,
    payout_check,
    play_type_for,
    score_slip,
)
from sportstradamus.helpers.io import PAIR_MODIFIER_COLS
from sportstradamus.leg_schema import leg_label
from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.payouts import (
    PAYOUT_CLIP_HI,
    PAYOUT_CLIP_LO,
    SLEEPER_FULL_REFUND_MAX_SIZE,
    expected_payout_with_pushes,
    payout_curve_for,
)
from sportstradamus.prediction.stories.context import ctxs_from_frame


def _leg(player, market, bet, line, p, boost, game, push=0.0):
    """A leg whose fitted chance is the model's claim; a test that needs the two apart
    overrides ``fit_prob``."""
    return {
        "player": player,
        "market": market,
        "bet": bet,
        "line": line,
        "win_prob": p,
        "fit_prob": p,
        "push_prob": push,
        "boost": boost,
        "game": game,
    }


def _corr(rows):
    return ctxs_from_frame(None, pd.DataFrame(rows, columns=["Game", "leg_a", "leg_b", "rho"]))


def _mods(rows):
    return modifier_map(pd.DataFrame(rows, columns=PAIR_MODIFIER_COLS))


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


def test_two_leg_slip_prices_ev_on_the_fit_and_model_ev_on_the_claim():
    """The reuse pin: both numbers equal a hand-built copula call, ``ev`` (and the joint)
    on the legs' ``fit_prob`` and ``model_ev`` on their ``win_prob``. The fit and the claim
    rank the two legs in opposite orders, so reading one field for the other fails."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y") | {"fit_prob": 0.45},
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y") | {"fit_prob": 0.62},
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, _mods([]), platform="Underdog", bankroll=Decimal("1000"))

    sig = psd_or_none(np.array([[1.0, 0.4], [0.4, 1.0]]))
    search, full = payout_curve_for("Underdog", "pooled")
    base = float(search[0])
    payout = float(np.clip(1.0 * base, 1.0, 100.0))

    def value(p):
        return float(parlay_payout_prob(np.array(p), np.zeros(2), sig, 2, 1.0, payout, full, base))

    assert score.ev == pytest.approx(value([0.45, 0.62]), rel=1e-9)
    assert score.model_ev == pytest.approx(value([0.6, 0.55]), rel=1e-9)
    assert score.indep_p == pytest.approx(0.45 * 0.62)
    # A two-leg Power slip without pushes pays only when both hit, so its value is the
    # fitted joint times the payout.
    assert score.joint_p == pytest.approx(score.ev / payout, rel=1e-9)
    assert score.joint_p > score.indep_p  # positive within-game rho lifts the joint
    assert score.play_type == "Power"
    assert isinstance(score.stake, Decimal)
    assert not score.payout_approximate


def test_a_leg_the_fit_did_not_price_leaves_the_slip_unpriced():
    """No ``ev``, no fitted joint and no stake; the claim is priced as on its priced twin."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y") | {"fit_prob": None},
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, _mods([]), platform="Underdog", bankroll=Decimal("1000"))
    twin = score_slip(
        [legs[0], legs[1] | {"fit_prob": 0.55}],
        corr,
        _mods([]),
        platform="Underdog",
        bankroll=Decimal("1000"),
    )

    assert (score.ev, score.joint_p, score.indep_p) == (None, None, None)
    assert score.model_ev == pytest.approx(twin.model_ev, rel=1e-9)
    assert score.payout == twin.payout
    assert twin.stake > Decimal("0")
    assert score.stake == Decimal("0")


def test_cross_game_pair_scores_as_independent():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "Z/W"),
    ]
    score = score_slip(legs, _corr([]), _mods([]), platform="Underdog", bankroll=Decimal("1000"))
    assert score.joint_p == pytest.approx(score.indep_p, rel=1e-9)


def test_sleeper_payout_is_product_of_boosts():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.8, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 2.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, _mods([]), platform="Sleeper", bankroll=Decimal("1000"))
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
    score = score_slip(legs, _corr([]), _mods([]), platform="Sleeper", bankroll=Decimal("1000"))
    search, _ = payout_curve_for("Sleeper", "pooled")
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
    score = score_slip(legs, _corr([]), _mods([]), platform="Sleeper", bankroll=Decimal("1000"))
    search, _ = payout_curve_for("Sleeper", "pooled")
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
    two_score = score_slip(two, _corr([]), _mods([]), platform="Underdog", bankroll=Decimal("1000"))
    four_score = score_slip(
        four, _corr([]), _mods([]), platform="Underdog", bankroll=Decimal("1000")
    )
    assert two_score.play_type == "Power"
    assert four_score.play_type == "Flex"


def test_play_type_for_by_platform_and_size():
    assert [play_type_for("Underdog", n) for n in (2, 3, 4)] == ["Power", "Power", "Flex"]
    assert [play_type_for("Sleeper", n) for n in (2, 3, 4)] == ["Max", "Max", "Flex"]


def test_pays_at_least_is_one_over_the_fitted_joint_on_an_all_or_nothing_slip():
    cross_game = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "Z/W"),
    ]
    assert payout_check(cross_game, "Power", 0.25, 3.5) == [
        "Worth a bet if the app pays at least 4.00x (we assume 3.50x)."
    ]
    assert payout_check(cross_game, "Max", 0.4, 3.0) == [
        "Worth a bet if the app pays at least 2.50x (we assume 3.00x)."
    ]


def test_a_flex_or_unpriced_slip_has_no_pays_at_least():
    """A Flex entry has no single payout to check; an unpriced one has no fitted joint."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
    ]
    assert payout_check(legs, "Flex", 0.25, 3.5) == []
    assert payout_check(legs, "Power", None, 3.5) == []


def test_legs_sharing_a_game_add_that_the_value_is_unproven():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "Z/W"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
    ]
    assert payout_check(legs, "Power", 0.2, 6.0) == [
        "Worth a bet if the app pays at least 5.00x (we assume 6.00x).",
        "Legs here share a game, so this value is unproven until you check the app's payout "
        "against that number.",
    ]


def test_stake_follows_the_fitted_value_not_the_claim():
    """Kelly sizes on ``ev``: nothing under 1 however high the claim, a stake above 1
    however low the claim."""
    claimed = [
        _leg("A", "PTS", "Over", 20.5, 0.9, 1.0, "X/Y") | {"fit_prob": 0.3},
        _leg("B", "REB", "Over", 8.5, 0.9, 1.0, "X/Y") | {"fit_prob": 0.3},
    ]
    fitted = [
        _leg("A", "PTS", "Over", 20.5, 0.2, 1.0, "X/Y") | {"fit_prob": 0.8},
        _leg("B", "REB", "Over", 8.5, 0.2, 1.0, "X/Y") | {"fit_prob": 0.8},
    ]
    bankroll = Decimal("1000")
    claimed_score = score_slip(
        claimed, _corr([]), _mods([]), platform="Underdog", bankroll=bankroll
    )
    fitted_score = score_slip(fitted, _corr([]), _mods([]), platform="Underdog", bankroll=bankroll)

    assert claimed_score.model_ev > 1 > claimed_score.ev
    assert claimed_score.stake == Decimal("0")
    assert fitted_score.ev > 1 > fitted_score.model_ev
    assert fitted_score.stake > Decimal("0")


def test_push_leg_routes_finite_ev():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y", push=0.05),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.3)])
    score = score_slip(legs, corr, _mods([]), platform="Underdog", bankroll=Decimal("1000"))
    assert np.isfinite(score.ev)
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
    score = score_slip(legs, _corr([]), _mods([]), platform="Sleeper", bankroll=Decimal("1000"))

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
    assert score.ev == pytest.approx(refunded, abs=1e-4)


def test_refused_pair_prices_at_zero_before_the_payout_clip():
    """A 0.0 modifier is the app refusing the pair: the slip is worth $0, not a clipped 1x."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    mods = _mods([("Underdog", "NBA", "X/Y", "A|PTS|Over", "B|REB|Over", 0.0)])
    score = score_slip(legs, corr, mods, platform="Underdog", bankroll=Decimal("1000"))
    assert score.banned
    assert score.pair_mods == ((0, 1, 0.0),)
    assert score.payout == 0.0
    assert score.ev == score.model_ev == 0.0
    assert score.stake == Decimal("0")


def test_repriced_pair_matches_direct_parlay_call_with_modifier_in_boost():
    """Parity with the story menu's pricing: boost = leg boosts x pair modifiers.

    The slip lists B before A, the reverse of the row's ``leg_a < leg_b`` order, so the
    match can't lean on leg order.
    """
    legs = [
        _leg("B", "REB", "Over", 8.5, 0.55, 0.9, "X/Y"),
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.1, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    mods = _mods([("Underdog", "NBA", "X/Y", "A|PTS|Over", "B|REB|Over", 0.85)])
    score = score_slip(legs, corr, mods, platform="Underdog", bankroll=Decimal("1000"))

    search, full = payout_curve_for("Underdog", "pooled")
    base = float(search[0])
    boost = 0.9 * 1.1 * 0.85
    payout = float(np.clip(boost * base, PAYOUT_CLIP_LO, PAYOUT_CLIP_HI))
    sig = psd_or_none(np.array([[1.0, 0.4], [0.4, 1.0]]))
    p = np.array([0.55, 0.6])
    expected = float(parlay_payout_prob(p, np.zeros(2), sig, 2, boost, payout, full, base))

    assert not score.banned
    assert score.pair_mods == ((0, 1, 0.85),)
    assert score.payout == pytest.approx(payout)
    assert score.ev == pytest.approx(expected, rel=1e-9)


def test_flex_loss_tier_prices_the_largest_multipliers_times_the_pair_modifier():
    """One sure loss on a 4-pick Flex pays the 1-loss tier on the three largest pick
    multipliers, whichever pick lost, times the same-game pair modifier."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 1.0, 0.9, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 1.0, 1.1, "X/Y"),
        _leg("C", "AST", "Over", 5.5, 0.0, 1.2, "X/Y"),  # the loser holds the largest multiplier
        _leg("D", "STL", "Over", 1.5, 1.0, 0.8, "X/Y"),
    ]
    mods = _mods([("Underdog", "NBA", "X/Y", "A|PTS|Over", "B|REB|Over", 0.85)])
    score = score_slip(legs, _corr([]), mods, platform="Underdog", bankroll=Decimal("1000"))

    one_loss_tier = payout_curve_for("Underdog", "pooled")[1][4][1]
    # score_slip clips each probability a hair inside (0, 1), so a stray sample can
    # land in another tier: the tolerance covers that, not a pricing difference.
    assert score.ev == pytest.approx(one_loss_tier * 1.2 * 1.1 * 0.9 * 0.85, rel=1e-3)


def test_mods_off_the_slips_game_or_platform_are_ignored():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    mods = _mods(
        [
            ("Underdog", "NBA", "Z/W", "A|PTS|Over", "B|REB|Over", 0.0),
            ("Sleeper", "NBA", "X/Y", "A|PTS|Over", "B|REB|Over", 0.0),
        ]
    )
    score = score_slip(legs, corr, mods, platform="Underdog", bankroll=Decimal("1000"))
    clean = score_slip(legs, corr, _mods([]), platform="Underdog", bankroll=Decimal("1000"))
    assert score.pair_mods == ()
    assert score.payout == clean.payout
    assert score.ev == pytest.approx(clean.ev, rel=1e-9)


def test_banned_partners_marks_a_satellites_partner_in_its_own_game():
    """Only the platform's refusals in the leg's own game count — not a reprice, not the
    same key in another game, not the other platform."""
    focus = _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y")
    satellite = _leg("C", "AST", "Over", 5.5, 0.5, 1.0, "Z/W")
    mods = _mods(
        [
            ("Underdog", "NBA", "Z/W", "C|AST|Over", "D|STL|Over", 0.0),
            ("Underdog", "NBA", "Z/W", "C|AST|Over", "E|BLK|Over", 0.85),
            ("Underdog", "NBA", "X/Y", "C|AST|Over", "F|REB|Over", 0.0),
            ("Sleeper", "NBA", "Z/W", "C|AST|Over", "G|PTS|Over", 0.0),
        ]
    )
    assert banned_partners([focus, satellite], mods, platform="Underdog") == {
        "D|STL|Over": f"Underdog won't pair this with {leg_label(satellite)}"
    }


def test_banned_partners_marks_both_legs_of_a_refused_slip_pair():
    a = _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y")
    b = _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y")
    mods = _mods(
        [
            ("Underdog", "NBA", "X/Y", "A|PTS|Over", "B|REB|Over", 0.0),
            ("Underdog", "NBA", "X/Y", "A|PTS|Over", "D|STL|Over", 0.0),
            ("Underdog", "NBA", "X/Y", "B|REB|Over", "D|STL|Over", 0.0),
        ]
    )
    assert banned_partners([a, b], mods, platform="Underdog") == {
        "A|PTS|Over": f"Underdog won't pair this with {leg_label(b)}",
        "B|REB|Over": f"Underdog won't pair this with {leg_label(a)}",
        "D|STL|Over": f"Underdog won't pair this with {leg_label(a)}, {leg_label(b)}",
    }


def test_astrolabe_payload_draws_the_fitted_numbers():
    """The win dials, the EV dial and the Kelly dial read the fit, not the claim."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.7, 1.0, "X/Y") | {"fit_prob": 0.6},
        _leg("B", "REB", "Over", 8.5, 0.65, 1.0, "X/Y") | {"fit_prob": 0.55},
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, _mods([]), platform="Underdog", bankroll=Decimal("1000"))

    payload = astrolabe_payload(score)

    assert payload == {
        "legs": 2,
        "play_type": "Power",
        "payout": pytest.approx(3.5),
        "payout_approximate": False,
        "win_corr": pytest.approx(score.joint_p),
        "win_indep": pytest.approx(0.6 * 0.55),
        "ev": pytest.approx(score.ev - 1),
        "kelly": pytest.approx((score.ev - 1) / (score.payout - 1)),
        "crowns": {"win": 0.30, "ev": 0.12, "kelly": 0.03},
    }
    assert score.ev != pytest.approx(score.model_ev)


def test_astrolabe_payload_clamps_negative_kelly():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.2, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.2, 1.0, "X/Y"),
    ]
    score = score_slip(legs, _corr([]), _mods([]), platform="Underdog", bankroll=Decimal("1000"))
    raw_kelly = (score.ev - 1) / (score.payout - 1)
    assert raw_kelly < 0.0

    payload = astrolabe_payload(score)
    assert payload["kelly"] == 0.0


def test_astrolabe_payload_rests_on_an_unpriced_slip():
    """No fitted dials: the bare rest payload the builder sends below two legs."""
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.0, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 1.0, "X/Y") | {"fit_prob": None},
    ]
    score = score_slip(legs, _corr([]), _mods([]), platform="Underdog", bankroll=Decimal("1000"))
    assert astrolabe_payload(score) == {"legs": 2}


def test_astrolabe_payload_sleeper_play_type_and_payout_passes_through():
    legs = [
        _leg("A", "PTS", "Over", 20.5, 0.6, 1.8, "X/Y"),
        _leg("B", "REB", "Over", 8.5, 0.55, 2.0, "X/Y"),
    ]
    corr = _corr([("X/Y", "A|PTS|Over", "B|REB|Over", 0.4)])
    score = score_slip(legs, corr, _mods([]), platform="Sleeper", bankroll=Decimal("1000"))
    assert not score.payout_approximate

    payload = astrolabe_payload(score)
    assert payload["payout"] == pytest.approx(1.8 * 2.0)
    assert payload["payout_approximate"] is False
    assert payload["play_type"] == "Max"
