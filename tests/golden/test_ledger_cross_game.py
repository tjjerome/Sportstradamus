"""Pin cross-game candidate generation for the simulated-bettor ledger --
game-span guard, per-pick multiplier pricing, RNG determinism, beam-width cap,
and the min_ev floor of ``_ledger_cross_game``
(docs/handoffs/sim-bettor-ledger.md §10).
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import stat_map
from sportstradamus.prediction.payouts import expected_payout_with_pushes, payout_curve_for
from sportstradamus.strategies import _ledger_cross_game as xg
from sportstradamus.strategies.underdog_pickem import PickemConfig

DATE = datetime.date(2026, 7, 12)


def _offer_row(
    player: str,
    game: str,
    *,
    market: str = "Rebounds",
    league: str = "NBA",
    win_prob: float = 0.60,
    market_prob: float = 0.56,
    line: float = 4.5,
    boost: float = 1.0,
    push_prob: float = 0.0,
    platform: str = "Underdog",
) -> dict:
    team, opp = game.split("/")
    return {
        "Player": player,
        "Team": team,
        "Opponent": opp,
        "Market": market,
        "League": league,
        "Game": game,
        "Platform": platform,
        "Date": DATE.isoformat(),
        "Line": line,
        "Bet": "Over",
        "Win Prob": win_prob,
        "Market Prob": market_prob,
        "Boost": boost,
        "Push Prob": push_prob,
    }


def _offers_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- game_span guard: cross-game only ------------------------------------------


def test_never_returns_same_game_combos() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA"),
            _offer_row("Player B", "BOS/MIA"),
            _offer_row("Player C", "LAL/DEN"),
            _offer_row("Player D", "LAL/DEN"),
        ]
    )
    config = PickemConfig(min_ev=-1.0)

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning")

    assert candidates
    assert all(c.game_span >= 2 for c in candidates)
    assert all(c.game_span != 1 for c in candidates)


def test_produces_at_least_one_two_leg_cross_game_candidate() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA"),
            _offer_row("Player B", "LAL/DEN"),
        ]
    )
    config = PickemConfig(min_ev=-1.0)

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning")

    sizes = {c.entry_size for c in candidates}
    assert 2 in sizes
    two_leg = next(c for c in candidates if c.entry_size == 2)
    assert two_leg.game_span == 2
    assert two_leg.players == frozenset({"Player A", "Player B"})


def test_canonical_legs_stat_field_is_canonicalized_not_raw_market() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", market="Rebounds"),
            _offer_row("Player B", "LAL/DEN", market="Rebounds"),
        ]
    )
    config = PickemConfig(min_ev=-1.0)

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning")

    canonical_market = stat_map["Underdog"]["Rebounds"]
    assert candidates
    for candidate in candidates:
        for leg in candidate.canonical_legs:
            assert leg["stat"] == canonical_market


# --- platform tagging -----------------------------------------------------------


def test_build_cross_game_candidates_tags_sleeper_platform() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", platform="Sleeper"),
            _offer_row("Player B", "LAL/DEN", platform="Sleeper"),
        ]
    )
    config = PickemConfig(min_ev=-1.0)

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning", platform="Sleeper")

    assert candidates
    assert all(c.platform == "Sleeper" for c in candidates)


def test_build_cross_game_candidates_defaults_to_underdog_platform() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA"),
            _offer_row("Player B", "LAL/DEN"),
        ]
    )
    config = PickemConfig(min_ev=-1.0)

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning")

    assert candidates
    assert all(c.platform == "Underdog" for c in candidates)


# --- hand-computed no-push 2-leg sanity check -----------------------------------


def test_price_combo_matches_hand_derived_no_push_expectation() -> None:
    p1, p2 = 0.60, 0.55
    payout_mult = 3.5  # 2-leg power payout, confirmed via payout_curve_for
    rng = np.random.default_rng(12345)

    ev_payout = expected_payout_with_pushes(
        p_win=np.array([p1, p2]),
        p_push=np.array([0.0, 0.0]),
        sigma=np.eye(2),
        bet_size=2,
        boost=1.0,
        payout_curve={2: [payout_mult, 0.0]},
        rng=rng,
    )

    expected = p1 * p2 * payout_mult
    assert ev_payout == pytest.approx(expected, abs=0.02)


def test_price_combo_wiring_matches_direct_call() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", win_prob=0.60, market_prob=0.56),
            _offer_row("Player B", "LAL/DEN", win_prob=0.55, market_prob=0.53),
        ]
    )
    config = PickemConfig(min_ev=-1.0)
    eligible = offers  # both legs clear filter_legs at these edges/thresholds
    legs = xg._score_legs(eligible, "Underdog")
    rng_direct = xg._pricing_rng(DATE, "morning")
    ev_payout = xg._price_combo(tuple(legs), rng_direct, "Underdog")

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning")
    two_leg = next(c for c in candidates if c.entry_size == 2)

    assert two_leg.ev == pytest.approx(ev_payout - 1.0, abs=0.05)


# --- Sleeper 2-pick full-refund-on-push divergence --------------------------------


def _scored_leg(
    idx: int, *, win_prob: float, push_prob: float = 0.0, boost: float = 1.0
) -> xg._ScoredLeg:
    return xg._ScoredLeg(
        idx=idx,
        player=f"Player {idx}",
        game=f"G{idx}A/G{idx}B",
        win_prob=win_prob,
        push_prob=push_prob,
        book_devig=0.5,
        line=4.5,
        boost=boost,
        display=f"Player {idx} Over 4.5 Rebounds",
        canonical_leg={},
    )


def _guaranteed_push_and_loss_legs() -> tuple[xg._ScoredLeg, xg._ScoredLeg]:
    return _scored_leg(0, win_prob=0.0, push_prob=1.0), _scored_leg(1, win_prob=0.0)


def test_price_combo_sleeper_two_leg_push_refunds_in_full() -> None:
    legs = _guaranteed_push_and_loss_legs()
    rng = np.random.default_rng(12345)

    ev_payout = xg._price_combo(legs, rng, "Sleeper")

    assert ev_payout == pytest.approx(1.0, abs=0.02)


def test_price_combo_underdog_two_leg_push_with_loss_still_busts() -> None:
    legs = _guaranteed_push_and_loss_legs()
    rng = np.random.default_rng(12345)

    ev_payout = xg._price_combo(legs, rng, "Underdog")

    assert ev_payout == pytest.approx(0.0, abs=0.02)


# --- each pick's own multiplier: a Flex loss tier pays on the largest that remain ---


def test_price_combo_flex_loss_tier_pays_on_the_largest_remaining_multipliers() -> None:
    """Four coin-flip picks at 0.8x / 1.0x / 1.0x / 1.5x on Underdog's 4-pick Flex
    (7.2 all hit, 1.8 with one miss). All four hit one time in 16 and pay
    7.2 * 1.2 = 8.64. One miss comes four times in 16 and pays on the three largest
    multipliers whichever pick missed: 1.8 * 1.5 = 2.7. One fused 1.2x on every tier
    would pay 2.16 there and price the entry at 1.08."""
    legs = tuple(
        _scored_leg(idx, win_prob=0.5, boost=boost)
        for idx, boost in enumerate((0.8, 1.0, 1.0, 1.5))
    )
    rng = np.random.default_rng(12345)

    ev_payout = xg._price_combo(legs, rng, "Underdog")

    assert ev_payout == pytest.approx(8.64 / 16 + 2.7 * 4 / 16, abs=0.05)


def test_candidate_is_sized_on_its_all_hit_payout_and_its_priced_ev(monkeypatch) -> None:
    """Four 0.60 picks on Underdog's 4-pick Flex, one of them at 1.25x. All four hit
    13% of the time, and 13% of the bare 7.2 tier is under a dollar back, so a stake
    sized on that is nothing. The entry pays 7.2 * 1.25 = 9 when all four hit and is
    worth holding for its one-miss tier; the stake has to see both."""
    monkeypatch.setattr(xg, "resolve_market_shrinkage", lambda league, market: (1.0, "fallback"))
    offers = _offers_df(
        [
            _offer_row(f"Player {i}", f"G{i}A/G{i}B", market_prob=0.58, boost=boost)
            for i, boost in enumerate((1.0, 1.0, 1.0, 1.25))
        ]
    )

    candidates = xg.build_cross_game_candidates(offers, PickemConfig(), DATE, "morning")

    four_leg = next(c for c in candidates if c.entry_size == 4)
    assert four_leg.payout_multiplier == pytest.approx(9.0)
    assert four_leg.joint_prob * four_leg.payout_multiplier == pytest.approx(four_leg.ev + 1)
    assert four_leg.joint_prob > 0.6**4
    assert four_leg.stake > 0


def test_canonical_legs_carry_each_picks_own_multiplier() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", boost=0.87),
            _offer_row("Player B", "LAL/DEN", boost=1.16),
        ]
    )

    candidates = xg.build_cross_game_candidates(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    (two_leg,) = candidates
    assert {leg["player"]: leg["boost"] for leg in two_leg.canonical_legs} == {
        "Player A": 0.87,
        "Player B": 1.16,
    }


def test_side_the_platform_does_not_post_never_enters_a_candidate() -> None:
    """An unposted side arrives with a zero multiplier. No entry can hold it, and a
    Flex loss tier would otherwise drop that zero as the smallest multiplier and
    pay on the rest."""
    offers = _offers_df(
        [_offer_row(f"Posted {i}", f"G{i}A/G{i}B") for i in range(4)]
        + [_offer_row("Unposted", "BOS/MIA", boost=0.0)]
    )

    candidates = xg.build_cross_game_candidates(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert {c.entry_size for c in candidates} == {2, 3, 4}
    assert all("Unposted" not in c.players for c in candidates)


# --- _pricing_rng / _entropy_from determinism -----------------------------------


def test_pricing_rng_reproducible_across_separate_calls() -> None:
    first = xg._pricing_rng(DATE, "morning")
    second = xg._pricing_rng(DATE, "morning")

    assert np.array_equal(first.random(5), second.random(5))


def test_pricing_rng_differs_across_run_slots() -> None:
    morning = xg._pricing_rng(DATE, "morning")
    afternoon = xg._pricing_rng(DATE, "afternoon")

    assert not np.array_equal(morning.random(5), afternoon.random(5))


def test_pricing_rng_differs_from_replicate_stream_salt() -> None:
    from sportstradamus.strategies._ledger_selection import _entropy_from

    pricing_entropy = _entropy_from(DATE, "morning", salt="universe_pricing")
    replicate_entropy = _entropy_from(DATE, "morning")

    assert pricing_entropy != replicate_entropy


# --- beam pruning respects _CROSS_GAME_BEAM_WIDTH -------------------------------


def test_beam_width_cap_holds_with_monkeypatched_narrow_width(monkeypatch) -> None:
    monkeypatch.setattr(xg, "_CROSS_GAME_BEAM_WIDTH", 3)
    rows = []
    for i in range(10):
        game = f"G{i}A/G{i}B"
        rows.append(_offer_row(f"Player {i}", game, win_prob=0.55 + i * 0.01))
    offers = _offers_df(rows)
    legs = xg._score_legs(offers, "Underdog")

    by_size = xg._enumerate_cross_game_combos(legs)

    for size, combos in by_size.items():
        assert len(combos) <= 3, f"size {size} returned {len(combos)} combos, expected <=3"


# --- candidates below min_ev are dropped ----------------------------------------


def test_candidates_below_min_ev_are_dropped() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", win_prob=0.58, market_prob=0.54),
            _offer_row("Player B", "LAL/DEN", win_prob=0.58, market_prob=0.54),
        ]
    )
    lenient_config = PickemConfig(min_ev=-1.0)
    strict_config = PickemConfig(min_ev=0.05)

    lenient = xg.build_cross_game_candidates(offers, lenient_config, DATE, "morning")
    strict = xg.build_cross_game_candidates(offers, strict_config, DATE, "morning")

    assert lenient
    assert strict == []


# --- Sleeper payout curve wired through end-to-end --------------------------------


def test_build_cross_game_candidates_sleeper_uses_sleeper_curve_for_payout_multiplier() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", platform="Sleeper"),
            _offer_row("Player B", "LAL/DEN", platform="Sleeper"),
        ]
    )
    config = PickemConfig(min_ev=-1.0)

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning", platform="Sleeper")
    two_leg = next(c for c in candidates if c.entry_size == 2)

    _, sleeper_curve = payout_curve_for("Sleeper", "pooled")
    assert two_leg.payout_multiplier == pytest.approx(sleeper_curve[2][0])
    _, underdog_curve = payout_curve_for("Underdog", "pooled")
    assert two_leg.payout_multiplier != pytest.approx(underdog_curve[2][0])
