"""Pin cross-game candidate generation for the simulated-bettor ledger --
game-span guard, per-pick multiplier pricing, RNG determinism, beam-width cap,
the min_ev floor, the ``even_picks`` pool, a leg's league-own stat key and its
price at the fitted chance of ``_ledger_cross_game``
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
from sportstradamus.strategies._ledger_selection import FLAT_STAKE
from sportstradamus.strategies.underdog_pickem import PickemConfig

DATE = datetime.date(2026, 7, 12)


def _offer_row(
    player: str,
    game: str,
    *,
    market: str = "Rebounds",
    league: str = "NBA",
    win_prob: float = 0.60,
    fit_prob: float | None = None,
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
        "Fit Prob": win_prob if fit_prob is None else fit_prob,
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
    legs = xg._served_legs(offers, "Underdog")
    rng_direct = xg._pricing_rng(DATE, "morning")
    ev_payout = xg._price_combo(tuple(legs), rng_direct, "Underdog")

    candidates = xg.build_cross_game_candidates(offers, config, DATE, "morning")
    two_leg = next(c for c in candidates if c.entry_size == 2)

    assert two_leg.ev == pytest.approx(ev_payout - 1.0, abs=0.05)


# --- Sleeper 2-pick full-refund-on-push divergence --------------------------------


def _scored_leg(
    idx: int, *, fit_prob: float, push_prob: float = 0.0, boost: float = 1.0
) -> xg._ScoredLeg:
    return xg._ScoredLeg(
        idx=idx,
        player=f"Player {idx}",
        game=f"G{idx}A/G{idx}B",
        fit_prob=fit_prob,
        push_prob=push_prob,
        book_devig=0.5,
        line=4.5,
        boost=boost,
        display=f"Player {idx} Over 4.5 Rebounds",
        canonical_leg={},
    )


def _guaranteed_push_and_loss_legs() -> tuple[xg._ScoredLeg, xg._ScoredLeg]:
    return _scored_leg(0, fit_prob=0.0, push_prob=1.0), _scored_leg(1, fit_prob=0.0)


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
    (7.2 all hit, 1.4 with one miss). All four hit one time in 16 and pay
    7.2 * 1.2 = 8.64. One miss comes four times in 16 and pays on the three largest
    multipliers whichever pick missed: 1.4 * 1.5 = 2.1. One fused 1.2x on every tier
    would pay 1.68 there and price the entry at 0.96."""
    legs = tuple(
        _scored_leg(idx, fit_prob=0.5, boost=boost)
        for idx, boost in enumerate((0.8, 1.0, 1.0, 1.5))
    )
    rng = np.random.default_rng(12345)

    ev_payout = xg._price_combo(legs, rng, "Underdog")

    assert ev_payout == pytest.approx(8.64 / 16 + 2.1 * 4 / 16, abs=0.05)


def test_candidate_is_recorded_on_its_all_hit_payout_and_its_priced_ev() -> None:
    """Four 0.60 picks on Underdog's 4-pick Flex, one of them at 1.25x. All four hit
    13% of the time, and 13% of the bare 7.2 tier is under a dollar back, so a Kelly
    stake sized on that is nothing. The entry pays 7.2 * 1.25 = 9 when all four hit and
    is worth holding for its one-miss tier; the record kelly_growth sizes on has to see
    both."""
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
    assert four_leg.stake == FLAT_STAKE


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
    legs = xg._served_legs(offers, "Underdog")

    by_size = xg._enumerate_cross_game_combos(legs)

    for size, combos in by_size.items():
        assert len(combos) <= 3, f"size {size} returned {len(combos)} combos, expected <=3"


# --- candidates below min_ev are dropped ----------------------------------------


def test_candidates_below_min_ev_are_dropped() -> None:
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", win_prob=0.58, fit_prob=0.54, market_prob=0.54),
            _offer_row("Player B", "LAL/DEN", win_prob=0.58, fit_prob=0.54, market_prob=0.54),
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


# --- even picks: the recommended ones, at their served read, a game per leg -------


def test_recommended_even_pick_the_shared_gate_refuses_is_in_the_even_pick_pool() -> None:
    """An even pick's book sits at a coin flip. A read strong enough to recommend the
    pick (0.60 x 1.83 - 1 = 0.098) is then further from the book than the shared leg
    gate allows, so the shared builder never sees a recommended even pick."""
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", market_prob=0.50),
            _offer_row("Player B", "LAL/DEN", market_prob=0.50),
        ]
    )
    config = PickemConfig()

    assert xg.build_cross_game_candidates(offers, config, DATE, "morning") == []
    (two_leg,) = xg.build_even_pick_candidates(offers, config, DATE, "morning")
    assert two_leg.entry_size == 2
    assert two_leg.players == frozenset({"Player A", "Player B"})


def test_only_recommended_even_picks_enter_the_even_pick_pool() -> None:
    """1.05 / 1.83 = 0.5738 is the weakest read recommended on an even pick. The
    discounted and the boosted pick clear the rule at their own payouts and are left
    out for the multiplier alone."""
    offers = _offers_df(
        [
            _offer_row("Even A", "BOS/MIA"),
            _offer_row("Even B", "LAL/DEN"),
            _offer_row("Weak Read", "NYK/CHI", win_prob=0.57),
            _offer_row("Discounted", "DAL/PHX", win_prob=0.70, boost=0.9),
            _offer_row("Boosted", "GSW/SAC", win_prob=0.70, boost=1.1),
        ]
    )

    candidates = xg.build_even_pick_candidates(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert {c.players for c in candidates} == {frozenset({"Even A", "Even B"})}


def test_even_pick_entry_never_holds_two_legs_of_one_game() -> None:
    """Underdog taxes some same-game pairs and a cross-game record carries pair_modifier
    1.0, so the table payout is exact only with a game per leg. Three games on the
    board then cap an entry at three legs."""
    offers = _offers_df(
        [_offer_row(f"Stack {i}", "BOS/MIA") for i in range(3)]
        + [_offer_row("Player D", "LAL/DEN"), _offer_row("Player E", "NYK/CHI")]
    )

    candidates = xg.build_even_pick_candidates(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    for candidate in candidates:
        games = [leg["game"] for leg in candidate.canonical_legs]
        assert len(set(games)) == len(games)
    assert {c.entry_size for c in candidates} == {2, 3}


def test_even_pick_candidate_is_staked_and_pays_its_sizes_table_tier() -> None:
    offers = _offers_df([_offer_row(f"Player {i}", f"G{i}A/G{i}B") for i in range(4)])
    _, underdog_curve = payout_curve_for("Underdog", "pooled")

    candidates = xg.build_even_pick_candidates(offers, PickemConfig(), DATE, "morning")

    assert {c.entry_size for c in candidates} == {2, 3, 4}
    for candidate in candidates:
        assert candidate.payout_multiplier == pytest.approx(underdog_curve[candidate.entry_size][0])
        assert candidate.stake == FLAT_STAKE


# --- even picks: the slate date's legs, dealt evenly ------------------------------


def test_even_pick_pool_takes_only_legs_of_the_slate_dates_games() -> None:
    """A pick for a later game is taken on its own day, at that day's read. Taken today
    as well, it is staked on two days and stays open past the next settlement."""
    today = [_offer_row("Player A", "BOS/MIA"), _offer_row("Player B", "LAL/DEN")]
    pick = _offer_row("Player C", "NYK/CHI")
    tomorrow = (DATE + datetime.timedelta(days=1)).isoformat()
    config = PickemConfig(min_ev=-1.0)

    later = xg.build_even_pick_candidates(
        _offers_df([*today, pick | {"Date": tomorrow}]), config, DATE, "morning"
    )
    game_day = xg.build_even_pick_candidates(_offers_df([*today, pick]), config, DATE, "morning")

    assert {c.players for c in later} == {frozenset({"Player A", "Player B"})}
    assert any("Player C" in c.players for c in game_day)


def _thirty_even_picks() -> pd.DataFrame:
    """Thirty recommended even picks in thirty games, reads rising from 0.58 to 0.70."""
    return _offers_df(
        [
            _offer_row(f"Player {i:02d}", f"G{i:02d}A/G{i:02d}B", win_prob=float(read))
            for i, read in enumerate(np.linspace(0.58, 0.70, 30))
        ]
    )


def test_even_pick_entries_are_dealt_evenly_over_the_legs(monkeypatch) -> None:
    """The shared beam keeps the entries with the highest joint read, and the strongest
    leg is in most of them. Dealt evenly, a leg is in about a fifth of thirty legs'
    six-pick entries, and no leg is left out of the pool."""
    # A thousand entries take ten seconds to price, and the price is not what is read here.
    monkeypatch.setattr(xg, "_price_combo", lambda legs, rng, platform: 2.0)

    candidates = xg.build_even_pick_candidates(
        _thirty_even_picks(), PickemConfig(), DATE, "morning"
    )

    six_pick = [c for c in candidates if c.entry_size == 6]
    assert sum("Player 29" in c.players for c in six_pick) < 0.4 * len(six_pick)
    assert len(six_pick) == xg._CROSS_GAME_BEAM_WIDTH
    assert set().union(*(c.players for c in candidates)) == {f"Player {i:02d}" for i in range(30)}


def test_small_slate_gives_every_entry_it_can_hold_and_none_larger() -> None:
    offers = _offers_df([_offer_row(f"Player {i}", f"G{i}A/G{i}B") for i in range(3)])

    candidates = xg.build_even_pick_candidates(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert sorted(sorted(c.players) for c in candidates) == [
        ["Player 0", "Player 1"],
        ["Player 0", "Player 1", "Player 2"],
        ["Player 0", "Player 2"],
        ["Player 1", "Player 2"],
    ]


def test_even_picks_of_one_game_alone_make_no_entry() -> None:
    offers = _offers_df([_offer_row("Player A", "BOS/MIA"), _offer_row("Player B", "BOS/MIA")])

    assert xg.build_even_pick_candidates(offers, PickemConfig(min_ev=-1.0), DATE, "morning") == []


def test_even_pick_deal_is_fixed_by_the_date_and_the_run_slot(monkeypatch) -> None:
    monkeypatch.setattr(xg, "_CROSS_GAME_BEAM_WIDTH", 5)  # five entries a size: quick to price
    offers = _thirty_even_picks()
    config = PickemConfig(min_ev=-1.0)

    def _ids(run_slot: str) -> list[str]:
        return [c.id for c in xg.build_even_pick_candidates(offers, config, DATE, run_slot)]

    morning = _ids("morning")
    assert _ids("morning") == morning
    assert _ids("afternoon") != morning


# --- a leg's stat is its league's own key, on both builders -----------------------

_BUILDERS = [xg.build_cross_game_candidates, xg.build_even_pick_candidates]


def _nhl_assists_and_points() -> list[dict]:
    return [
        _offer_row("Skater A", "BOS/TOR", market="Assists", league="NHL"),
        _offer_row("Skater B", "NYR/MTL", market="Points", league="NHL"),
    ]


@pytest.mark.parametrize("build", _BUILDERS)
def test_nhl_leg_carries_the_gamelogs_stat_not_the_platform_maps(build) -> None:
    """``stat_map`` names a market the same for every league: Underdog's Assists is AST.
    The NHL gamelog has no AST column, and settlement reads the gamelog by this key,
    where a missing column settles the leg as a push."""
    offers = _offers_df(_nhl_assists_and_points())

    (two_leg,) = build(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert {leg["player"]: leg["stat"] for leg in two_leg.canonical_legs} == {
        "Skater A": "assists",
        "Skater B": "points",
    }


@pytest.mark.parametrize("build", _BUILDERS)
def test_market_the_platform_map_lacks_still_builds_a_leg(build) -> None:
    """There is no mapped name to rename, so the leg keeps the stat ``build_leg`` gave it."""
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", market="Not A Market"),
            _offer_row("Player B", "LAL/DEN"),
        ]
    )

    (two_leg,) = build(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert {leg["player"]: leg["stat"] for leg in two_leg.canonical_legs} == {
        "Player A": "Not A Market",
        "Player B": stat_map["Underdog"]["Rebounds"],
    }


# --- every pool prices a leg at its fitted chance, and has no leg without one -----


@pytest.mark.parametrize("build", _BUILDERS)
def test_every_pool_prices_and_records_a_leg_at_its_fitted_chance(build) -> None:
    """The served read is the model's claim. The fitted chance, fit on settled posted
    picks, is what an entry is priced on and what its record keeps."""
    fitted = {"Player A": 0.52, "Player B": 0.49}
    offers = _offers_df(
        [
            _offer_row("Player A", "BOS/MIA", fit_prob=fitted["Player A"]),
            _offer_row("Player B", "LAL/DEN", fit_prob=fitted["Player B"]),
        ]
    )
    _, underdog_curve = payout_curve_for("Underdog", "pooled")

    (two_leg,) = build(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert two_leg.model_probs == tuple(fitted[leg["player"]] for leg in two_leg.canonical_legs)
    assert two_leg.ev + 1 == pytest.approx(0.52 * 0.49 * underdog_curve[2][0], abs=0.05)


@pytest.mark.parametrize("build", _BUILDERS)
def test_leg_without_a_fitted_chance_never_enters_a_candidate(build) -> None:
    """The fit wants both sides posted. A record with one side posted has no fitted chance,
    and neither has the unposted side's record, which carries a zero multiplier."""
    one_sided = _offer_row("One Side", "BOS/MIA", fit_prob=float("nan"))
    offers = _offers_df(
        [_offer_row(f"Player {i}", f"G{i}A/G{i}B") for i in range(3)]
        + [one_sided, one_sided | {"Bet": "Under", "Boost": 0.0}]
    )

    candidates = build(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert {c.entry_size for c in candidates} == {2, 3}
    assert all("One Side" not in c.players for c in candidates)


@pytest.mark.parametrize("build", _BUILDERS)
def test_every_candidate_is_staked_flat_though_worth_less_than_its_stake(build) -> None:
    """Kelly stakes nothing on an entry worth less than its stake, and every entry on this
    board is. The flat stake records them all the same."""
    offers = _offers_df([_offer_row(f"Player {i}", f"G{i}A/G{i}B", fit_prob=0.5) for i in range(3)])

    candidates = build(offers, PickemConfig(min_ev=-1.0), DATE, "morning")

    assert candidates
    assert all(c.ev < 0 for c in candidates)
    assert {c.stake for c in candidates} == {FLAT_STAKE}
