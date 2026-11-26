"""Behavioral tests for parlay search remediation (audit follow-up).

Covers the four targets from ``docs/PARLAY_AUDIT.md`` remediation:

* :func:`_nearest_psd` repairs non-PSD correlation submatrices instead of
  dropping them.
* :func:`expected_payout_with_pushes` produces hand-checkable EV for a
  small parlay with known win/push/lose probabilities.
* Switching ``contest_variant`` from "power" to "flex" changes EV in the
  expected direction (flex pays at non-zero misses; power doesn't).
* The half-integer-line fast path uses the analytical Gaussian-copula CDF
  (not the MC sampler), avoiding a perf regression for the common case.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from sportstradamus.helpers import no_vig_odds, sleeper_payouts, underdog_payouts
from sportstradamus.leg_schema import leg_label
from sportstradamus.prediction import parlay
from sportstradamus.prediction.joint import _nearest_psd
from sportstradamus.prediction.parlay import (
    GameArrays,
    _leg_boost_payout,
    beam_search_parlays,
    resolve_leg_stat,
)
from sportstradamus.prediction.payouts import (
    PAYOUT_CLIP_HI,
    SLEEPER_FLEX_CAP,
    SLEEPER_FLEX_MIN_MULTIPLIER,
    SLEEPER_FLEX_MIN_SIZE,
    SLEEPER_MAX_SIZE,
    UNDERDOG_FLEX_CAP,
    expected_payout_with_pushes,
    payout_curve_for,
    poisson_binomial_pmf,
    sleeper_flex_payout_curve,
)


def test_nearest_psd_repairs_negative_eigenvalue() -> None:
    """A correlation-shaped matrix with one negative eigenvalue is projected
    onto the PSD cone with unit diagonal preserved."""
    bad = np.array(
        [
            [1.0, 0.95, 0.95, 0.95],
            [0.95, 1.0, 0.95, 0.95],
            [0.95, 0.95, 1.0, -0.95],
            [0.95, 0.95, -0.95, 1.0],
        ]
    )
    assert np.min(np.linalg.eigvalsh(bad)) < 0, "test setup invalid"

    repaired = _nearest_psd(bad)

    eigvals = np.linalg.eigvalsh(repaired)
    # After eigenvalue clipping + diagonal rescale, the min eigenvalue is
    # positive but may be smaller than the raw clip threshold (rescaling
    # changes the spectrum). PSD-ness — not the exact threshold — is the
    # invariant the rest of the pipeline relies on.
    assert np.min(eigvals) > 0
    assert np.allclose(np.diag(repaired), 1.0, atol=1e-9)
    assert np.allclose(repaired, repaired.T, atol=1e-12)


def test_nearest_psd_passes_through_already_psd() -> None:
    """A PSD matrix with unit diagonal is returned essentially unchanged."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 8))
    sigma = a @ a.T
    d = 1.0 / np.sqrt(np.diag(sigma))
    sigma = sigma * d[:, None] * d[None, :]

    repaired = _nearest_psd(sigma)
    assert np.allclose(repaired, sigma, atol=1e-9)


def test_push_aware_payout_independent_two_leg_power() -> None:
    """Hand-computed EV for a 2-leg power parlay with independent legs.

    Outcomes: both-win pays 3x; one-push-one-win drops to a 1-leg parlay
    which falls below the size minimum with no losses (refund ×1); any-loss
    pays 0. Independent-leg arithmetic gives the closed-form expectation.
    """
    p_win = np.array([0.6, 0.55])
    p_push = np.array([0.05, 0.10])
    p_lose = 1.0 - p_win - p_push
    sigma = np.eye(2)
    payout_curve = {2: [3.0, 0.0]}

    rng = np.random.default_rng(42)
    ev = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size=2,
        boost=1.0,
        payout_curve=payout_curve,
        rng=rng,
    )

    # Expected: 3 * P(both win) + 1 * P(one or both push, no losses).
    # No-loss outcomes other than both-win: WIN-PUSH, PUSH-WIN, PUSH-PUSH.
    refund_prob = p_win[0] * p_push[1] + p_push[0] * p_win[1] + p_push[0] * p_push[1]
    expected = 3.0 * p_win[0] * p_win[1] + 1.0 * refund_prob
    assert ev == pytest.approx(expected, rel=0.02), (
        f"expected ~{expected:.4f}, got {ev:.4f}; p_lose={p_lose}"
    )


def test_push_aware_payout_one_leg_pushes_promotes_to_refund() -> None:
    """A 2-leg parlay where leg 0 always pushes and leg 1 always wins drops
    below the minimum size with no losses → refund (×1)."""
    p_win = np.array([0.0, 1.0])
    p_push = np.array([1.0, 0.0])
    sigma = np.eye(2)
    payout_curve = {2: [3.0, 0.0]}

    rng = np.random.default_rng(7)
    ev = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size=2,
        boost=1.0,
        payout_curve=payout_curve,
        rng=rng,
    )
    assert abs(ev - 1.0) < 1e-6


def test_flex_vs_power_diverges_under_misses() -> None:
    """A 5-leg parlay with non-trivial miss probability earns more EV under
    flex than power because flex pays at 1+ misses and power doesn't."""
    bet_size = 5
    p_win = np.full(bet_size, 0.55)
    p_push = np.zeros(bet_size)
    sigma = np.eye(bet_size)

    power_curve = {5: [20.0, 0.0, 0.0, 0.0, 0.0]}
    flex_curve = {5: [10.0, 2.0, 0.4, 0.4, 0.0]}

    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    ev_power = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size,
        boost=1.0,
        payout_curve=power_curve,
        rng=rng_a,
    )
    ev_flex = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        bet_size,
        boost=1.0,
        payout_curve=flex_curve,
        rng=rng_b,
    )

    # Power expected ~ 20 * 0.55^5 ≈ 1.006; flex picks up the 1- and 2-miss tiers.
    assert ev_flex > ev_power
    assert ev_power == pytest.approx(20.0 * 0.55**5, rel=0.05)


def test_payout_curve_loader_returns_power_by_default() -> None:
    """``payout_curve_for("Underdog", "power")`` reads the
    JSON config and returns the 0-misses entries as the search list."""
    search, full = payout_curve_for("Underdog", "power")
    assert search[0] == pytest.approx(3.5)  # 2-leg power
    assert full[2][0] == pytest.approx(3.5)
    assert full[2][1] == 0.0  # power: no payout at 1 miss


def test_underdog_payout_file_carries_the_confirmed_table() -> None:
    """The live table the owner confirmed in the app (docs/underdog_api.md §6.8): Power for
    2 to 8 picks, the Flex tiers Underdog offers, and no other variant."""
    assert underdog_payouts == {
        "power": {2: 3.5, 3: 6.5, 4: 12.0, 5: 20.0, 6: 35.0, 7: 65.0, 8: 120.0},
        "flex": {
            3: [3.25, 1.09],
            4: [7.2, 1.4],
            5: [10.0, 2.5],
            6: [25.0, 2.6, 0.25],
            7: [40.0, 2.75, 0.5],
            8: [80.0, 3.0, 1.0],
        },
    }


def test_underdog_curves_cover_two_to_eight_picks_and_build_to_the_cap() -> None:
    """Every Underdog payout table runs to the app's 8 picks, while the search list, whose
    length sets the largest entry beam search builds, still stops at UNDERDOG_FLEX_CAP."""
    for variant in ("pooled", "power", "flex"):
        search, full = payout_curve_for("Underdog", variant)
        assert sorted(full) == list(range(2, 9)), variant
        assert search == [full[size][0] for size in range(2, UNDERDOG_FLEX_CAP + 1)], variant


def _even_pick_breakeven(row: list[float], size: int) -> float:
    """Per-pick hit rate at which ``size`` independent even-money picks return the stake."""

    def edge(p: float) -> float:
        tiers = (
            mult * comb(size, k) * p ** (size - k) * (1 - p) ** k for k, mult in enumerate(row)
        )
        return sum(tiers) - 1.0

    return brentq(edge, 0.01, 0.99)


def test_underdog_pooled_curve_takes_the_lower_breakeven_schedule() -> None:
    """The pooled Power-versus-Flex choice at the live terms: each size pays on the schedule
    an entry of even-money picks beats at the lower per-pick hit rate
    (docs/archive/researcher_parlay_engine.md S1-a). A payout change that flips a size fails
    here; the split in ``payouts._pooled_underdog_curve`` then moves with it."""
    _, pooled = payout_curve_for("Underdog", "pooled")
    for size, mult in underdog_payouts["power"].items():
        best = [float(mult), 0.0]
        flex = underdog_payouts["flex"].get(size)
        if flex and _even_pick_breakeven(flex, size) < _even_pick_breakeven(best, size):
            best = flex
        assert pooled[size] == best, f"{size} picks"


def test_psd_repair_keeps_correlation_units_diagonal() -> None:
    """After repair, the matrix is still a correlation matrix (unit diag),
    so downstream norm.ppf/mvn-cdf calls remain valid."""
    bad = np.array([[1.0, 1.5], [1.5, 1.0]])  # |rho| > 1, not valid
    repaired = _nearest_psd(bad)
    assert np.allclose(np.diag(repaired), 1.0, atol=1e-9)
    # Off-diagonal must be in [-1, 1] for a correlation matrix.
    assert -1.0 - 1e-9 <= repaired[0, 1] <= 1.0 + 1e-9


def _beam_inputs(p_fit=(0.82,) * 4, p_model=None, payouts=(3.0, 6.0, 10.0)) -> dict:
    """A two-team game, the first half of its legs on team A, each its own player.

    ``p_model`` is the fitted chance unless given. The pair grid is built on the fitted
    chances, as ``_build_correlation_matrices`` builds it. Identity correlation (PSD,
    analytical mvn.cdf path) + zero pushes + a single-tier power curve keep the output
    fully deterministic, so it pins ``beam_search_parlays`` end-to-end without touching
    the stochastic push-aware Monte-Carlo branch (covered separately above).
    """
    p_fit = np.asarray(p_fit, dtype=float)
    p_model = p_fit if p_model is None else np.asarray(p_model, dtype=float)
    n = len(p_fit)
    boosts = np.ones(n)
    C = np.eye(n)
    M = np.ones((n, n))
    P = p_fit.reshape(n, 1) * p_fit
    var = p_fit * (1 - p_fit)
    V = np.sqrt(var.reshape(n, 1) * var)
    EV = np.exp(C * V) * P * (boosts.reshape(n, 1) * M * boosts) * payouts[0]
    bet_df = {
        i: {
            "Player": f"P{i}",
            "Team": "A" if i < n // 2 else "B",
            "League": "NBA",
            "Game": "A/B",
            "Date": "2026-01-01",
            "Line": 1.5,
            "Market": f"Market{i}",
            "Bet": "Over",
            "Win Prob": p_model[i],
        }
        for i in range(n)
    }
    idx = pd.DataFrame(
        [{"Player": f"P{i}", "Team": bet_df[i]["Team"]} for i in range(n)], index=range(n)
    )
    g = GameArrays(
        C=C,
        M=M,
        EV=EV,
        EVb=np.zeros((n, n)),
        V=V,
        p_model=p_model,
        p_fit=p_fit,
        p_books=p_model - 0.2,
        p_push=np.zeros(n),
        boosts=boosts,
        shrinkage=np.ones(n),
        opp_boost=np.full(n, np.nan),
    )
    return {
        "idx": idx,
        "g": g,
        "payouts": list(payouts),
        "full_payouts": {size: [mult, 0.0] for size, mult in enumerate(payouts, start=2)},
        "max_boost": 10.0,
        "bet_df": bet_df,
        "info": {"Game": "A/B", "Date": "2026-01-01", "League": "NBA", "Platform": "Underdog"},
        "team": "A",
        "opp": "B",
    }


def test_a_game_with_every_entry_under_break_even_still_builds_one_at_each_size() -> None:
    """Selection is by rank: no floor on an entry's value stands between the beam and the
    board. The floors this replaced kept no entry the model put under 2 for 1."""
    res = beam_search_parlays(**_beam_inputs(p_fit=(0.5,) * 4, p_model=(0.55,) * 4))

    assert {r["Bet Size"] for r in res} == {2, 3, 4}
    assert max(r["Model EV"] for r in res) < 1


def test_an_entry_carries_its_fitted_value_beside_the_models_claim() -> None:
    """``EV`` prices an entry on its legs' fitted chances and ``Model EV`` on the model's:
    with independent legs, the size's payout times the product of each."""
    inputs = _beam_inputs(p_fit=(0.50, 0.52, 0.54, 0.56), p_model=(0.76, 0.74, 0.72, 0.70))
    g, payouts = inputs["g"], inputs["payouts"]

    res = beam_search_parlays(**inputs)

    assert {r["Bet Size"] for r in res} == {2, 3, 4}
    for row in res:
        legs = list(row["Bet ID"])
        payout = payouts[len(legs) - 2]
        assert row["EV"] == pytest.approx(payout * g.p_fit[legs].prod(), rel=1e-4)
        assert row["Model EV"] == pytest.approx(payout * g.p_model[legs].prod(), rel=1e-4)


def test_each_size_exact_prices_the_head_of_the_beam_and_keeps_its_best_few(monkeypatch) -> None:
    """Per size the search exact-prices the first ``_EXACT_PRICED_PER_SIZE`` admissible
    candidates of the beam on the fitted chances, keeps the ``_KEPT_PER_SIZE`` that price
    highest, prices only those again on the model's chances, and returns the rows best
    fitted value first.

    The legs are independent, so the pair grid ranks a set by the product of its fitted
    chances. The stand-in pricer values a set higher the lower that product, the reverse of
    the beam's order: keeping the head of the beam, or pricing every candidate, picks other
    sets. The stronger team's legs come last, so the head of each beam is one-team sets
    the search has to pass over without counting them.
    """
    n_priced, n_kept = 5, 2
    monkeypatch.setattr(parlay, "_EXACT_PRICED_PER_SIZE", n_priced)
    monkeypatch.setattr(parlay, "_KEPT_PER_SIZE", n_kept)
    p_fit = np.array([0.46, 0.48, 0.50, 0.52, 0.64, 0.66, 0.68, 0.70])
    inputs = _beam_inputs(p_fit=p_fit, p_model=p_fit + 0.25)
    priced = []

    def low_chances_price_high(p, *args, **kwargs):
        priced.append(tuple(p))
        return 2.0 - float(np.prod(p))

    monkeypatch.setattr(parlay, "parlay_payout_prob", low_chances_price_high)

    res = beam_search_parlays(**inputs)

    def product(legs):
        return p_fit[list(legs)].prod()

    for size in (2, 3, 4):
        both_teams = [c for c in combinations(range(8), size) if c[0] < 4 <= c[-1]]
        head = sorted(both_teams, key=product, reverse=True)[:n_priced]
        on_fit = {p for p in priced if len(p) == size and set(p) <= set(p_fit)}
        on_model = [p for p in priced if len(p) == size and not set(p) <= set(p_fit)]
        assert on_fit == {tuple(p_fit[list(c)]) for c in head}
        assert len(on_model) == n_kept
        kept = [r["Bet ID"] for r in res if r["Bet Size"] == size]
        assert kept == sorted(head, key=product)[:n_kept]
    assert len(priced) == 3 * (n_priced + n_kept)
    assert [r["EV"] for r in res] == sorted((r["EV"] for r in res), reverse=True)


def test_resolve_leg_stat_uses_market_map_when_present() -> None:
    """A remapped market resolves through ``stat_by_label``; an unmapped one falls back."""
    stat_by_label = {"Fantasy Points": "fantasy points prizepicks"}
    assert resolve_leg_stat("Fantasy Points", stat_by_label) == "fantasy points prizepicks"
    assert resolve_leg_stat("PTS", stat_by_label) == "PTS"


def test_resolve_leg_stat_strips_h2h_prefix_before_lookup() -> None:
    """``"H2H "`` is stripped before the map lookup, mirroring the cMarket idiom."""
    stat_by_label = {"Points": "points"}
    assert resolve_leg_stat("H2H Points", stat_by_label) == "points"


def test_a_rows_legs_carry_the_leagues_own_stat_key() -> None:
    """The search hands its label map to each leg: Underdog's NHL ``Assists`` is the
    ``assists`` cell, the key a leg's ``stat`` holds beside the platform's label."""
    inputs = _beam_inputs()
    inputs["info"]["League"] = "NHL"
    for leg in inputs["bet_df"].values():
        leg.update(League="NHL", Market="Assists")

    res = beam_search_parlays(**inputs)

    assert {(leg["market"], leg["stat"]) for row in res for leg in row["legs"]} == {
        ("Assists", "assists")
    }


def test_a_row_carries_these_columns() -> None:
    """The row contract ``find_correlation``'s empty frame and the parlay history follow:
    the fitted ``EV`` beside the model's ``Model EV``, and no sort-view score."""
    row = beam_search_parlays(**_beam_inputs())[0]

    assert list(row) == [
        "Game",
        "Date",
        "League",
        "Platform",
        "EV",
        "Model EV",
        "Market EV",
        "Boost",
        "Rec Bet",
        "legs",
        "Bet ID",
        "P",
        "PB",
        "Bet Size",
        "Corr Pairs",
        "Boost Pairs",
        "Indep P",
        "Indep PB",
    ]


def test_beam_search_keeps_each_sizes_best_entries_by_fitted_value() -> None:
    """Pin ``beam_search_parlays`` on a deterministic slate: which entries each size keeps,
    the order they come back in, and what a row's value columns hold.

    Independent legs on a single-tier curve, so an entry is worth its size's payout times
    the product of its legs' chances. Sizes 2 and 3 each have four entries holding a leg of
    both teams and keep the three worth most at the fitted chances; size 4 has one. The
    best leg is last, so the entry a size drops is its first by leg order.
    """
    fit = np.array([0.80, 0.82, 0.84, 0.86])
    claim = fit + 0.04
    book = claim - 0.2
    inputs = _beam_inputs(p_fit=fit, p_model=claim)

    res = beam_search_parlays(**inputs)

    kept = [(0, 1, 2, 3), (1, 2, 3), (0, 2, 3), (0, 1, 3), (1, 3), (1, 2), (0, 3)]
    assert [r["Bet ID"] for r in res] == kept
    for row, bet_id in zip(res, kept, strict=True):
        legs = list(bet_id)
        payout = inputs["payouts"][len(legs) - 2]
        assert row["Bet Size"] == len(legs)
        assert row["Boost"] == payout
        assert row["EV"] == pytest.approx(payout * fit[legs].prod(), rel=1e-4)
        assert row["Model EV"] == pytest.approx(payout * claim[legs].prod(), rel=1e-4)
        assert row["Market EV"] == pytest.approx(payout * book[legs].prod(), rel=1e-4)
        assert row["P"] == pytest.approx(payout * claim[legs].prod())
        assert row["PB"] == pytest.approx(payout * book[legs].prod())
        assert [leg_label(leg) for leg in row["legs"]] == [f"P{i} Over 1.5 Market{i}" for i in legs]
    assert res[0]["EV"] == pytest.approx(10.0 * 0.80 * 0.82 * 0.84 * 0.86, rel=1e-4)
    assert res[-1]["EV"] == pytest.approx(3.0 * 0.80 * 0.86, rel=1e-4)


def test_fitted_value_equals_the_models_claim_where_the_chances_agree() -> None:
    """One pricing run on two chance vectors: where the fitted chance is the model's on
    every leg, ``EV`` is ``Model EV``."""
    res = beam_search_parlays(**_beam_inputs(p_fit=(0.60, 0.70, 0.65, 0.75)))

    assert len(res) == 7
    assert [r["EV"] for r in res] == pytest.approx([r["Model EV"] for r in res], rel=1e-4)


def test_no_entry_carries_a_stake() -> None:
    """``Rec Bet`` is 0 on every row, entries the fit values above break-even included: an
    entry gets no stake until the entries' money record supports one
    (docs/handoffs/profit-engine.md 5.3)."""
    res = beam_search_parlays(**_beam_inputs(p_fit=(0.82,) * 4))

    assert min(r["EV"] for r in res) > 1
    assert [r["Rec Bet"] for r in res] == [0.0] * len(res)


def test_a_size_the_table_pays_nothing_at_builds_no_entry() -> None:
    """A variant's payout list is zero-padded below its smallest entry: Underdog Flex starts
    at 3 picks. The search builds nothing at a padded size and still reaches the sizes
    that pay."""
    res = beam_search_parlays(**_beam_inputs(payouts=(0.0, 6.0, 10.0)))

    assert {r["Bet Size"] for r in res} == {3, 4}


# --- Sleeper payout curve (§1/§2) -------------------------------------------


def test_sleeper_payout_curve_max_applies_power_bonus() -> None:
    """Sleeper Max payout curve pulls the confirmed size-only power bonus:
    size 2 has no bonus (1.0x), size 3 carries the confirmed ~1.0797x."""
    search, full = payout_curve_for("Sleeper", "pooled")
    assert search[0] == pytest.approx(1.0)
    assert search[1] == pytest.approx(1.0797)
    assert full[2][0] == pytest.approx(1.0)
    assert full[3][0] == pytest.approx(1.0797)


def test_sleeper_payout_curve_variant_zeroes_out_of_scope_sizes() -> None:
    """A single-variant Sleeper call restricts the curve to its own size
    range (Fix 0: previously ``_sleeper_curve`` ignored ``contest_variant``
    and always returned the full 2-6 curve, which not only mispriced the
    pre-check gate but left ``search`` at length 5 regardless of variant --
    doubling ``beam_search_parlays``'s ``max_bet_size`` for every "power"
    call). "power" must drop the flex sizes entirely (not just zero their
    value) so ``max_bet_size`` actually shrinks; "flex" keeps the full range
    but zero-pads the power sizes below it, since ``search`` always indexes
    from size 2."""
    search_power, full_power = payout_curve_for("Sleeper", "power")
    assert set(full_power.keys()) == set(range(2, SLEEPER_MAX_SIZE + 1))
    assert len(search_power) == SLEEPER_MAX_SIZE - 1
    assert full_power[SLEEPER_MAX_SIZE][0] == pytest.approx(1.0797)

    search_flex, full_flex = payout_curve_for("Sleeper", "flex")
    assert set(full_flex.keys()) == set(range(2, SLEEPER_FLEX_CAP + 1))
    assert len(search_flex) == SLEEPER_FLEX_CAP - 1
    for sz in range(2, SLEEPER_FLEX_MIN_SIZE):
        assert full_flex[sz][0] == 0.0
    assert full_flex[SLEEPER_FLEX_MIN_SIZE][0] != 0.0
    assert search_flex[0] == 0.0  # size 2 slot zeroed, not power's bonus

    search_pooled, full_pooled = payout_curve_for("Sleeper", "pooled")
    assert set(full_pooled.keys()) == set(range(2, SLEEPER_FLEX_CAP + 1))
    assert len(search_pooled) == SLEEPER_FLEX_CAP - 1
    assert full_pooled[SLEEPER_MAX_SIZE][0] == pytest.approx(1.0797)  # real, not zeroed
    assert full_pooled[SLEEPER_FLEX_MIN_SIZE][0] != 0.0  # real, not zeroed


def test_sleeper_payout_curve_flex_k_loads_from_config() -> None:
    """sleeper_payouts['flex_k'] loads with int size keys and the confirmed
    K(n,k) constants (n=5/6 high confidence, n=4 provisional)."""
    assert set(sleeper_payouts["flex_k"].keys()) == {4, 5, 6}
    assert sleeper_payouts["flex_k"][4] == pytest.approx([0.40, 0.35])
    assert sleeper_payouts["flex_k"][5] == pytest.approx([0.300, 0.313, 0.138])
    assert sleeper_payouts["flex_k"][6] == pytest.approx([0.386, 0.186, 0.097])


def test_flex_k_table_ideal_hold_matches_documented_range() -> None:
    """sum(flex_k[size]) is an exact composition-independent identity equal to
    the entry's expected payout ratio; 1 - that sum should land in the
    ~[20%,35%] range Context derived (≈24.9% at n=4/5, ≈33% at n=6). This is
    a config-sanity check on the table, distinct from the runtime per-tier
    payout math covered by the sleeper_flex_payout_curve tests below."""
    for size in (4, 5, 6):
        ideal_hold = 1.0 - sum(sleeper_payouts["flex_k"][size])
        assert 0.20 <= ideal_hold <= 0.35, f"n={size} ideal hold {ideal_hold:.3f} out of range"


# --- expected_payout_with_pushes array-boost (§3) ---------------------------


def test_expected_payout_with_pushes_array_boost_matches_scalar_when_uniform() -> None:
    """A per-leg boost array whose product equals a scalar boost gives the
    same EV when no leg pushes -- proves the array branch's np.prod(axis=1)
    correctly reconstructs the fused scalar in the dominant no-push case."""
    p_win = np.array([0.7, 0.6])
    p_push = np.zeros(2)
    sigma = np.eye(2)
    curve = {2: [3.0, 0.0]}
    per_leg = np.array([1.5, 1.2])  # product = 1.8, matches the scalar case below

    rng_scalar = np.random.default_rng(3)
    ev_scalar = expected_payout_with_pushes(
        p_win, p_push, sigma, 2, boost=1.8, payout_curve=curve, rng=rng_scalar
    )
    rng_array = np.random.default_rng(3)
    ev_array = expected_payout_with_pushes(
        p_win, p_push, sigma, 2, boost=per_leg, payout_curve=curve, rng=rng_array
    )
    assert ev_array == pytest.approx(ev_scalar)


def test_expected_payout_with_pushes_array_boost_excludes_pushed_leg_multiplier() -> None:
    """A pushed leg's own multiplier must not survive into the sample's boost
    product -- only the surviving (non-pushed) legs' multipliers should.

    3 legs, all deterministic: leg 0 always pushes (boost=2.0, must be
    excluded), leg 1 always wins (boost=1.5, survives), leg 2 always loses
    (boost=1.0, survives but neutral). Effective size after the push is 2,
    with 1 loss -> payout_curve[2][1] = 4.0. Surviving product = 1.0(masked)
    * 1.5 * 1.0 = 1.5 (NOT the naive full 3-leg product 2.0*1.5*1.0=3.0).
    Expected EV = 1.5 * 4.0 = 6.0.
    """
    p_win = np.array([0.0, 1.0, 0.0])
    p_push = np.array([1.0, 0.0, 0.0])
    sigma = np.eye(3)
    boost = np.array([2.0, 1.5, 1.0])
    curve = {2: [10.0, 4.0]}

    rng = np.random.default_rng(5)
    ev = expected_payout_with_pushes(
        p_win, p_push, sigma, 3, boost=boost, payout_curve=curve, rng=rng
    )
    assert ev == pytest.approx(6.0, abs=1e-6)


def test_expected_payout_with_pushes_pair_modifier_restores_the_fused_boost() -> None:
    """Per-pick multipliers plus ``pair_modifier`` price like the fused scalar when no leg
    pushes: M's pairwise product travels beside the per-leg array, which has to stay clean
    for the Flex loss-tier ranking. The naive alternative -- passing boosts_leg alone,
    silently dropping M's contribution -- does NOT match (asserted below as the regression
    this test guards against).
    """
    p_win = np.array([0.7, 0.6])
    p_push = np.zeros(2)
    sigma = np.eye(2)
    curve = {2: [3.0, 0.0]}

    boosts_leg = np.array([1.2, 1.3])
    m_pairwise = 0.8  # simulates GameArrays.M's pairwise contribution, M != 1
    boost = m_pairwise * np.prod(boosts_leg)  # true fused scalar _parlay_admissible computes

    rng_scalar = np.random.default_rng(9)
    ev_scalar = expected_payout_with_pushes(
        p_win, p_push, sigma, 2, boost=boost, payout_curve=curve, rng=rng_scalar
    )
    rng_split = np.random.default_rng(9)
    ev_split = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        2,
        boost=boosts_leg,
        payout_curve=curve,
        rng=rng_split,
        pair_modifier=m_pairwise,
    )
    assert ev_split == pytest.approx(ev_scalar)

    rng_naive = np.random.default_rng(9)
    ev_naive = expected_payout_with_pushes(
        p_win, p_push, sigma, 2, boost=boosts_leg, payout_curve=curve, rng=rng_naive
    )
    assert ev_naive != pytest.approx(ev_scalar), "naive boosts_leg (dropping M) should not match"


def test_flex_loss_tier_multiplies_the_largest_remaining_multipliers() -> None:
    """Underdog quotes a k-loss Flex tier on the n - k largest pick multipliers, whichever
    picks lose (docs/underdog_api.md §6.8: picks at 0.87 / 1.16 / 0.74 quote the 1-loss
    tier at 1.09 x 0.87 x 1.16).

    Here the 1.16 pick is the one that loses, so the tier differs from both the product of
    all three multipliers and the product of the two picks that hit.
    """
    boost = np.array([0.87, 1.16, 0.74])
    curve = {3: [3.25, 1.09]}

    ev_one_loss = expected_payout_with_pushes(
        np.array([1.0, 0.0, 1.0]),
        np.zeros(3),
        np.eye(3),
        3,
        boost=boost,
        payout_curve=curve,
        rng=np.random.default_rng(21),
    )
    assert ev_one_loss == pytest.approx(1.09 * 0.87 * 1.16)

    ev_all_hit = expected_payout_with_pushes(
        np.ones(3),
        np.zeros(3),
        np.eye(3),
        3,
        boost=boost,
        payout_curve=curve,
        rng=np.random.default_rng(21),
    )
    assert ev_all_hit == pytest.approx(3.25 * 0.87 * 1.16 * 0.74)


def test_leg_boost_payout_applies_the_pair_modifier_on_the_monte_carlo_path() -> None:
    """Beam search hands the Monte-Carlo scorer the legs' own multipliers and M's pairwise
    product apart, so a taxed stack prices at its modifier product times the untaxed price.

    Three legs at multiplier 1.0, so the fused boost is the pair product alone: three pairs
    at 0.8. The Flex curve routes to the unseeded 50k-draw sampler, hence the loose
    tolerance against the 0.8 ** 3 = 0.512 ratio.
    """
    g = _beam_inputs()["g"]
    bet_id = (0, 1, 2)
    prices = {
        boost: _leg_boost_payout(
            g,
            bet_id,
            3,
            boost,
            3.25 * boost,
            g.p_model[np.ix_(bet_id)],
            np.eye(3),
            {3: [3.25, 1.09]},
            3.25,
            None,
        )
        for boost in (1.0, 0.8**3)
    }
    assert prices[0.8**3] == pytest.approx(0.8**3 * prices[1.0], rel=0.03)


# --- Sleeper 2-leg full-refund special case (§4) ----------------------------


def test_sleeper_two_leg_push_refunds_in_full_even_with_a_loss() -> None:
    """Sleeper's 2-pick divergence: ANY push refunds in full, even with a loss
    elsewhere in the entry (Underdog's generic rule would bust this instead)."""
    p_win = np.array([0.0, 0.0])
    p_push = np.array([1.0, 0.0])
    sigma = np.eye(2)
    curve = {2: [3.0, 0.0]}  # miss tier would otherwise bust (0.0)

    rng = np.random.default_rng(13)
    ev_refund = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        2,
        boost=1.0,
        payout_curve=curve,
        rng=rng,
        full_refund_below_size=2,
    )
    assert ev_refund == pytest.approx(1.0, abs=1e-6)

    rng2 = np.random.default_rng(13)
    ev_generic = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        2,
        boost=1.0,
        payout_curve=curve,
        rng=rng2,
        full_refund_below_size=None,
    )
    assert ev_generic == pytest.approx(0.0, abs=1e-6)


def test_sleeper_three_leg_push_drops_and_reprices_not_full_refund() -> None:
    """A 3-leg entry's push still drops-and-reprices even with
    full_refund_below_size=2 set -- the special case only fires at
    bet_size <= full_refund_below_size, never above it."""
    p_win = np.array([0.0, 1.0, 1.0])
    p_push = np.array([1.0, 0.0, 0.0])
    sigma = np.eye(3)
    curve = {2: [5.0, 0.0]}  # 2 surviving legs, both win -> 0-miss tier

    rng = np.random.default_rng(17)
    ev = expected_payout_with_pushes(
        p_win,
        p_push,
        sigma,
        3,
        boost=1.0,
        payout_curve=curve,
        rng=rng,
        full_refund_below_size=2,
    )
    assert ev == pytest.approx(5.0, abs=1e-6)


@pytest.mark.parametrize("full_refund_below_size", [None, 2])
def test_refund_returns_the_stake_not_the_surviving_pick_multiplier(
    full_refund_below_size,
) -> None:
    """A 2-pick whose other leg pushed is refunded at 1.0x on both platforms: the winning
    survivor's 1.8x multiplier and the pair modifier belong to a payout, not to a refund."""
    ev = expected_payout_with_pushes(
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.eye(2),
        2,
        boost=np.array([1.5, 1.8]),
        payout_curve={2: [3.5, 0.0]},
        rng=np.random.default_rng(17),
        full_refund_below_size=full_refund_below_size,
        pair_modifier=0.9,
    )
    assert ev == pytest.approx(1.0, abs=1e-9)


# --- poisson_binomial_pmf (§4b-i) -------------------------------------------


def test_poisson_binomial_pmf_matches_hand_computation() -> None:
    """3 legs at distinct probabilities; hand-computed against all C(3,k) terms."""
    probs = np.array([0.6, 0.5, 0.3])
    pmf = poisson_binomial_pmf(probs)
    expected = np.array([0.14, 0.41, 0.36, 0.09])
    assert pmf == pytest.approx(expected)
    assert pmf.sum() == pytest.approx(1.0)


def test_poisson_binomial_pmf_degenerates_to_binomial_at_equal_probs() -> None:
    """Equal per-trial probabilities reduce to the standard binomial pmf --
    cross-checks against an independently-known-correct distribution."""
    from scipy.stats import binom

    n, p = 5, 0.35
    pmf = poisson_binomial_pmf(np.full(n, p))
    expected = np.array([binom.pmf(k, n, p) for k in range(n + 1)])
    assert pmf == pytest.approx(expected)


# --- sleeper_flex_payout_curve (§4b-ii) -------------------------------------


def test_sleeper_flex_payout_curve_recovers_confirmed_k_ratio() -> None:
    """5 legs all at devigged p=0.7: the 0-miss entry equals
    flex_k[5][0] / poisson_binomial_pmf(...)[5] -- proves the formula wiring."""
    devigged_p = np.full(5, 0.7)
    curve = sleeper_flex_payout_curve(devigged_p, sleeper_payouts["flex_k"])
    pmf = poisson_binomial_pmf(devigged_p)
    assert curve[5][0] == pytest.approx(sleeper_payouts["flex_k"][5][0] / pmf[5])


def test_sleeper_flex_payout_curve_composition_dependence() -> None:
    """Two 4-leg probability vectors with the same mean but different
    Poisson-binomial all-hit probability must price differently -- regression
    guard against a future 'simplify to mean-p' shortcut (the research
    brief's own leave-one-out comparison showed that fits worse than the real
    per-composition calculation). Both still recover the same flex_k[4][0]
    via payout*pmf[hits] -- an exact identity by construction, not itself
    proof of composition-awareness (see the raw-payout divergence below)."""
    flex_k = sleeper_payouts["flex_k"]
    uniform_p = np.array([0.7, 0.7, 0.7, 0.7])
    skewed_p = np.array([0.9, 0.9, 0.5, 0.5])  # same mean (0.7), different product

    curve_uniform = sleeper_flex_payout_curve(uniform_p, flex_k)
    curve_skewed = sleeper_flex_payout_curve(skewed_p, flex_k)

    pmf_uniform = poisson_binomial_pmf(uniform_p)[4]
    pmf_skewed = poisson_binomial_pmf(skewed_p)[4]
    assert pmf_uniform != pytest.approx(pmf_skewed)  # composition genuinely changes the pmf

    assert curve_uniform[4][0] * pmf_uniform == pytest.approx(flex_k[4][0])
    assert curve_skewed[4][0] * pmf_skewed == pytest.approx(flex_k[4][0])
    assert curve_uniform[4][0] != pytest.approx(curve_skewed[4][0])


def test_sleeper_flex_payout_curve_floors_at_1_25x() -> None:
    """A tier whose K(n,k)/pmf ratio computes below the documented 1.25x
    Sleeper Flex minimum is floored, not passed through under-priced."""
    flex_k = sleeper_payouts["flex_k"]
    devigged_p = np.full(6, 0.5)
    curve = sleeper_flex_payout_curve(devigged_p, flex_k)

    pmf = poisson_binomial_pmf(devigged_p)
    raw = flex_k[6][2] / pmf[6 - 2]
    assert raw < SLEEPER_FLEX_MIN_MULTIPLIER, "test setup should exercise the floor"
    assert curve[6][2] == pytest.approx(SLEEPER_FLEX_MIN_MULTIPLIER)


def test_sleeper_flex_payout_curve_caps_at_payout_clip_hi() -> None:
    """A near-zero-pmf tier that would otherwise produce a runaway payout is
    capped at PAYOUT_CLIP_HI (the fix for the algebraically-degenerate
    hold-ratio clamp originally specced for this -- see implementation notes)."""
    flex_k = sleeper_payouts["flex_k"]
    # 6 legs all near-certain to hit -> "exactly 2 misses" (tier index 2) is
    # vanishingly unlikely, driving pmf -> ~0 and k_val/pmf -> huge.
    devigged_p = np.full(6, 0.999)
    curve = sleeper_flex_payout_curve(devigged_p, flex_k)

    pmf = poisson_binomial_pmf(devigged_p)
    raw = flex_k[6][2] / pmf[6 - 2]
    assert raw > PAYOUT_CLIP_HI, "test setup should exercise the ceiling"
    assert curve[6][2] == pytest.approx(PAYOUT_CLIP_HI)


# --- Sleeper Flex dispatch integration (§4b-iv) -----------------------------


def _sleeper_flex_beam_inputs(*, m_pair: float = 1.0, p_push_leg0: float = 0.0) -> dict:
    """4-leg Sleeper Flex fixture spanning 2 teams; ``m_pair`` (GameArrays.M's
    off-diagonal) and ``p_push_leg0`` are the knobs the dispatch tests vary to
    prove Flex pricing is invariant to both."""
    n = 4
    p_model = np.full(n, 0.85)
    p_books = np.full(n, 0.75)
    boosts = np.full(n, 1.8)
    opp_boost = np.full(n, 2.2)
    C = np.eye(n)
    M = np.full((n, n), m_pair)
    np.fill_diagonal(M, 0.0)
    P = p_model.reshape(n, 1) * p_model
    var = p_model * (1 - p_model)
    V = np.sqrt(var.reshape(n, 1) * var)
    search, full_payouts = payout_curve_for("Sleeper", "pooled")
    search = search[:3]
    full_payouts = {sz: v for sz, v in full_payouts.items() if sz <= 4}
    EV = np.exp(C * V) * P * (boosts.reshape(n, 1) * M * boosts) * search[0]
    p_push = np.zeros(n)
    p_push[0] = p_push_leg0
    bet_df = {
        i: {
            "Player": f"P{i}",
            "Team": "A" if i < 2 else "B",
            "League": "NBA",
            "Game": "A/B",
            "Date": "2026-01-01",
            "Line": 1.5,
            "Market": f"Market{i}",
            "Bet": "Over",
            "Win Prob": p_model[i],
        }
        for i in range(n)
    }
    idx = pd.DataFrame(
        [{"Player": f"P{i}", "Team": bet_df[i]["Team"]} for i in range(n)], index=range(n)
    )
    g = GameArrays(
        C=C,
        M=M,
        EV=EV,
        EVb=np.zeros((n, n)),
        V=V,
        p_model=p_model,
        p_fit=p_model,
        p_books=p_books,
        p_push=p_push,
        boosts=boosts,
        shrinkage=np.ones(n),
        opp_boost=opp_boost,
    )
    return {
        "idx": idx,
        "g": g,
        "payouts": search,
        "full_payouts": full_payouts,
        "max_boost": 60.0,
        "bet_df": bet_df,
        "info": {"Game": "A/B", "Date": "2026-01-01", "League": "NBA", "Platform": "Sleeper"},
        "team": "A",
        "opp": "B",
    }


def _size4_row(res: list[dict]) -> dict:
    rows = [r for r in res if r["Bet Size"] == 4]
    assert len(rows) == 1, f"expected exactly one 4-leg candidate, got {len(rows)}"
    return rows[0]


def test_sleeper_flex_bypasses_boost_product_mechanism() -> None:
    """A candidate priced under Sleeper Flex must not scale with M's pairwise
    modifier product -- devig already consumes the same per-leg multipliers
    as pricing input, so boost/M staying inert (not double-counted) is the
    dispatch's whole point (§4b-iv). Two runs differing only in M's
    off-diagonal (1.0 vs 1.05, both admissible) must price identically.

    ``Boost`` (== the Flex-curve payout, a pure function of devigged_p/flex_k
    with zero dependence on M/boost) is the exact, deterministic proof.
    ``Model EV`` is a Monte-Carlo estimate re-sampled fresh (unseeded) on
    every call -- pre-existing behavior, not something this stage changes --
    so it's checked only loosely, as a sanity bound against a real bug.
    """
    res_neutral = beam_search_parlays(**_sleeper_flex_beam_inputs(m_pair=1.0))
    res_bumped = beam_search_parlays(**_sleeper_flex_beam_inputs(m_pair=1.05))

    row_neutral = _size4_row(res_neutral)
    row_bumped = _size4_row(res_bumped)
    assert row_neutral["Boost"] == pytest.approx(row_bumped["Boost"])
    assert row_neutral["Model EV"] == pytest.approx(row_bumped["Model EV"], rel=0.05)


def test_sleeper_flex_values_an_entry_on_the_fitted_chances() -> None:
    """The Flex dispatch prices on the chances it is handed: ``EV`` at the fitted 0.60 a
    leg, ``Model EV`` at the model's 0.85, both on the entry's own devigged curve.

    Independent legs, so each is the 0-miss tier times P(all four hit) plus the 1-miss tier
    times P(exactly three hit). Both are 50k-draw Monte-Carlo estimates, hence the loose
    tolerance.
    """
    inputs = _sleeper_flex_beam_inputs()
    inputs["g"] = replace(inputs["g"], p_fit=np.full(4, 0.60))
    devigged = np.full(4, no_vig_odds(1.8, 2.2)[0])
    all_hit, one_miss = sleeper_flex_payout_curve(devigged, sleeper_payouts["flex_k"])[4][:2]

    def value(p: float) -> float:
        return all_hit * p**4 + one_miss * 4 * p**3 * (1 - p)

    row = _size4_row(beam_search_parlays(**inputs))

    assert row["EV"] == pytest.approx(value(0.60), rel=0.05)
    assert row["Model EV"] == pytest.approx(value(0.85), rel=0.05)


def test_sleeper_flex_ignores_push_probability_by_design() -> None:
    """A Flex candidate with a leg carrying real p_push > 0 must price
    identically to the same candidate with that leg's p_push forced to 0 --
    regression guard for the deliberate no-push Flex scope decision
    (docs/handoffs/sleeper-parity.md §4b): Flex always resolves WIN/LOSS
    only, never honoring push probability, until a future stage builds the
    K(n,k)-at-push-reduced-size extrapolation this would otherwise require.

    ``Boost`` is the exact, deterministic proof (_sleeper_flex_payout never
    even receives p_push). ``Model EV`` is a Monte-Carlo estimate re-sampled
    fresh (unseeded) on every call, so it's checked only loosely.
    """
    res_no_push = beam_search_parlays(**_sleeper_flex_beam_inputs(p_push_leg0=0.0))
    res_with_push = beam_search_parlays(**_sleeper_flex_beam_inputs(p_push_leg0=0.3))

    row_no_push = _size4_row(res_no_push)
    row_with_push = _size4_row(res_with_push)
    assert row_no_push["Boost"] == pytest.approx(row_with_push["Boost"])
    assert row_no_push["Model EV"] == pytest.approx(row_with_push["Model EV"], rel=0.05)
