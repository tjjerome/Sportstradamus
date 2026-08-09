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

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import sleeper_payouts
from sportstradamus.leg_schema import leg_label
from sportstradamus.prediction.joint import _PSD_EIG_TOLERANCE, _nearest_psd
from sportstradamus.prediction.parlay import (
    GameArrays,
    beam_search_parlays,
    resolve_leg_stat,
)
from sportstradamus.prediction.payouts import (
    PAYOUT_CLIP_HI,
    SLEEPER_FLEX_CAP,
    SLEEPER_FLEX_MIN_MULTIPLIER,
    SLEEPER_FLEX_MIN_SIZE,
    SLEEPER_MAX_SIZE,
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
    assert search[0] == pytest.approx(3.0)  # 2-leg power
    assert full[2][0] == pytest.approx(3.0)
    assert full[2][1] == 0.0  # power: no payout at 1 miss


def test_psd_repair_keeps_correlation_units_diagonal() -> None:
    """After repair, the matrix is still a correlation matrix (unit diag),
    so downstream norm.ppf/mvn-cdf calls remain valid."""
    bad = np.array([[1.0, 1.5], [1.5, 1.0]])  # |rho| > 1, not valid
    repaired = _nearest_psd(bad)
    assert np.allclose(np.diag(repaired), 1.0, atol=1e-9)
    # Off-diagonal must be in [-1, 1] for a correlation matrix.
    assert -1.0 - 1e-9 <= repaired[0, 1] <= 1.0 + 1e-9


def _beam_inputs() -> dict:
    """A 4-leg / 2-team game whose legs clear every beam-search EV gate.

    Identity correlation (PSD, analytical mvn.cdf path) + zero pushes + a
    single-tier power curve keep the output fully deterministic, so it pins
    ``beam_search_parlays`` end-to-end without touching the stochastic
    push-aware Monte-Carlo branch (covered separately above).
    """
    n = 4
    p_model = np.array([0.82, 0.82, 0.82, 0.82])
    p_books = np.array([0.62, 0.62, 0.62, 0.62])
    boosts = np.ones(n)
    C = np.eye(n)
    M = np.ones((n, n))
    P = p_model.reshape(n, 1) * p_model
    var = p_model * (1 - p_model)
    V = np.sqrt(var.reshape(n, 1) * var)
    payouts = [3.0, 6.0, 10.0]
    EV = np.exp(C * V) * P * (boosts.reshape(n, 1) * M * boosts) * payouts[0]
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
        V=V,  # EVb is annotation-only, unused here
        p_model=p_model,
        p_books=p_books,
        p_push=np.zeros(n),
        boosts=boosts,
        shrinkage=np.ones(n),
        opp_boost=np.full(n, np.nan),
    )
    return {
        "idx": idx,
        "g": g,
        "payouts": payouts,
        "full_payouts": {2: [3.0, 0.0], 3: [6.0, 0.0], 4: [10.0, 0.0]},
        "max_boost": 10.0,
        "bet_df": bet_df,
        "info": {"Game": "A/B", "Date": "2026-01-01", "League": "NBA", "Platform": "Underdog"},
        "team": "A",
        "opp": "B",
        "stat_map": {"Underdog": {}},
    }


def test_resolve_leg_stat_uses_market_map_when_present() -> None:
    """A remapped market resolves through ``new_map``; an unmapped one falls back."""
    new_map = {"Fantasy Points": "fantasy points prizepicks"}
    assert resolve_leg_stat("Fantasy Points", new_map) == "fantasy points prizepicks"
    assert resolve_leg_stat("PTS", new_map) == "PTS"


def test_resolve_leg_stat_strips_h2h_prefix_before_lookup() -> None:
    """``"H2H "`` is stripped before the map lookup, mirroring the cMarket idiom."""
    new_map = {"Points": "points"}
    assert resolve_leg_stat("H2H Points", new_map) == "points"


def test_beam_search_characterization() -> None:
    """Pin beam_search_parlays output on a deterministic slate (decomposition guard).

    ``Rec Bet`` (units) hand-derivation under the sleeper-parity §5
    shrinkage-aware Kelly gate, with the fixture's full-trust
    ``shrinkage=np.ones(n)`` (a no-op — ``kelly_edge``'s effective
    ``p`` equals ``win_prob`` exactly) and ``DEFAULT_KELLY_FRACTION=0.25``,
    ``MAX_FRACTION_OF_BANKROLL=0.005``:

    ``win_prob = p / payout``; ``b = payout - 1``;
    ``raw_kelly = (b*win_prob - (1-win_prob)) / b``;
    ``units = raw_kelly * 0.25 / 0.005 = raw_kelly * 50``.

    - size 2: p=2.0172, payout=3.0 → win_prob=0.6724, b=2.0,
      raw_kelly=(2*0.6724-0.3276)/2=0.5086 → units=0.5086*50=25.43
    - size 3: p=3.308208, payout=6.0 → win_prob=0.551368, b=5.0,
      raw_kelly=(5*0.551368-0.448632)/5=0.4616416 → units=23.08208
    - size 4: p=4.5212176, payout=10.0 → win_prob=0.45212176, b=9.0,
      raw_kelly=(9*0.45212176-0.54787824)/9=0.3912464 → units=19.56232
    """
    res = beam_search_parlays(**_beam_inputs())

    def sig(r):
        return (
            r["Bet Size"],
            round(r["Model EV"], 6),
            round(r["Market EV"], 6),
            round(r["Rec Bet"], 6),
            round(r["P"], 6),
            round(r["PB"], 6),
            round(r["Boost"], 6),
        )

    expected = (
        [(2, 2.0172, 1.1532, 25.43, 2.0172, 1.1532, 3.0)] * 4
        + [(3, 3.308208, 1.429968, 23.08208, 3.308208, 1.429968, 6.0)] * 4
        + [(4, 4.521218, 1.477634, 19.56232, 4.521218, 1.477634, 10.0)]
    )
    assert sorted(map(sig, res)) == sorted(expected)
    # Each size-2 parlay must span both teams (covers-team/opp gate).
    legs2 = {tuple(leg_label(leg) for leg in r["legs"]) for r in res if r["Bet Size"] == 2}
    assert legs2 == {
        ("P0 Over 1.5 Market0", "P2 Over 1.5 Market2"),
        ("P0 Over 1.5 Market0", "P3 Over 1.5 Market3"),
        ("P1 Over 1.5 Market1", "P2 Over 1.5 Market2"),
        ("P1 Over 1.5 Market1", "P3 Over 1.5 Market3"),
    }


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


def test_expected_payout_with_pushes_array_boost_preserves_full_survival_product() -> None:
    """Proves the §3b fold-in construction: leg_boost[0] *= boost / prod(boosts_leg)
    reconstructs the true fused scalar (M's pairwise product included) exactly,
    so the array path matches the scalar path when no leg pushes. The naive
    alternative -- passing boosts_leg alone, silently dropping M's contribution
    -- does NOT match (asserted below as the regression this test guards against).
    """
    p_win = np.array([0.7, 0.6])
    p_push = np.zeros(2)
    sigma = np.eye(2)
    curve = {2: [3.0, 0.0]}

    boosts_leg = np.array([1.2, 1.3])
    m_pairwise = 1.25  # simulates GameArrays.M's pairwise contribution, M != 1
    boost = m_pairwise * np.prod(boosts_leg)  # true fused scalar _parlay_admissible computes

    leg_boost = boosts_leg.copy()
    leg_boost[0] *= boost / np.prod(boosts_leg)
    assert np.prod(leg_boost) == pytest.approx(boost)

    rng_scalar = np.random.default_rng(9)
    ev_scalar = expected_payout_with_pushes(
        p_win, p_push, sigma, 2, boost=boost, payout_curve=curve, rng=rng_scalar
    )
    rng_folded = np.random.default_rng(9)
    ev_folded = expected_payout_with_pushes(
        p_win, p_push, sigma, 2, boost=leg_boost, payout_curve=curve, rng=rng_folded
    )
    assert ev_folded == pytest.approx(ev_scalar)

    rng_naive = np.random.default_rng(9)
    ev_naive = expected_payout_with_pushes(
        p_win, p_push, sigma, 2, boost=boosts_leg, payout_curve=curve, rng=rng_naive
    )
    assert ev_naive != pytest.approx(ev_scalar), "naive boosts_leg (dropping M) should not match"


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
        "stat_map": {"Sleeper": {}},
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
