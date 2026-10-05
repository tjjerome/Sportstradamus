"""Platform payout tables/curves and payout math for parlay pricing."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

from sportstradamus.helpers import sleeper_payouts, underdog_payouts

# Payout multiplier clip. Caps runaway boost products; matches legacy.
PAYOUT_CLIP_LO: float = 1.0
PAYOUT_CLIP_HI: float = 100.0

# Monte-Carlo sample count for the push-aware copula (audit follow-up).
_PUSH_MC_SAMPLES: int = 50_000

# Pooled-variant split: slips of this size or smaller pay on the all-or-nothing
# ``power`` schedule; larger slips, up to UNDERDOG_FLEX_CAP, pay on the
# partial-hit ``flex`` schedule.
POWER_MAX_SIZE: int = 3

# Largest Underdog entry the engine builds, and the last size the pooled variant
# prices on ``flex``. OUR construction convention, like SLEEPER_FLEX_CAP: the app
# takes 8 picks and the payout table runs that far.
UNDERDOG_FLEX_CAP: int = 6

# Sleeper Max/Flex split: OUR recommendation-construction convention (not a
# platform rule -- the real app allows toggling either mode up to 8 legs).
# Mirrors POWER_MAX_SIZE's Underdog split. See docs/handoffs/sleeper-parity.md.
SLEEPER_MAX_SIZE: int = 3
SLEEPER_FLEX_CAP: int = 6
SLEEPER_FLEX_MIN_SIZE: int = SLEEPER_MAX_SIZE + 1

# Sleeper's own documented minimum Flex payout (support.sleeper.com/en/articles/9261402).
SLEEPER_FLEX_MIN_MULTIPLIER: float = 1.25

# Sleeper's 2-pick divergence: ANY push refunds the entry in full, rather than
# the generic drop-and-reprice rule every larger size uses. See
# docs/handoffs/sleeper-parity.md §3 item 3.
SLEEPER_FULL_REFUND_MAX_SIZE: int = 2

# One leg's result, as the pricer samples it and settlement resolves it.
LEG_LOSS, LEG_PUSH, LEG_WIN = 0, 1, 2


def _pooled_underdog_curve() -> dict[int, list[float]]:
    """Build the single combined Underdog payout pool keyed by entry size.

    Every size in the table, 2 to 8 picks, gets one schedule: the one an entry
    of even-money picks breaks even on at the lower per-pick hit rate. That is
    ``flex`` from ``POWER_MAX_SIZE + 1`` to ``UNDERDOG_FLEX_CAP`` picks and the
    all-or-nothing ``power`` multiplier at every other size. The split is fixed
    here because the play-type labels and the ledger share it;
    ``tests/golden/test_parlay_search.py`` re-derives it from the table, so a
    payout change that moves it fails there.
    """
    curve = {sz: [float(mult), 0.0] for sz, mult in underdog_payouts["power"].items()}
    for sz in range(POWER_MAX_SIZE + 1, UNDERDOG_FLEX_CAP + 1):
        curve[sz] = [float(v) for v in underdog_payouts["flex"][sz]]
    return curve


def _sleeper_curve(
    contest_variant: Literal["pooled", "power", "flex"],
) -> dict[int, list[float]]:
    """Sleeper payout curve keyed by bet size.

    Max (2..SLEEPER_MAX_SIZE): the real per-entry payout is dominated by the
    live per-leg payout_multiplier product, threaded through as the `boost`
    argument, not payout_base -- but it is NOT a bare product. Confirmed
    from two independent in-app trials: size 3 carries a
    real ~1.0797x bonus on top of the raw product; size 2 does not (1.0x).
    `sleeper_payouts["power"]` holds this confirmed, fully-populated
    two-entry table (mirrors underdog_payouts["power"]'s scalar-per-size
    shape). payout_base is this bonus, not a placeholder -- ranking still
    multiplies the boost product in, it's just no longer assumed neutral.

    Flex (SLEEPER_MAX_SIZE+1..SLEEPER_FLEX_CAP): pulls from
    sleeper_payouts["flex"] -- NOT the priced value (see sleeper_flex_payout_curve),
    just a coarse composition-independent stand-in (sum of the real K(n,k)
    constants) used by beam search's pre-filter ranking before the exact
    per-candidate price is computed. Unpopulated sizes pad to [0.0] so beam
    search's EV floors reject them instead of mispricing on invented
    numbers -- not exercised today since sizes 4-6 are populated, but kept
    as the safety behavior if a future size is ever added unpopulated.

    Like ``_pooled_underdog_curve``, ``"pooled"`` builds the full combined
    curve (power range + flex range). A single-variant call restricts the
    curve to only its own range: ``"power"`` returns keys
    2..SLEEPER_MAX_SIZE and no keys above it at all, so
    ``payout_curve_for``'s ``max(full_curve.keys())`` shrinks and
    ``beam_search_parlays``'s ``max_bet_size`` stops at SLEEPER_MAX_SIZE
    instead of exploring up through SLEEPER_FLEX_CAP on every variant call.
    ``"flex"`` returns its own range plus zero-padded keys below
    SLEEPER_MAX_SIZE+1 -- ``payout_curve_for``'s search list always indexes
    from size 2, so those keys must exist, just priced at 0.0 so the EV
    floor rejects them (mirrors the "unpopulated sizes pad to 0.0" behavior
    above, applied below the variant's range instead of within it).
    """
    curve: dict[int, list[float]] = {}
    if contest_variant in ("pooled", "power"):
        power = sleeper_payouts.get("power", {})
        for sz in range(2, SLEEPER_MAX_SIZE + 1):
            curve[sz] = [float(power[sz]), 0.0]
    if contest_variant in ("pooled", "flex"):
        flex = sleeper_payouts.get("flex", {})
        for sz in range(SLEEPER_MAX_SIZE + 1, SLEEPER_FLEX_CAP + 1):
            row = flex.get(sz)
            curve[sz] = [float(v) for v in row] if row else [0.0]
        if contest_variant == "flex":
            for sz in range(2, SLEEPER_MAX_SIZE + 1):
                curve.setdefault(sz, [0.0])
    return curve


def payout_curve_for(
    platform: str,
    contest_variant: Literal["pooled", "power", "flex"],
) -> tuple[list[float], dict[int, list[float]]]:
    """Build the (per-size search list, per-(size,misses) payout table) for a platform.

    The first return drives beam-search ranking (single multiplier per size,
    indexed ``[bet_size - 2]``), and its length sets the largest entry the
    search builds. The second drives push-aware EV, settlement and the display
    ``Boost`` column (full payout curve indexed by miss count).

    Underdog pulls from ``data/config/underdog_payouts.json``. The default
    ``"pooled"`` variant builds one combined pool
    (:func:`_pooled_underdog_curve`); the single-variant names serve the
    ``pickem-build`` path and settlement. An Underdog payout table carries
    every size the app takes, while the search list stops at
    ``UNDERDOG_FLEX_CAP``. PrizePicks, ParlayPlay and Chalkboard keep the
    legacy single-payout table.
    """
    if platform == "Sleeper":
        full_curve = _sleeper_curve(contest_variant)
        max_size = max(full_curve.keys())
        search = [full_curve[sz][0] for sz in range(2, max_size + 1)]
        return search, full_curve

    legacy_tables: dict[str, list[float]] = {
        "PrizePicks": [3.0, 5.3, 10.0, 20.8, 38.8],
        "ParlayPlay": [1.0, 1.0, 1.0, 1.0, 1.0],
        "Chalkboard": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    if platform != "Underdog":
        lst = legacy_tables[platform]
        full = {i + 2: [mult, 0.0] for i, mult in enumerate(lst)}
        return lst, full

    if contest_variant == "pooled":
        full_curve = _pooled_underdog_curve()
    elif contest_variant == "flex":
        full_curve = {sz: [float(v) for v in row] for sz, row in underdog_payouts["flex"].items()}
    else:
        full_curve = {
            sz: [float(mult), 0.0] for sz, mult in underdog_payouts[contest_variant].items()
        }

    # Flex starts at 3 picks: zero-pad below a variant's minimum so the search
    # list indexes by ``size - 2`` and the EV pre-checks reject the padded size.
    for sz in range(2, UNDERDOG_FLEX_CAP + 1):
        full_curve.setdefault(sz, [0.0])
    search = [full_curve[sz][0] for sz in range(2, UNDERDOG_FLEX_CAP + 1)]
    return search, full_curve


def outcome_payouts(
    outcomes: np.ndarray,
    boost: float | np.ndarray,
    payout_curve: dict[int, list[float]],
    *,
    full_refund_below_size: int | None = None,
    pair_modifier: float = 1.0,
) -> np.ndarray:
    """Payout per $1 staked for each entry outcome: the rule pricing and settlement share.

    The pricer hands in its sampled outcomes and averages the result;
    settlement hands in the one real outcome. Each row pays the payout curve
    at its (effective size, losses) cell, where a push drops the entry one
    leg per Underdog rules, times the picks' own multipliers and the pair
    modifier. An entry pushed below the minimum size with no losses is a
    refund (×1).

    Args:
        outcomes: ``(n_outcomes, bet_size)`` leg results, each ``LEG_LOSS``,
            ``LEG_PUSH`` or ``LEG_WIN``.
        boost: The picks' own payout multipliers. A per-leg ``np.ndarray``
            (shape ``(bet_size,)``) pays each outcome the way Underdog quotes
            it: a pushed leg's multiplier drops out, and a tier with ``k``
            losses leaves out the ``k`` smallest of the remaining
            multipliers, whichever legs lost (docs/underdog_api.md §6.8). A
            scalar is one fused product applied to every outcome.
        payout_curve: ``{size: [mult_at_0_misses, mult_at_1_miss, ...]}``.
        full_refund_below_size: When set, any outcome with at least one push
            on an entry sized at or below this threshold refunds in full
            (×1) regardless of losses — Sleeper's 2-pick divergence from the
            generic drop-and-reprice rule (docs/handoffs/sleeper-parity.md §4).
        pair_modifier: Product of the slip's same-game pair modifiers
            (``GameArrays.M``). Multiplies every outcome; it travels beside a
            per-leg ``boost`` because folding it into one leg's multiplier
            would change which multipliers rank smallest.

    Returns:
        np.ndarray: One payout multiple per outcome row.
    """
    bet_size = outcomes.shape[1]
    pushes = (outcomes == LEG_PUSH).sum(axis=1)
    losses = (outcomes == LEG_LOSS).sum(axis=1)
    eff_size = bet_size - pushes

    if isinstance(boost, np.ndarray):
        # Walk the non-pushed legs from the smallest multiplier up and skip one
        # per loss: what is left is the set Underdog quotes the tier on.
        order = np.argsort(boost)
        live = outcomes[:, order] != LEG_PUSH
        kept = live & (np.cumsum(live, axis=1) > losses[:, None])
        sample_boost = np.where(kept, boost[order], 1.0).prod(axis=1)
    else:
        sample_boost = boost
    sample_boost = np.clip(sample_boost * pair_modifier, 0.0, PAYOUT_CLIP_HI)

    # lookup[size, misses] → payout multiplier.
    max_idx = bet_size + 1
    lookup = np.zeros((max_idx, max_idx), dtype=float)
    for sz in range(2, bet_size + 1):
        curve = payout_curve.get(sz)
        if curve is None:
            continue
        for miss_idx, mult in enumerate(curve):
            if miss_idx < max_idx:
                lookup[sz, miss_idx] = float(mult)

    payouts = sample_boost * lookup[eff_size, losses]

    # Pushed below the two-pick minimum, an entry refunds when nothing lost and busts
    # otherwise. A refund is the stake back: no pick multiplier or pair modifier applies.
    payouts = np.where(eff_size < 2, np.where(losses == 0, 1.0, 0.0), payouts)
    if full_refund_below_size is not None and bet_size <= full_refund_below_size:
        payouts = np.where(pushes >= 1, 1.0, payouts)
    return payouts


def expected_payout_with_pushes(
    p_win: np.ndarray,
    p_push: np.ndarray,
    sigma: np.ndarray,
    bet_size: int,
    boost: float | np.ndarray,
    payout_curve: dict[int, list[float]],
    rng: np.random.Generator | None = None,
    *,
    full_refund_below_size: int | None = None,
    pair_modifier: float = 1.0,
) -> float:
    """Expected payout for a parlay where some legs may push.

    Samples ``_PUSH_MC_SAMPLES`` draws from the multivariate normal copula,
    classifies each leg as WIN / PUSH / LOSS via inverse-CDF cuts, and
    averages what :func:`outcome_payouts` pays those sampled outcomes.

    Args:
        p_win: Per-leg chosen-side probability (already direction-adjusted).
        p_push: Per-leg push probability. Zeros where push is impossible.
        sigma: PSD-repaired correlation matrix for the parlay's legs.
        bet_size: Number of legs (``len(p_win)``).
        boost: The picks' own payout multipliers, per leg or one fused scalar.
        payout_curve: ``{size: [mult_at_0_misses, mult_at_1_miss, ...]}``.
        rng: Optional ``np.random.Generator`` for deterministic tests.
        full_refund_below_size: Entry size at or below which any push refunds
            the entry in full.
        pair_modifier: Product of the slip's same-game pair modifiers.

    Returns:
        float: Expected payout, ready to be compared against the EV floor.
    """
    rng = rng if rng is not None else np.random.default_rng()
    samples = rng.multivariate_normal(np.zeros(bet_size), sigma, size=_PUSH_MC_SAMPLES)

    # Per leg: cuts split the standard normal into LOSS / PUSH / WIN bands so
    # the marginal probabilities match (p_lose, p_push, p_win) exactly.
    p_lose = np.clip(1.0 - p_win - p_push, 0.0, 1.0)
    cut_lose = norm.ppf(np.clip(p_lose, 1e-9, 1 - 1e-9))
    cut_push_top = norm.ppf(np.clip(p_lose + p_push, 1e-9, 1 - 1e-9))

    outcomes = np.where(
        samples < cut_lose,
        LEG_LOSS,
        np.where(samples < cut_push_top, LEG_PUSH, LEG_WIN),
    )
    payouts = outcome_payouts(
        outcomes,
        boost,
        payout_curve,
        full_refund_below_size=full_refund_below_size,
        pair_modifier=pair_modifier,
    )
    return float(np.mean(payouts))


def poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """Distribution of successes over independent Bernoulli trials with
    per-trial probabilities ``probs`` (n <= SLEEPER_FLEX_CAP=6 here, so the
    O(n^2) DP fold is negligible).

    Returns:
        np.ndarray: ``pmf``, shape ``(len(probs) + 1,)``, ``pmf[k]`` = P(exactly k successes).
    """
    pmf = np.array([1.0])
    for p in probs:
        new = np.zeros(len(pmf) + 1)
        new[:-1] += pmf * (1.0 - p)
        new[1:] += pmf * p
        pmf = new
    return pmf


def sleeper_flex_payout_curve(
    devigged_p: np.ndarray, flex_k: dict[int, list[float]]
) -> dict[int, list[float]]:
    """Per-candidate Sleeper Flex payout curve: K(n,k) / P_devigged(exactly n-k hit).

    Unlike payout_curve_for()'s output, this is NOT reusable across other
    candidates of the same size -- it is keyed to this specific candidate's
    devigged leg probabilities and must be rebuilt per candidate.
    """
    n = len(devigged_p)
    k_row = flex_k.get(n)
    if k_row is None:
        return {n: [0.0] * (n + 1)}
    pmf = poisson_binomial_pmf(devigged_p)
    row = []
    for misses, k_val in enumerate(k_row):
        p_exact = pmf[n - misses]
        if p_exact <= 0.0:
            row.append(0.0)  # tier this candidate's legs will never realize
            continue
        # Absolute ceiling, not a hold-ratio clamp: k_val/p_exact is exact by
        # construction (Context's sum_k K(n,k) identity), so any per-tier
        # "implied hold" derived from the same p_exact used to build
        # raw_payout cancels out algebraically (1 - p_exact*(k_val/p_exact)
        # == 1-k_val always) and can never actually bound a thin-tier blowup.
        # PAYOUT_CLIP_HI is the same runaway-payout guard used everywhere
        # else in this pipeline (expected_payout_with_pushes, _evaluate_parlay).
        raw_payout = min(float(k_val) / float(p_exact), PAYOUT_CLIP_HI)
        row.append(max(raw_payout, SLEEPER_FLEX_MIN_MULTIPLIER))
    row += [0.0] * (n + 1 - len(row))
    return {n: row}
