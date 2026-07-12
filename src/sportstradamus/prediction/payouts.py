"""Platform payout tables/curves and payout math for parlay pricing."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

from sportstradamus.helpers import underdog_payouts

# Payout multiplier clip. Caps runaway boost products; matches legacy.
PAYOUT_CLIP_LO: float = 1.0
PAYOUT_CLIP_HI: float = 100.0

# Monte-Carlo sample count for the push-aware copula (audit follow-up).
_PUSH_MC_SAMPLES: int = 50_000

# Pooled-variant split: slips of this size or smaller pay on the all-or-nothing
# ``power`` schedule; larger slips pay on the partial-hit ``flex`` schedule.
POWER_MAX_SIZE: int = 3


def _pooled_underdog_curve() -> dict[int, list[float]]:
    """Build the single combined Underdog payout pool keyed by bet size.

    Underdog entries are not separate interchangeable contests: a 2- or
    3-leg slip pays out on the all-or-nothing ``power`` schedule, while a
    4+-leg slip pays out on the partial-hit ``flex`` schedule. ``rivals``
    legs carry no special schedule — they sit in the same pool as any
    other leg. The legacy ``insurance`` table is an old alias of ``flex``
    and is intentionally not consulted here.
    """
    power = underdog_payouts["power"]
    flex = underdog_payouts["flex"]
    curve: dict[int, list[float]] = {}
    for sz_str, mult in power.items():
        sz = int(sz_str)
        if sz <= POWER_MAX_SIZE:
            curve[sz] = [float(mult), 0.0]
    for sz_str, row in flex.items():
        sz = int(sz_str)
        if sz > POWER_MAX_SIZE:
            curve[sz] = [float(v) for v in row]
    return curve


def payout_curve_for(
    platform: str,
    contest_variant: Literal["pooled", "power", "flex", "insurance", "rivals"],
    *,
    legacy: bool,
) -> tuple[list[float], dict[int, list[float]]]:
    """Build the (per-size search list, per-(size,misses) payout table) for a platform.

    The first return drives beam-search ranking (single multiplier per size,
    indexed ``[bet_size - 2]``). The second drives push-aware EV and the
    display ``Boost`` column (full payout curve indexed by miss count).

    Underdog pulls from ``data/underdog_payouts.json``. The default
    ``"pooled"`` variant builds one combined pool (``power`` for sizes 2-3,
    ``flex`` for sizes 4+); the legacy single-variant names are still
    accepted for the ``pickem-build`` path. Other platforms (PrizePicks,
    Sleeper, ParlayPlay, Chalkboard) keep the legacy single-payout table.
    """
    # Deferred import: legacy shims live in ``correlation.py`` per
    # CONTRIBUTING.md §Package Map; importing at module scope would create a
    # cycle with ``correlation``'s top-level ``beam_search_parlays`` import.
    from sportstradamus.prediction.correlation import (
        _legacy_underdog_overwrite_payouts,
        _legacy_underdog_search_payouts,
    )

    if legacy and platform == "Underdog":
        search = _legacy_underdog_search_payouts()
        # Display table mirrors the legacy line-498 overwrite (mixed regime).
        legacy_overwrite = _legacy_underdog_overwrite_payouts()
        full_curve = {sz: [legacy_overwrite[sz], 0.0] for sz in legacy_overwrite}
        return search, full_curve

    legacy_tables: dict[str, list[float]] = {
        "PrizePicks": [3.0, 5.3, 10.0, 20.8, 38.8],
        # Sleeper caps real parlays at 3 legs — sizes 4-6 not enumerated.
        "Sleeper": [1.0, 1.0],
        "ParlayPlay": [1.0, 1.0, 1.0, 1.0, 1.0],
        "Chalkboard": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    if platform != "Underdog":
        lst = legacy_tables[platform]
        full = {i + 2: [lst[i], 0.0] for i in range(len(lst))}
        return lst, full

    if contest_variant == "pooled":
        full_curve = _pooled_underdog_curve()
        max_size = max(full_curve.keys())
        for sz in range(2, max_size + 1):
            full_curve.setdefault(sz, [0.0])
        search = [full_curve[sz][0] for sz in range(2, max_size + 1)]
        return search, full_curve

    variant_table = underdog_payouts[contest_variant]
    if contest_variant in ("flex", "insurance"):
        full_curve = {int(sz): [float(v) for v in row] for sz, row in variant_table.items()}
    else:
        full_curve = {int(sz): [float(variant_table[sz]), 0.0] for sz in variant_table}

    # Pad sizes below the variant's minimum with zero-payout placeholders so
    # the beam-search ranking heuristic indexes by ``size - 2`` consistently
    # and the EV pre-checks naturally reject those sizes (insurance only
    # exists at 5/6, flex at 3-6).
    max_size = max(full_curve.keys())
    for sz in range(2, max_size + 1):
        full_curve.setdefault(sz, [0.0])
    search = [full_curve[sz][0] for sz in range(2, max_size + 1)]
    return search, full_curve


def expected_payout_with_pushes(
    p_win: np.ndarray,
    p_push: np.ndarray,
    sigma: np.ndarray,
    bet_size: int,
    boost: float,
    payout_curve: dict[int, list[float]],
    rng: np.random.Generator | None = None,
) -> float:
    """Expected payout for a parlay where some legs may push.

    Samples ``_PUSH_MC_SAMPLES`` draws from the multivariate normal copula,
    classifies each leg as WIN / PUSH / LOSS via inverse-CDF cuts, and applies
    the variant payout curve at the resulting (effective_size, misses) cell.
    Pushes drop the parlay one leg per Underdog rules; an entry that pushes
    below the minimum bet size with no losses is treated as a refund (×1).

    Args:
        p_win: Per-leg chosen-side probability (already direction-adjusted).
        p_push: Per-leg push probability. Zeros where push is impossible.
        sigma: PSD-repaired correlation matrix for the parlay's legs.
        bet_size: Number of legs (``len(p_win)``).
        boost: Modifier-product boost for this parlay.
        payout_curve: ``{size: [mult_at_0_misses, mult_at_1_miss, ...]}``.
        rng: Optional ``np.random.Generator`` for deterministic tests.

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

    # Classification: 0 = LOSS, 1 = PUSH, 2 = WIN.
    classification = np.where(
        samples < cut_lose,
        0,
        np.where(samples < cut_push_top, 1, 2),
    )

    pushes = (classification == 1).sum(axis=1)
    losses = (classification == 0).sum(axis=1)
    eff_size = bet_size - pushes

    # Flat lookup table: lookup[size, misses] → payout multiplier. Sizes below
    # the parlay minimum are treated as refund (×1) iff there are no losses.
    max_idx = bet_size + 1
    lookup = np.zeros((max_idx, max_idx), dtype=float)
    for sz in range(2, bet_size + 1):
        curve = payout_curve.get(sz)
        if curve is None:
            continue
        for miss_idx, mult in enumerate(curve):
            if miss_idx < max_idx:
                lookup[sz, miss_idx] = float(mult)

    safe_size = np.clip(eff_size, 0, max_idx - 1)
    safe_miss = np.clip(losses, 0, max_idx - 1)
    payouts = lookup[safe_size, safe_miss]

    # Special-case the "all pushes (no losses)" outcome: refund.
    all_push_no_loss = (eff_size < 2) & (losses == 0)
    payouts = np.where(all_push_no_loss, 1.0, payouts)
    # Sub-minimum size with at least one loss → bust.
    payouts = np.where((eff_size < 2) & (losses > 0), 0.0, payouts)

    return float(np.clip(boost, 0.0, PAYOUT_CLIP_HI) * payouts.mean())
