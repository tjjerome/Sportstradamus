"""Beam-search parlay enumeration and push-aware EV evaluation.

:func:`beam_search_parlays` is the parlay constructor consumed by
:func:`sportstradamus.prediction.correlation.find_correlation`; the helpers
:func:`_payout_curve_for`, :func:`_expected_payout_with_pushes`, and
:func:`_nearest_psd` live alongside it.
"""

from __future__ import annotations

from itertools import combinations
from operator import itemgetter
from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from sportstradamus.helpers import stat_cv, stat_std, underdog_payouts

# --- Beam-search constants (audit §2.3 — promoted from inline magic numbers) -

# Beam width: max parlay candidates carried between sizes. Empirically large
# enough that the survivors at sizes 5/6 don't get pruned by ranking jitter.
_BEAM_WIDTH: int = 1000

# Per-step pairwise-EV floor. Below this, the candidate is too noisy to keep.
_PARLAY_GEO_MEAN_FLOOR: float = 1.05

# Boost-product gates: drop entries whose modifier product implies hard-banned
# pairs (low) or runaway promo stacking (high).
_MIN_PRODUCT_BOOST: float = 0.7

# Final EV gates (post-copula evaluation).
_BOOKS_EV_FLOOR: float = 0.9
_MODEL_EV_PRECHECK_FLOOR: float = 1.5
_MODEL_EV_FINAL_FLOOR: float = 2.0
_KELLY_UNITS_FLOOR: float = 0.5

# PSD repair: floor on minimum eigenvalue. Sub-floor matrices are projected
# to the nearest PSD instead of dropped (audit §2.1, finding bullet 3).
_PSD_EIG_TOLERANCE: float = 1e-4

# Payout multiplier clip. Caps runaway boost products; matches legacy.
_PAYOUT_CLIP_LO: float = 1.0
_PAYOUT_CLIP_HI: float = 100.0

# Push-handling thresholds.
# Below this, a leg's push prob is treated as zero so the fast (analytical)
# mvn.cdf path runs. Set just above floating-point noise from ``get_push_prob``.
_PUSH_PROB_FLOOR: float = 1e-6
# Monte-Carlo sample count for the push-aware copula (audit follow-up).
_PUSH_MC_SAMPLES: int = 50_000

# Kelly sizing denominator: 5% bankroll per unit. Legacy.
_KELLY_BANKROLL_FRACTION: float = 0.05

# Pooled-variant split: slips of this size or smaller pay on the all-or-nothing
# ``power`` schedule; larger slips pay on the partial-hit ``flex`` schedule.
_POWER_MAX_SIZE: int = 3

# --- Parlay family clustering -----------------------------------------------

# Upper bound on independence families per game. Auto-selected count (by
# silhouette) may be anywhere in [1, this]; keeps the dashboard picker short.
_PARLAY_MAX_FAMILIES: int = 4

# Silhouette floor: a k>=2 split must separate at least this well, otherwise
# the survivors are one blob and we honestly report a single family.
_PARLAY_MIN_SILHOUETTE: float = 0.15

# Below this many survivor parlays, clustering is noise — all one family.
_PARLAY_MIN_FOR_CLUSTERING: int = 5

# Self-kernel floor: a parlay whose 1ᵀC1 is at/below this is treated as having
# no usable correlation signal (distance 1 to everything) instead of dividing
# by ~0.
_KERNEL_SELF_FLOOR: float = 1e-9


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
        if sz <= _POWER_MAX_SIZE:
            curve[sz] = [float(mult), 0.0]
    for sz_str, row in flex.items():
        sz = int(sz_str)
        if sz > _POWER_MAX_SIZE:
            curve[sz] = [float(v) for v in row]
    return curve


def _payout_curve_for(
    platform: str,
    contest_variant: Literal["pooled", "power", "flex", "insurance", "rivals"],
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


def _nearest_psd(sigma: np.ndarray, tol: float = _PSD_EIG_TOLERANCE) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD via eigenvalue clipping.

    Symmetrize, clip eigenvalues at ``tol``, then rescale the diagonal back to
    1 so single-leg variances stay unit (the inputs are correlation matrices).

    Args:
        sigma: Symmetric ``(n, n)`` matrix from ``C[bet_id, bet_id]``.
        tol: Eigenvalue floor; matches the PSD acceptance threshold elsewhere.

    Returns:
        np.ndarray: PSD ``(n, n)`` matrix with unit diagonal.
    """
    sigma = (sigma + sigma.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, tol, None)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    diag_scale = 1.0 / np.sqrt(np.diag(repaired))
    return repaired * diag_scale[:, None] * diag_scale[None, :]


def assign_parlay_families(
    bet_ids: list[tuple[int, ...]],
    C: np.ndarray,
    *,
    max_families: int = _PARLAY_MAX_FAMILIES,
) -> np.ndarray:
    """Cluster a game's survivor parlays into independence families.

    Each parlay is the indicator vector ``1_p`` over its legs; parlay
    similarity is the RKHS cosine ``(1_pᵀ C 1_q) / sqrt((1_pᵀ C 1_p)·(1_qᵀ C
    1_q))`` — the diagonal enters numerator-self and denominator consistently,
    so it is a true bounded cosine (unlike the legacy mean-with-diagonal vs.
    mean-without). ``C`` is projected to the nearest PSD first so every
    self-kernel is non-negative and the cosine is well defined; non-finite
    distances are floored to 1. Families are formed by average-linkage
    hierarchical clustering and the cluster count is silhouette-selected in
    ``[1, max_families]`` — weak separation honestly collapses to one family.

    Args:
        bet_ids: One tuple of integer leg indices into ``C`` per parlay.
        C: Per-game leg×leg signed correlation matrix (unit diagonal).
        max_families: Upper bound on the auto-selected family count.

    Returns:
        np.ndarray: Integer labels ``1..k`` (length ``len(bet_ids)``).
    """
    n = len(bet_ids)
    single = np.ones(n, dtype=int)
    if n <= _PARLAY_MIN_FOR_CLUSTERING:
        return single

    Cpsd = _nearest_psd(np.asarray(C, dtype=float))
    idx = [np.asarray(b, dtype=int) for b in bet_ids]
    self_k = np.array([Cpsd[np.ix_(b, b)].sum() for b in idx])
    self_k = np.where(np.isfinite(self_k) & (self_k > _KERNEL_SELF_FLOOR), self_k, np.nan)

    dist = np.zeros((n, n))
    for i in range(n):
        bi = idx[i]
        for j in range(i + 1, n):
            denom = np.sqrt(self_k[i] * self_k[j])
            sim = Cpsd[np.ix_(bi, idx[j])].sum() / denom if denom > 0 else np.nan
            d = 1.0 - np.clip(sim, -1.0, 1.0)
            if not np.isfinite(d):
                d = 1.0
            dist[i, j] = dist[j, i] = d

    condensed = squareform(dist, checks=False)
    if not np.all(np.isfinite(condensed)):
        return single

    z = linkage(condensed, method="average")
    best_k, best_score = 1, -1.0
    for k in range(2, min(max_families, n - 1) + 1):
        labels = fcluster(z, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        # Ascending k with strict-improvement keeps the smallest k on ties.
        score = silhouette_score(dist, labels, metric="precomputed")
        if score > best_score:
            best_score, best_k = score, k

    if best_k >= 2 and best_score >= _PARLAY_MIN_SILHOUETTE:
        return fcluster(z, best_k, criterion="maxclust").astype(int)
    return single


def _expected_payout_with_pushes(
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

    return float(np.clip(boost, 0.0, _PAYOUT_CLIP_HI) * payouts.mean())


def beam_search_parlays(
    idx,
    EV,
    C,
    M,
    p_model,
    p_books,
    p_push,
    boosts,
    payouts,
    full_payouts,
    max_boost,
    bet_df,
    info,
    team,
    opp,
    *,
    contest_variant: Literal["pooled", "power", "flex", "insurance", "rivals"] = "pooled",
    legacy: bool = False,
):
    """Enumerate top parlay combinations via beam search.

    Extends parlays leg-by-leg up to ``len(payouts) + 1`` legs, keeping
    only the top ``_BEAM_WIDTH`` candidates at each size by geometric-mean EV.
    Full correlated-probability evaluation runs on the survivors.

    Args:
        idx: DataFrame of candidate legs (filtered from the game DataFrame).
        EV: Pairwise EV matrix for the full game DataFrame.
        C: Pairwise correlation matrix.
        M: Pairwise boost-modifier matrix.
        p_model: Model probability vector (chosen-side win probability).
        p_books: Books probability vector.
        p_push: Per-leg push probability (P(stat == line)). Zero for
            non-integer lines and for continuous-distribution markets.
        boosts: Boost multiplier vector.
        payouts: Search-list payout multipliers (length = max_size - 1),
            indexed by ``bet_size - 2``. Drives the ranking heuristic and the
            books-EV pre-check.
        full_payouts: Per-(size, miss-count) payout curve. Drives push-aware
            EV and the displayed ``Boost`` column.
        max_boost: Max allowed product boost for a parlay.
        bet_df: Dict of leg metadata keyed by DataFrame index.
        info: ``{Game, Date, League, Platform}`` metadata dict.
        team: Home team abbreviation.
        opp: Away team abbreviation.
        contest_variant: Underdog contest variant. Affects payout curve
            interpretation in :func:`_expected_payout_with_pushes`.
        legacy: When True, reproduce pre-2026.05 scoring (no PSD repair, no
            push-aware EV, bare modifier-product Boost in the output).

    Returns:
        list[dict]: Parlay candidate dicts ready for DataFrame construction.
    """
    K = _BEAM_WIDTH
    max_bet_size = len(payouts) + 1
    leg_indices = sorted(idx.index.to_numpy())
    leg_players = {i: bet_df[i]["Player"] for i in leg_indices}
    leg_teams = {i: bet_df[i]["Team"] for i in leg_indices}

    candidates = [(i,) for i in leg_indices]
    all_results = []

    for target_size in tqdm(
        range(2, max_bet_size + 1), desc=f"{info['League']}, {team}/{opp} Parlays", leave=False
    ):
        next_candidates = []

        for parlay in candidates:
            used_players = {leg_players[i] for i in parlay}
            last_idx = parlay[-1]

            for new_leg in leg_indices:
                if new_leg <= last_idx:
                    continue
                new_player = leg_players[new_leg]
                if new_player in used_players:
                    continue
                if any(new_player in p or p in new_player for p in used_players):
                    continue

                extended = (*parlay, new_leg)

                n_pairs = target_size * (target_size - 1) // 2
                ev_prod = np.prod(EV[np.ix_(extended, extended)][np.triu_indices(target_size, 1)])
                geo_mean = ev_prod ** (1 / n_pairs)
                if geo_mean < _PARLAY_GEO_MEAN_FLOOR:
                    continue

                next_candidates.append((extended, geo_mean))

        next_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = next_candidates[:K]

        payout_base = payouts[target_size - 2]

        for parlay, _ in top_candidates:
            bet_id = parlay
            bet_size = target_size

            covers_team = any(team in leg_teams[i] for i in bet_id)
            covers_opp = any(opp in leg_teams[i] for i in bet_id)
            if not (covers_team and covers_opp):
                continue

            boost = np.prod(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]) * np.prod(
                boosts[np.ix_(bet_id)]
            )
            if boost <= _MIN_PRODUCT_BOOST or boost > max_boost:
                continue

            pb = p_books[np.ix_(bet_id)]
            prev_pb = np.prod(pb) * boost * payout_base
            if prev_pb < _BOOKS_EV_FLOOR:
                continue

            p = p_model[np.ix_(bet_id)]
            prev_p = np.prod(p) * boost * payout_base
            if prev_p < _MODEL_EV_PRECHECK_FLOOR:
                continue

            SIG = C[np.ix_(bet_id, bet_id)]
            min_eig = np.min(np.linalg.eigvalsh(SIG))
            if legacy:
                # Legacy: drop non-PSD subsets instead of repairing.
                if min_eig < _PSD_EIG_TOLERANCE:
                    continue
            elif min_eig < _PSD_EIG_TOLERANCE:
                SIG = _nearest_psd(SIG)

            payout = np.clip(payout_base * boost, _PAYOUT_CLIP_LO, _PAYOUT_CLIP_HI)

            push_legs = p_push[np.ix_(bet_id)]
            has_pushes = bool(np.any(push_legs > _PUSH_PROB_FLOOR))

            # Curves with payouts at multiple miss-counts (e.g. Underdog flex
            # and insurance) need the MC path even with zero pushes — the
            # analytical mvn.cdf only gives P(all hit), which discards the
            # partial-hit tiers. Pure all-hit curves (power, rivals, legacy
            # platform tables) keep the fast analytical path.
            curve = full_payouts.get(bet_size, [payout_base, 0.0])
            multi_tier = sum(1 for v in curve if v > 0) > 1

            if (has_pushes or multi_tier) and not legacy:
                p = _expected_payout_with_pushes(
                    p,
                    push_legs,
                    SIG,
                    bet_size,
                    boost=boost,
                    payout_curve=full_payouts,
                )
            else:
                p = payout * multivariate_normal.cdf(norm.ppf(p), np.zeros(bet_size), SIG)

            pb = p / prev_p * prev_pb
            units = (p - 1) / (payout - 1) / _KELLY_BANKROLL_FRACTION

            if units < _KELLY_UNITS_FLOOR or p < _MODEL_EV_FINAL_FLOOR or pb < _BOOKS_EV_FLOOR:
                continue

            bet = itemgetter(*bet_id)(bet_df)
            # Display Boost: under legacy, return the bare modifier product
            # (the post-search line-498 overwrite multiplies by per-size
            # payout). Under the new path, return the payout-inclusive value
            # so the column reflects the same EV that drove ranking.
            display_boost = boost if legacy else payout
            parlay_dict = info | {
                "Model EV": p,
                "Books EV": pb,
                "Boost": display_boost,
                "Rec Bet": units,
                "Leg 1": "",
                "Leg 2": "",
                "Leg 3": "",
                "Leg 4": "",
                "Leg 5": "",
                "Leg 6": "",
                "Legs": ", ".join([leg["Desc"] for leg in bet]),
                "Bet ID": bet_id,
                "P": prev_p,
                "PB": prev_pb,
                "Fun": np.sum(
                    [
                        3
                        - (
                            np.abs(leg["Line"])
                            / stat_std.get(info["League"], {}).get(leg["Market"], 1)
                        )
                        if ("H2H" in leg["Desc"])
                        else 2
                        - 1 / stat_cv.get(info["League"], {}).get(leg["Market"], 1)
                        + leg["Line"] / stat_std.get(info["League"], {}).get(leg["Market"], 1)
                        for leg in bet
                        if (leg["Bet"] == "Over") or ("H2H" in leg["Desc"])
                    ]
                ),
                "Bet Size": bet_size,
                "Leg Probs": tuple(bet_df[i]["Model P"] for i in bet_id),
                "Corr Pairs": tuple(SIG[np.triu_indices(bet_size, 1)]),
                "Boost Pairs": tuple(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]),
                "Indep P": float(np.prod(p_model[np.ix_(bet_id)]) * payout),
                "Indep PB": float(np.prod(p_books[np.ix_(bet_id)]) * payout),
            }
            for j in range(bet_size):
                parlay_dict["Leg " + str(j + 1)] = bet[j]["Desc"]

            all_results.append(parlay_dict)

        candidates = [p for p, _ in top_candidates]

    return all_results
