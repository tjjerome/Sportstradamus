"""Beam-search parlay enumeration and push-aware EV evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter
from typing import Literal

import numpy as np
from tqdm import tqdm

from sportstradamus.analysis import _leg_market_map
from sportstradamus.helpers import stat_cv, stat_std
from sportstradamus.leg_schema import build_leg
from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.payouts import PAYOUT_CLIP_HI, PAYOUT_CLIP_LO


@dataclass(frozen=True)
class GameArrays:
    """Per-game leg×leg and per-leg numeric grids for one matchup.

    Built once by ``correlation._build_correlation_matrices`` and consumed
    read-only by correlation-column annotation and parlay beam search.
    """

    C: np.ndarray  # leg×leg correlation
    M: np.ndarray  # leg×leg boost modifier
    EV: np.ndarray  # leg×leg model pairwise EV
    EVb: np.ndarray  # leg×leg book pairwise EV
    V: np.ndarray  # model std-dev outer product
    p_model: np.ndarray
    p_books: np.ndarray
    p_push: np.ndarray
    boosts: np.ndarray


@dataclass(frozen=True)
class GameScoringContext:
    """One game's scoring bundle, captured so an arbitrary leg-subset can be priced.

    ``find_correlation`` builds the per-game :class:`GameArrays` ``g`` and the
    payout tables for the beam search; the story-menu generator reuses the same
    objects to score story subsets that beam search never enumerates (it only
    keeps both-team parlays). ``leg_indices`` are positions into ``g``'s matrices
    (the bet-eligible legs); ``bet_df`` maps the same positions to their offer
    records (``Desc``/``Model``/``Bet``/``Team``/…). Collected per game into the
    opt-in ``story_sink``; one context per ``(platform, game)``.
    """

    platform: str
    league: str
    game: str
    date: str
    g: GameArrays
    bet_df: dict
    leg_indices: tuple[int, ...]
    full_payouts: dict[int, list[float]]
    payout_base_by_size: dict[int, float]
    max_size: int


# --- Beam-search constants --------------------------------------------------

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

# Kelly sizing denominator: 5% bankroll per unit. Legacy.
_KELLY_BANKROLL_FRACTION: float = 0.05


def _expand_candidates(candidates, leg_indices, leg_players, EV, target_size, k):
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
    return [parlay for parlay, _ in next_candidates[:k]]


def _parlay_admissible(bet_id, leg_teams, team, opp, M, boosts, bet_size, max_boost):
    covers_team = any(team in leg_teams[i] for i in bet_id)
    covers_opp = any(opp in leg_teams[i] for i in bet_id)
    if not (covers_team and covers_opp):
        return None
    boost = np.prod(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]) * np.prod(
        boosts[np.ix_(bet_id)]
    )
    if boost <= _MIN_PRODUCT_BOOST or boost > max_boost:
        return None
    return boost


def _parlay_fun(bet, league):
    """Heuristic 'fun' score: line distance in std units, only Over / H2H legs count."""
    return np.sum(
        [
            3 - (np.abs(leg["Line"]) / stat_std.get(league, {}).get(leg["Market"], 1))
            if ("H2H" in leg["Market"])
            else 2
            - 1 / stat_cv.get(league, {}).get(leg["Market"], 1)
            + leg["Line"] / stat_std.get(league, {}).get(leg["Market"], 1)
            for leg in bet
            if (leg["Bet"] == "Over") or ("H2H" in leg["Market"])
        ]
    )


def resolve_leg_stat(market: str, new_map: dict) -> str:
    """Gamelog stat key for a leg's display ``Market``, mirroring the ``cMarket`` idiom."""
    stripped = market.replace("H2H ", "")
    return new_map.get(stripped, stripped)


def _evaluate_parlay(
    bet_id,
    bet_size,
    payout_base,
    g,
    full_payouts,
    max_boost,
    bet_df,
    info,
    team,
    opp,
    leg_teams,
    legacy,
    new_map,
):
    """Score one parlay candidate; return its row dict, or None if it fails a gate."""
    C, M, p_model, p_books, p_push, boosts = g.C, g.M, g.p_model, g.p_books, g.p_push, g.boosts
    boost = _parlay_admissible(bet_id, leg_teams, team, opp, M, boosts, bet_size, max_boost)
    if boost is None:
        return None

    pb = p_books[np.ix_(bet_id)]
    prev_pb = np.prod(pb) * boost * payout_base
    if prev_pb < _BOOKS_EV_FLOOR:
        return None

    p = p_model[np.ix_(bet_id)]
    prev_p = np.prod(p) * boost * payout_base
    if prev_p < _MODEL_EV_PRECHECK_FLOOR:
        return None

    SIG = psd_or_none(C[np.ix_(bet_id, bet_id)], legacy)
    if SIG is None:
        return None

    payout = np.clip(payout_base * boost, PAYOUT_CLIP_LO, PAYOUT_CLIP_HI)
    p = parlay_payout_prob(
        p, p_push[np.ix_(bet_id)], SIG, bet_size, boost, payout, full_payouts, payout_base, legacy
    )
    pb = p / prev_p * prev_pb
    units = (p - 1) / (payout - 1) / _KELLY_BANKROLL_FRACTION

    if units < _KELLY_UNITS_FLOOR or p < _MODEL_EV_FINAL_FLOOR or pb < _BOOKS_EV_FLOOR:
        return None

    bet = itemgetter(*bet_id)(bet_df)
    # Display Boost: under legacy, the bare modifier product (the post-search
    # line-498 overwrite multiplies by per-size payout); otherwise the
    # payout-inclusive value so the column matches the EV that drove ranking.
    display_boost = boost if legacy else payout
    legs = [
        build_leg(
            {
                **leg,
                "Platform": info["Platform"],
                "Stat": resolve_leg_stat(leg["Market"], new_map),
            }
        )
        for leg in bet
    ]
    return info | {
        "Model EV": p,
        "Market EV": pb,
        "Boost": display_boost,
        "Rec Bet": units,
        "legs": legs,
        "Bet ID": bet_id,
        "P": prev_p,
        "PB": prev_pb,
        # Transient — used only for the Fun sort in _append_parlay_rows and
        # dropped by persist._PARLAY_DROP_COLS before current_parlays.parquet.
        "Fun": _parlay_fun(bet, info["League"]),
        "Bet Size": bet_size,
        "Corr Pairs": tuple(SIG[np.triu_indices(bet_size, 1)]),
        "Boost Pairs": tuple(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]),
        "Indep P": float(np.prod(p_model[np.ix_(bet_id)]) * payout),
        "Indep PB": float(np.prod(p_books[np.ix_(bet_id)]) * payout),
    }


def beam_search_parlays(
    idx,
    g,
    payouts,
    full_payouts,
    max_boost,
    bet_df,
    info,
    team,
    opp,
    stat_map,
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
        g: Per-game leg arrays (correlation/boost matrices, EV grid, per-leg
            model/book/push probabilities and boosts). See :class:`GameArrays`.
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
        stat_map: ``{platform: {display_market: gamelog_stat}}`` — resolved once
            per game into the league-specific leg ``Stat`` lookup via
            :func:`sportstradamus.analysis._leg_market_map`.
        contest_variant: Underdog contest variant. Affects payout curve
            interpretation in :func:`payouts.expected_payout_with_pushes`.
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
    new_map = _leg_market_map(info["League"], info["Platform"], stat_map)

    candidates = [(i,) for i in leg_indices]
    all_results = []

    for target_size in tqdm(
        range(2, max_bet_size + 1), desc=f"{info['League']}, {team}/{opp} Parlays", leave=False
    ):
        top_candidates = _expand_candidates(
            candidates, leg_indices, leg_players, g.EV, target_size, K
        )
        payout_base = payouts[target_size - 2]
        for parlay in top_candidates:
            result = _evaluate_parlay(
                parlay,
                target_size,
                payout_base,
                g,
                full_payouts,
                max_boost,
                bet_df,
                info,
                team,
                opp,
                leg_teams,
                legacy,
                new_map,
            )
            if result is not None:
                all_results.append(result)
        candidates = top_candidates

    return all_results
