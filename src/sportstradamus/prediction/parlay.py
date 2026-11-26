"""Beam-search parlay enumeration and push-aware EV evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter

import numpy as np
from tqdm import tqdm

from sportstradamus.helpers import cell_market, no_vig_odds, sleeper_payouts, stat_map
from sportstradamus.leg_schema import build_leg
from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.payouts import (
    PAYOUT_CLIP_HI,
    PAYOUT_CLIP_LO,
    SLEEPER_FLEX_MIN_SIZE,
    sleeper_flex_payout_curve,
)


@dataclass(frozen=True)
class GameArrays:
    """Per-game leg×leg and per-leg numeric grids for one matchup.

    Built once by ``correlation._build_correlation_matrices`` and consumed
    read-only by correlation-column annotation and parlay beam search.
    """

    C: np.ndarray  # leg×leg correlation
    M: np.ndarray  # leg×leg boost modifier
    EV: np.ndarray  # leg×leg pair value at the lead chance (fitted, else the model's)
    V: np.ndarray  # lead-chance std-dev outer product
    p_model: np.ndarray
    p_fit: np.ndarray  # the EV stamp's Fit Prob; nan on a leg the fit did not price
    p_books: np.ndarray
    p_push: np.ndarray
    boosts: np.ndarray
    opp_boost: np.ndarray  # opposite-side payout multiplier; nan unless Sleeper


@dataclass(frozen=True)
class GameScoringContext:
    """One game's scoring bundle, captured so an arbitrary leg-subset can be priced.

    ``find_correlation`` builds the per-game :class:`GameArrays` ``g`` and the
    payout tables for the beam search; the story-menu generator reuses the same
    objects to score story subsets that beam search never enumerates (it only
    keeps both-team parlays). ``leg_indices`` are positions into ``g``'s matrices
    (the legs of the game's pool); ``bet_df`` maps the same positions to their offer
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

# Leg sets carried from one entry size to the next. A size's admissible sets can sit hundreds
# deep behind one-team and over-boosted ones (a 250-wide beam priced no six-leg entry on a
# synthetic pool of boosted legs); at this width a game at the pool cap costs about 1.4 s.
_BEAM_WIDTH: int = 1000

# Boost-product gates: drop entries whose modifier product implies hard-banned
# pairs (low) or runaway promo stacking (high).
_MIN_PRODUCT_BOOST: float = 0.7

# Admissible sets exact-priced per entry size, from the head of the beam. Each costs a copula
# call of up to 20 ms, so with no value floor left this is what bounds a game's run time.
_EXACT_PRICED_PER_SIZE: int = 12

# Entries a game keeps per size: the exact-priced sets worth most at the fitted chances.
_KEPT_PER_SIZE: int = 3


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
            next_candidates.append((extended, ev_prod ** (1 / n_pairs)))
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


def leg_market_map(league: str, platform: str) -> dict[str, str]:
    """Each of ``platform``'s market labels mapped to ``league``'s own cell key."""
    return {label: cell_market(league, platform, label) for label in stat_map[platform]}


def resolve_leg_stat(market: str, stat_by_label: dict) -> str:
    """Gamelog stat key for a leg's display ``Market``: the league's own name, ``H2H`` dropped."""
    stripped = market.replace("H2H ", "")
    return stat_by_label.get(stripped, stripped)


def _is_sleeper_flex_candidate(info, bet_size):
    return info["Platform"] == "Sleeper" and bet_size >= SLEEPER_FLEX_MIN_SIZE


def _sleeper_flex_payout(g, bet_id, bet_size, payout_base, p, SIG):
    """Price a Sleeper Flex candidate: devig each leg, then K(n,k)/P_devigged.

    boost/M's running-product mechanism does not apply here -- devig already
    consumes the same per-leg multipliers as pricing input, so folding boost
    in on top would double-count that information (sleeper-parity §4b-iv).
    boost=1.0 passed to parlay_payout_prob is an inert signature placeholder.
    """
    boosts_leg = g.boosts[np.ix_(bet_id)]
    devigged_p = np.array(
        [
            no_vig_odds(float(over), float(under))[0]
            for over, under in zip(boosts_leg, g.opp_boost[np.ix_(bet_id)], strict=True)
        ]
    )
    flex_curve = sleeper_flex_payout_curve(devigged_p, sleeper_payouts["flex_k"])
    payout = flex_curve[bet_size][0]
    p = parlay_payout_prob(
        p, np.zeros(bet_size), SIG, bet_size, 1.0, payout, flex_curve, payout_base
    )
    return payout, p


def _leg_boost_payout(
    g,
    bet_id,
    bet_size,
    boost,
    payout,
    p,
    SIG,
    full_payouts,
    payout_base,
    full_refund_below_size,
):
    """Price a Max/Underdog candidate on its legs' own multipliers and the pair-modifier product."""
    boosts_leg = g.boosts[np.ix_(bet_id)]
    return parlay_payout_prob(
        p,
        g.p_push[np.ix_(bet_id)],
        SIG,
        bet_size,
        boosts_leg,
        payout,
        full_payouts,
        payout_base,
        full_refund_below_size=full_refund_below_size,
        # M's pairwise product, passed apart from the per-leg multipliers so a
        # Flex loss tier can rank them. Division is safe: _parlay_admissible
        # already gated boost > _MIN_PRODUCT_BOOST (0.7), so no boosts_leg
        # entry can be zero here without the candidate already rejected.
        pair_modifier=boost / np.prod(boosts_leg),
    )


def _price_parlay(
    g, p, bet_id, boost, sig, payout_base, info, full_payouts, full_refund_below_size
):
    """``(payout, value of a unit staked)`` for one admissible candidate at leg chances ``p``."""
    bet_size = len(bet_id)
    p_legs = p[np.ix_(bet_id)]
    if _is_sleeper_flex_candidate(info, bet_size):
        return _sleeper_flex_payout(g, bet_id, bet_size, payout_base, p_legs, sig)
    payout = np.clip(payout_base * boost, PAYOUT_CLIP_LO, PAYOUT_CLIP_HI)
    value = _leg_boost_payout(
        g,
        bet_id,
        bet_size,
        boost,
        payout,
        p_legs,
        sig,
        full_payouts,
        payout_base,
        full_refund_below_size,
    )
    return payout, value


def _parlay_row(
    g, bet_id, boost, sig, payout_base, payout, ev, model_ev, bet_df, info, stat_by_label
):
    """The board row of one kept entry: its fitted value beside the model's claim."""
    bet_size = len(bet_id)
    p_model, p_books = g.p_model[np.ix_(bet_id)], g.p_books[np.ix_(bet_id)]
    prev_p = np.prod(p_model) * boost * payout_base
    prev_pb = np.prod(p_books) * boost * payout_base
    legs = [
        build_leg(
            {
                **leg,
                "Platform": info["Platform"],
                "Stat": resolve_leg_stat(leg["Market"], stat_by_label),
            }
        )
        for leg in itemgetter(*bet_id)(bet_df)
    ]
    return info | {
        "EV": ev,
        "Model EV": model_ev,
        "Market EV": model_ev / prev_p * prev_pb,
        # Display Boost: the payout-inclusive value, so the column matches the
        # EV that drove ranking.
        "Boost": payout,
        # An entry gets no stake, and no "+EV" mark, until the entries' money record
        # supports one (docs/handoffs/profit-engine.md 5.3).
        "Rec Bet": 0.0,
        "legs": legs,
        "Bet ID": bet_id,
        "P": prev_p,
        "PB": prev_pb,
        "Bet Size": bet_size,
        "Corr Pairs": tuple(sig[np.triu_indices(bet_size, 1)]),
        "Boost Pairs": tuple(g.M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]),
        "Indep P": float(np.prod(p_model) * payout),
        "Indep PB": float(np.prod(p_books) * payout),
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
    *,
    full_refund_below_size: int | None = None,
):
    """Build a game's entries: a beam ranks leg sets, the fitted chances price the best.

    Extends leg sets one leg at a time up to ``len(payouts) + 1`` legs, carrying the
    ``_BEAM_WIDTH`` sets of each size with the highest geometric-mean pair value
    (``g.EV``). At each size the first ``_EXACT_PRICED_PER_SIZE`` admissible sets of the
    beam (a leg of each team, boost product in range) are priced on the fitted chances
    through the copula, and the ``_KEPT_PER_SIZE`` worth most become rows. No value floor
    applies: an entry's rank decides.

    Args:
        idx: DataFrame of candidate legs (filtered from the game DataFrame); every one
            carries a fitted chance.
        g: Per-game leg arrays (correlation/boost matrices, pair-value grid, per-leg
            model/fitted/book/push probabilities and boosts). See :class:`GameArrays`.
        payouts: Search-list payout multipliers (length = max_size - 1),
            indexed by ``bet_size - 2``.
        full_payouts: Per-(size, miss-count) payout curve. Drives push-aware
            EV and the displayed ``Boost`` column.
        max_boost: Max allowed product boost for a parlay.
        bet_df: Dict of leg metadata keyed by DataFrame index.
        info: ``{Game, Date, League, Platform}`` metadata dict.
        team: One of the game's two team abbreviations.
        opp: The other; a parlay must hold a leg of each.
        full_refund_below_size: Threaded to :func:`payouts.expected_payout_with_pushes`
            — entries at or below this size refund in full on any push
            (Sleeper's 2-pick divergence). ``None`` keeps the generic
            drop-and-reprice rule at every size.

    Returns:
        list[dict]: One row per kept entry, highest ``EV`` first. ``EV`` is the value of
        a unit staked at the fitted chances; ``Model EV`` is the same pricing at the
        model's own chances.
    """
    max_bet_size = len(payouts) + 1
    leg_indices = sorted(idx.index.to_numpy())
    leg_players = {i: bet_df[i]["Player"] for i in leg_indices}
    leg_teams = {i: bet_df[i]["Team"] for i in leg_indices}
    stat_by_label = leg_market_map(info["League"], info["Platform"])

    candidates = [(i,) for i in leg_indices]
    parlay_rows = []

    for target_size in tqdm(
        range(2, max_bet_size + 1), desc=f"{info['League']}, {team}/{opp} Parlays", leave=False
    ):
        candidates = _expand_candidates(
            candidates, leg_indices, leg_players, g.EV, target_size, _BEAM_WIDTH
        )
        payout_base = payouts[target_size - 2]
        if payout_base == 0:
            # A variant's list is zero-padded at the sizes it does not offer.
            continue
        pricing = (payout_base, info, full_payouts, full_refund_below_size)
        priced = []
        for bet_id in candidates:
            boost = _parlay_admissible(
                bet_id, leg_teams, team, opp, g.M, g.boosts, target_size, max_boost
            )
            if boost is None:
                continue
            sig = psd_or_none(g.C[np.ix_(bet_id, bet_id)])
            payout, ev = _price_parlay(g, g.p_fit, bet_id, boost, sig, *pricing)
            priced.append((ev, payout, bet_id, boost, sig))
            if len(priced) == _EXACT_PRICED_PER_SIZE:
                break
        priced.sort(key=itemgetter(0), reverse=True)
        for ev, payout, bet_id, boost, sig in priced[:_KEPT_PER_SIZE]:
            _, model_ev = _price_parlay(g, g.p_model, bet_id, boost, sig, *pricing)
            parlay_rows.append(
                _parlay_row(
                    g,
                    bet_id,
                    boost,
                    sig,
                    payout_base,
                    payout,
                    ev,
                    model_ev,
                    bet_df,
                    info,
                    stat_by_label,
                )
            )

    return sorted(parlay_rows, key=itemgetter("EV"), reverse=True)
