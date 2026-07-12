"""Cross-game candidate builder for the simulated-bettor ledger (Policy v1).

Builds Underdog Pick'em combinations that span >=2 distinct games directly
from already-scored single-leg offers, under an explicit independence
(rho=0) assumption between legs from different games. The existing
same-game parlay search (``prediction.parlay`` / ``prediction.correlation``)
structurally cannot produce these -- it only ever combines legs sharing one
``Game`` -- so this module is a deliberately separate, independence-assumed
path, not a duplicate of that search. See ``docs/handoffs/sim-bettor-ledger.md``
for the surrounding Policy v1 design.
"""

from __future__ import annotations

import datetime
import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

from sportstradamus.helpers import stat_map
from sportstradamus.leg_schema import build_leg, leg_label
from sportstradamus.prediction.payouts import expected_payout_with_pushes, payout_curve_for
from sportstradamus.strategies._ledger_selection import (
    BANKROLL_PER_REPLICATE,
    LedgerCandidate,
    _entropy_from,
)
from sportstradamus.strategies.kelly import fractional_kelly_stake
from sportstradamus.strategies.underdog_pickem import (
    PickemConfig,
    filter_legs,
    resolve_market_shrinkage,
)

_CROSS_GAME_BEAM_WIDTH: int = 200  # v1 default, narrower than the same-game search's beam
_MAX_ENTRY_SIZE: int = 6

_, _POOLED_CURVE = payout_curve_for("Underdog", "pooled", legacy=False)


@dataclass(frozen=True)
class _ScoredLeg:
    idx: int
    player: str
    game: str
    win_prob: float  # shrinkage-adjusted: 0.5 + (raw_p - 0.5) * shrinkage
    push_prob: float
    book_devig: float  # "Market Prob" column passthrough
    line: float
    boost: float
    display: str  # leg_label(leg) -- canonical rendering, do not reformat it yourself
    canonical_leg: dict  # build_leg() output, with leg["stat"] patched to the canonical market


def _score_legs(eligible: pd.DataFrame) -> list[_ScoredLeg]:
    out = []
    for i, row in eligible.reset_index(drop=True).iterrows():
        leg = build_leg(row)
        canonical_market = stat_map["Underdog"].get(row["Market"])
        if canonical_market:
            leg["stat"] = canonical_market
        shrinkage, _source = resolve_market_shrinkage(str(row["League"]), canonical_market)
        adj_p = 0.5 + (leg["win_prob"] - 0.5) * max(0.0, min(1.0, shrinkage))
        out.append(
            _ScoredLeg(
                idx=i,
                player=leg["player"],
                game=leg["game"],
                win_prob=adj_p,
                push_prob=leg["push_prob"],
                book_devig=float(row["Market Prob"]),
                line=leg["line"],
                boost=leg["boost"],
                display=leg_label(leg),
                canonical_leg=leg,
            )
        )
    return out


def _expand(legs: list[_ScoredLeg], beam: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    by_idx = {leg.idx: leg for leg in legs}
    scored: list[tuple[tuple[int, ...], float]] = []
    for combo in beam:
        used_players = {by_idx[i].player for i in combo}
        last_idx = combo[-1]
        for leg in legs:
            if leg.idx <= last_idx or leg.player in used_players:
                continue
            if any(leg.player in p or p in leg.player for p in used_players):
                continue  # guards against "A. Player" vs "A. Player Jr." collisions
            extended = (*combo, leg.idx)
            if len({by_idx[i].game for i in extended}) < 2:
                continue  # single-game combos are the same-game search's job
            heuristic = math.prod(
                by_idx[i].win_prob for i in extended
            )  # cheap beam-pruning rank only
            scored.append((extended, heuristic))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [combo for combo, _ in scored[:_CROSS_GAME_BEAM_WIDTH]]


def _enumerate_cross_game_combos(legs: list[_ScoredLeg]) -> dict[int, list[tuple[int, ...]]]:
    beam = [(leg.idx,) for leg in legs]
    by_size: dict[int, list[tuple[int, ...]]] = {}
    for size in range(2, _MAX_ENTRY_SIZE + 1):
        beam = _expand(legs, beam)
        by_size[size] = beam
    return by_size


def _pricing_rng(date: datetime.date, run_slot: str) -> np.random.Generator:
    seed_seq = np.random.SeedSequence(_entropy_from(date, run_slot, salt="universe_pricing"))
    return np.random.default_rng(seed_seq)


def _price_combo(legs: tuple[_ScoredLeg, ...], rng: np.random.Generator) -> tuple[float, float]:
    """Returns (joint_prob, model_ev). joint_prob is the plain product of
    shrinkage-adjusted win probs (the "all legs hit" probability -- correct on
    its own, but NOT sufficient to price flex's partial-payout tiers, which is
    why model_ev goes through the push-aware pooled-curve pricer instead of
    joint_prob * curve[0]).
    """
    n = len(legs)
    p_win = np.array([leg.win_prob for leg in legs])
    p_push = np.array([leg.push_prob for leg in legs])
    boost = math.prod(leg.boost for leg in legs)
    joint_prob = float(np.prod(p_win))
    ev_payout = expected_payout_with_pushes(
        p_win=p_win,
        p_push=p_push,
        sigma=np.eye(n),
        bet_size=n,
        boost=boost,
        payout_curve=_POOLED_CURVE,
        rng=rng,
    )
    return joint_prob, ev_payout


def build_cross_game_candidates(
    offers_df: pd.DataFrame, config: PickemConfig, date: datetime.date, run_slot: str
) -> list[LedgerCandidate]:
    """Build independence-assumed (rho=0) parlay candidates spanning >=2 games.

    Filters ``offers_df`` through the same leg gate ``construct_entries`` uses,
    beam-expands cross-game combos up to ``_MAX_ENTRY_SIZE`` legs, prices each
    via the pooled payout curve, and drops anything below ``config.min_ev``.
    See the module docstring for why this independence assumption is safe.

    Returns:
        One :class:`LedgerCandidate` per surviving combo, in no particular
        order (the caller's selection layer does the ranking/sampling).
    """
    eligible = filter_legs(offers_df, config)
    if eligible.empty:
        return []
    legs = _score_legs(eligible)
    rng = _pricing_rng(date, run_slot)
    by_size = _enumerate_cross_game_combos(legs)
    by_idx = {leg.idx: leg for leg in legs}

    out: list[LedgerCandidate] = []
    for size, combos in by_size.items():
        payout_mult = _POOLED_CURVE[size][0]
        for combo in combos:
            combo_legs = tuple(by_idx[i] for i in combo)
            joint_prob, ev_payout = _price_combo(combo_legs, rng)
            ev = ev_payout - 1.0
            if ev < config.min_ev:
                continue
            stake = fractional_kelly_stake(
                bankroll=BANKROLL_PER_REPLICATE,
                win_prob=joint_prob,
                payout_multiplier=Decimal(repr(payout_mult)),
                fraction=config.kelly_fraction,
                max_fraction_of_bankroll=config.max_stake_pct_bankroll,
            )
            games = {leg.game for leg in combo_legs}
            out.append(
                LedgerCandidate(
                    id=hashlib.sha1(
                        "|".join(sorted(leg.display for leg in combo_legs)).encode()
                    ).hexdigest()[:16],
                    contest_variant="power" if size <= 3 else "flex",
                    entry_size=size,
                    legs=tuple(leg.display for leg in combo_legs),
                    canonical_legs=tuple(leg.canonical_leg for leg in combo_legs),
                    players=frozenset(leg.player for leg in combo_legs),
                    game_span=len(games),
                    joint_prob=joint_prob,
                    payout_multiplier=payout_mult,
                    ev=ev,
                    stake=stake,
                    lines=tuple(leg.line for leg in combo_legs),
                    model_probs=tuple(leg.win_prob for leg in combo_legs),
                    book_devig=tuple(leg.book_devig for leg in combo_legs),
                )
            )
    return out
