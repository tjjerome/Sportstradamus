"""Selection/RNG/persona-scoring layer for the simulated-bettor ledger (Policy v1).

Adapts same-game :class:`~sportstradamus.strategies.underdog_pickem.RecommendedEntry`
candidates (and, later, cross-game candidates from a sibling module) into one
common :class:`LedgerCandidate` shape, derives reproducible per-replicate RNG
streams, and runs the Jaccard-similarity-weighted sequential draw described in
``docs/handoffs/sim-bettor-ledger.md`` §10. No public API beyond what stage 1's
orchestrator (not built yet) needs; underscore-prefixed helpers are shared
within the ``strategies`` subpackage, not the outside world.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from sportstradamus.strategies import _ledger_store
from sportstradamus.strategies.underdog_pickem import RecommendedEntry, leg_player

LEDGER_REPLICATES: int = 40
PERSONAS: tuple[str, ...] = (
    "safe",
    "high_ev",
    "kelly_growth",
)  # fixed consumption order, load-bearing
RUN_SLOTS: tuple[str, ...] = ("morning", "afternoon")

K: float = 3.0  # Jaccard decay rate -- named v1 default, not fitted
DRAW_CONTINUE_PROB: float = 0.75  # stop-after-each-draw probability -- named v1 default

MAX_ENTRIES_PER_DAY: int = 5
KELLY_GROWTH_FRACTION: float = 0.25
BANKROLL_PER_REPLICATE: Decimal = Decimal("5000")


@dataclass(frozen=True)
class LedgerCandidate:
    """Unified shape for same-game (adapted from RecommendedEntry) and
    cross-game (built by a sibling module, _ledger_cross_game) candidates, so
    selection code never special-cases the source.
    """

    id: str
    contest_variant: str
    entry_size: int
    legs: tuple[str, ...]
    players: frozenset[str]
    game_span: int
    joint_prob: float
    payout_multiplier: float
    ev: float
    stake: Decimal
    lines: tuple[float, ...] = ()
    model_probs: tuple[float, ...] = ()
    book_devig: tuple[float, ...] = ()


def from_recommended_entry(entry: RecommendedEntry) -> LedgerCandidate:
    """extras['game'] is always a single game string -- game_span is always 1
    for this source (confirmed against RecommendedEntry's own construction:
    it always sets exactly one "TEAM1/TEAM2" key, never multi-game).
    """
    players = frozenset(leg_player(leg) for leg in entry.legs)
    return LedgerCandidate(
        id=entry.id,
        contest_variant=entry.contest_variant,
        entry_size=entry.entry_size,
        legs=entry.legs,
        players=players,
        game_span=1,
        joint_prob=entry.joint_prob,
        payout_multiplier=entry.payout_multiplier,
        ev=entry.ev,
        stake=entry.recommended_stake,
    )


def ev_per_dollar(ev: float, joint_prob: float, stake: Decimal) -> float:
    """Mirrors _pickem_emit.rank_and_dedupe's ranking key exactly (that file
    is untouched -- this is an independent, intentional re-derivation so two
    personas below can share one formula instead of two copies).
    """
    return (ev * joint_prob) / float(max(stake, Decimal("0.01")))


def _entropy_from(date: datetime.date, run_slot: str, salt: str = "replicates") -> int:
    """Deterministic entropy int from (date, run_slot[, salt]). sha256 (not
    Python's salted hash()) is what makes this reproducible across separate
    process invocations. `salt` lets a sibling module (_ledger_cross_game, not
    built yet) mint an independent, non-colliding sub-stream for its own
    Monte-Carlo pricing step under a different salt value -- that's why salt
    is a parameter here rather than hardcoded, even though every call site in
    THIS file uses the default.
    """
    key = f"{date.isoformat()}|{run_slot}|{salt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def replicate_rngs(date: datetime.date, run_slot: str) -> list[np.random.Generator]:
    """One stream per replicate_id (0-39), spawned fresh every call from a
    fresh SeedSequence -- this function keeps NO state across calls, so
    "same (date, run_slot) -> identical 40 children, every time" holds
    without needing a persistent parent object anywhere. Callers must pass
    the SAME Generator instance to all 3 personas for one replicate_id, in a
    fixed order -- this function only hands out streams, consumption order
    is the caller's responsibility (a different module, not built yet).
    """
    parent = np.random.SeedSequence(_entropy_from(date, run_slot))
    return [np.random.default_rng(child) for child in parent.spawn(LEDGER_REPLICATES)]


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _percentile_scores(candidates: list[LedgerCandidate], score_fn: Callable) -> np.ndarray:
    raw = np.array([score_fn(c) for c in candidates], dtype=float)
    ranks = np.empty_like(raw)
    ranks[np.argsort(raw)] = np.arange(len(raw))
    return ranks / max(len(raw) - 1, 1)


def draw_entries(
    candidates: list[LedgerCandidate],
    score_fn: Callable[[LedgerCandidate], float],
    already_chosen_players: list[frozenset[str]],
    budget: int,
    rng: np.random.Generator,
) -> list[LedgerCandidate]:
    """Sequential weighted sampling without replacement: normalize remaining
    pool weights, rng.choice, pop, repeat -- rhymes with this repo's
    strategies/profit_sim.py:_pick_day_bets (private, do not import it, this
    is an independent reimplementation of the same *shape*).

    Weight formula per candidate: percentile_score * exp(-K * max_jaccard *
    (1 - percentile_score)), where max_jaccard is against the most-similar
    entry in already_chosen_players union anything already picked THIS call.
    Stops at `budget`, when the pool empties, or (after each successful draw,
    if budget not yet reached and pool not empty) when a DRAW_CONTINUE_PROB
    coin flip fails -- so a full draw doesn't robotically fill budget every
    time.
    """
    if not candidates or budget <= 0:
        return []
    percentiles = _percentile_scores(candidates, score_fn)
    pool = list(range(len(candidates)))
    pool_pct = percentiles.copy()
    chosen: list[LedgerCandidate] = []
    chosen_players = list(already_chosen_players)

    while pool and len(chosen) < budget:
        weights = np.empty(len(pool))
        for i, p_idx in enumerate(pool):
            pct = pool_pct[i]
            max_j = max(
                (_jaccard(candidates[p_idx].players, s) for s in chosen_players), default=0.0
            )
            weights[i] = pct * np.exp(-K * max_j * (1.0 - pct))
        total = weights.sum()
        if total <= 0.0:
            break
        pick = rng.choice(len(pool), p=weights / total)
        picked = candidates[pool[pick]]
        chosen.append(picked)
        chosen_players.append(picked.players)
        pool.pop(pick)
        pool_pct = np.delete(pool_pct, pick)
        if pool and len(chosen) < budget and rng.random() >= DRAW_CONTINUE_PROB:
            break
    return chosen


# kelly_growth's scorer is a draw-order proxy only -- actual sizing for this
# persona comes from a portfolio solve elsewhere (ledger._resize_kelly_growth),
# not from this score.
PERSONA_SCORERS: dict[str, Callable[[LedgerCandidate], float]] = {
    "safe": lambda c: c.joint_prob,
    "high_ev": lambda c: ev_per_dollar(c.ev, c.joint_prob, c.stake),
    "kelly_growth": lambda c: ev_per_dollar(c.ev, c.joint_prob, c.stake),
}


def remaining_budget_and_seen_players(
    date: datetime.date, persona: str, replicate_id: int
) -> tuple[int, list[frozenset[str]]]:
    """Called unconditionally at the top of BOTH daily run_slots -- a no-op
    read for the morning run (nothing committed yet today), the real budget
    enforcement point for the afternoon run.
    """
    committed = _ledger_store.already_committed_entries(date, persona, replicate_id)
    remaining = max(MAX_ENTRIES_PER_DAY - len(committed), 0)
    seen = [frozenset(rec["legs_players"]) for rec in committed]
    return remaining, seen


def remaining_kelly_fraction(date: datetime.date, replicate_id: int) -> float:
    """Kelly-growth budget left today for this replicate, after prior commits.

    Reads only this persona's own committed stakes today (hardcoded
    ``"kelly_growth"``), never the shared per-persona entry budget
    :func:`remaining_budget_and_seen_players` tracks.
    """
    committed = _ledger_store.already_committed_entries(date, "kelly_growth", replicate_id)
    consumed = sum(Decimal(rec["stake"]) for rec in committed) / BANKROLL_PER_REPLICATE
    return max(KELLY_GROWTH_FRACTION - float(consumed), 0.0)
