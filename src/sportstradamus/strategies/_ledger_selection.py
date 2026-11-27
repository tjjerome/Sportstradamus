"""Selection/RNG/persona-scoring layer for the simulated-bettor ledger.

Adapts same-game :class:`~sportstradamus.strategies.underdog_pickem.RecommendedEntry`
candidates into the :class:`LedgerCandidate` shape ``_ledger_cross_game``'s
builders also produce, derives reproducible per-replicate RNG streams, and runs
the Jaccard-similarity-weighted sequential draw described in
``docs/handoffs/sim-bettor-ledger.md`` §10. No public API beyond what the
orchestrator (``ledger``) needs; underscore-prefixed helpers are shared
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
from sportstradamus.strategies.kelly import MAX_FRACTION_OF_BANKROLL
from sportstradamus.strategies.underdog_pickem import RecommendedEntry, leg_player

# Ensemble size at the knee of the ensemble-mean convergence curve, which goes as 1 + 1/M
# (docs/handoffs/sim-bettor-ledger.md §10, Ensemble sizing).
LEDGER_REPLICATES: int = 40
# Fixed consumption order on each replicate's one random stream. A new persona goes
# last: it draws after the others, so their draws for a given seed never move.
PERSONAS: tuple[str, ...] = (
    "safe",
    "high_ev",
    "kelly_growth",
    "even_picks",
    "ev_picks",
)
RUN_SLOTS: tuple[str, ...] = ("morning", "afternoon")

K: float = 3.0  # Jaccard decay rate -- named v1 default, not fitted
DRAW_CONTINUE_PROB: float = 0.75  # chance of another draw after each one -- named v1 default

# Entries a persona may commit per replicate and day, across both run slots.
MAX_ENTRIES_PER_DAY: int = 5
# kelly_growth's Kelly budget for the day; the afternoon run solves for what the morning left.
KELLY_GROWTH_FRACTION: float = 0.25
# Every persona and replicate stakes against this fixed paper bankroll, never its settled balance.
BANKROLL_PER_REPLICATE: Decimal = Decimal("5000")
# What every persona but kelly_growth stakes on an entry: the cap v4's Kelly stakes were
# clipped at. Kelly on a fitted value under 1 stakes nothing, so it would record nothing.
FLAT_STAKE: Decimal = (BANKROLL_PER_REPLICATE * Decimal(repr(MAX_FRACTION_OF_BANKROLL))).quantize(
    Decimal("0.01")
)


@dataclass(frozen=True)
class LedgerCandidate:
    """A candidate entry, adapted from a same-game ``RecommendedEntry`` or built cross-game.

    One shape for both (cross-game candidates come from ``_ledger_cross_game``), so selection
    code never special-cases the source.
    """

    id: str
    contest_variant: str
    entry_size: int
    legs: tuple[str, ...]
    canonical_legs: tuple[dict, ...]
    players: frozenset[str]
    game_span: int
    joint_prob: float
    payout_multiplier: float
    ev: float
    stake: Decimal
    platform: str = "Underdog"
    lines: tuple[float, ...] = ()
    model_probs: tuple[float, ...] = ()
    book_devig: tuple[float, ...] = ()
    # Product of the entry's same-game pair modifiers. Each leg's own multiplier is the
    # raw ``boost`` on its ``canonical_legs`` entry; settlement reads both.
    pair_modifier: float = 1.0


def from_recommended_entry(entry: RecommendedEntry) -> LedgerCandidate:
    """A same-game pick'em entry as a ledger candidate, staked ``FLAT_STAKE``.

    Pick'em stakes no entry (``entry.recommended_stake`` is 0), so the stake is the ledger's
    own. ``game_span`` is 1: the same-game search builds an entry from one game's legs.
    """
    players = frozenset(leg_player(leg) for leg in entry.legs)
    return LedgerCandidate(
        id=entry.id,
        contest_variant=entry.contest_variant,
        entry_size=entry.entry_size,
        legs=entry.legs,
        canonical_legs=entry.canonical_legs,
        players=players,
        game_span=1,
        joint_prob=entry.joint_prob,
        payout_multiplier=entry.payout_multiplier,
        ev=entry.ev,
        stake=FLAT_STAKE,
        platform=entry.platform,
        pair_modifier=entry.pair_modifier,
    )


def _entropy_from(date: datetime.date, run_slot: str, salt: str = "replicates") -> int:
    """Deterministic entropy int from ``(date, run_slot, salt)``.

    sha256 (not Python's salted ``hash()``) is what makes this reproducible across separate
    process invocations. ``salt`` lets ``_ledger_cross_game`` mint independent, non-colliding
    sub-streams for its Monte-Carlo pricing and its deals -- that's why salt is a parameter
    here rather than hardcoded, even though every call site in THIS file uses the default.
    """
    key = f"{date.isoformat()}|{run_slot}|{salt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def replicate_rngs(date: datetime.date, run_slot: str) -> list[np.random.Generator]:
    """One random stream per replicate, spawned fresh from the (date, run_slot) entropy.

    The function keeps NO state across calls, so the same ``(date, run_slot)`` gives the same
    ``LEDGER_REPLICATES`` streams every time, without a persistent parent object anywhere.
    Callers must pass the SAME Generator to every persona for one replicate, in ``PERSONAS``
    order: this function only hands out streams, and consumption order is the caller's
    responsibility (``ledger.run_commit``).
    """
    parent = np.random.SeedSequence(_entropy_from(date, run_slot))
    return [np.random.default_rng(child) for child in parent.spawn(LEDGER_REPLICATES)]


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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
    """Draw up to ``budget`` entries by sequential weighted sampling without replacement.

    Each step normalizes the remaining pool weights, picks one with ``rng.choice``, pops it and
    repeats. That rhymes with this repo's ``strategies/profit_sim.py:_pick_day_bets`` (private,
    do not import it; this is an independent reimplementation of the same *shape*).

    Weight formula per candidate: ``percentile_score * exp(-K * max_jaccard * (1 -
    percentile_score))``, where ``max_jaccard`` is against the most-similar entry in
    ``already_chosen_players`` union anything already picked THIS call. Stops at ``budget``,
    when the pool empties, or (after each successful draw, if budget not yet reached and pool
    not empty) when a ``DRAW_CONTINUE_PROB`` coin flip fails -- so a full draw doesn't
    robotically fill the budget every time.
    """
    if not candidates or budget <= 0:
        return []
    pool = list(range(len(candidates)))
    pool_pct = _percentile_scores(candidates, score_fn)
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


# safe ranks by an entry's hit chance; every other persona by its fitted value, as pick'em
# ranks its entries (_pickem_emit.rank_and_dedupe). kelly_growth's rank is a draw order
# only: its stakes come from a portfolio solve (ledger._resize_kelly_growth). even_picks
# and ev_picks rank like high_ev; what sets each apart is its pool
# (ledger.build_candidate_universe): entries of Underdog's recommended even picks, and
# entries dealt evenly over each platform's picks.
PERSONA_SCORERS: dict[str, Callable[[LedgerCandidate], float]] = {
    "safe": lambda c: c.joint_prob,
    "high_ev": lambda c: c.ev,
    "kelly_growth": lambda c: c.ev,
    "even_picks": lambda c: c.ev,
    "ev_picks": lambda c: c.ev,
}


def remaining_budget_and_seen_players(
    date: datetime.date, persona: str, replicate_id: int
) -> tuple[int, list[frozenset[str]]]:
    """The entries ``persona`` may still draw today on this replicate, and its committed players.

    Called unconditionally at the top of BOTH daily run_slots -- a no-op read for the morning
    run (nothing committed yet today), the real budget enforcement point for the afternoon run.

    Returns:
        ``(remaining, seen)``: the entries left of ``MAX_ENTRIES_PER_DAY``, and each committed
        entry's player set, which the draw's overlap decay treats as already chosen.
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
