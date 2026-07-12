"""Pin RNG reproducibility, Jaccard-draw weighting, and budget arithmetic of
``_ledger_selection`` -- the selection/RNG/persona-scoring layer for the
simulated-bettor ledger (Policy v1, docs/handoffs/sim-bettor-ledger.md §10).
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np

from sportstradamus.strategies import _ledger_selection as sel

DATE = datetime.date(2026, 7, 12)


def _candidate(
    cand_id: str,
    players: frozenset[str],
    *,
    joint_prob: float = 0.5,
    ev: float = 0.1,
    stake: Decimal = Decimal("10"),
) -> sel.LedgerCandidate:
    return sel.LedgerCandidate(
        id=cand_id,
        contest_variant="power",
        entry_size=len(players),
        legs=tuple(f"{p} Over 4.5 rebounds - 60.0%, 1.6x" for p in players),
        players=players,
        game_span=1,
        joint_prob=joint_prob,
        payout_multiplier=3.0,
        ev=ev,
        stake=stake,
    )


# --- RNG reproducibility -------------------------------------------------------


def test_replicate_rngs_reproducible_across_separate_calls() -> None:
    first = sel.replicate_rngs(DATE, "morning")
    second = sel.replicate_rngs(DATE, "morning")

    assert len(first) == sel.LEDGER_REPLICATES
    assert len(second) == sel.LEDGER_REPLICATES
    for rng_a, rng_b in zip(first, second, strict=True):
        draws_a = rng_a.random(5)
        draws_b = rng_b.random(5)
        assert np.array_equal(draws_a, draws_b)


def test_replicate_rngs_differ_across_run_slots() -> None:
    morning = sel.replicate_rngs(DATE, "morning")
    afternoon = sel.replicate_rngs(DATE, "afternoon")

    morning_draws = morning[0].random(5)
    afternoon_draws = afternoon[0].random(5)
    assert not np.array_equal(morning_draws, afternoon_draws)


# --- draw_entries budget/empty-pool behavior -----------------------------------


def test_draw_entries_never_exceeds_budget() -> None:
    candidates = [_candidate(f"c{i}", frozenset({f"Player{i}"})) for i in range(20)]
    rng = np.random.default_rng(0)

    chosen = sel.draw_entries(candidates, sel.PERSONA_SCORERS["safe"], [], budget=5, rng=rng)

    assert len(chosen) <= 5


def test_draw_entries_zero_budget_returns_empty() -> None:
    candidates = [_candidate("c0", frozenset({"Player0"}))]
    rng = np.random.default_rng(0)

    assert sel.draw_entries(candidates, sel.PERSONA_SCORERS["safe"], [], budget=0, rng=rng) == []


def test_draw_entries_empty_candidates_returns_empty() -> None:
    rng = np.random.default_rng(0)

    assert sel.draw_entries([], sel.PERSONA_SCORERS["safe"], [], budget=5, rng=rng) == []


# --- _jaccard basic cases -------------------------------------------------------


def test_jaccard_disjoint_sets_is_zero() -> None:
    assert sel._jaccard(frozenset({"A"}), frozenset({"B"})) == 0.0


def test_jaccard_identical_nonempty_sets_is_one() -> None:
    assert sel._jaccard(frozenset({"A", "B"}), frozenset({"A", "B"})) == 1.0


def test_jaccard_empty_set_either_side_is_zero() -> None:
    assert sel._jaccard(frozenset(), frozenset({"A"})) == 0.0
    assert sel._jaccard(frozenset({"A"}), frozenset()) == 0.0
    assert sel._jaccard(frozenset(), frozenset()) == 0.0


# --- weight formula: overlap penalty is material -------------------------------


def test_overlapping_candidate_gets_lower_effective_weight_than_disjoint() -> None:
    overlapping = _candidate("dup", frozenset({"Player A", "Player B"}), joint_prob=0.5)
    disjoint = _candidate("new", frozenset({"Player C", "Player D"}), joint_prob=0.5)
    candidates = [overlapping, disjoint]
    already_chosen = [frozenset({"Player A", "Player B"})]

    percentiles = sel._percentile_scores(candidates, sel.PERSONA_SCORERS["safe"])

    def effective_weight(cand: sel.LedgerCandidate, pct: float) -> float:
        max_j = max((sel._jaccard(cand.players, s) for s in already_chosen), default=0.0)
        return pct * np.exp(-sel.K * max_j * (1.0 - pct))

    overlapping_weight = effective_weight(overlapping, percentiles[0])
    disjoint_weight = effective_weight(disjoint, percentiles[1])

    assert overlapping_weight < disjoint_weight


# --- remaining budget / seen players / kelly fraction --------------------------


def test_remaining_budget_and_seen_players_reads_committed_entries(monkeypatch) -> None:
    fake_records = [
        {"legs_players": ["Player A", "Player B"], "stake": "10"},
        {"legs_players": ["Player C"], "stake": "20"},
    ]
    monkeypatch.setattr(
        sel._ledger_store,
        "already_committed_entries",
        lambda date, persona, replicate_id: fake_records,
    )

    remaining, seen = sel.remaining_budget_and_seen_players(DATE, "safe", 0)

    assert remaining == sel.MAX_ENTRIES_PER_DAY - 2
    assert seen == [frozenset({"Player A", "Player B"}), frozenset({"Player C"})]


def test_remaining_budget_floors_at_zero_when_over_committed(monkeypatch) -> None:
    fake_records = [{"legs_players": [f"Player{i}"], "stake": "1"} for i in range(10)]
    monkeypatch.setattr(
        sel._ledger_store,
        "already_committed_entries",
        lambda date, persona, replicate_id: fake_records,
    )

    remaining, seen = sel.remaining_budget_and_seen_players(DATE, "safe", 0)

    assert remaining == 0
    assert len(seen) == 10


def test_remaining_kelly_fraction_subtracts_consumed_stake(monkeypatch) -> None:
    fake_records = [
        {"stake": "500"},
        {"stake": "250"},
    ]
    monkeypatch.setattr(
        sel._ledger_store,
        "already_committed_entries",
        lambda date, persona, replicate_id: fake_records,
    )

    fraction = sel.remaining_kelly_fraction(DATE, 0)

    consumed = (Decimal("500") + Decimal("250")) / sel.BANKROLL_PER_REPLICATE
    expected = float(Decimal(str(sel.KELLY_GROWTH_FRACTION)) - consumed)
    assert fraction == expected or abs(fraction - expected) < 1e-9


def test_remaining_kelly_fraction_floors_at_zero_when_overspent(monkeypatch) -> None:
    fake_records = [{"stake": "5000"}]
    monkeypatch.setattr(
        sel._ledger_store,
        "already_committed_entries",
        lambda date, persona, replicate_id: fake_records,
    )

    fraction = sel.remaining_kelly_fraction(DATE, 0)

    assert fraction == 0.0
