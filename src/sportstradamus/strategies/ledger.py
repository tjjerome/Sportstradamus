"""Twice-daily commit orchestrator for the simulated-bettor ledger (Policy v1).

Implements ``docs/handoffs/sim-bettor-ledger.md`` §10: builds the shared
same-game + cross-game candidate universe once per run, then draws and sizes
entries for 3 personas × 40 Monte-Carlo replicates and appends them to the
append-only JSONL ledger. Pure orchestration over
:mod:`sportstradamus.strategies._ledger_selection`,
:mod:`sportstradamus.strategies._ledger_cross_game`, and
:mod:`sportstradamus.strategies._ledger_store` — no selection/pricing logic
lives here.
"""

from __future__ import annotations

import dataclasses
import datetime
from decimal import Decimal

import click

from sportstradamus.helpers.logging import get_logger
from sportstradamus.helpers.provenance import git_sha
from sportstradamus.strategies import _ledger_cross_game, _ledger_selection, _ledger_store
from sportstradamus.strategies.kelly import KellyCandidate, joint_kelly_portfolio
from sportstradamus.strategies.underdog_pickem import PickemConfig, construct_entries, live_load

_logger = get_logger("ledger-commit")

POLICY_VERSION = "policy_v1"
BANKROLL_PER_REPLICATE = Decimal("5000")

_SHARED_CONFIG = PickemConfig(
    entry_sizes=(2, 3, 4, 5, 6),
    contest_variants=("power", "flex"),
    min_ev=0.05,
    kelly_fraction=0.25,
    max_stake_pct_bankroll=0.005,
    top_k=100,
    max_overlap=2,
)
_POWER_SIZES = frozenset({2, 3})
_FLEX_SIZES = frozenset({4, 5, 6})


def _partition_by_size_rule(
    candidates: list[_ledger_selection.LedgerCandidate],
) -> list[_ledger_selection.LedgerCandidate]:
    """§10's post-hoc partition on construct_entries's output, which searches
    BOTH variants across the full entry_sizes tuple and needs this to discard
    invalid combinations (e.g. a 5-leg "power" search hit). The cross-game
    builder does NOT need this call -- its pooled payout curve already
    encodes the same partition internally, so it never generates a combo this
    would discard in the first place. Only apply this to construct_entries's
    output, not to build_cross_game_candidates's output.
    """
    return [
        c
        for c in candidates
        if (c.contest_variant == "power" and c.entry_size in _POWER_SIZES)
        or (c.contest_variant == "flex" and c.entry_size in _FLEX_SIZES)
    ]


def build_candidate_universe(
    date: datetime.date, run_slot: str
) -> list[_ledger_selection.LedgerCandidate]:
    """Exactly ONE live scrape (via live_load), reused by both candidate
    sources. Do not call construct_entries without parlay_dfs/offers_df here
    -- that would trigger a second, redundant scrape.
    """
    parlay_dfs, offers_df = live_load(_SHARED_CONFIG)
    same_game = _partition_by_size_rule(
        [
            _ledger_selection.from_recommended_entry(e)
            for e in construct_entries(
                date,
                BANKROLL_PER_REPLICATE,
                _SHARED_CONFIG,
                parlay_dfs=parlay_dfs,
                offers_df=offers_df,
            )
        ]
    )
    cross_game = _ledger_cross_game.build_cross_game_candidates(
        offers_df, _SHARED_CONFIG, date, run_slot
    )
    return same_game + cross_game


def _committed_record(
    candidate: _ledger_selection.LedgerCandidate,
    *,
    date: datetime.date,
    run_slot: str,
    persona: str,
    replicate_id: int,
) -> dict:
    """Build one JSONL record matching ``_ledger_store``'s expected schema.

    This is the ONLY place that constructs these records — ``_ledger_store``
    itself treats them as opaque dicts.
    """
    return {
        "id": candidate.id,
        "legs": list(candidate.legs),
        "legs_players": sorted(candidate.players),
        "lines": list(candidate.lines),
        "model_probs": list(candidate.model_probs),
        "book_devig": list(candidate.book_devig),
        "stake": str(candidate.stake),
        "policy_version": POLICY_VERSION,
        "git_sha": git_sha(),
        "committed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "persona": persona,
        "run_slot": run_slot,
        "replicate_id": replicate_id,
        "game_span": candidate.game_span,
        "contest_variant": candidate.contest_variant,
        "entry_size": candidate.entry_size,
        "joint_prob": candidate.joint_prob,
        "payout_multiplier": candidate.payout_multiplier,
        "ev": candidate.ev,
        "date": date.isoformat(),
    }


def _resize_kelly_growth(
    drawn: list[_ledger_selection.LedgerCandidate], date: datetime.date, replicate_id: int
) -> list[_ledger_selection.LedgerCandidate]:
    """Re-solve ``joint_kelly_portfolio`` over the already-Jaccard-de-overlapped
    ``drawn`` candidates, budgeted by the day's remaining Kelly fraction.

    Drops any candidate the portfolio solve zeroed out or omitted entirely --
    expected behavior for this persona, not a bug to work around.
    """
    remaining_fraction = _ledger_selection.remaining_kelly_fraction(date, replicate_id)
    if remaining_fraction <= 0.0:
        return []

    kelly_candidates = [
        KellyCandidate(
            bet_id=c.id, win_prob=c.joint_prob, payout_multiplier=Decimal(repr(c.payout_multiplier))
        )
        for c in drawn
    ]
    stakes = joint_kelly_portfolio(
        BANKROLL_PER_REPLICATE, kelly_candidates, fraction=remaining_fraction
    )
    return [dataclasses.replace(c, stake=stakes[c.id]) for c in drawn if c.id in stakes]


def run_commit(date: datetime.date, run_slot: str) -> int:
    """Build candidates, draw + size per persona/replicate, and append.

    Returns the count of newly-appended records (0 on an empty candidate
    universe or when a re-run finds nothing new to commit).
    """
    if run_slot not in _ledger_selection.RUN_SLOTS:
        msg = f"run_slot must be one of {_ledger_selection.RUN_SLOTS}, got {run_slot!r}"
        raise ValueError(msg)

    universe = build_candidate_universe(date, run_slot)
    if not universe:
        _logger.info(
            "empty candidate universe for %s %s -- recording as an empty decision, not a failure",
            date.isoformat(),
            run_slot,
        )
        return 0

    rngs = _ledger_selection.replicate_rngs(date, run_slot)
    total = 0
    for replicate_id in range(_ledger_selection.LEDGER_REPLICATES):
        rng = rngs[replicate_id]
        records: list[dict] = []
        for persona in _ledger_selection.PERSONAS:
            remaining, seen = _ledger_selection.remaining_budget_and_seen_players(
                date, persona, replicate_id
            )
            if remaining <= 0:
                continue
            drawn = _ledger_selection.draw_entries(
                universe, _ledger_selection.PERSONA_SCORERS[persona], seen, remaining, rng
            )
            if not drawn:
                continue
            if persona == "kelly_growth":
                drawn = _resize_kelly_growth(drawn, date, replicate_id)
            already_ids = _ledger_store.already_committed_ids(date, run_slot, persona, replicate_id)
            records.extend(
                _committed_record(
                    c, date=date, run_slot=run_slot, persona=persona, replicate_id=replicate_id
                )
                for c in drawn
                if c.id not in already_ids
            )
        total += _ledger_store.append_entries(date, records)
    return total


@click.command()
@click.option("--run-slot", type=click.Choice(_ledger_selection.RUN_SLOTS), required=True)
@click.option("--date", default="today")
def ledger_commit(run_slot: str, date: str) -> None:
    """Twice-daily policy_v1 commit: build candidates, draw, size, append."""
    slate_date = datetime.date.today() if date == "today" else datetime.date.fromisoformat(date)
    appended = run_commit(slate_date, run_slot)
    click.echo(
        f"committed {appended} new entries -> data/ledger/entries/{slate_date.isoformat()}.jsonl"
    )


if __name__ == "__main__":
    ledger_commit()
