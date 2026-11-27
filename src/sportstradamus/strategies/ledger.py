"""Twice-daily commit orchestrator for the simulated-bettor ledger.

Implements ``docs/handoffs/sim-bettor-ledger.md`` §10. Each run builds the candidate pools
once: one live scrape per platform (Underdog and Sleeper) feeds a shared pool of same-game and
cross-game entries, and two personas get pools of their own -- ``even_picks``'s, cross-game
entries of the Underdog even picks the product recommends, and ``ev_picks``'s, cross-game
entries dealt evenly over each platform's picks. Every persona then draws from its pool on each
Monte-Carlo replicate, and the entries are appended to the append-only JSONL ledger. Every
persona but ``kelly_growth`` stakes ``_ledger_selection.FLAT_STAKE``; ``kelly_growth``'s draw is
re-sized by a portfolio solve.

Pure orchestration over :mod:`sportstradamus.strategies._ledger_selection`,
:mod:`sportstradamus.strategies._ledger_cross_game`, and
:mod:`sportstradamus.strategies._ledger_store`: no selection or pricing logic lives here.
"""

from __future__ import annotations

import dataclasses
import datetime
from decimal import Decimal

import click
import pandas as pd

from sportstradamus.helpers.logging import get_logger
from sportstradamus.helpers.provenance import git_sha
from sportstradamus.prediction.payouts import POWER_MAX_SIZE, UNDERDOG_FLEX_CAP
from sportstradamus.strategies import _ledger_cross_game, _ledger_selection, _ledger_store
from sportstradamus.strategies._ledger_selection import LedgerCandidate
from sportstradamus.strategies.kelly import KellyCandidate, joint_kelly_portfolio
from sportstradamus.strategies.underdog_pickem import PickemConfig, construct_entries, live_load

_logger = get_logger("ledger-commit")

# Names the rule a record was drawn and sized under; each version keeps its own paper
# bankrolls, so two rules are never pooled (docs/handoffs/sim-bettor-ledger.md §10).
POLICY_VERSION = "policy_v5"

# The variant split prediction.payouts prices: power takes 2 to POWER_MAX_SIZE legs, flex the
# sizes above it up to UNDERDOG_FLEX_CAP. Sleeper's Max/Flex split mirrors it.
_POWER_SIZES = tuple(range(2, POWER_MAX_SIZE + 1))
_FLEX_SIZES = tuple(range(POWER_MAX_SIZE + 1, UNDERDOG_FLEX_CAP + 1))
# Wider than the PickemConfig defaults ``bet pickem`` runs on: every size either variant pays,
# and a top_k deep enough that the cut ahead of the size partition keeps the valid entries
# (docs/handoffs/sim-bettor-ledger.md §10).
_SHARED_CONFIG = PickemConfig(
    entry_sizes=_POWER_SIZES + _FLEX_SIZES,
    contest_variants=("power", "flex"),
    top_k=100,
    max_overlap=2,
)
_PLATFORMS: tuple[str, ...] = ("Underdog", "Sleeper")


def _partition_by_size_rule(candidates: list[LedgerCandidate]) -> list[LedgerCandidate]:
    """Drop the entries whose size their variant does not pay, like a 5-leg ``power`` hit.

    ``construct_entries`` searches both variants over every size, so its output needs this. The
    cross-game and dealt builders set an entry's variant from its size, so theirs does not.
    """
    return [
        c
        for c in candidates
        if (c.contest_variant == "power" and c.entry_size in _POWER_SIZES)
        or (c.contest_variant == "flex" and c.entry_size in _FLEX_SIZES)
    ]


def _platform_candidates(
    parlay_dfs: dict[str, pd.DataFrame],
    offers_df: pd.DataFrame,
    date: datetime.date,
    run_slot: str,
    platform: str,
) -> list[LedgerCandidate]:
    """One platform's same-game entries off ``parlay_dfs`` plus cross-game combos of its picks.

    ``construct_entries`` gets ``parlay_dfs`` so it does not scrape the platform a second time.
    """
    entries = construct_entries(_SHARED_CONFIG, parlay_dfs=parlay_dfs, platform=platform)
    same_game = _partition_by_size_rule(
        [_ledger_selection.from_recommended_entry(e) for e in entries]
    )
    cross_game = _ledger_cross_game.build_cross_game_candidates(
        offers_df, date, run_slot, platform=platform
    )
    return same_game + cross_game


def build_candidate_universe(
    date: datetime.date, run_slot: str
) -> dict[str, list[LedgerCandidate]]:
    """Build each persona's candidate pool from one league load and one scrape per platform.

    ``live_load`` scrapes each platform once and every candidate builder reuses that scrape.
    Every persona shares one universe except ``even_picks`` and ``ev_picks``, whose pools are
    built beside it. ``even_picks``'s comes from Underdog's offers alone: cross-game entries of
    the even picks the product recommends on ``date``'s games, whether or not the board picks
    them. ``ev_picks``'s holds both platforms' picks on ``date``'s games, dealt into entries of
    one platform each. No pool has a value floor: the personas' ranks choose among its entries.

    Returns:
        The candidates each persona draws from, keyed by persona.
    """
    # Function-local, so a patch of sportstradamus.stats.load_live_leagues takes effect.
    from sportstradamus.stats import load_live_leagues

    # A league whose gamelog update failed is logged by the loader and sits this build
    # out; the board run of the same hour is what raises the alarm.
    stats, _ = load_live_leagues()
    universe: list[LedgerCandidate] = []
    even_picks_pool: list[LedgerCandidate] = []
    ev_picks_pool: list[LedgerCandidate] = []
    for platform in _PLATFORMS:
        parlay_dfs, offers_df = live_load(_SHARED_CONFIG, platform, stats)
        universe.extend(_platform_candidates(parlay_dfs, offers_df, date, run_slot, platform))
        ev_picks_pool.extend(
            _ledger_cross_game.build_ev_pick_candidates(offers_df, date, run_slot, platform)
        )
        if platform == "Underdog":
            even_picks_pool = _ledger_cross_game.build_even_pick_candidates(
                offers_df, date, run_slot
            )
    return dict.fromkeys(_ledger_selection.PERSONAS, universe) | {
        "even_picks": even_picks_pool,
        "ev_picks": ev_picks_pool,
    }


def _committed_record(
    candidate: LedgerCandidate,
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
        "canonical_legs": list(candidate.canonical_legs),
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
        "platform": candidate.platform,
        "pair_modifier": candidate.pair_modifier,
    }


def _resize_kelly_growth(
    drawn: list[LedgerCandidate], date: datetime.date, replicate_id: int
) -> list[LedgerCandidate]:
    """Size ``drawn`` by one ``joint_kelly_portfolio`` solve on the day's remaining Kelly fraction.

    The draw already de-overlapped the candidates. A candidate the solve zeroes out or omits is
    dropped -- expected for this persona, not a bug to work around.
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
        _ledger_selection.BANKROLL_PER_REPLICATE, kelly_candidates, fraction=remaining_fraction
    )
    return [dataclasses.replace(c, stake=stakes[c.id]) for c in drawn if c.id in stakes]


def run_commit(date: datetime.date, run_slot: str) -> int:
    """Build the pools, draw each persona's entries on every replicate, and append them.

    A replicate that already holds a record from ``run_slot`` has drawn in it
    and is left alone, so a re-run of the slot adds nothing to it.

    Returns:
        The count of newly-appended records (0 on an empty candidate universe or when a
        re-run finds nothing new to commit).

    Raises:
        ValueError: If ``run_slot`` is not one of ``_ledger_selection.RUN_SLOTS``.
    """
    if run_slot not in _ledger_selection.RUN_SLOTS:
        msg = f"run_slot must be one of {_ledger_selection.RUN_SLOTS}, got {run_slot!r}"
        raise ValueError(msg)

    pools = build_candidate_universe(date, run_slot)
    if not any(pools.values()):
        _logger.info(
            "empty candidate universe for %s %s -- recording as an empty decision, not a failure",
            date.isoformat(),
            run_slot,
        )
        return 0

    rngs = _ledger_selection.replicate_rngs(date, run_slot)
    already_drew = _ledger_store.committed_replicates(date, run_slot)
    total = 0
    for replicate_id, rng in enumerate(rngs):
        if replicate_id in already_drew:
            continue
        records: list[dict] = []
        for persona in _ledger_selection.PERSONAS:
            remaining, seen = _ledger_selection.remaining_budget_and_seen_players(
                date, persona, replicate_id
            )
            if remaining <= 0:
                continue
            drawn = _ledger_selection.draw_entries(
                pools[persona], _ledger_selection.PERSONA_SCORERS[persona], seen, remaining, rng
            )
            if not drawn:
                continue
            if persona == "kelly_growth":
                drawn = _resize_kelly_growth(drawn, date, replicate_id)
            records.extend(
                _committed_record(
                    c, date=date, run_slot=run_slot, persona=persona, replicate_id=replicate_id
                )
                for c in drawn
            )
        total += _ledger_store.append_entries(date, records)
    return total


@click.command()
@click.option("--run-slot", type=click.Choice(_ledger_selection.RUN_SLOTS), required=True)
@click.option("--date", default="today")
def ledger_commit(run_slot: str, date: str) -> None:
    """Twice-daily ledger commit: build candidates, draw, size, append."""
    slate_date = datetime.date.today() if date == "today" else datetime.date.fromisoformat(date)
    appended = run_commit(slate_date, run_slot)
    click.echo(f"committed {appended} new entries -> {_ledger_store.entries_path(slate_date)}")


if __name__ == "__main__":
    ledger_commit()
