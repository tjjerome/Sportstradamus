"""Pin the MERGE design for cross-platform candidate combination in
strategies.ledger: Sleeper and Underdog candidates share one universe and
one persona budget (docs/handoffs/sleeper-parity.md Stage 4 ledger line),
rather than tracking independent per-platform bankrolls/budgets. See
test_ledger.py for the rest of this module's coverage (idempotency, record
schema, kelly-growth resizing) -- this file is scoped to the platform-merge
behavior only.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pandas as pd

from sportstradamus.strategies import _ledger_selection, ledger
from sportstradamus.strategies._ledger_selection import LedgerCandidate
from sportstradamus.strategies.underdog_pickem import RecommendedEntry

DATE = datetime.date(2026, 7, 12)


def _redirect_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ledger._ledger_store,
        "entries_path",
        lambda date: tmp_path / f"{date.isoformat()}.jsonl",
    )


def _canonical_leg(player: str, platform: str = "Underdog") -> dict:
    return {
        "player": player,
        "team": "",
        "market": "",
        "stat": "PTS",
        "bet": "Over",
        "line": 4.5,
        "league": "NBA",
        "game": "",
        "date": DATE.isoformat(),
        "platform": platform,
        "win_prob": 0.6,
        "boost": 1.0,
        "push_prob": 0.0,
        "kelly": 0.0,
    }


def _candidate(
    cand_id: str,
    players: frozenset[str],
    *,
    platform: str,
    joint_prob: float,
    ev: float = 0.1,
    stake: Decimal = Decimal("10"),
) -> LedgerCandidate:
    return LedgerCandidate(
        id=cand_id,
        contest_variant="power",
        entry_size=2,
        legs=tuple(f"{p} Over 4.5 rebounds - 60.0%, 1.6x" for p in players),
        canonical_legs=tuple(_canonical_leg(p, platform=platform) for p in players),
        players=players,
        game_span=1,
        joint_prob=joint_prob,
        payout_multiplier=3.0,
        ev=ev,
        stake=stake,
        platform=platform,
    )


def _recommended_entry(cand_id: str, player: str, platform: str) -> RecommendedEntry:
    return RecommendedEntry(
        id=cand_id,
        contest_variant="power",
        entry_size=2,
        legs=(f"{player} Over 4.5 rebounds - 60.0%, 1.6x",),
        joint_prob=0.5,
        payout_multiplier=3.0,
        ev=0.1,
        recommended_stake=Decimal("10"),
        canonical_legs=(_canonical_leg(player, platform=platform),),
        platform=platform,
    )


# --- build_candidate_universe combines both platforms' scrapes -----------------


def test_build_candidate_universe_combines_both_platforms(monkeypatch) -> None:
    def _fake_live_load(config, platform):
        return {}, pd.DataFrame()

    def _fake_construct_entries(date, bankroll, config, *, parlay_dfs, offers_df, platform):
        prefix = "ud" if platform == "Underdog" else "sl"
        return [_recommended_entry(f"{prefix}-same-1", f"{prefix.upper()} Same Player", platform)]

    def _fake_cross_game(offers_df, config, date, run_slot, *, platform):
        prefix = "ud" if platform == "Underdog" else "sl"
        return [
            _candidate(
                f"{prefix}-cross-1",
                frozenset({f"{prefix.upper()} Cross Player"}),
                platform=platform,
                joint_prob=0.5,
            )
        ]

    monkeypatch.setattr(ledger, "live_load", _fake_live_load)
    monkeypatch.setattr(ledger, "construct_entries", _fake_construct_entries)
    monkeypatch.setattr(ledger._ledger_cross_game, "build_cross_game_candidates", _fake_cross_game)

    universe = ledger.build_candidate_universe(DATE, "morning")

    assert len(universe) == 4
    by_id = {c.id: c for c in universe}
    assert set(by_id) == {"ud-same-1", "ud-cross-1", "sl-same-1", "sl-cross-1"}
    for cand_id, candidate in by_id.items():
        expected_platform = "Underdog" if cand_id.startswith("ud-") else "Sleeper"
        assert candidate.platform == expected_platform


# --- run_commit draws from one shared pool/budget per persona ------------------


def _six_candidate_universe() -> list[LedgerCandidate]:
    """3 Underdog + 3 Sleeper candidates, distinct players (no Jaccard penalty
    against each other) and slightly staggered joint_prob (breaks exact score
    ties so the weighted draw isn't degenerate) -- all comfortably above the
    "safe" persona's implicit bar since its scorer is bare joint_prob.
    """
    underdog = [
        _candidate(
            f"ud-{i}", frozenset({f"UD Player {i}"}), platform="Underdog", joint_prob=0.5 + i * 0.01
        )
        for i in range(3)
    ]
    sleeper = [
        _candidate(
            f"sl-{i}", frozenset({f"SL Player {i}"}), platform="Sleeper", joint_prob=0.5 + i * 0.01
        )
        for i in range(3)
    ]
    return underdog + sleeper


def _force_full_draw(monkeypatch) -> None:
    """draw_entries stops early on a DRAW_CONTINUE_PROB coin flip after every
    successful draw (see _ledger_selection.py), so a 6-candidate pool doesn't
    reliably fill a 5-slot budget under the real per-replicate RNG stream --
    most replicates stop after 1-2 draws, which would leave the "6 candidates
    contend for 5 shared slots" claim unproven by accident. Pushing the
    threshold above 1.0 makes `rng.random() >= DRAW_CONTINUE_PROB` always
    False, i.e. never stop early, so the draw runs until the budget (5) or the
    pool (6) is exhausted -- deterministic given DATE's fixed RNG seed.
    """
    monkeypatch.setattr(_ledger_selection, "DRAW_CONTINUE_PROB", 2.0)


def _safe_persona_records(tmp_path) -> list[dict]:
    path = tmp_path / f"{DATE.isoformat()}.jsonl"
    records = ledger._ledger_store.read_records(DATE)
    assert path.exists()  # sanity: we're reading the redirected file, not the real archive
    return [rec for rec in records if rec["persona"] == "safe" and rec["replicate_id"] == 0]


def test_run_commit_persona_budget_shared_across_platforms(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    _force_full_draw(monkeypatch)
    universe = _six_candidate_universe()
    monkeypatch.setattr(ledger, "build_candidate_universe", lambda date, run_slot: universe)

    ledger.run_commit(DATE, "morning")

    records = _safe_persona_records(tmp_path)
    assert len(records) == _ledger_selection.MAX_ENTRIES_PER_DAY
    platforms_drawn = {rec["platform"] for rec in records}
    assert platforms_drawn == {"Underdog", "Sleeper"}, (
        "expected both platforms to compete for and fill the one shared 5-slot "
        f"budget; got {platforms_drawn}"
    )


# --- committed records carry the correct platform after the shared draw --------


def test_committed_records_carry_correct_platform_after_shared_draw(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    _force_full_draw(monkeypatch)
    universe = _six_candidate_universe()
    id_to_platform = {c.id: c.platform for c in universe}
    monkeypatch.setattr(ledger, "build_candidate_universe", lambda date, run_slot: universe)

    ledger.run_commit(DATE, "morning")

    records = _safe_persona_records(tmp_path)
    assert len(records) == _ledger_selection.MAX_ENTRIES_PER_DAY
    for rec in records:
        assert rec["platform"] == id_to_platform[rec["id"]]
