"""Pin the twice-daily commit orchestrator of ``strategies.ledger``:
idempotency across repeat calls, the committed-record schema, invalid
run_slot handling, empty-universe no-op, and Kelly-growth resizing.

Every test monkeypatches ``build_candidate_universe`` to a small, fixed
synthetic ``LedgerCandidate`` list and redirects ``_ledger_store.entries_path``
into ``tmp_path`` -- this repo runs pytest-xdist with ``-n auto``, so a test
touching the real ``data/ledger/`` path would race across parallel workers.
No test hits the network or runs a real scrape.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from sportstradamus.strategies import _ledger_store, ledger
from sportstradamus.strategies._ledger_selection import LedgerCandidate

DATE = datetime.date(2026, 7, 12)


def _redirect_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        _ledger_store,
        "entries_path",
        lambda date: tmp_path / f"{date.isoformat()}.jsonl",
    )


def _canonical_leg(player: str) -> dict:
    return {
        "player": player,
        "team": "",
        "market": "",
        "stat": "PTS",
        "bet": "Over",
        "line": 4.5,
        "league": "NBA",
        "game": "",
        "date": "2026-07-12",
        "platform": "Underdog",
        "win_prob": 0.6,
        "boost": 1.0,
        "push_prob": 0.0,
        "kelly": 0.0,
    }


def _candidate(
    cand_id: str,
    players: frozenset[str],
    *,
    contest_variant: str = "power",
    entry_size: int = 2,
    joint_prob: float = 0.5,
    ev: float = 0.1,
    stake: Decimal = Decimal("10"),
) -> LedgerCandidate:
    return LedgerCandidate(
        id=cand_id,
        contest_variant=contest_variant,
        entry_size=entry_size,
        legs=tuple(f"{p} Over 4.5 rebounds - 60.0%, 1.6x" for p in players),
        canonical_legs=tuple(_canonical_leg(p) for p in players),
        players=players,
        game_span=1,
        joint_prob=joint_prob,
        payout_multiplier=3.0,
        ev=ev,
        stake=stake,
        lines=tuple(4.5 for _ in players),
        model_probs=tuple(0.6 for _ in players),
        book_devig=tuple(0.55 for _ in players),
    )


def _small_universe() -> list[LedgerCandidate]:
    return [
        _candidate("cand-1", frozenset({"Player A", "Player B"})),
        _candidate("cand-2", frozenset({"Player C", "Player D"}), joint_prob=0.4, ev=0.2),
        _candidate("cand-3", frozenset({"Player E", "Player F"}), joint_prob=0.45, ev=0.15),
        _candidate("cand-4", frozenset({"Player G", "Player H"}), joint_prob=0.35, ev=0.3),
    ]


def _patch_universe(monkeypatch, universe: list[LedgerCandidate]) -> None:
    monkeypatch.setattr(ledger, "build_candidate_universe", lambda date, run_slot: universe)


# --- acceptance criterion: repeat run_commit is idempotent ---------------------


def test_run_commit_repeat_call_appends_nothing_new(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, _small_universe())

    first = ledger.run_commit(DATE, "morning")
    second = ledger.run_commit(DATE, "morning")

    assert first > 0
    assert second == 0


# --- _committed_record schema ---------------------------------------------------


def test_committed_record_has_full_schema_with_correct_types(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    candidate = _candidate("cand-schema", frozenset({"Player A", "Player B"}))

    record = ledger._committed_record(
        candidate, date=DATE, run_slot="morning", persona="safe", replicate_id=3
    )

    required_fields = {
        "id",
        "legs",
        "canonical_legs",
        "legs_players",
        "lines",
        "model_probs",
        "book_devig",
        "stake",
        "policy_version",
        "git_sha",
        "committed_at",
        "persona",
        "run_slot",
        "replicate_id",
        "game_span",
        "contest_variant",
        "entry_size",
        "joint_prob",
        "payout_multiplier",
        "ev",
        "date",
    }
    assert required_fields.issubset(record.keys())

    assert Decimal(record["stake"])  # round-trips without raising
    assert record["replicate_id"] in range(40)
    assert record["persona"] in {"safe", "high_ev", "kelly_growth"}
    assert record["run_slot"] in {"morning", "afternoon"}
    assert record["policy_version"] == ledger.POLICY_VERSION
    assert record["date"] == DATE.isoformat()
    assert record["legs_players"] == sorted(candidate.players)


# --- invalid run_slot ------------------------------------------------------------


def test_run_commit_raises_value_error_for_invalid_run_slot(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, _small_universe())

    try:
        ledger.run_commit(DATE, "midday")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid run_slot")


# --- empty candidate universe ----------------------------------------------------


def test_run_commit_empty_universe_returns_zero_without_raising(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, [])

    assert ledger.run_commit(DATE, "morning") == 0


# --- _resize_kelly_growth ----------------------------------------------------------


def test_resize_kelly_growth_drops_candidate_zeroed_by_solver(monkeypatch) -> None:
    drawn = [
        _candidate("keep-1", frozenset({"Player A"})),
        _candidate("drop-1", frozenset({"Player B"})),
    ]
    monkeypatch.setattr(
        ledger._ledger_selection, "remaining_kelly_fraction", lambda date, replicate_id: 0.25
    )
    monkeypatch.setattr(
        ledger,
        "joint_kelly_portfolio",
        lambda bankroll, candidates, fraction: {"keep-1": Decimal("12.34")},
    )

    result = ledger._resize_kelly_growth(drawn, DATE, 0)

    assert [c.id for c in result] == ["keep-1"]
    assert result[0].stake == Decimal("12.34")


def test_resize_kelly_growth_returns_empty_without_calling_solver_when_fraction_zero(
    monkeypatch,
) -> None:
    drawn = [_candidate("cand-1", frozenset({"Player A"}))]
    monkeypatch.setattr(
        ledger._ledger_selection, "remaining_kelly_fraction", lambda date, replicate_id: 0.0
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("joint_kelly_portfolio must not be called when fraction is 0")

    monkeypatch.setattr(ledger, "joint_kelly_portfolio", _fail_if_called)

    assert ledger._resize_kelly_growth(drawn, DATE, 0) == []
