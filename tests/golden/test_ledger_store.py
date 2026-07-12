"""Pin JSONL append/idempotency/Decimal-codec behavior of ``_ledger_store``.

Every test monkeypatches ``entries_path`` to resolve inside pytest's
``tmp_path`` fixture -- this suite runs under pytest-xdist with ``-n auto``
(parallel workers), so a test that touched the real ``data/ledger/`` path
would race across workers. See ``test_dashboard_no_archive_lock.py`` for this
codebase's established pattern of redirecting a path away from its real
location for tests.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from sportstradamus.strategies import _ledger_store

DATE = datetime.date(2026, 7, 12)


def _redirect(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        _ledger_store,
        "entries_path",
        lambda date: tmp_path / f"{date.isoformat()}.jsonl",
    )


def _record(
    rec_id: str,
    run_slot: str = "morning",
    persona: str = "safe",
    replicate_id: int = 0,
    stake: str = "12.34",
) -> dict:
    return {
        "id": rec_id,
        "legs": ["Player A Over 20.5 PTS"],
        "legs_players": ["Player A"],
        "lines": [20.5],
        "model_probs": [0.55],
        "book_devig": [0.52],
        "stake": stake,
        "policy_version": "policy_v1",
        "git_sha": "deadbeef",
        "committed_at": "2026-07-12T12:00:00Z",
        "persona": persona,
        "run_slot": run_slot,
        "replicate_id": replicate_id,
        "game_span": 1,
        "contest_variant": "power",
        "entry_size": 2,
        "joint_prob": 0.3,
        "payout_multiplier": 3.0,
        "ev": 0.1,
        "date": DATE.isoformat(),
    }


def test_append_entries_idempotent_on_repeat_call(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    records = [_record("id-1"), _record("id-2")]

    first = _ledger_store.append_entries(DATE, records)
    second = _ledger_store.append_entries(DATE, records)

    assert first == 2
    assert second == 0
    stored = _ledger_store._read_records(DATE)
    assert sorted(rec["id"] for rec in stored) == ["id-1", "id-2"]


def test_decode_stake_round_trips_decimal_exactly(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    record = _record("id-1", stake="12.34")
    _ledger_store.append_entries(DATE, [record])

    stored = _ledger_store._read_records(DATE)[0]
    decoded = _ledger_store.decode_stake(stored)

    assert decoded == Decimal("12.34")
    assert isinstance(decoded, Decimal)
    assert str(decoded) == "12.34"


def test_already_committed_entries_spans_both_run_slots(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    morning = _record("id-morning", run_slot="morning", persona="safe", replicate_id=0)
    afternoon = _record("id-afternoon", run_slot="afternoon", persona="safe", replicate_id=0)
    _ledger_store.append_entries(DATE, [morning, afternoon])

    entries = _ledger_store.already_committed_entries(DATE, persona="safe", replicate_id=0)
    assert sorted(rec["id"] for rec in entries) == ["id-afternoon", "id-morning"]

    morning_ids = _ledger_store.already_committed_ids(
        DATE, run_slot="morning", persona="safe", replicate_id=0
    )
    afternoon_ids = _ledger_store.already_committed_ids(
        DATE, run_slot="afternoon", persona="safe", replicate_id=0
    )
    assert morning_ids == {"id-morning"}
    assert afternoon_ids == {"id-afternoon"}


def test_missing_file_returns_empty_without_error(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)

    assert _ledger_store._read_records(DATE) == []
    assert (
        _ledger_store.already_committed_ids(
            DATE, run_slot="morning", persona="safe", replicate_id=0
        )
        == set()
    )
    assert _ledger_store.already_committed_entries(DATE, persona="safe", replicate_id=0) == []


def test_append_entries_writes_distinct_ids_for_same_slot(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    same_slot_records = [
        _record("id-a", run_slot="morning", persona="safe", replicate_id=0),
        _record("id-b", run_slot="morning", persona="safe", replicate_id=0),
        _record("id-c", run_slot="morning", persona="safe", replicate_id=0),
    ]

    written = _ledger_store.append_entries(DATE, same_slot_records)

    assert written == 3
    ids = _ledger_store.already_committed_ids(
        DATE, run_slot="morning", persona="safe", replicate_id=0
    )
    assert ids == {"id-a", "id-b", "id-c"}
