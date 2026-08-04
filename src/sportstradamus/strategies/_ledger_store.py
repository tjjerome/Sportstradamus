"""JSONL storage for the simulated-bettor ledger: append, idempotency-check reads.

One file per slate date at ``data/ledger/entries/{date}.jsonl``. Records are
opaque dicts to this module — schema ownership lives with the orchestrator
that builds them; this module only stores/reads/appends by ``id``.
"""

from __future__ import annotations

import datetime
import json
from decimal import Decimal
from pathlib import Path

ENTRIES_DIR = Path("data") / "ledger" / "entries"


def entries_path(date: datetime.date) -> Path:
    return ENTRIES_DIR / f"{date.isoformat()}.jsonl"


def read_records(date: datetime.date) -> list[dict]:
    path = entries_path(date)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def already_committed_ids(
    date: datetime.date, run_slot: str, persona: str, replicate_id: int
) -> set[str]:
    """IDs already committed for this EXACT (run_slot, persona, replicate_id) today.

    Scoped to one run_slot -- this run's own idempotency/retry-safety check.
    """
    return {
        rec["id"]
        for rec in read_records(date)
        if rec["run_slot"] == run_slot
        and rec["persona"] == persona
        and rec["replicate_id"] == replicate_id
    }


def already_committed_entries(date: datetime.date, persona: str, replicate_id: int) -> list[dict]:
    """Full records for (persona, replicate_id) across BOTH run_slots committed so far today.

    NOT scoped to run_slot -- this is the function a later afternoon-run
    budget check reads to see what a morning run already committed. Kept
    separate from :func:`already_committed_ids` (rather than one function with
    an optional run_slot filter) because the two answer genuinely different
    questions and every call site knows up front which one it needs.
    """
    return [
        rec
        for rec in read_records(date)
        if rec["persona"] == persona and rec["replicate_id"] == replicate_id
    ]


def append_entries(date: datetime.date, records: list[dict]) -> int:
    """Append-only write; never rewrites existing lines.

    Re-checks each record's "id" against what's already on disk before
    writing -- belt-and-suspenders idempotency that closes the gap even if a
    caller forgets to pre-filter via :func:`already_committed_ids`.
    """
    existing_ids = {rec["id"] for rec in read_records(date)}
    to_write = [rec for rec in records if rec["id"] not in existing_ids]
    if not to_write:
        return 0

    path = entries_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in to_write:
            fh.write(json.dumps(rec, default=_json_default) + "\n")
    return len(to_write)


def _json_default(obj: object) -> str:
    """json.dumps `default=` hook: Decimal -> str, so `stake` round-trips exactly."""
    if isinstance(obj, Decimal):
        return str(obj)
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def decode_stake(record: dict) -> Decimal:
    return Decimal(record["stake"])
