"""JSONL storage for the simulated-bettor ledger: append, idempotency-check reads.

One file per slate date at ``data/ledger/entries/{date}.jsonl``. Records are
opaque dicts to this module — schema ownership lives with the orchestrator
that builds them; this module only stores/reads/appends by ``id`` and the
bettor copy (``persona``, ``replicate_id``) that holds it.
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


def entry_key(record: dict) -> tuple[str, str, int]:
    """What tells one entry from another within a slate date: its ``id`` and the copy holding it.

    A persona's replicates are independent bettors drawing from one candidate
    pool, and a candidate's ``id`` is the same whoever draws it, so two copies
    that draw the same candidate each hold an entry of their own.
    """
    return (record["id"], record["persona"], record["replicate_id"])


def committed_replicates(date: datetime.date, run_slot: str) -> set[int]:
    """Replicates that already hold a record from ``run_slot`` on ``date``.

    The commit run's retry check. Such a replicate has drawn in this slot;
    drawn again it would count its own entries against the day's budget and
    draw others, so a retry leaves it alone.
    """
    return {rec["replicate_id"] for rec in read_records(date) if rec["run_slot"] == run_slot}


def already_committed_entries(date: datetime.date, persona: str, replicate_id: int) -> list[dict]:
    """Full records for (persona, replicate_id) across BOTH run_slots committed so far today.

    NOT scoped to run_slot -- this is the function a later afternoon-run
    budget check reads to see what a morning run already committed.
    """
    return [
        rec
        for rec in read_records(date)
        if rec["persona"] == persona and rec["replicate_id"] == replicate_id
    ]


def append_entries(date: datetime.date, records: list[dict]) -> int:
    """Append-only write; never rewrites existing lines.

    Refuses a record whose :func:`entry_key` is already held that day, on disk
    or earlier in ``records``: a copy holds a candidate once, and what another
    copy holds does not count against it.
    """
    held = {entry_key(rec) for rec in read_records(date)}
    to_write = []
    for rec in records:
        if entry_key(rec) not in held:
            held.add(entry_key(rec))
            to_write.append(rec)
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
