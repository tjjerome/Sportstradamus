"""Persistence layer for the simulated-bettor ledger's settlement results.

Writes each day's settled entries to ``data/ledger/settled_entries.parquet``
(one accumulating table, matching ``history.parquet``/``parlay_hist.parquet``'s
read-modify-write idiom) and derives a compounding per-``(policy_version,
persona, replicate_id)`` paper bankroll trajectory in
``data/ledger/bankroll.parquet``: a new policy version starts its own
trajectory from the seed bankroll, so two versions are never pooled.
Pure sink -- nothing upstream (``_ledger_selection``, ``ledger``) imports this
module or reads its output back; sizing must never be a function of settled
P&L. Idempotent against re-running the same date twice: a re-run's
``settled_rows`` is empty once ``_ledger_settlement.settle_day`` has left out
what :func:`read_settled_entries` shows as settled, and both writers below
no-op on empty input.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from sportstradamus.helpers.io import _atomic_write_parquet, read_parquet_safe

SETTLED_ENTRIES_PATH = Path("data") / "ledger" / "settled_entries.parquet"
BANKROLL_PATH = Path("data") / "ledger" / "bankroll.parquet"

STARTING_BANKROLL = Decimal("5000")  # matches _ledger_selection.BANKROLL_PER_REPLICATE

_SETTLED_ENTRIES_COLUMNS = (
    "id",
    "date",
    "persona",
    "run_slot",
    "replicate_id",
    "contest_variant",
    "entry_size",
    "effective_size",
    "misses",
    "pushes",
    "realized_multiplier",
    "stake",
    "payout",
    "pnl",
    "game_span",
    "committed_at",
    "policy_version",
    "git_sha",
    "clv_leg_count",
    "model_clv_mean",
    "market_clv_mean",
    "settled_at",
)

_BANKROLL_COLUMNS = (
    "date",
    "policy_version",
    "persona",
    "replicate_id",
    "starting_bankroll",
    "daily_pnl",
    "ending_bankroll",
    "n_entries_settled",
    "n_entries_pending",
)

# Bankroll rows written before the table carried ``policy_version`` all came from this one.
_FIRST_POLICY_VERSION = "policy_v1"


def read_settled_entries(date: datetime.date) -> list[dict]:
    """Rows of the settled-entries table for slate ``date``.

    Empty-safe: returns ``[]`` before the first settlement ever runs.
    ``_ledger_settlement.settleable_entries`` reads these to leave what has
    settled out of its resolve pass -- the idempotency check for the whole
    settlement pipeline, not just this module's half.
    """
    df = read_parquet_safe(SETTLED_ENTRIES_PATH)
    if df.empty:
        return []
    return df.loc[df["date"] == date.isoformat()].to_dict("records")


def write_settled_entries(rows: list[dict]) -> None:
    """Append settlement results to the accumulating settled-entries parquet.

    No-op on empty input. ``stake``/``payout``/``pnl`` arrive as ``Decimal``
    and are cast to ``float`` only here, at the parquet boundary.
    """
    if not rows:
        return
    new_rows = pd.DataFrame(
        [
            {
                **row,
                "stake": float(row["stake"]),
                "payout": float(row["payout"]),
                "pnl": float(row["pnl"]),
            }
            for row in rows
        ]
    )[list(_SETTLED_ENTRIES_COLUMNS)]
    existing = read_parquet_safe(SETTLED_ENTRIES_PATH)
    combined = (
        pd.concat([existing, new_rows], ignore_index=True) if not existing.empty else new_rows
    )
    _atomic_write_parquet(combined, SETTLED_ENTRIES_PATH)


def update_bankroll(date: datetime.date, settled_rows: list[dict]) -> None:
    """Append one bankroll row per ``(policy_version, persona, replicate_id)`` settled today.

    No-op on empty input -- a gap day writes nothing, and the next active day's
    ``starting_bankroll`` carries forward from the last row written for that
    trajectory (not necessarily yesterday's; see :func:`_latest_ending_bankroll`).
    """
    if not settled_rows:
        return
    existing = read_parquet_safe(BANKROLL_PATH)
    if not existing.empty and "policy_version" not in existing:
        existing["policy_version"] = _FIRST_POLICY_VERSION
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in settled_rows:
        key = (row["policy_version"], row["persona"], row["replicate_id"])
        grouped.setdefault(key, []).append(row)

    new_rows = []
    for (policy_version, persona, replicate_id), rows in grouped.items():
        daily_pnl = sum((Decimal(str(row["pnl"])) for row in rows), Decimal("0"))
        starting_bankroll = _latest_ending_bankroll(existing, policy_version, persona, replicate_id)
        ending_bankroll = starting_bankroll + daily_pnl
        new_rows.append(
            {
                "date": date.isoformat(),
                "policy_version": policy_version,
                "persona": persona,
                "replicate_id": replicate_id,
                "starting_bankroll": float(starting_bankroll),
                "daily_pnl": float(daily_pnl),
                "ending_bankroll": float(ending_bankroll),
                "n_entries_settled": len(rows),
                "n_entries_pending": 0,  # placeholder: needs a committed-vs-settled cross-reference a future stage can add
            }
        )
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    _atomic_write_parquet(combined[list(_BANKROLL_COLUMNS)], BANKROLL_PATH)


def _latest_ending_bankroll(
    existing: pd.DataFrame, policy_version: str, persona: str, replicate_id: int
) -> Decimal:
    """``ending_bankroll`` of the last row written for this trajectory, or the seed value if none.

    Write order, not slate date, is the chain. Rows are appended as passes
    settle, so the last one written holds every earlier pass's P&L. The row of
    the latest ``date`` need not: a date can settle in two passes, and a late
    pass of an older date is written after a newer date's row.
    """
    if existing.empty:
        return STARTING_BANKROLL
    prior_rows = existing.loc[
        (existing["policy_version"] == policy_version)
        & (existing["persona"] == persona)
        & (existing["replicate_id"] == replicate_id)
    ]
    if prior_rows.empty:
        return STARTING_BANKROLL
    return Decimal(str(prior_rows["ending_bankroll"].iloc[-1]))
