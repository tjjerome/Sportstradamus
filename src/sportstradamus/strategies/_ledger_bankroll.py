"""Persistence layer for the simulated-bettor ledger's settlement results.

Writes each day's settled entries to ``data/ledger/settled_entries.parquet``
(one accumulating table, matching ``history.parquet``/``parlay_hist.parquet``'s
read-modify-write idiom) and derives a compounding per-``(persona,
replicate_id)`` paper bankroll trajectory in ``data/ledger/bankroll.parquet``.
Pure sink -- nothing upstream (``_ledger_selection``, ``ledger``) imports this
module or reads its output back; sizing must never be a function of settled
P&L. Idempotent against re-running the same date twice: a re-run's
``settled_rows`` is empty once ``_ledger_settlement.settle_day`` has excluded
already-settled ids via :func:`already_settled_ids`, and both writers below
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
    "persona",
    "replicate_id",
    "starting_bankroll",
    "daily_pnl",
    "ending_bankroll",
    "n_entries_settled",
    "n_entries_pending",
)


def already_settled_ids(date: datetime.date | None = None) -> set[str]:
    """IDs already present in the settled-entries table, optionally filtered to ``date``.

    Empty-safe: returns an empty set before the first settlement ever runs.
    ``_ledger_settlement.settle_day`` calls this directly to exclude
    already-settled entries from its resolve pass -- the idempotency check
    for the whole settlement pipeline, not just this module's half.
    """
    df = read_parquet_safe(SETTLED_ENTRIES_PATH)
    if df.empty:
        return set()
    if date is not None:
        df = df.loc[df["date"] == date.isoformat()]
    return set(df["id"])


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
    """Derive and append one bankroll row per ``(persona, replicate_id)`` settled today.

    No-op on empty input -- a gap day writes nothing, and the next active day's
    ``starting_bankroll`` naturally carries forward from the most recent prior
    row for that pair (not necessarily yesterday's), since it looks up the max
    existing row rather than assuming a contiguous daily series.
    """
    if not settled_rows:
        return
    existing = read_parquet_safe(BANKROLL_PATH)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in settled_rows:
        grouped.setdefault((row["persona"], row["replicate_id"]), []).append(row)

    new_rows = []
    for (persona, replicate_id), rows in grouped.items():
        daily_pnl = sum((Decimal(str(row["pnl"])) for row in rows), Decimal("0"))
        starting_bankroll = _latest_ending_bankroll(existing, persona, replicate_id)
        ending_bankroll = starting_bankroll + daily_pnl
        new_rows.append(
            {
                "date": date.isoformat(),
                "persona": persona,
                "replicate_id": replicate_id,
                "starting_bankroll": float(starting_bankroll),
                "daily_pnl": float(daily_pnl),
                "ending_bankroll": float(ending_bankroll),
                "n_entries_settled": len(rows),
                "n_entries_pending": 0,  # placeholder: needs a committed-vs-settled cross-reference a future stage can add
            }
        )
    new_df = pd.DataFrame(new_rows)[list(_BANKROLL_COLUMNS)]
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    _atomic_write_parquet(combined, BANKROLL_PATH)


def _latest_ending_bankroll(existing: pd.DataFrame, persona: str, replicate_id: int) -> Decimal:
    """Most recent prior ``ending_bankroll`` for this pair, or the seed value if none."""
    if existing.empty:
        return STARTING_BANKROLL
    pair_rows = existing.loc[
        (existing["persona"] == persona) & (existing["replicate_id"] == replicate_id)
    ]
    if pair_rows.empty:
        return STARTING_BANKROLL
    latest = pair_rows.loc[pair_rows["date"].idxmax()]
    return Decimal(str(latest["ending_bankroll"]))
