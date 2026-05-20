#!/usr/bin/env python3
"""One-shot backfill of ``data/live_metrics_per_market.parquet``.

Walks backward through ``history.parquet`` for the last ``--days`` days,
invoking :func:`sportstradamus.nightly._compute_live_metrics` once per
``--step``-day endpoint, then merges the resulting rows with any existing
output parquet (deduplicated on ``(league, market, computed_at, window_days)``,
last-write wins). Stage 0 deliverable 0.5 — never called from cron.

Usage
-----
    poetry run backfill-live-metrics --days 90
    poetry run backfill-live-metrics --days 30 --step 7 --output /tmp/live.parquet
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from sportstradamus.helpers.io import (
    LIVE_METRICS_PATH,
    _atomic_write_parquet,
    read_history,
)
from sportstradamus.nightly import LIVE_METRICS_COLUMNS, _compute_live_metrics

_KEY_COLUMNS = ("league", "market", "computed_at", "window_days")


def _endpoints(days: int, step: int, now: datetime) -> list[datetime]:
    """Walk backwards from ``now`` by ``step``-day strides for ``days`` total."""
    count = max(1, days // step)
    return [now - timedelta(days=i * step) for i in range(count)]


def _merge_with_existing(new_rows: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Concat existing parquet (if any) with new rows, dedupe on the canonical key."""
    if output.exists():
        prior = pd.read_parquet(output, engine="pyarrow")
        combined = pd.concat([prior, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = combined.drop_duplicates(subset=list(_KEY_COLUMNS), keep="last")
    return combined.reset_index(drop=True)


@click.command()
@click.option("--days", default=90, show_default=True, help="Backfill window length in days.")
@click.option(
    "--step",
    default=1,
    show_default=True,
    help="Days between successive computed_at endpoints walking backward.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the output parquet path (defaults to data/live_metrics_per_market.parquet).",
)
def main(days: int, step: int, output: Path | None) -> None:
    """Backfill rolling live metrics over the last ``--days`` days."""
    if step <= 0:
        raise click.UsageError("--step must be a positive integer.")
    out_path = Path(output) if output is not None else Path(str(LIVE_METRICS_PATH))
    history = read_history()
    if history.empty:
        click.echo("history.parquet is empty; nothing to backfill.")
        return

    # Round to the day so re-runs within the same day produce identical
    # computed_at values — the backfill is idempotent under (league, market,
    # computed_at, window_days) deduplication only if endpoints align exactly.
    now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    endpoints = _endpoints(days, step, now)
    batches: list[pd.DataFrame] = []
    for endpoint in tqdm(endpoints, desc="Backfilling live metrics"):
        batch = _compute_live_metrics(history, now=endpoint)
        if not batch.empty:
            batches.append(batch)

    if not batches:
        click.echo("No settled offers found in the backfill window.")
        return

    new_rows = pd.concat(batches, ignore_index=True)
    merged = _merge_with_existing(new_rows, out_path)
    _atomic_write_parquet(merged, out_path)
    click.echo(f"Wrote {len(merged)} rows ({len(new_rows)} new) to {out_path}.")


if __name__ == "__main__":
    main()
