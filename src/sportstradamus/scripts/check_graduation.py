#!/usr/bin/env python3
"""Print the lifecycle status table for every (league, market) cell.

Stage 0 deliverable 0.4. Joins ``data/model_stats.parquet`` (Gate 1) and
``data/live_metrics_per_market.parquet`` (Gate 2) per (league, market) via
``training.graduation`` and prints the classification colored to stdout. The
8-metric body shows 4 Gate 1 and 4 Gate 2 columns alongside the lifecycle state.

Usage
-----
    poetry run check-graduation
    poetry run check-graduation --league NBA
"""

from __future__ import annotations

import math
from pathlib import Path

import click
import pandas as pd

from sportstradamus.helpers.io import LIVE_METRICS_PATH, MODEL_STATS_PATH
from sportstradamus.training.graduation import lifecycle_table

# Color map for the lifecycle states, applied by click.secho per row.
_STATE_COLORS = {
    "graduated": "green",
    "in-test": "yellow",
    "demoted": "red",
    "not-shipped": "cyan",
}
# Column order used by the printed table. 3 keys + 4 Gate 1 + 4 Gate 2 + 1 state.
_DISPLAY_COLUMNS = (
    "league",
    "market",
    "distribution",
    "gate1_bss",
    "predicted_over_rate",
    "empirical_over_rate",
    "kelly_shrinkage",
    "gate2_book_bss",
    "predicted_over_rate_live",
    "empirical_over_rate_live",
    "profit_sim_yield",
    "lifecycle_state",
)


def _format_metric(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "    nan"
    return f"{float(value):+7.3f}"


def _print_header() -> None:
    click.echo(
        f"{'league':<6} {'market':<22} {'dist':<10} "
        f"{'g1_bss':>7} {'g1_p_or':>7} {'g1_e_or':>7} {'g1_kelly':>8} "
        f"{'g2_bss':>7} {'g2_p_or':>7} {'g2_e_or':>7} {'g2_yield':>8} "
        f"{'state':<12}"
    )


def _print_row(row: pd.Series) -> None:
    line = (
        f"{row['league']!s:<6} {row['market']!s:<22} "
        f"{row.get('distribution', '')!s:<10} "
        f"{_format_metric(row.get('gate1_bss')):>7} "
        f"{_format_metric(row.get('predicted_over_rate')):>7} "
        f"{_format_metric(row.get('empirical_over_rate')):>7} "
        f"{_format_metric(row.get('kelly_shrinkage')):>8} "
        f"{_format_metric(row.get('gate2_book_bss')):>7} "
        f"{_format_metric(row.get('predicted_over_rate_live')):>7} "
        f"{_format_metric(row.get('empirical_over_rate_live')):>7} "
        f"{_format_metric(row.get('profit_sim_yield')):>8} "
        f"{row['lifecycle_state']:<12}"
    )
    color = _STATE_COLORS.get(row["lifecycle_state"])
    click.secho(line, fg=color)


def _print_summary(states: pd.Series) -> None:
    counts = states.value_counts().to_dict()
    parts = [f"{counts.get(s, 0)} {s}" for s in ("graduated", "in-test", "demoted", "not-shipped")]
    click.echo("")
    click.echo(f"Summary: {', '.join(parts)} (n={len(states)})")


@click.command()
@click.option("--league", default=None, help="Filter to one league (e.g. NBA).")
@click.option(
    "--model-stats-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override Gate 1 parquet path (defaults to data/model_stats.parquet).",
)
@click.option(
    "--live-metrics-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override Gate 2 parquet path (defaults to data/live_metrics_per_market.parquet).",
)
def main(league: str | None, model_stats_path: Path | None, live_metrics_path: Path | None) -> None:
    """Print the lifecycle status table joining Gate 1 (offline) and Gate 2 (live)."""
    gate1_path = model_stats_path if model_stats_path else Path(str(MODEL_STATS_PATH))
    gate2_path = live_metrics_path if live_metrics_path else Path(str(LIVE_METRICS_PATH))

    merged = lifecycle_table(gate1_path, gate2_path, league)
    if merged.empty:
        click.echo("No model_stats rows match the filter; nothing to classify.")
        return

    for col in _DISPLAY_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")
    merged = merged.sort_values(["league", "market"]).reset_index(drop=True)

    _print_header()
    for _, row in merged.iterrows():
        _print_row(row)
    _print_summary(merged["lifecycle_state"])


if __name__ == "__main__":
    main()
