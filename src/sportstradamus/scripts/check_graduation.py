#!/usr/bin/env python3
"""Print the lifecycle status table for every (league, market) cell.

Stage 0 deliverable 0.4. Joins ``data/model_stats.parquet`` (Gate 1) and
``data/live_metrics_per_market.parquet`` (Gate 2) per (league, market) and
classifies each cell into one of ``not-shipped`` / ``in-test`` / ``graduated``
/ ``demoted`` so future sessions can read graduation state without inspecting
dashboards.

Output is printed colored to stdout (locked decision #9 — no CSV/parquet
sink, no dashboard page in Stage 0). The 8-metric body shows 4 Gate 1 and
4 Gate 2 columns alongside the lifecycle classification.

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

# Graduation requires at least this many settled offers in the 30d window
# before the live BSS signal is trustworthy.
_MIN_SETTLED_FOR_GRADUATION = 200
# The 30d window is the canonical graduation gate (7d is too noisy for state).
_GRADUATION_WINDOW_DAYS = 30
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


def _classify_lifecycle(gate1_bss: float, n_settled: float, book_bss_30d: float) -> str:
    """Map (Gate 1 BSS, n_settled, Gate 2 BSS) to a lifecycle state.

    See plan locked decision #15 for the rule table; in short:
    NaN/negative Gate 1 → ``not-shipped``; positive Gate 1 but insufficient
    live data → ``in-test``; live BSS non-negative → ``graduated``; live BSS
    negative → ``demoted``.
    """
    if gate1_bss is None or (isinstance(gate1_bss, float) and math.isnan(gate1_bss)):
        return "not-shipped"
    if gate1_bss < 0:
        return "not-shipped"
    n_settled_nan = n_settled is None or (isinstance(n_settled, float) and math.isnan(n_settled))
    n_int = 0 if n_settled_nan else int(n_settled)
    if n_int < _MIN_SETTLED_FOR_GRADUATION:
        return "in-test"
    if book_bss_30d is None or (isinstance(book_bss_30d, float) and math.isnan(book_bss_30d)):
        return "in-test"
    if book_bss_30d < 0:
        return "demoted"
    return "graduated"


def _read_gate1(path: Path, league: str | None) -> pd.DataFrame:
    """Read model_stats.parquet and project to the calibrated Gate 1 view."""
    if not path.exists():
        raise click.UsageError(f"model_stats parquet not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    df = df[(df["row_kind"] == "model") & (df["metric_row"] == "calibrated")]
    if league:
        df = df[df["league"] == league]
    keep = ["league", "market", "distribution", "brier_skill_score",
            "predicted_over_rate", "empirical_over_rate", "kelly_shrinkage"]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    return df.rename(columns={"brier_skill_score": "gate1_bss"})


def _read_gate2(path: Path) -> pd.DataFrame:
    """Read live_metrics_per_market.parquet and project to the 30d Gate 2 view.

    Returns an empty frame (correct columns, zero rows) when the parquet is
    missing — the outer merge then classifies every Gate 1 row as ``in-test``
    until the live aggregator catches up.
    """
    cols = ["league", "market", "n_settled", "gate2_book_bss",
            "predicted_over_rate_live", "empirical_over_rate_live", "profit_sim_yield"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(path, engine="pyarrow")
    df = df[df["window_days"] == _GRADUATION_WINDOW_DAYS]
    df = df.rename(
        columns={
            "book_bss": "gate2_book_bss",
            "predicted_over_rate": "predicted_over_rate_live",
            "empirical_over_rate": "empirical_over_rate_live",
        }
    )
    return df[cols]


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
    color = _STATE_COLORS.get(row["lifecycle_state"], None)
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
    gate1_path = Path(model_stats_path) if model_stats_path else Path(str(MODEL_STATS_PATH))
    gate2_path = Path(live_metrics_path) if live_metrics_path else Path(str(LIVE_METRICS_PATH))

    gate1 = _read_gate1(gate1_path, league)
    if gate1.empty:
        click.echo("No model_stats rows match the filter; nothing to classify.")
        return
    gate2 = _read_gate2(gate2_path)

    merged = gate1.merge(gate2, on=["league", "market"], how="left")
    merged["lifecycle_state"] = merged.apply(
        lambda r: _classify_lifecycle(
            r.get("gate1_bss", float("nan")),
            r.get("n_settled", float("nan")),
            r.get("gate2_book_bss", float("nan")),
        ),
        axis=1,
    )

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
