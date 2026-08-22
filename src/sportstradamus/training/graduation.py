"""Shared lifecycle classification for (league, market) cells.

Joins the offline Gate-1 view (``data/model_stats.parquet``) with the live
Gate-2 view (``data/live_metrics_per_market.parquet``) and classifies each cell
into ``not-shipped`` / ``in-test`` / ``graduated`` / ``demoted``. Both the
``check-graduation`` display CLI and the ``generate-ship-config`` generator
import these functions so the two share one definition of "graduated".

The Gate-2 rule here (positive Gate-1 BSS + at least
:data:`MIN_SETTLED_FOR_GRADUATION` settled offers in the
:data:`GRADUATION_WINDOW_DAYS`-day window + non-negative live book-BSS) is a
*simplified proxy* of the full Gate 2 in ``docs/ship_gate.md``; see that doc's
"Known gap" note. ``main`` is dormant (no live metrics yet), so the proxy is
acceptable until the live aggregator is producing data.
"""

from __future__ import annotations

import math
from pathlib import Path

import click
import pandas as pd

# Graduation requires at least this many settled offers in the window before the
# live BSS signal is trustworthy.
MIN_SETTLED_FOR_GRADUATION = 200
# The 30d window is the canonical graduation gate (7d is too noisy for state).
GRADUATION_WINDOW_DAYS = 30
# Demote a cell when the Bet=Over hit rate (precision_over) falls below this.
# Precision_over = P(Result=Over | Bet=Over) is the like-for-like metric that
# matches the training pipeline's `precision_over` column — comparing the
# aggregate `predicted_over_rate` vs `empirical_over_rate` instead conflates
# publication-selection bias (Sleeper/UD push Over-skewed offers) with real
# model bias and produces false-positive demotes (see
# `/tmp/researcher_fg3m_calibration_divergence.md`, 2026-05-24). Below 0.50
# the model's Over recommendations lose more often than they win, which is a
# losing strategy at any boost ≤ 2.0×. Live precision_over is NaN when fewer
# than `_MIN_BETS_FOR_PRECISION` Over bets exist in the window (see
# `nightly.py`); the check is skipped in that case.
MIN_PRECISION_OVER = 0.50


def _is_nan_like(x: object) -> bool:
    # Treats None and float NaN as "missing". Pandas merges produce NaN (not None)
    # for missing join keys, but callers may also pass None explicitly.
    return x is None or (isinstance(x, float) and math.isnan(x))


def classify_lifecycle(
    gate1_bss: float,
    n_settled: float,
    book_bss_30d: float,
    precision_over_live: float = float("nan"),
) -> str:
    """Map (Gate-1 BSS, n_settled, Gate-2 BSS, precision_over) to a lifecycle state.

    NaN/negative Gate-1 BSS -> ``not-shipped``; positive Gate-1 BSS but
    insufficient live data -> ``in-test``; negative live BSS or live
    precision_over below :data:`MIN_PRECISION_OVER` -> ``demoted``;
    otherwise -> ``graduated``.

    Args:
        gate1_bss: Offline calibrated brier-skill-score vs the book baseline.
        n_settled: Settled offer count in the graduation window.
        book_bss_30d: Live 30-day book-relative brier-skill-score.
        precision_over_live: Live 30d hit rate among ``Bet="Over"`` rows
            (i.e. P(Result=Over | Bet=Over)). NaN -> precision check skipped
            (typically because the cell has fewer Over bets than
            ``nightly._MIN_BETS_FOR_PRECISION``).

    Returns:
        One of ``"not-shipped"``, ``"in-test"``, ``"graduated"``, ``"demoted"``.
    """
    if _is_nan_like(gate1_bss):
        return "not-shipped"
    if gate1_bss < 0:
        return "not-shipped"
    n_int = 0 if _is_nan_like(n_settled) else int(n_settled)
    if n_int < MIN_SETTLED_FOR_GRADUATION:
        return "in-test"
    if _is_nan_like(book_bss_30d):
        return "in-test"
    if book_bss_30d < 0:
        return "demoted"
    if not _is_nan_like(precision_over_live) and precision_over_live < MIN_PRECISION_OVER:
        return "demoted"
    return "graduated"


def read_gate1(path: Path, league: str | None = None) -> pd.DataFrame:
    """Read ``model_stats.parquet`` and project to the calibrated Gate-1 view.

    Args:
        path: Path to the model-stats parquet.
        league: Optional league filter (e.g. ``"NBA"``).

    Returns:
        DataFrame with ``brier_skill_score`` renamed to ``gate1_bss``, limited
        to the real model's calibrated rows.

    Raises:
        click.UsageError: If the parquet is missing (Gate 1 is required).
    """
    if not path.exists():
        raise click.UsageError(f"model_stats parquet not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    if league:
        df = df[df["league"] == league]
    keep = [
        "league",
        "market",
        "distribution",
        "brier_skill_score",
        "predicted_over_rate",
        "empirical_over_rate",
        "kelly_shrinkage",
    ]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()
    return df.rename(columns={"brier_skill_score": "gate1_bss"})


def read_gate2(path: Path) -> pd.DataFrame:
    """Read ``live_metrics_per_market.parquet`` and project to the 30d Gate-2 view.

    Returns an empty frame (correct columns, zero rows) when the parquet is
    missing — the outer merge then classifies every Gate-1 row as ``in-test``
    until the live aggregator catches up.

    Args:
        path: Path to the live-metrics parquet.

    Returns:
        DataFrame limited to the 30d window, with ``book_bss`` renamed to
        ``gate2_book_bss`` and the over-rate columns suffixed ``_live``.
    """
    cols = [
        "league",
        "market",
        "n_settled",
        "gate2_book_bss",
        "predicted_over_rate_live",
        "empirical_over_rate_live",
        "precision_over_live",
        "precision_under_live",
        "profit_sim_yield",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(path, engine="pyarrow")
    df = df[df["window_days"] == GRADUATION_WINDOW_DAYS]
    df = df.rename(
        columns={
            "book_bss": "gate2_book_bss",
            "predicted_over_rate": "predicted_over_rate_live",
            "empirical_over_rate": "empirical_over_rate_live",
        }
    )
    # Older live_metrics parquets predate the precision_{over,under}_live
    # columns; default to NaN so classify_lifecycle's precision gate skips
    # those rows instead of erroring.
    for missing in ("precision_over_live", "precision_under_live"):
        if missing not in df.columns:
            df[missing] = float("nan")
    return df[cols]


def lifecycle_table(
    model_stats_path: Path,
    live_metrics_path: Path,
    league: str | None = None,
) -> pd.DataFrame:
    """Join Gate 1 + Gate 2 and add a ``lifecycle_state`` column per cell.

    Args:
        model_stats_path: Gate-1 parquet path (required to exist).
        live_metrics_path: Gate-2 parquet path (may be missing).
        league: Optional league filter.

    Returns:
        The merged frame with a ``lifecycle_state`` column. Empty (but carrying
        ``lifecycle_state``) when no Gate-1 rows match the filter.
    """
    gate1 = read_gate1(model_stats_path, league)
    if gate1.empty:
        out = gate1.copy()
        out["lifecycle_state"] = pd.Series(dtype="object")
        return out
    gate2 = read_gate2(live_metrics_path)
    merged = gate1.merge(gate2, on=["league", "market"], how="left")
    merged["lifecycle_state"] = merged.apply(
        lambda r: classify_lifecycle(
            r.get("gate1_bss", float("nan")),
            r.get("n_settled", float("nan")),
            r.get("gate2_book_bss", float("nan")),
            r.get("precision_over_live", float("nan")),
        ),
        axis=1,
    )
    return merged


def graduated_cells(model_stats_path: Path, live_metrics_path: Path) -> set[tuple[str, str]]:
    """Return the set of ``(league, market)`` cells classified ``graduated``.

    Tolerant of a missing model-stats parquet (returns an empty set) so the
    ``main`` branch — dormant until live data arrives — yields no graduated
    cells rather than erroring.

    Args:
        model_stats_path: Gate-1 parquet path.
        live_metrics_path: Gate-2 parquet path.

    Returns:
        Set of graduated ``(league, market)`` tuples.
    """
    if not model_stats_path.exists():
        return set()
    table = lifecycle_table(model_stats_path, live_metrics_path)
    if table.empty:
        return set()
    grad = table[table["lifecycle_state"] == "graduated"]
    return set(zip(grad["league"], grad["market"], strict=False))


def served_cells_failing_ship(
    meta: dict[str, dict[str, dict]],
    model_stats_path: Path,
) -> list[tuple[str, str]]:
    """Return ``(league, market)`` cells marked ``shipped`` devel/main yet failing ship.

    Feeds the two warning surfaces — meditate's post-run table and the
    ``generate-ship-config --branch devel`` summary — that tell the human which
    served cells fail the offline gates (``ship == False``; see
    ``training.scorecard``); the cull itself is a manual decision. A missing
    parquet or an ``NA`` ``ship`` (no scored evidence) is not a failure — only
    an explicit ``ship == False`` flags.
    """
    if not model_stats_path.exists():
        return []
    stats = pd.read_parquet(model_stats_path, columns=["league", "market", "ship"])
    failed = stats["ship"].eq(False).fillna(False).astype(bool)
    failing_rows = set(zip(stats.loc[failed, "league"], stats.loc[failed, "market"], strict=False))
    return [
        (league, market)
        for league, markets in meta.items()
        for market, cell in markets.items()
        if cell.get("shipped") in ("devel", "main") and (league, market) in failing_rows
    ]


def free_passer_cells(
    meta: dict[str, dict[str, dict]],
    model_stats_path: Path,
) -> list[tuple[str, str]]:
    """Return ``(league, market)`` cells passing ship yet still ``shipped: "withheld"``.

    The Gate-4 redefinition demoted cells but never auto-promoted the ones the
    new gate newly passes (the free-passer sweep, model_improvement_track §6.0).
    These are candidates for a manual flip to ``"devel"`` after a re-confirm on
    the official scorecard — auto-flipping is unsafe, since a sweep pass that
    disagrees with the scorecard is a scorer bug to chase. A missing parquet or
    an ``NA`` ``ship`` (no scored evidence) is not a pass — only an explicit
    ``ship == True`` qualifies.
    """
    if not model_stats_path.exists():
        return []
    stats = pd.read_parquet(model_stats_path, columns=["league", "market", "ship"])
    passed = stats["ship"].eq(True).fillna(False).astype(bool)
    passing_rows = set(zip(stats.loc[passed, "league"], stats.loc[passed, "market"], strict=False))
    return [
        (league, market)
        for league, markets in meta.items()
        for market, cell in markets.items()
        if cell.get("shipped") == "withheld" and (league, market) in passing_rows
    ]
