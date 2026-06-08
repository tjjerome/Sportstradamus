# Gate-driven serving control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ship_config.json` a generated artifact that withholds (dark-outs) every `(league, market)` cell that has not cleared its branch's gate — Gate 1 on `devel`, Gate 2 (live graduation) on `main` — so `prophecize` serves only gate-passing cells.

**Architecture:** A new canonical, human-curated `data/gate1_decisions.json` records "passed Gate 1 + which strategy" (seeded from the current 36 cells). `check_graduation`'s lifecycle classifier is extracted into an importable `training/graduation.py`. A new `generate-ship-config --branch {devel|main}` CLI reads the decisions file plus (for `main`) the live graduation parquets and writes an **exhaustive** `ship_config.json` over all 96 `ALL_MARKETS` cells: active cells get their decisions strategy, every other cell gets `"withheld"`. A monthly `run_job.sh gate-status` cron regenerates `main`'s config from live data and opens a PR (a human merges). This is a deliberate **default-deny** behavior change: cells previously `absent` (served with the default strategy) become explicitly `"withheld"`.

**Tech Stack:** Python 3.11, click (CLI), pandas + pyarrow (parquet I/O), pytest + click.testing.CliRunner (tests), poetry (console scripts), bash + git + `gh` (cron job).

---

## Background the implementer needs

- **Serving = a pickle exists on disk.** `prophecize`/`model_prob` never read `ship_config.json`; they decode the strategy baked into each pickle (`target_strategy` + `offset_meta`). The lever that controls *which pickles exist* is `ship_config.json`: a cell marked `"withheld"` makes `meditate` call `prune_model_pickle` and skip training, so inference dark-outs that market. See [src/sportstradamus/training/cli.py:269-286](../../../src/sportstradamus/training/cli.py#L269-L286).
- **`ship_config.json` has three states** (see [src/sportstradamus/training/ship_config.py](../../../src/sportstradamus/training/ship_config.py)): a real strategy slug (shipped), `"withheld"` (skip + prune), or absent (train with the run's default). The loader `load_ship_config` validates every value is a known slug or `"withheld"`.
- **`ALL_MARKETS` = 96 cells** (NFL 20, NBA 21, WNBA 18, MLB 22, NHL 15) — [src/sportstradamus/training/markets.py:5-112](../../../src/sportstradamus/training/markets.py#L5-L112).
- **Today's 36 Gate-1 cells** live in `ship_config.json` on the unmerged ship branch `ship/baselines-36-cells` (NBA 13, WNBA 10, NFL 13). On `devel`, `ship_config.json` is currently `{}`.
- **`check_graduation` prints only** — no state file. Its classifier and parquet readers are what Task 2 extracts. See [src/sportstradamus/scripts/check_graduation.py](../../../src/sportstradamus/scripts/check_graduation.py).
- **Design spec:** [docs/superpowers/specs/2026-05-22-gate-driven-serving-control-design.md](../specs/2026-05-22-gate-driven-serving-control-design.md) (approved).
- **Quality gates (CLAUDE.md):** `poetry run ruff check src/sportstradamus/`, `poetry run pytest tests/golden/`, `poetry run pytest -m integration` must all pass. The `refactoring-specialist` subagent must run on every touched `.py` before any push/PR/review.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/sportstradamus/data/gate1_decisions.json` | Canonical, branch-independent record of Gate-1 passers + their strategy (the 36). | Create |
| `src/sportstradamus/training/graduation.py` | Shared lifecycle classification + Gate-1/Gate-2 parquet readers + graduated-set helper. | Create |
| `src/sportstradamus/scripts/check_graduation.py` | Display only — imports the classifier from `graduation.py`. | Modify |
| `src/sportstradamus/scripts/generate_ship_config.py` | Synthesize an exhaustive per-branch `ship_config.json` from decisions + (for main) graduation. | Create |
| `src/sportstradamus/training/ship_config.py` | Add `GATE1_DECISIONS_PATH`; docstring note on default-deny generated configs. | Modify |
| `pyproject.toml` | Register the `generate-ship-config` console script. | Modify |
| `scripts/gate_status_update.sh` | Monthly: regenerate `main`'s config from live graduation, open a PR if changed. | Create |
| `scripts/run_job.sh` | Add the `gate-status` job. | Modify |
| `src/sportstradamus/data/ship_config.json` | Regenerated for `devel` (36 active + 60 withheld). | Regenerate |
| `docs/ship_gate.md`, `docs/gbdt_mean_regression_plan.md`, `CLAUDE.md` | Record default-deny + the generator workflow + the cron line. | Modify |
| `tests/test_gate1_decisions.py`, `tests/test_graduation.py`, `tests/test_generate_ship_config.py`, `tests/test_gate_status_job.py` | Tests. | Create |
| `tests/test_check_graduation.py` | Drop the moved unit test; fix the import. | Modify |

---

## Phase 1 — Canonical decisions file

### Task 1: Seed `gate1_decisions.json` + guard test

**Files:**
- Create: `src/sportstradamus/data/gate1_decisions.json`
- Test: `tests/test_gate1_decisions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate1_decisions.py
"""Guard test for the canonical Gate-1 decisions file (data/gate1_decisions.json)."""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

from sportstradamus import data
from sportstradamus.training.baselines import STRATEGY_SLUGS
from sportstradamus.training.markets import ALL_MARKETS

# The Gate-1 lock-in count seeded from ship/baselines-36-cells (NBA 13, WNBA 10, NFL 13).
_EXPECTED_DECISION_COUNT = 36


def _load_decisions() -> dict[str, dict[str, str]]:
    path = pkg_resources.files(data) / "gate1_decisions.json"
    with open(str(path)) as fh:
        return json.load(fh)


def test_decisions_has_expected_cell_count():
    decisions = _load_decisions()
    n_cells = sum(len(markets) for markets in decisions.values())
    assert n_cells == _EXPECTED_DECISION_COUNT


def test_every_decision_is_a_real_strategy():
    decisions = _load_decisions()
    for league, markets in decisions.items():
        for market, strategy in markets.items():
            assert strategy in STRATEGY_SLUGS, f"{league}/{market}={strategy!r}"


def test_every_decision_cell_in_all_markets():
    decisions = _load_decisions()
    for league, markets in decisions.items():
        assert league in ALL_MARKETS, league
        for market in markets:
            assert market in ALL_MARKETS[league], f"{league}/{market} not in ALL_MARKETS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_gate1_decisions.py -v`
Expected: FAIL — `FileNotFoundError` / `No such file` for `gate1_decisions.json`.

- [ ] **Step 3: Create the data file**

Write `src/sportstradamus/data/gate1_decisions.json` with exactly this content (the current 36 Gate-1 cells):

```json
{
    "NBA": {
        "DREB": "ratio_meanyr",
        "FG3A": "ratio_meanyr",
        "FGA": "ratio_meanyr",
        "FGM": "ratio_meanyr",
        "FTM": "ratio_meanyr",
        "MIN": "ratio_meanyr",
        "PA": "ratio_meanyr",
        "PR": "ratio_meanyr",
        "PRA": "ratio_meanyr",
        "PTS": "ratio_meanyr",
        "RA": "ratio_meanyr",
        "REB": "ratio_meanyr",
        "fantasy points prizepicks": "ratio_meanyr"
    },
    "WNBA": {
        "DREB": "centered_additive_mean10",
        "FGA": "ratio_meanyr",
        "MIN": "ratio_meanyr",
        "PA": "ratio_meanyr",
        "PR": "ratio_meanyr",
        "PRA": "ratio_meanyr",
        "PTS": "ratio_meanyr",
        "RA": "ratio_meanyr",
        "REB": "ratio_meanyr",
        "fantasy points prizepicks": "ratio_meanyr"
    },
    "NFL": {
        "attempts": "ratio_meanyr",
        "carries": "ratio_meanyr",
        "completions": "ratio_meanyr",
        "fantasy points prizepicks": "ratio_meanyr",
        "fantasy points underdog": "ratio_meanyr",
        "passing first downs": "centered_additive_eb_meanyr_k10",
        "passing yards": "ratio_meanyr",
        "qb yards": "ratio_meanyr",
        "receiving yards": "ratio_meanyr",
        "receptions": "centered_additive_mean10",
        "targets": "ratio_meanyr",
        "tds": "ratio_meanyr",
        "yards": "ratio_meanyr"
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_gate1_decisions.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/data/gate1_decisions.json tests/test_gate1_decisions.py
git commit -m "feat(gate): canonical gate1_decisions.json (36 Gate-1 cells)"
```

---

## Phase 2 — Shared graduation classifier

### Task 2: Extract the classifier into `training/graduation.py`

**Files:**
- Create: `src/sportstradamus/training/graduation.py`
- Test: `tests/test_graduation.py`

This is a pure move of `_classify_lifecycle`, `_read_gate1`, `_read_gate2` (and the merge/classify step) out of `check_graduation.py` into an importable module, with the names made public and one new helper (`graduated_cells`) the generator needs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graduation.py
"""Unit tests for the shared graduation classifier (training/graduation.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training.graduation import (
    classify_lifecycle,
    graduated_cells,
    read_gate1,
    read_gate2,
)


@pytest.mark.parametrize(
    ("gate1_bss", "n_settled", "book_bss_30d", "expected"),
    [
        (math.nan, 500, 0.10, "not-shipped"),
        (-0.05, 500, 0.10, "not-shipped"),
        (0.10, 50, 0.10, "in-test"),
        (0.10, 250, math.nan, "in-test"),  # live row missing entirely
        (0.10, 250, 0.05, "graduated"),
        (0.10, 250, 0.0, "graduated"),
        (0.10, 250, -0.04, "demoted"),
    ],
)
def test_classify_lifecycle(gate1_bss, n_settled, book_bss_30d, expected):
    assert classify_lifecycle(gate1_bss, n_settled, book_bss_30d) == expected


def _seed_model_stats(path):
    rows = []
    for league, market, bss in [("NBA", "PTS", 0.12), ("NBA", "FG3M", 0.08), ("WNBA", "PTS", -0.05)]:
        rows.append(
            {
                "league": league,
                "market": market,
                "distribution": "SkewNormal",
                "row_kind": "model",
                "metric_row": "calibrated",
                "brier_skill_score": bss,
                "predicted_over_rate": 0.52,
                "empirical_over_rate": 0.50,
                "kelly_shrinkage": 0.1,
            }
        )
        # A book_baseline row the reader must filter out.
        rows.append(
            {
                "league": league,
                "market": market,
                "distribution": "SkewNormal",
                "row_kind": "book_baseline",
                "metric_row": None,
                "brier_skill_score": 0.0,
                "predicted_over_rate": 0.50,
                "empirical_over_rate": 0.50,
                "kelly_shrinkage": 0.0,
            }
        )
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def _seed_live_metrics(path):
    rows = []
    for league, market, n, bss in [("NBA", "PTS", 300, 0.05), ("NBA", "FG3M", 220, -0.03)]:
        for window in (7, 30):
            rows.append(
                {
                    "league": league,
                    "market": market,
                    "computed_at": pd.Timestamp("2026-05-20"),
                    "window_days": np.int16(window),
                    "n_settled": np.int64(n),
                    "book_bss": bss,
                    "empirical_over_rate": 0.51,
                    "predicted_over_rate": 0.53,
                    "top_decile_mae": 2.1,
                    "profit_sim_yield": 0.04,
                }
            )
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def test_read_gate1_filters_and_renames(tmp_path):
    p = tmp_path / "model_stats.parquet"
    _seed_model_stats(p)
    df = read_gate1(p)
    assert "gate1_bss" in df.columns
    assert "brier_skill_score" not in df.columns
    assert len(df) == 3  # one model+calibrated row per cell; book_baseline filtered out
    assert set(zip(df["league"], df["market"], strict=False)) == {
        ("NBA", "PTS"),
        ("NBA", "FG3M"),
        ("WNBA", "PTS"),
    }


def test_read_gate1_league_filter(tmp_path):
    p = tmp_path / "model_stats.parquet"
    _seed_model_stats(p)
    df = read_gate1(p, league="WNBA")
    assert set(df["league"]) == {"WNBA"}


def test_read_gate2_window_filter_and_rename(tmp_path):
    p = tmp_path / "live.parquet"
    _seed_live_metrics(p)
    df = read_gate2(p)
    assert "gate2_book_bss" in df.columns
    assert len(df) == 2  # only the 30d window rows survive
    assert set(df["n_settled"]) == {300, 220}


def test_read_gate2_missing_returns_empty_with_columns(tmp_path):
    df = read_gate2(tmp_path / "nope.parquet")
    assert df.empty
    assert "gate2_book_bss" in df.columns


def test_graduated_cells(tmp_path):
    ms = tmp_path / "model_stats.parquet"
    lm = tmp_path / "live.parquet"
    _seed_model_stats(ms)
    _seed_live_metrics(lm)
    # NBA/PTS: gate1 0.12, n 300, book 0.05 -> graduated.
    # NBA/FG3M: gate1 0.08, n 220, book -0.03 -> demoted.
    # WNBA/PTS: gate1 -0.05 -> not-shipped.
    assert graduated_cells(ms, lm) == {("NBA", "PTS")}


def test_graduated_cells_missing_model_stats_is_empty(tmp_path):
    assert graduated_cells(tmp_path / "nope.parquet", tmp_path / "nolive.parquet") == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_graduation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sportstradamus.training.graduation'`.

- [ ] **Step 3: Write the module**

```python
# src/sportstradamus/training/graduation.py
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


def classify_lifecycle(gate1_bss: float, n_settled: float, book_bss_30d: float) -> str:
    """Map (Gate-1 BSS, n_settled, Gate-2 BSS) to a lifecycle state.

    NaN/negative Gate-1 BSS -> ``not-shipped``; positive Gate-1 BSS but
    insufficient live data -> ``in-test``; non-negative live BSS ->
    ``graduated``; negative live BSS -> ``demoted``.

    Args:
        gate1_bss: Offline calibrated brier-skill-score vs the book baseline.
        n_settled: Settled offer count in the graduation window.
        book_bss_30d: Live 30-day book-relative brier-skill-score.

    Returns:
        One of ``"not-shipped"``, ``"in-test"``, ``"graduated"``, ``"demoted"``.
    """
    if gate1_bss is None or (isinstance(gate1_bss, float) and math.isnan(gate1_bss)):
        return "not-shipped"
    if gate1_bss < 0:
        return "not-shipped"
    n_settled_nan = n_settled is None or (isinstance(n_settled, float) and math.isnan(n_settled))
    n_int = 0 if n_settled_nan else int(n_settled)
    if n_int < MIN_SETTLED_FOR_GRADUATION:
        return "in-test"
    if book_bss_30d is None or (isinstance(book_bss_30d, float) and math.isnan(book_bss_30d)):
        return "in-test"
    if book_bss_30d < 0:
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
    if not Path(str(path)).exists():
        raise click.UsageError(f"model_stats parquet not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    df = df[(df["row_kind"] == "model") & (df["metric_row"] == "calibrated")]
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
        "profit_sim_yield",
    ]
    if not Path(str(path)).exists():
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
    if not Path(str(model_stats_path)).exists():
        return set()
    table = lifecycle_table(model_stats_path, live_metrics_path)
    if table.empty:
        return set()
    grad = table[table["lifecycle_state"] == "graduated"]
    return set(zip(grad["league"], grad["market"], strict=False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_graduation.py -v`
Expected: PASS (all parametrized + reader + graduated_cells tests).

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/training/graduation.py tests/test_graduation.py
git commit -m "feat(gate): extract shared lifecycle classifier into training/graduation.py"
```

### Task 3: Point `check_graduation.py` at the shared module

**Files:**
- Modify: `src/sportstradamus/scripts/check_graduation.py`
- Modify: `tests/test_check_graduation.py`

- [ ] **Step 1: Rewrite `check_graduation.py` to import from `graduation.py`**

Replace the whole file with the display-only version (the classifier, the two readers, and the merge/classify step now live in `graduation.py`):

```python
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
```

- [ ] **Step 2: Fix the test import (drop the moved unit test)**

In `tests/test_check_graduation.py`, change the import line (top of file):

```python
from sportstradamus.scripts.check_graduation import _classify_lifecycle, main
```

to:

```python
from sportstradamus.scripts.check_graduation import main
```

Then **delete** the `test_lifecycle_classification` function and its `@pytest.mark.parametrize` decorator (lines ~15-28) — that unit test now lives in `tests/test_graduation.py`. The four CLI tests (`test_check_graduation_synthetic_parquets`, `test_check_graduation_filters_by_league`, `test_check_graduation_missing_live_parquet_classifies_all_as_in_test`, `test_check_graduation_missing_model_stats_errors`) stay unchanged. If `pytest`/`math` become unused after the deletion, remove those imports too (keep `numpy`, `pandas`, `CliRunner`).

- [ ] **Step 3: Run the check_graduation tests to verify behavior is unchanged**

Run: `poetry run pytest tests/test_check_graduation.py tests/test_graduation.py -v`
Expected: PASS — the four CLI tests are green (pure move) and the relocated classifier tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/sportstradamus/scripts/check_graduation.py tests/test_check_graduation.py
git commit -m "refactor(gate): check_graduation imports shared classifier (no behavior change)"
```

---

## Phase 3 — The generator CLI

### Task 4: Add `GATE1_DECISIONS_PATH` + the generator's pure functions

**Files:**
- Modify: `src/sportstradamus/training/ship_config.py`
- Create: `src/sportstradamus/scripts/generate_ship_config.py`
- Test: `tests/test_generate_ship_config.py`

- [ ] **Step 1: Add the decisions path constant to `ship_config.py`**

In `src/sportstradamus/training/ship_config.py`, directly below the existing `SHIP_CONFIG_PATH` line (line 34), add:

```python
# Canonical, branch-independent Gate-1 record (which cells passed + their
# strategy). generate-ship-config reads this to synthesize ship_config.json.
GATE1_DECISIONS_PATH = pkg_resources.files(data) / "gate1_decisions.json"
```

- [ ] **Step 2: Write the failing tests for the pure functions**

```python
# tests/test_generate_ship_config.py
"""Unit tests for the gate-driven ship_config generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sportstradamus.scripts.generate_ship_config import (
    active_cells,
    build_ship_config,
    load_decisions,
)
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.ship_config import WITHHELD, load_ship_config

_N_ALL_MARKETS = sum(len(markets) for markets in ALL_MARKETS.values())  # 96


def _write(path, mapping):
    Path(path).write_text(json.dumps(mapping))


def test_load_decisions_rejects_withheld(tmp_path):
    p = tmp_path / "d.json"
    _write(p, {"NBA": {"PTS": WITHHELD}})
    with pytest.raises(ValueError):
        load_decisions(p)


def test_load_decisions_accepts_real_slugs(tmp_path):
    p = tmp_path / "d.json"
    _write(p, {"NBA": {"PTS": "ratio_meanyr"}})
    assert load_decisions(p) == {"NBA": {"PTS": "ratio_meanyr"}}


def test_build_ship_config_is_exhaustive_and_withholds_the_rest():
    cfg = build_ship_config({"NBA": {"PTS": "ratio_meanyr"}}, {("NBA", "PTS")})
    assert sum(len(markets) for markets in cfg.values()) == _N_ALL_MARKETS
    assert cfg["NBA"]["PTS"] == "ratio_meanyr"
    assert cfg["NBA"]["REB"] == WITHHELD
    assert cfg["MLB"]["hits"] == WITHHELD


def test_build_ship_config_rejects_cell_not_in_all_markets():
    with pytest.raises(ValueError):
        build_ship_config({"NBA": {"NOT_A_MARKET": "ratio_meanyr"}}, set())


def test_build_ship_config_is_deterministic():
    decisions = {"NBA": {"PTS": "ratio_meanyr"}}
    first = build_ship_config(decisions, {("NBA", "PTS")})
    second = build_ship_config(decisions, {("NBA", "PTS")})
    # Same inputs -> byte-identical serialized output (sorted keys).
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_ship_config_output_passes_load_ship_config(tmp_path):
    cfg = build_ship_config({"NBA": {"PTS": "ratio_meanyr"}}, {("NBA", "PTS")})
    p = tmp_path / "ship_config.json"
    p.write_text(json.dumps(cfg, indent=4, sort_keys=True))
    assert load_ship_config(p) == cfg


def test_active_cells_devel_is_all_decisions(tmp_path):
    decisions = {"NBA": {"PTS": "ratio_meanyr", "REB": "ratio_meanyr"}}
    got = active_cells("devel", decisions, tmp_path / "ms", tmp_path / "lm")
    assert got == {("NBA", "PTS"), ("NBA", "REB")}


def test_active_cells_main_no_data_is_empty(tmp_path):
    decisions = {"NBA": {"PTS": "ratio_meanyr"}}
    got = active_cells("main", decisions, tmp_path / "ms", tmp_path / "lm")
    assert got == set()


def test_active_cells_main_intersects_graduated(tmp_path):
    decisions = {"NBA": {"PTS": "ratio_meanyr"}}
    ms = tmp_path / "ms.parquet"
    lm = tmp_path / "lm.parquet"
    pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "PTS",
                "distribution": "SkewNormal",
                "row_kind": "model",
                "metric_row": "calibrated",
                "brier_skill_score": 0.12,
                "predicted_over_rate": 0.5,
                "empirical_over_rate": 0.5,
                "kelly_shrinkage": 0.1,
            }
        ]
    ).to_parquet(ms, engine="pyarrow", index=False)
    pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "PTS",
                "computed_at": pd.Timestamp("2026-05-20"),
                "window_days": np.int16(30),
                "n_settled": np.int64(300),
                "book_bss": 0.05,
                "empirical_over_rate": 0.5,
                "predicted_over_rate": 0.5,
                "top_decile_mae": 2.0,
                "profit_sim_yield": 0.03,
            }
        ]
    ).to_parquet(lm, engine="pyarrow", index=False)
    assert active_cells("main", decisions, ms, lm) == {("NBA", "PTS")}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/test_generate_ship_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sportstradamus.scripts.generate_ship_config'`.

- [ ] **Step 4: Write the module's pure functions (CLI added in Task 5)**

```python
# src/sportstradamus/scripts/generate_ship_config.py
#!/usr/bin/env python3
"""Generate an exhaustive, gate-driven ship_config.json for one branch.

Reads the canonical ``gate1_decisions.json`` (which cells passed Gate 1 + their
strategy) and writes ``ship_config.json`` over **all** ``ALL_MARKETS`` cells:
active cells get their decisions strategy, every other cell gets ``"withheld"``.
This is default-deny serving control — only gate-passing cells keep a pickle and
are served by prophecize.

Active set by branch:

* ``devel`` — every cell in the decisions file (Gate-1 passers).
* ``main`` — decisions cells that are also live-``graduated`` (Gate 2), per
  ``training.graduation``. Dormant (empty) until live metrics exist.

Usage
-----
    poetry run generate-ship-config --branch devel
    poetry run generate-ship-config --branch main --dry-run
    poetry run generate-ship-config --branch devel --prune
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from sportstradamus.helpers.io import (
    LIVE_METRICS_PATH,
    MODEL_STATS_PATH,
    prune_model_pickle,
)
from sportstradamus.training.baselines import STRATEGY_SLUGS
from sportstradamus.training.graduation import graduated_cells
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.ship_config import (
    GATE1_DECISIONS_PATH,
    SHIP_CONFIG_PATH,
    WITHHELD,
    ShipConfig,
)


def load_decisions(path: Path) -> ShipConfig:
    """Load and validate ``gate1_decisions.json``.

    Args:
        path: Path to the decisions JSON.

    Returns:
        Nested ``{league: {market: strategy}}`` map.

    Raises:
        ValueError: If any value is not a known strategy slug. The decisions
            file records real strategies only — ``"withheld"`` is a generated
            ship_config value, never a decision.
    """
    with open(str(path)) as fh:
        decisions: ShipConfig = json.load(fh)
    for league, markets in decisions.items():
        for market, strategy in markets.items():
            if strategy not in STRATEGY_SLUGS:
                raise ValueError(
                    f"gate1_decisions.json: {league}/{market} has non-strategy "
                    f"value {strategy!r}; valid: {STRATEGY_SLUGS}"
                )
    return decisions


def active_cells(
    branch: str,
    decisions: ShipConfig,
    model_stats_path: Path,
    live_metrics_path: Path,
) -> set[tuple[str, str]]:
    """Return the set of cells that serve on ``branch``.

    Args:
        branch: ``"devel"`` (all decisions) or ``"main"`` (decisions that are
            also live-graduated).
        decisions: Loaded decisions map.
        model_stats_path: Gate-1 parquet path (only read for ``main``).
        live_metrics_path: Gate-2 parquet path (only read for ``main``).

    Returns:
        Set of active ``(league, market)`` tuples.

    Raises:
        ValueError: If ``branch`` is neither ``"devel"`` nor ``"main"``.
    """
    decision_cells = {
        (league, market) for league, markets in decisions.items() for market in markets
    }
    if branch == "devel":
        return decision_cells
    if branch == "main":
        return decision_cells & graduated_cells(model_stats_path, live_metrics_path)
    raise ValueError(f"unknown branch {branch!r}; expected 'devel' or 'main'")


def build_ship_config(decisions: ShipConfig, active: set[tuple[str, str]]) -> ShipConfig:
    """Build an exhaustive ship_config over ``ALL_MARKETS``.

    Active cells get their decisions strategy; every other ``ALL_MARKETS`` cell
    gets ``"withheld"``. Output is deterministic (leagues and markets sorted).

    Args:
        decisions: Loaded decisions map (its strategies fill the active cells).
        active: The set of active ``(league, market)`` tuples.

    Returns:
        Nested ``{league: {market: strategy-or-withheld}}`` over all 96 cells.

    Raises:
        ValueError: If a decisions cell is not in ``ALL_MARKETS`` (typo guard).
    """
    for league, markets in decisions.items():
        for market in markets:
            if league not in ALL_MARKETS or market not in ALL_MARKETS[league]:
                raise ValueError(f"decisions cell {league}/{market} not in ALL_MARKETS")
    config: ShipConfig = {}
    for league in sorted(ALL_MARKETS):
        cell: dict[str, str] = {}
        for market in sorted(ALL_MARKETS[league]):
            if (league, market) in active:
                cell[market] = decisions[league][market]
            else:
                cell[market] = WITHHELD
        config[league] = cell
    return config
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/test_generate_ship_config.py -v`
Expected: PASS (load_decisions, build_ship_config, active_cells tests).

- [ ] **Step 6: Commit**

```bash
git add src/sportstradamus/training/ship_config.py src/sportstradamus/scripts/generate_ship_config.py tests/test_generate_ship_config.py
git commit -m "feat(gate): generate-ship-config core (decisions -> exhaustive config)"
```

### Task 5: Add the generator CLI + console script

**Files:**
- Modify: `src/sportstradamus/scripts/generate_ship_config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_generate_ship_config.py`

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/test_generate_ship_config.py`:

```python
from click.testing import CliRunner  # noqa: E402  (grouped with CLI tests)

from sportstradamus.scripts.generate_ship_config import main  # noqa: E402


def _invoke(args):
    return CliRunner().invoke(main, args)


def test_cli_devel_writes_active_plus_withheld(tmp_path):
    dpath = tmp_path / "decisions.json"
    _write(dpath, {"NBA": {"PTS": "ratio_meanyr"}})
    out = tmp_path / "ship_config.json"
    result = _invoke(
        [
            "--branch", "devel",
            "--decisions", str(dpath),
            "--out", str(out),
            "--model-stats", str(tmp_path / "ms.parquet"),
            "--live-metrics", str(tmp_path / "lm.parquet"),
        ]
    )
    assert result.exit_code == 0, result.output
    cfg = json.loads(out.read_text())
    assert cfg["NBA"]["PTS"] == "ratio_meanyr"
    assert cfg["NBA"]["REB"] == WITHHELD
    n_active = sum(1 for lg in cfg for mk in cfg[lg] if cfg[lg][mk] != WITHHELD)
    assert n_active == 1


def test_cli_main_no_data_all_withheld(tmp_path):
    dpath = tmp_path / "decisions.json"
    _write(dpath, {"NBA": {"PTS": "ratio_meanyr"}})
    out = tmp_path / "ship_config.json"
    result = _invoke(
        [
            "--branch", "main",
            "--decisions", str(dpath),
            "--out", str(out),
            "--model-stats", str(tmp_path / "ms.parquet"),
            "--live-metrics", str(tmp_path / "lm.parquet"),
        ]
    )
    assert result.exit_code == 0, result.output
    cfg = json.loads(out.read_text())
    assert all(cfg[lg][mk] == WITHHELD for lg in cfg for mk in cfg[lg])


def test_cli_dry_run_does_not_write(tmp_path):
    dpath = tmp_path / "decisions.json"
    _write(dpath, {"NBA": {"PTS": "ratio_meanyr"}})
    out = tmp_path / "ship_config.json"
    result = _invoke(
        [
            "--branch", "devel",
            "--decisions", str(dpath),
            "--out", str(out),
            "--model-stats", str(tmp_path / "ms.parquet"),
            "--live-metrics", str(tmp_path / "lm.parquet"),
            "--dry-run",
        ]
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert "ratio_meanyr" in result.output


def test_cli_prune_deletes_only_non_active_pickles(tmp_path, monkeypatch):
    import sportstradamus.helpers.io as io_mod

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(io_mod, "MODELS_DIR", models)
    (models / "NBA_PTS.mdl").write_text("x")  # active -> kept
    (models / "NBA_REB.mdl").write_text("x")  # non-active -> pruned

    dpath = tmp_path / "decisions.json"
    _write(dpath, {"NBA": {"PTS": "ratio_meanyr"}})
    out = tmp_path / "ship_config.json"
    result = _invoke(
        [
            "--branch", "devel",
            "--decisions", str(dpath),
            "--out", str(out),
            "--model-stats", str(tmp_path / "ms.parquet"),
            "--live-metrics", str(tmp_path / "lm.parquet"),
            "--prune",
        ]
    )
    assert result.exit_code == 0, result.output
    assert (models / "NBA_PTS.mdl").exists()
    assert not (models / "NBA_REB.mdl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_generate_ship_config.py -k cli -v`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Append the CLI to `generate_ship_config.py`**

```python
@click.command()
@click.option(
    "--branch",
    type=click.Choice(["devel", "main"]),
    required=True,
    help="Which branch's gate to apply: 'devel' = Gate 1 passers, 'main' = graduated.",
)
@click.option(
    "--prune/--no-prune",
    default=False,
    help="Also delete every non-active cell's model pickle (immediate dark-out on this machine).",
)
@click.option(
    "--decisions",
    type=click.Path(path_type=Path),
    default=None,
    help="Decisions JSON path (defaults to data/gate1_decisions.json).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output ship_config.json path (defaults to data/ship_config.json).",
)
@click.option(
    "--model-stats",
    type=click.Path(path_type=Path),
    default=None,
    help="Gate 1 parquet path (defaults to data/model_stats.parquet). Only read for --branch main.",
)
@click.option(
    "--live-metrics",
    type=click.Path(path_type=Path),
    default=None,
    help="Gate 2 parquet path (defaults to data/live_metrics_per_market.parquet). main only.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Print the config; do not write or prune.")
def main(branch, prune, decisions, out, model_stats, live_metrics, dry_run) -> None:
    """Write an exhaustive, gate-driven ship_config.json for one branch."""
    decisions_path = Path(decisions) if decisions else Path(str(GATE1_DECISIONS_PATH))
    out_path = Path(out) if out else Path(str(SHIP_CONFIG_PATH))
    model_stats_path = Path(model_stats) if model_stats else Path(str(MODEL_STATS_PATH))
    live_metrics_path = Path(live_metrics) if live_metrics else Path(str(LIVE_METRICS_PATH))

    decisions_map = load_decisions(decisions_path)
    active = active_cells(branch, decisions_map, model_stats_path, live_metrics_path)
    config = build_ship_config(decisions_map, active)

    n_active = sum(1 for lg in config for mk in config[lg] if config[lg][mk] != WITHHELD)
    n_withheld = sum(1 for lg in config for mk in config[lg] if config[lg][mk] == WITHHELD)
    payload = json.dumps(config, indent=4, sort_keys=True)

    if dry_run:
        click.echo(payload)
        click.echo(f"# branch={branch} active={n_active} withheld={n_withheld} (dry-run, not written)")
        return

    out_path.write_text(payload + "\n")
    click.echo(f"wrote {out_path}: active={n_active} withheld={n_withheld} (branch={branch})")

    if prune:
        pruned = 0
        for league in ALL_MARKETS:
            for market in ALL_MARKETS[league]:
                if (league, market) not in active and prune_model_pickle(league, market):
                    pruned += 1
        click.echo(f"pruned {pruned} non-active pickles")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the console script in `pyproject.toml`**

In `[tool.poetry.scripts]`, directly below the `check-graduation` line (line 51), add:

```toml
generate-ship-config = "sportstradamus.scripts.generate_ship_config:main"
```

- [ ] **Step 5: Re-install so the console script resolves, then run the CLI tests**

```bash
poetry install
poetry run pytest tests/test_generate_ship_config.py -v
```
Expected: PASS (all pure-function + CLI tests). Also confirm the entry point exists:

```bash
poetry run generate-ship-config --help
```
Expected: help text listing `--branch`, `--prune/--no-prune`, `--decisions`, `--out`, `--model-stats`, `--live-metrics`, `--dry-run`.

- [ ] **Step 6: Commit**

```bash
git add src/sportstradamus/scripts/generate_ship_config.py pyproject.toml tests/test_generate_ship_config.py
git commit -m "feat(gate): generate-ship-config CLI + console script"
```

---

## Phase 4 — Apply to `devel`

### Task 6: Regenerate `devel`'s `ship_config.json` (36 active + 60 withheld)

**Files:**
- Regenerate: `src/sportstradamus/data/ship_config.json`

> **Branch note:** This supersedes the lean 36-cell `ship_config.json` carried on `ship/baselines-36-cells`. Run this on the branch that ships the gate system to `devel` (i.e. after Phases 1-3 are present). The result darkens MLB (0/22) and NHL (0/15) entirely, plus the non-baselined NFL/NBA/WNBA cells.

- [ ] **Step 1: Generate the devel config**

```bash
poetry run generate-ship-config --branch devel
```
Expected stdout: `wrote .../ship_config.json: active=36 withheld=60 (branch=devel)`.

- [ ] **Step 2: Verify the counts and that it loads**

```bash
poetry run python -c "
from sportstradamus.training.ship_config import load_ship_config, WITHHELD, SHIP_CONFIG_PATH
from pathlib import Path
cfg = load_ship_config(Path(str(SHIP_CONFIG_PATH)))
active = sum(1 for lg in cfg for mk in cfg[lg] if cfg[lg][mk] != WITHHELD)
withheld = sum(1 for lg in cfg for mk in cfg[lg] if cfg[lg][mk] == WITHHELD)
print('active', active, 'withheld', withheld, 'total', active + withheld)
assert (active, withheld) == (36, 60), (active, withheld)
print('OK')
"
```
Expected: `active 36 withheld 60 total 96` then `OK`. (`load_ship_config` raising would mean a bad value — it cannot, by construction.)

- [ ] **Step 3: Confirm the three non-default cells survived as their strategy**

```bash
poetry run python -c "
import json
from pathlib import Path
from sportstradamus.training.ship_config import SHIP_CONFIG_PATH
cfg = json.loads(Path(str(SHIP_CONFIG_PATH)).read_text())
assert cfg['WNBA']['DREB'] == 'centered_additive_mean10'
assert cfg['NFL']['passing first downs'] == 'centered_additive_eb_meanyr_k10'
assert cfg['NFL']['receptions'] == 'centered_additive_mean10'
assert cfg['MLB']['hits'] == 'withheld' and cfg['NHL']['goals'] == 'withheld'
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/sportstradamus/data/ship_config.json
git commit -m "feat(gate): regenerate devel ship_config (36 active + 60 withheld, default-deny)"
```

---

## Phase 5 — Monthly cron (`main` only)

### Task 7: `gate-status` job — regenerate `main`'s config + open a PR

**Files:**
- Create: `scripts/gate_status_update.sh`
- Modify: `scripts/run_job.sh`
- Test: `tests/test_gate_status_job.py`

> **Why only `main` is on a timer:** `devel`'s active set = Gate-1 passers, which change only when a human edits `gate1_decisions.json`; regenerate `devel` manually then (Task 6). `main`'s active set = graduated cells, which evolve with live data — so only `main` needs a periodic refresh. The job opens a PR (never pushes to `main` directly; `main` is the public branch).
>
> **Deploy prerequisites (operational, not code):** the production box needs `gh` authenticated (`GH_TOKEN` or `gh auth login`) and push rights for a `gate-status/*` branch. Set `HEALTHCHECK_URL_GATE_STATUS` for the monthly job's healthcheck. These are documented in Task 8, not configured here.

- [ ] **Step 1: Write the failing fake-mode test**

```python
# tests/test_gate_status_job.py
"""Fake-mode test for the monthly gate-status cron wrapper (no git, no network)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "gate_status_update.sh"


@pytest.mark.integration
def test_gate_status_dry_run_invokes_generator_and_skips_git(tmp_path):
    env = dict(os.environ)
    env["GATE_STATUS_DRY_RUN"] = "1"
    # Stub the generator so the test touches neither poetry nor the real data.
    env["GENERATE_SHIP_CONFIG_CMD"] = "echo generate-ship-config-stub"
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    assert result.returncode == 0, result.stderr
    assert "generate-ship-config-stub" in result.stdout
    assert "dry-run" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest -m integration tests/test_gate_status_job.py -v`
Expected: FAIL — the script does not exist (`bash: .../gate_status_update.sh: No such file or directory`, non-zero exit).

- [ ] **Step 3: Write `scripts/gate_status_update.sh`**

```bash
#!/usr/bin/env bash
# Monthly gate-status job: regenerate main's ship_config.json from live
# graduation metrics and, if it changed, open a PR for a human to merge.
# main is the public branch, so this NEVER pushes to main directly. Invoked
# via `run_job.sh gate-status` (which adds flock + healthchecks).
#
# Environment (optional):
#   GATE_STATUS_DRY_RUN       non-empty -> generate (or stub) and stop before git.
#   GENERATE_SHIP_CONFIG_CMD  generator command (default: "poetry run generate-ship-config").
#   GATE_STATUS_WORKTREE      worktree dir for the main checkout (default: $PROJECT_DIR/.gate-status-main).
#   SPORTSTRADAMUS_DIR        project root (default: parent of this script).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${SPORTSTRADAMUS_DIR:-$(dirname -- "$SCRIPT_DIR")}"
SHIP_CONFIG_REL="src/sportstradamus/data/ship_config.json"
GEN_CMD="${GENERATE_SHIP_CONFIG_CMD:-poetry run generate-ship-config}"

cd "$PROJECT_DIR"

# Dry/fake mode (tests, manual dry runs): generate or stub, then stop before git.
if [[ -n "${GATE_STATUS_DRY_RUN:-}" ]]; then
    $GEN_CMD --branch main --dry-run
    echo "gate-status: dry-run complete (no PR)."
    exit 0
fi

git fetch origin main

TMP_CONFIG="$(mktemp)"
TMP_MAIN="$(mktemp)"
trap 'rm -f "$TMP_CONFIG" "$TMP_MAIN"' EXIT

$GEN_CMD --branch main --out "$TMP_CONFIG"

# No change vs main's committed config -> nothing to do.
if git show "origin/main:$SHIP_CONFIG_REL" > "$TMP_MAIN" 2>/dev/null \
        && diff -q "$TMP_MAIN" "$TMP_CONFIG" >/dev/null 2>&1; then
    echo "gate-status: main ship_config unchanged; no PR."
    exit 0
fi

# A change exists -> land it on a fresh branch off origin/main and open a PR.
BRANCH="gate-status/main-$(date -u +%Y%m%d)"
WORKTREE_DIR="${GATE_STATUS_WORKTREE:-$PROJECT_DIR/.gate-status-main}"
git worktree add --force -B "$BRANCH" "$WORKTREE_DIR" origin/main
cp "$TMP_CONFIG" "$WORKTREE_DIR/$SHIP_CONFIG_REL"
git -C "$WORKTREE_DIR" add "$SHIP_CONFIG_REL"
git -C "$WORKTREE_DIR" commit -m "chore(gate): refresh main ship_config from live graduation"
git -C "$WORKTREE_DIR" push -u origin "$BRANCH"
gh pr create --base main --head "$BRANCH" \
    --title "Gate-status: refresh main ship_config from live graduation" \
    --body "Automated monthly regeneration of main's ship_config.json from live graduation metrics (training/graduation.py). Review the active/withheld diff before merging; main is the public branch."
git worktree remove --force "$WORKTREE_DIR"
echo "gate-status: opened PR on branch $BRANCH."
```

Make it executable:

```bash
chmod +x scripts/gate_status_update.sh
```

- [ ] **Step 4: Add the `gate-status` job to `run_job.sh`**

In `scripts/run_job.sh`, add a comment line in the Jobs block (after the `reflect` line, ~line 13):

```bash
#   gate-status        scripts/gate_status_update.sh (monthly: refresh main ship_config, open PR)
```

Update the usage echo (line 38) from:

```bash
    echo "usage: $(basename "$0") <prophecize|confer|close-lines|meditate|reflect> [args...]" >&2
```
to:
```bash
    echo "usage: $(basename "$0") <prophecize|confer|close-lines|meditate|reflect|gate-status> [args...]" >&2
```

Add a case branch (after the `reflect)` case, ~line 50). `$SCRIPT_DIR` is already defined at the top of `run_job.sh`:

```bash
    gate-status)  CMD=(bash "$SCRIPT_DIR/gate_status_update.sh") ;;
```

(The healthcheck slug resolves automatically: `gate-status` -> `HEALTHCHECK_URL_GATE_STATUS` via the existing `tr '[:lower:]-' '[:upper:]_'` mapping.)

- [ ] **Step 5: Run the fake-mode test to verify it passes**

Run: `poetry run pytest -m integration tests/test_gate_status_job.py -v`
Expected: PASS — the dry-run path echoes the stubbed generator and "dry-run", exits 0, touches no git.

- [ ] **Step 6: Smoke-check the run_job.sh dispatch (still fake-mode)**

```bash
GATE_STATUS_DRY_RUN=1 GENERATE_SHIP_CONFIG_CMD="echo stub" bash scripts/run_job.sh gate-status
tail -n 5 logs/gate-status.log
```
Expected: the log shows `START job=gate-status ...` and `OK job=gate-status ...`; the dry-run printed `stub --branch main --dry-run` and `gate-status: dry-run complete`.

- [ ] **Step 7: Commit**

```bash
git add scripts/gate_status_update.sh scripts/run_job.sh tests/test_gate_status_job.py
git commit -m "feat(gate): monthly gate-status cron (regenerate main ship_config, open PR)"
```

---

## Phase 6 — Documentation (default-deny)

### Task 8: Record the behavior change + workflow + cron line

**Files:**
- Modify: `docs/ship_gate.md`
- Modify: `docs/gbdt_mean_regression_plan.md`
- Modify: `CLAUDE.md`

No code; no tests. Pull `origin` first (per the standing instruction) in case docs moved, then edit.

- [ ] **Step 1: `docs/ship_gate.md` — add a "Serving control" section**

After the "two-tier ship model" section, add:

```markdown
## Serving control — default-deny via generated ship_config

`ship_config.json` is a **generated artifact**, not hand-edited. The canonical
human-curated source is `data/gate1_decisions.json` (`{league: {market:
strategy}}` for Gate-1 passers). `generate-ship-config --branch {devel|main}`
writes `ship_config.json` exhaustively over **all** `ALL_MARKETS` cells:

- a cell that passed the branch's gate gets its decisions strategy (served);
- every other cell gets `"withheld"` — `meditate` prunes its pickle so
  `prophecize` dark-outs the market.

This is **default-deny**: only gate-passing cells serve. `--branch devel` =
Gate-1 passers (regenerate manually when `gate1_decisions.json` changes);
`--branch main` = Gate-2 graduated cells (regenerated monthly by the
`run_job.sh gate-status` cron, which opens a PR a human merges).

**Known gap:** the graduated classifier (`training/graduation.py`) uses a
proxy of Gate 2 — positive Gate-1 BSS + ≥ 200 settled offers in the 30d window
+ non-negative live book-BSS — not the full metric set below. `main` is dormant
until the live aggregator produces data, so the proxy is acceptable for now.
```

- [ ] **Step 2: `docs/gbdt_mean_regression_plan.md` — update the ship-mechanism / lifecycle section**

Find the "Ship mechanism — per-cell strategy config on devel" section and the lifecycle line `absent ──(begin rework)──▶ "withheld" ──(Gate 1 pass)──▶ "<strategy>" ──(Gate 2 pass)──▶ graduated`. Add a paragraph directly under that section:

```markdown
**Default-deny (gate-driven serving control).** `ship_config.json` is now
generated by `generate-ship-config --branch {devel|main}` from the canonical
`data/gate1_decisions.json`. Generated configs are **exhaustive over
`ALL_MARKETS`** — a cell that has not passed the branch's gate is explicitly
`"withheld"`, not absent. The `absent → "withheld" → "<strategy>" → graduated`
lifecycle is realized across branches: `"<strategy>"` on `devel` = Gate-1
shipped; `"<strategy>"` on `main` = Gate-2 graduated. The `absent = serve the
default strategy` path still exists in the `load_ship_config` loader, but the
generator never emits it for `devel`/`main`. Regenerate `devel` manually on a
`gate1_decisions.json` change; `main` is refreshed monthly by `run_job.sh
gate-status` (opens a PR).
```

- [ ] **Step 3: `CLAUDE.md` — add the cron line + deploy note**

In the "Production crontab" code block, add a line:

```cron
0 2 1 * *              /home/sportstradamus/Sportstradamus/scripts/run_job.sh gate-status
```

Below the crontab block, add:

```markdown
The `gate-status` job runs monthly: it regenerates `main`'s `ship_config.json`
from live graduation and opens a PR (a human merges — `main` is the public
branch). It needs `gh` authenticated on the box (`GH_TOKEN` or `gh auth`) and
`HEALTHCHECK_URL_GATE_STATUS` set. `devel`'s `ship_config.json` is regenerated
manually with `generate-ship-config --branch devel` whenever
`gate1_decisions.json` changes.
```

- [ ] **Step 4: Verify the docs render and reference real paths**

```bash
grep -n "generate-ship-config" docs/ship_gate.md docs/gbdt_mean_regression_plan.md CLAUDE.md
grep -n "gate-status" CLAUDE.md scripts/run_job.sh
```
Expected: matches in each file; no broken references.

- [ ] **Step 5: Commit**

```bash
git add docs/ship_gate.md docs/gbdt_mean_regression_plan.md CLAUDE.md
git commit -m "docs(gate): record default-deny serving control + gate-status cron"
```

---

## Phase 7 — `main` foundation carve (orchestration, after Phases 1-6 land on `devel`)

### Task 9: Carve the serving system onto `main` via `devel-ship-curator`

This is a git-carve workstream, not TDD. `main` lacks the entire ship system; before `generate-ship-config --branch main` output is usable there, the foundation must reach `main`. Sequence this **after** Phases 1-6 are merged to `devel`, so the carve includes the new modules.

- [ ] **Step 1: Dispatch the curator targeting `main`**

Use the `Agent` tool with `subagent_type: "devel-ship-curator"`. Instruct it to branch off `main` (e.g. `ship/main-serving-foundation`) and bring exactly the production-runtime serving system from `devel`:

- `src/sportstradamus/training/ship_config.py` (loader + `WITHHELD` + `GATE1_DECISIONS_PATH`)
- `src/sportstradamus/training/baselines.py` (strategy registry + `get_strategy`)
- `src/sportstradamus/training/graduation.py` (shared classifier)
- `src/sportstradamus/training/markets.py` (`ALL_MARKETS`)
- `src/sportstradamus/scripts/generate_ship_config.py` + the `pyproject.toml` console-script line
- `src/sportstradamus/scripts/check_graduation.py` + `pyproject.toml` `check-graduation` line (if not already on main)
- `src/sportstradamus/helpers/io.py` (`model_pickle_path`, `prune_model_pickle`, `MODELS_DIR`, `MODEL_STATS_PATH`, `LIVE_METRICS_PATH`, `market_file_slug`)
- the `model_prob` self-describing-pickle decode path + the `pipeline`/`cli` per-cell strategy wiring (whatever the foundation needs to train + serve)
- `src/sportstradamus/data/gate1_decisions.json`
- `scripts/gate_status_update.sh` + the `run_job.sh` `gate-status` case
- the four test files (`tests/test_gate1_decisions.py`, `tests/test_graduation.py`, `tests/test_generate_ship_config.py`, `tests/test_gate_status_job.py`) + `tests/test_ship_config.py` if absent on main

**Hard-exclude** (denylist, enforced by the curator): `scripts/compression_eval.py`, `scripts/icc_diagnostics.py`, `scripts/zinb_routing_diagnostics.py` and their tests; the `statsmodels` dependency and the `zinb-routing-diagnostics` / `icc-diagnostics` console-scripts; any `/tmp` research harnesses.

- [ ] **Step 2: Run `refactoring-specialist` on every carried `.py`**

Per CLAUDE.md, before any push/PR. Hand it the exact list of `.py` files the carve touched.

- [ ] **Step 3: Verify the carve**

```bash
git diff main --stat                                   # only the files above, nothing diagnostic
grep -rn "compression_eval\|zinb_routing_diagnostics\|icc_diagnostics" src/ tests/ | grep import   # none
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration
poetry run python -c "from sportstradamus.scripts.generate_ship_config import main; print('ok')"
```
Expected: scoped diff, no diagnostic imports, all gates green, generator importable.

- [ ] **Step 4: Generate `main`'s first (all-withheld) config**

```bash
git checkout ship/main-serving-foundation
poetry run generate-ship-config --branch main
poetry run python -c "
import json; from pathlib import Path
from sportstradamus.training.ship_config import SHIP_CONFIG_PATH, WITHHELD
cfg = json.loads(Path(str(SHIP_CONFIG_PATH)).read_text())
assert all(cfg[lg][mk] == WITHHELD for lg in cfg for mk in cfg[lg]), 'expected all-withheld on dormant main'
print('all-withheld OK', sum(len(v) for v in cfg.values()), 'cells')
"
```
Expected: `all-withheld OK 96 cells` (no live metrics ⇒ 0 graduated). Commit this with the carve.

- [ ] **Step 5: Hand off to Trevor for push/merge**

The curator never pushes. Surface the branch, the scoped diff, and the green gates; Trevor pushes `ship/main-serving-foundation` and merges to `main` (the public branch). Once merged, the monthly `gate-status` cron's `--branch main` output is usable, and graduated cells will flip `"withheld" → "<strategy>"` on `main` as live data accrues.

---

## Final gate (run before any push / PR / review of this work)

- [ ] **Run the refactoring-specialist** on every `.py` touched this session:
  `src/sportstradamus/training/graduation.py`, `src/sportstradamus/scripts/check_graduation.py`,
  `src/sportstradamus/scripts/generate_ship_config.py`, `src/sportstradamus/training/ship_config.py`.
  (Tests-only `.py` files too if the specialist's scope includes them.)
- [ ] **Run all three quality gates:**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration
```
Expected: ruff clean; golden green (incl. the new `test_gate1_decisions.py`, `test_graduation.py`, `test_generate_ship_config.py`, and the unchanged `test_check_graduation.py`); integration green (incl. `test_gate_status_job.py`).
- [ ] **Confirm the determinism gate is untouched** — this work adds no deterministic-pipeline code, but run `poetry run pytest -m integration -k determinism` to be sure.

---

## Self-review notes (resolved while writing this plan)

- **Spec coverage:** decisions file (Task 1), graduation refactor (Tasks 2-3), generator CLI (Tasks 4-5), apply-to-devel (Task 6), monthly cron Option X (Task 7), default-deny docs (Task 8), main foundation carve (Task 9). All spec sections map to a task.
- **Type/name consistency:** the public names introduced in Task 2 (`classify_lifecycle`, `read_gate1`, `read_gate2`, `lifecycle_table`, `graduated_cells`, `MIN_SETTLED_FOR_GRADUATION`, `GRADUATION_WINDOW_DAYS`) are the exact names imported in Tasks 3-5. The generator functions (`load_decisions`, `active_cells`, `build_ship_config`, `main`) match between Task 4/5 code and their tests. `GATE1_DECISIONS_PATH` is defined in Task 4 and imported in the same task's module.
- **Proxy-vs-full-Gate-2 gap:** flagged in the `graduation.py` docstring and Task 8 docs; closing it is out of scope (main dormant), consistent with the spec's "Out of scope".
- **Sequencing:** Phases 1-6 are independent of `main` and run first; Phase 7 (main carve) follows because the carve must include the new modules. Stated in Task 9's preamble.
