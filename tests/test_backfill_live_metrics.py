"""Unit tests for the Stage 0 backfill CLI."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.nightly import LIVE_METRICS_COLUMNS
from sportstradamus.scripts.backfill_live_metrics import main

NOW = datetime(2026, 5, 20, 0, 0, 0)


def _offer(line, bet, model_p, books_p):
    return (line, 1.0, "Underdog", bet, model_p, books_p, float("nan"), float("nan"), float("nan"))


def _build_history(span_days: int = 100, n_per_cell: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    rows = []
    cells = [("NBA", "PTS"), ("WNBA", "PTS"), ("NFL", "rushing_yards")]
    for idx in range(n_per_cell):
        for league, market in cells:
            date = (NOW - timedelta(days=int(rng.integers(0, span_days)))).strftime("%Y-%m-%d")
            line = float(rng.uniform(8.0, 25.0))
            rows.append(
                {
                    "Player": f"Player_{league}_{idx}",
                    "League": league,
                    "Date": date,
                    "Market": market,
                    "Projection": line,
                    "Offers": [_offer(line, "Over", 0.55, 0.50)],
                    "Actual": float(rng.normal(line, 2.0)),
                }
            )
    return pd.DataFrame(rows)


def test_backfill_walks_window_endpoints(monkeypatch, tmp_path):
    history = _build_history(span_days=60, n_per_cell=10)
    monkeypatch.setattr(
        "sportstradamus.scripts.backfill_live_metrics.read_history", lambda: history
    )
    out = tmp_path / "live_metrics.parquet"
    runner = CliRunner()
    result = runner.invoke(main, ["--days", "30", "--step", "7", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    written = pd.read_parquet(out)
    assert set(written.columns) == set(LIVE_METRICS_COLUMNS)
    endpoints = sorted(written["computed_at"].unique())
    assert 3 <= len(endpoints) <= 6  # ~4 endpoints from a 30/7 walk
    # Both windows present per endpoint
    for ep in endpoints:
        ep_rows = written[written["computed_at"] == ep]
        assert set(ep_rows["window_days"].unique()) == {7, 30}


def test_backfill_idempotent_rerun_dedupes_on_key(monkeypatch, tmp_path):
    history = _build_history(span_days=40, n_per_cell=8)
    monkeypatch.setattr(
        "sportstradamus.scripts.backfill_live_metrics.read_history", lambda: history
    )
    out = tmp_path / "live_metrics.parquet"
    runner = CliRunner()
    runner.invoke(main, ["--days", "21", "--step", "7", "--output", str(out)])
    first = pd.read_parquet(out)
    runner.invoke(main, ["--days", "21", "--step", "7", "--output", str(out)])
    second = pd.read_parquet(out)
    key = ["league", "market", "computed_at", "window_days"]
    assert not second.duplicated(subset=key).any()
    assert len(second) == len(first)


def test_backfill_merges_with_existing_parquet(monkeypatch, tmp_path):
    history = _build_history(span_days=40, n_per_cell=8)
    monkeypatch.setattr(
        "sportstradamus.scripts.backfill_live_metrics.read_history", lambda: history
    )
    out = tmp_path / "live_metrics.parquet"
    runner = CliRunner()
    # Seed with a stale row that should be preserved as-is.
    seed = pd.DataFrame(
        [
            {
                "league": "LEGACY",
                "market": "OLDMKT",
                "computed_at": pd.Timestamp("2020-01-01"),
                "window_days": 30,
                "n_settled": 99,
                "book_bss": 0.5,
                "empirical_over_rate": 0.5,
                "predicted_over_rate": 0.5,
                "top_decile_mae": 1.0,
                "profit_sim_yield": 0.0,
            }
        ]
    )
    seed["window_days"] = seed["window_days"].astype("int16")
    seed["n_settled"] = seed["n_settled"].astype("int64")
    seed.to_parquet(out, engine="pyarrow", index=False)

    result = runner.invoke(main, ["--days", "21", "--step", "7", "--output", str(out)])
    assert result.exit_code == 0, result.output
    written = pd.read_parquet(out)
    assert (written["league"] == "LEGACY").any()


def test_backfill_empty_history_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setattr("sportstradamus.scripts.backfill_live_metrics.read_history", pd.DataFrame)
    runner = CliRunner()
    out = tmp_path / "out.parquet"
    result = runner.invoke(main, ["--days", "30", "--output", str(out)])
    assert result.exit_code == 0
    assert "empty" in result.output.lower() or "no" in result.output.lower()


def test_backfill_rejects_step_zero(monkeypatch, tmp_path):
    monkeypatch.setattr("sportstradamus.scripts.backfill_live_metrics.read_history", pd.DataFrame)
    runner = CliRunner()
    out = tmp_path / "out.parquet"
    result = runner.invoke(main, ["--days", "30", "--step", "0", "--output", str(out)])
    assert result.exit_code != 0
    assert "step" in result.output.lower()
