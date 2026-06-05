"""Characterization pin for the ``reflect`` CLI orchestrator ``nightly.run``.

CC-18 six-step pipeline (load stats -> resolve history -> CLV -> resolve parlays ->
write meta -> live metrics) with no direct test, though every step it calls is
covered elsewhere. This pins the *wiring*: with all step seams mocked to known
returns, it asserts the ``resolve_meta.json`` counts run derives (history/parlays
resolved, totals, pending) and that the live-metrics frame is written. That freezes
the orchestration so the decomposition into step helpers is behavior-preserving.

All league loads + history/parlay I/O + archive + clv + resolve_history +
check_bet + _compute_live_metrics are mocked; the real ``resolve_meta.json`` under
the package ``runtime`` dir is restored by the fixture.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus import data, nightly


class _FakeStats:
    def load(self):
        pass

    def update(self):
        pass

    @property
    def gamelog(self):
        return pd.DataFrame({"x": [1]})


def _history():
    return pd.DataFrame({
        "League": ["NBA", "NBA"],
        "Market": ["PTS", "REB"],
        "Date": ["2025-01-01", "2025-01-02"],
        "Offers": [[(10.0, 1.0, "Underdog", "Over", 0.6, 0.5, 0.0, 0.0, 0.0)],
                   [(8.0, 1.0, "Underdog", "Under", 0.55, 0.5, 0.0, 0.0, 0.0)]],
        "Actual": [np.nan, np.nan],
    })


def _resolve_history(history, stats):
    out = history.copy()
    out.loc[0, "Actual"] = 12.0  # resolve exactly one of the two pending rows
    return out


@pytest.fixture
def meta(monkeypatch):
    monkeypatch.setattr(nightly, "LEAGUE_CLASSES", {"NBA": _FakeStats})
    monkeypatch.setattr(nightly, "read_history", _history)
    monkeypatch.setattr(nightly, "resolve_history", _resolve_history)
    monkeypatch.setattr(nightly, "write_history", lambda h: None)
    monkeypatch.setattr(nightly, "Archive", object)
    monkeypatch.setattr(nightly.clv, "fill_from_archive", lambda h, a: h)
    monkeypatch.setattr(nightly.clv, "summarize", lambda h, archive=None: {"n": 0})
    monkeypatch.setattr(nightly.clv, "persist_segments", lambda s: None)
    monkeypatch.setattr(nightly, "read_parlay_hist",
                        lambda: pd.DataFrame({"Legs": [np.nan], "Misses": [np.nan]}))
    monkeypatch.setattr(nightly, "write_parlay_hist", lambda p: None)
    monkeypatch.setattr(nightly, "check_bet", lambda bet, stats, stat_map: (1, 0))
    written = {}
    monkeypatch.setattr(nightly, "_compute_live_metrics", lambda h: pd.DataFrame({"a": [1, 2, 3]}))
    monkeypatch.setattr(nightly, "_atomic_write_parquet",
                        lambda df, path: written.update({"rows": len(df)}))

    meta_path = Path(str(pkg_resources.files(data) / "runtime" / "resolve_meta.json"))
    saved = meta_path.read_bytes() if meta_path.is_file() else None
    try:
        result = CliRunner().invoke(nightly.run, ["--league", "NBA", "--skip-update"])
        assert result.exit_code == 0, result.output
        out = json.loads(meta_path.read_text())
        out["_live_rows"] = written.get("rows")
        out["_output"] = result.output
        yield out
    finally:
        if saved is None:
            meta_path.unlink(missing_ok=True)
        else:
            meta_path.write_bytes(saved)


def test_history_resolved_count(meta):
    assert meta["history_resolved"] == 1


def test_parlays_resolved_count(meta):
    assert meta["parlays_resolved"] == 1


def test_history_totals(meta):
    assert meta["history_total"] == 2
    assert meta["history_pending"] == 1


def test_live_metrics_written(meta):
    assert meta["_live_rows"] == 3


def test_echo_summary(meta):
    assert "History: 1 resolved" in meta["_output"]
    assert "Parlays: 1 resolved" in meta["_output"]
