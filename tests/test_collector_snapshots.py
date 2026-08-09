"""Leakage-guard contract tests for :class:`DatedSnapshotStore`.

The as-of selection is the load-bearing invariant: a snapshot captured on or after
the game date must never feed that game's features. These tests pin the strict-``<``
behavior and the graceful ``None`` when no snapshot qualifies.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.text import remove_accents
from sportstradamus.stats import collector_snapshots
from sportstradamus.stats.base import Stats
from sportstradamus.stats.collector_snapshots import DatedSnapshotStore, load_asof_features

_BASE_DIR = "player_data/NBA/cleaningtheglass"
_SEASON = 2026
_TOOL = "team_offense"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store whose data root is redirected to a tmp dir (never touches the repo)."""
    monkeypatch.setattr(collector_snapshots.pkg_resources, "files", lambda _pkg: tmp_path)
    return DatedSnapshotStore(base_dir=_BASE_DIR, label="CTG test")


def _write_snapshot(root, snap_date: str, rows: int) -> None:
    directory = root / _BASE_DIR / str(_SEASON) / snap_date
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value": range(rows)}).to_parquet(directory / f"{_TOOL}.parquet", index=False)


def _write_tool(root, snap_date: str, tool: str, frame: pd.DataFrame) -> None:
    directory = root / _BASE_DIR / str(_SEASON) / snap_date
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / f"{tool}.parquet", index=False)


def test_available_dates_sorted(store, tmp_path):
    _write_snapshot(tmp_path, "2026-01-15", 1)
    _write_snapshot(tmp_path, "2026-01-01", 1)
    _write_snapshot(tmp_path, "2026-01-08", 1)
    assert store.available_dates(_SEASON) == [
        date(2026, 1, 1),
        date(2026, 1, 8),
        date(2026, 1, 15),
    ]


def test_available_dates_empty_when_absent(store):
    assert store.available_dates(_SEASON) == []


def test_latest_before_is_strict(store, tmp_path):
    _write_snapshot(tmp_path, "2026-01-01", 1)
    _write_snapshot(tmp_path, "2026-01-08", 1)
    # A game on the 8th must NOT see the 8th's snapshot (same-day leakage).
    assert store.latest_before(_SEASON, date(2026, 1, 8)) == date(2026, 1, 1)
    # A game on the 9th may see the 8th's snapshot.
    assert store.latest_before(_SEASON, date(2026, 1, 9)) == date(2026, 1, 8)


def test_latest_before_accepts_datetime(store, tmp_path):
    _write_snapshot(tmp_path, "2026-01-01", 1)
    assert store.latest_before(_SEASON, datetime(2026, 1, 5, 19, 30)) == date(2026, 1, 1)


def test_latest_before_none_when_all_after(store, tmp_path):
    _write_snapshot(tmp_path, "2026-01-08", 1)
    assert store.latest_before(_SEASON, date(2026, 1, 1)) is None


def test_load_asof_returns_prior_snapshot(store, tmp_path):
    _write_snapshot(tmp_path, "2026-01-01", 3)
    _write_snapshot(tmp_path, "2026-01-08", 7)
    frame = store.load_asof(_SEASON, date(2026, 1, 8), _TOOL)
    assert frame is not None and len(frame) == 3  # the 1st, not the same-day 8th


def test_load_asof_none_when_no_prior(store, tmp_path):
    _write_snapshot(tmp_path, "2026-01-08", 3)
    assert store.load_asof(_SEASON, date(2026, 1, 1), _TOOL) is None


def test_load_asof_none_for_missing_tool(store, tmp_path):
    _write_snapshot(tmp_path, "2026-01-01", 3)
    assert store.load_asof(_SEASON, date(2026, 1, 8), "no_such_tool") is None


def test_available_tools_lists_sorted_stems(store, tmp_path):
    _write_tool(tmp_path, "2026-01-01", "player_usage", pd.DataFrame({"name": ["X"]}))
    _write_tool(tmp_path, "2026-01-01", "player_shooting", pd.DataFrame({"name": ["X"]}))
    assert store.available_tools(_SEASON, date(2026, 1, 1)) == ["player_shooting", "player_usage"]


def test_available_tools_empty_when_absent(store):
    assert store.available_tools(_SEASON, date(2026, 1, 1)) == []


def test_load_asof_features_none_when_key_unset(store, tmp_path):
    _write_tool(tmp_path, "2026-01-01", "player_usage", pd.DataFrame({"name": ["X"], "a": [1]}))
    assert load_asof_features(store, _SEASON, date(2026, 1, 8), "", set()) is None


def test_load_asof_features_renames_indexes_and_strips_accents(store, tmp_path):
    _write_tool(
        tmp_path,
        "2026-01-01",
        "player_usage",
        pd.DataFrame({"name": ["Luka Dončić"], "usg": [0.31], "team": ["DAL"]}),
    )
    out = load_asof_features(store, _SEASON, date(2026, 1, 8), "name", {"team"})
    assert list(out.columns) == ["usg_asof"]
    assert out.index.tolist() == [remove_accents("Luka Dončić")]
    assert out.loc[remove_accents("Luka Dončić"), "usg_asof"] == 0.31


def test_load_asof_features_merges_multiple_tools(store, tmp_path):
    _write_tool(tmp_path, "2026-01-01", "player_a", pd.DataFrame({"name": ["X"], "a": [1]}))
    _write_tool(tmp_path, "2026-01-01", "player_b", pd.DataFrame({"name": ["X"], "b": [2]}))
    out = load_asof_features(store, _SEASON, date(2026, 1, 8), "name", set())
    assert set(out.columns) == {"a_asof", "b_asof"}
    assert out.loc["X", "a_asof"] == 1 and out.loc["X", "b_asof"] == 2


def test_load_asof_features_skips_tool_without_key(store, tmp_path):
    _write_tool(tmp_path, "2026-01-01", "player_keyed", pd.DataFrame({"name": ["X"], "a": [1]}))
    _write_tool(tmp_path, "2026-01-01", "player_unkeyed", pd.DataFrame({"other": ["Y"], "b": [2]}))
    out = load_asof_features(store, _SEASON, date(2026, 1, 8), "name", set())
    assert set(out.columns) == {"a_asof"}


def test_load_asof_features_dedupes_keeping_most_complete(store, tmp_path):
    # Traded player: two rows share the key; keep the one with more non-null cells.
    _write_tool(
        tmp_path,
        "2026-01-01",
        "player_usage",
        pd.DataFrame({"name": ["Traded Guy", "Traded Guy"], "x": [1.0, np.nan], "y": [3.0, 2.0]}),
    )
    out = load_asof_features(store, _SEASON, date(2026, 1, 8), "name", set())
    assert len(out) == 1
    assert out.loc["Traded Guy", "x_asof"] == 1.0 and out.loc["Traded Guy", "y_asof"] == 3.0


def test_load_asof_features_strict_leakage(store, tmp_path):
    _write_tool(tmp_path, "2026-01-08", "player_usage", pd.DataFrame({"name": ["X"], "a": [9]}))
    # Same-day snapshot must not feed the 8th's features.
    assert load_asof_features(store, _SEASON, date(2026, 1, 8), "name", set()) is None
    # The 9th may see it.
    out = load_asof_features(store, _SEASON, date(2026, 1, 9), "name", set())
    assert out.loc["X", "a_asof"] == 9


def test_collector_asof_features_none_for_unset_key(store):
    """The base hook helper short-circuits to None when the source's key_col is unset."""
    holder = SimpleNamespace(_collector_asof_cache=None)
    out = Stats._collector_asof_features(holder, _BASE_DIR, "", set(), _SEASON, date(2026, 1, 8))
    assert out is None


def test_collector_asof_features_memoizes_per_key(store, tmp_path):
    """A second call for the same (grain, season, date) serves the cache, not a fresh read."""
    _write_tool(tmp_path, "2026-01-01", "player_usage", pd.DataFrame({"name": ["X"], "a": [5]}))
    holder = SimpleNamespace(_collector_asof_cache=None)
    first = Stats._collector_asof_features(
        holder, _BASE_DIR, "name", set(), _SEASON, date(2026, 1, 8)
    )
    assert first.loc["X", "a_asof"] == 5
    # A newer (still-prior) snapshot would win on a fresh read; the cache must ignore it.
    _write_tool(tmp_path, "2026-01-05", "player_usage", pd.DataFrame({"name": ["X"], "a": [99]}))
    again = Stats._collector_asof_features(
        holder, _BASE_DIR, "name", set(), _SEASON, date(2026, 1, 8)
    )
    assert again.loc["X", "a_asof"] == 5


def test_wnba_disables_inherited_nba_ctg_hooks():
    """WNBA subclasses StatsNBA but has no Cleaning the Glass source.

    It must override the inherited NBA CTG hooks back to the base no-op — the
    inherited ``_ctg_season`` slices ``_current_season_key`` as a string, which
    crashes on WNBA's integer season key.
    """
    from sportstradamus.stats.nba import StatsNBA
    from sportstradamus.stats.wnba import StatsWNBA

    game_date = date(2026, 7, 11)
    assert StatsWNBA._join_fp_player_features(SimpleNamespace(), game_date) is None
    assert StatsWNBA._join_fp_team_features(SimpleNamespace(), game_date) == (None, None)
    assert StatsWNBA._join_fp_player_features is not StatsNBA._join_fp_player_features
