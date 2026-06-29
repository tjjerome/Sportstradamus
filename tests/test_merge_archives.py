"""Lossless-merge guarantees for ``scripts.merge_archives``.

Each test builds two tiny DuckDB archives in ``tmp_path`` and calls
``merge_archives`` directly (no subprocess), then asserts the row sets with a
fresh read-only connection.
"""

from __future__ import annotations

import datetime

import duckdb
import pytest

from sportstradamus.scripts.merge_archives import merge_archives

_LINES_DDL = """
CREATE TABLE lines (
    league TEXT, market TEXT, game_date DATE, entity TEXT,
    line DOUBLE, observed_at TIMESTAMP
);
"""
# Pre-WS1 (7-col) and post-WS1 migrated (9-col, with the shape-free quote columns) odds schemas.
_DDL = (
    """
CREATE TABLE odds (
    league TEXT, market TEXT, game_date DATE, entity TEXT,
    book TEXT, ev DOUBLE, observed_at TIMESTAMP
);
"""
    + _LINES_DDL
)
_DDL_V2 = (
    """
CREATE TABLE odds (
    league TEXT, market TEXT, game_date DATE, entity TEXT,
    book TEXT, ev DOUBLE, observed_at TIMESTAMP, under_prob DOUBLE, line DOUBLE
);
"""
    + _LINES_DDL
)

_DATE = datetime.date(2026, 6, 1)
_T0 = datetime.datetime(2026, 6, 1, 0, 0, 0)
_T1 = datetime.datetime(2026, 6, 1, 1, 0, 0)

# A = shared, B = target-only, C = source-only; A2 shares A's logical key but a
# later observed_at (a distinct time-series observation that must survive).
A_ODDS = ("NBA", "PTS", _DATE, "Player A", "fanduel", 0.05, _T0)
A2_ODDS = ("NBA", "PTS", _DATE, "Player A", "fanduel", 0.05, _T1)
B_ODDS = ("NBA", "PTS", _DATE, "Player B", "fanduel", 0.10, _T0)
C_ODDS = ("NBA", "PTS", _DATE, "Player C", "fanduel", -0.03, _T0)

A_LINE = ("NBA", "PTS", _DATE, "Player A", 25.5, _T0)
B_LINE = ("NBA", "PTS", _DATE, "Player B", 18.5, _T0)
C_LINE = ("NBA", "PTS", _DATE, "Player C", 30.5, _T0)

# Same observations on a migrated (9-col) archive: the 7-col identity + (under_prob, line).
A_ODDS9 = (*A_ODDS, 0.55, 25.5)
B_ODDS9 = (*B_ODDS, 0.60, 18.5)
C_ODDS9 = (*C_ODDS, 0.48, 30.5)


def _build(path, odds_rows, lines_rows, *, shapefree=False):
    con = duckdb.connect(str(path))
    con.execute(_DDL_V2 if shapefree else _DDL)
    if odds_rows:
        placeholders = ", ".join("?" * len(odds_rows[0]))
        con.executemany(f"INSERT INTO odds VALUES ({placeholders})", odds_rows)
    if lines_rows:
        con.executemany("INSERT INTO lines VALUES (?, ?, ?, ?, ?, ?)", lines_rows)
    con.close()


def _read(path, table):
    con = duckdb.connect(str(path), read_only=True)
    rows = con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
    con.close()
    return rows


def test_merge_is_lossless_union_and_dedups(tmp_path):
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    _build(target, [A_ODDS, B_ODDS], [A_LINE, B_LINE])
    _build(source, [A_ODDS, C_ODDS], [A_LINE, C_LINE])

    report = merge_archives(source, target)

    assert set(_read(target, "odds")) == {A_ODDS, B_ODDS, C_ODDS}
    assert set(_read(target, "lines")) == {A_LINE, B_LINE, C_LINE}
    assert report["odds"] == {
        "target_before": 2,
        "source": 2,
        "merged": 3,
        "added": 1,
        "shared": 1,
    }
    assert report["lines"]["added"] == 1


def test_distinct_timestamps_survive_merge(tmp_path):
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    _build(target, [A_ODDS, A2_ODDS], [])
    _build(source, [A_ODDS], [])

    merge_archives(source, target)

    # A dedups against the source copy; A2 (later observed_at) is preserved.
    assert set(_read(target, "odds")) == {A_ODDS, A2_ODDS}


def test_merge_is_idempotent(tmp_path):
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    _build(target, [A_ODDS, B_ODDS], [])
    _build(source, [A_ODDS, C_ODDS], [])

    merge_archives(source, target)
    report = merge_archives(source, target)

    assert set(_read(target, "odds")) == {A_ODDS, B_ODDS, C_ODDS}
    assert report["odds"]["added"] == 0


def test_dry_run_writes_nothing(tmp_path):
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    _build(target, [A_ODDS, B_ODDS], [])
    _build(source, [A_ODDS, C_ODDS], [])

    report = merge_archives(source, target, dry_run=True)

    assert report["odds"]["added"] == 1
    assert report["odds"]["merged"] == 3
    assert len(_read(target, "odds")) == 2  # unchanged on disk
    assert list(tmp_path.glob("dev.duckdb.bak-*")) == []


def test_inplace_merge_backs_up_pre_merge_target(tmp_path):
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    _build(target, [A_ODDS, B_ODDS], [])
    _build(source, [A_ODDS, C_ODDS], [])

    merge_archives(source, target)

    backups = list(tmp_path.glob("dev.duckdb.bak-*"))
    assert len(backups) == 1
    assert len(_read(backups[0], "odds")) == 2  # pre-merge snapshot


def test_output_mode_leaves_inputs_untouched(tmp_path):
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    output = tmp_path / "merged.duckdb"
    _build(target, [A_ODDS, B_ODDS], [])
    _build(source, [A_ODDS, C_ODDS], [])

    merge_archives(source, target, output)

    assert set(_read(target, "odds")) == {A_ODDS, B_ODDS}  # untouched
    assert set(_read(source, "odds")) == {A_ODDS, C_ODDS}  # untouched
    assert set(_read(output, "odds")) == {A_ODDS, B_ODDS, C_ODDS}
    assert list(tmp_path.glob("dev.duckdb.bak-*")) == []


def test_refuses_self_merge(tmp_path):
    target = tmp_path / "dev.duckdb"
    _build(target, [A_ODDS], [])
    with pytest.raises(ValueError, match="same file"):
        merge_archives(target, target)


def test_missing_source_fails_loud(tmp_path):
    target = tmp_path / "dev.duckdb"
    _build(target, [A_ODDS], [])
    with pytest.raises(FileNotFoundError, match="source archive not found"):
        merge_archives(tmp_path / "nope.duckdb", target)


def test_merge_keeps_shapefree_quotes_over_unmigrated_source(tmp_path):
    """The dev (sync target) is migrated to the 9-col schema; prod (source) still has the 7-col
    schema until its own backfill. The merge must keep dev's under_prob/line (not wipe them to
    NULL via the old 7-col rebuild) and bring prod's ev-only rows in with NULL quotes."""
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    _build(target, [A_ODDS9, B_ODDS9], [A_LINE, B_LINE], shapefree=True)
    _build(source, [A_ODDS, C_ODDS], [A_LINE, C_LINE])

    merge_archives(source, target)

    rows = set(_read(target, "odds"))
    assert A_ODDS9 in rows  # shared observation keeps the dev quote, not wiped to NULL
    assert B_ODDS9 in rows  # dev-only observation intact
    assert (*C_ODDS, None, None) in rows  # prod-only ev-only row merges with NULL quote
    assert rows == {A_ODDS9, B_ODDS9, (*C_ODDS, None, None)}


def test_merge_both_migrated_dedups_on_identity(tmp_path):
    """Once both sides are migrated, identical observations dedup and each side's own quotes
    survive — the union keys on the observation identity, not the full (quote-bearing) row."""
    target = tmp_path / "dev.duckdb"
    source = tmp_path / "prod.duckdb"
    _build(target, [A_ODDS9, B_ODDS9], [], shapefree=True)
    _build(source, [A_ODDS9, C_ODDS9], [], shapefree=True)

    merge_archives(source, target)

    assert set(_read(target, "odds")) == {A_ODDS9, B_ODDS9, C_ODDS9}
