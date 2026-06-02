"""Unit tests for ``scripts.delete_corrupt_seed``.

The deletion targets only ``observed_at``-midnight migration-seed rows that are
either magnitude-blown (``ev`` far beyond the line) or degenerate (every book
stamped with the same ``ev``). Re-fetched (``01:00``) rows, live rows, clean
multi-book seed rows, and other cells must survive.
"""

from __future__ import annotations

import datetime

import duckdb
import pytest

from sportstradamus.scripts.delete_corrupt_seed import (
    count_blown,
    count_corrupt_seed,
    delete_blown,
    delete_corrupt_seed,
)

D = datetime.date(2025, 11, 2)
SEED = datetime.datetime(2025, 11, 2, 0, 0, 0)
REFETCH = datetime.datetime(2025, 11, 2, 1, 0, 0)
LIVE = datetime.datetime(2025, 11, 2, 14, 30, 0)


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute(
        "CREATE TABLE odds (league VARCHAR, market VARCHAR, game_date DATE, "
        "entity VARCHAR, book VARCHAR, ev DOUBLE, observed_at TIMESTAMP)"
    )
    c.execute(
        "CREATE TABLE lines (league VARCHAR, market VARCHAR, game_date DATE, "
        "entity VARCHAR, line DOUBLE, observed_at TIMESTAMP)"
    )
    yield c
    c.close()


def _odds(con, entity, book, ev, observed_at, *, market="FTM", league="NBA"):
    con.execute(
        "INSERT INTO odds VALUES (?, ?, ?, ?, ?, ?, ?)",
        [league, market, D, entity, book, float(ev), observed_at],
    )


def _line(con, entity, line, *, market="FTM", league="NBA"):
    con.execute(
        "INSERT INTO lines VALUES (?, ?, ?, ?, ?, ?)",
        [league, market, D, entity, float(line), SEED],
    )


def _seed_corpus(con):
    """Insert the canonical mix of corrupt and clean rows for cell (NBA, FTM)."""
    _odds(con, "P1", "dk", 40000, SEED)                      # blown seed (relative + absolute)
    _line(con, "P1", 3.0)
    _odds(con, "P2", "dk", 2.0, SEED)                        # degenerate group: both books
    _odds(con, "P2", "fd", 2.0, SEED)
    _line(con, "P2", 2.5)
    _odds(con, "P3", "dk", 2.0, SEED)                        # clean multi-book seed: keep both
    _odds(con, "P3", "fd", 3.5, SEED)
    _line(con, "P3", 3.0)
    _odds(con, "P4", "dk", 2.0, REFETCH)                     # refetch row: keep
    _odds(con, "P5", "dk", 2.0, LIVE)                        # live row: keep
    _odds(con, "P6", "dk", 2.0, SEED)                        # partial group: dk clean, fd blown
    _odds(con, "P6", "fd", 50.0, SEED)
    _line(con, "P6", 2.0)
    _odds(con, "P7", "dk", 9000, SEED, market="BLK")         # other cell: untouched
    _odds(con, "P8", "dk", 8000, SEED)                       # blown seed, no line: absolute ceiling
    _odds(con, "P9", "dk", 2.0, SEED)                        # clean single-book seed, no line: keep


def test_deletes_only_blown_and_degenerate_midnight_seed(con):
    _seed_corpus(con)

    assert count_corrupt_seed(con, "NBA", "FTM") == 5
    assert delete_corrupt_seed(con, "NBA", "FTM") == 5

    remaining = set(
        con.execute("SELECT entity, book FROM odds WHERE market = 'FTM'").fetchall()
    )
    assert remaining == {
        ("P3", "dk"),
        ("P3", "fd"),
        ("P4", "dk"),
        ("P5", "dk"),
        ("P6", "dk"),
        ("P9", "dk"),
    }


def test_other_cell_untouched(con):
    _seed_corpus(con)
    delete_corrupt_seed(con, "NBA", "FTM")
    assert con.execute("SELECT COUNT(*) FROM odds WHERE market = 'BLK'").fetchone()[0] == 1


def test_idempotent(con):
    _seed_corpus(con)
    delete_corrupt_seed(con, "NBA", "FTM")
    assert delete_corrupt_seed(con, "NBA", "FTM") == 0


def _blown_layer_corpus(con):
    """Magnitude-blown rows across every observed_at layer for cell (NBA, BLK)."""
    _odds(con, "B1", "ud", 14457, LIVE, market="BLK")        # blown live (DFS): delete
    _odds(con, "B2", "sl", 5000, REFETCH, market="BLK")      # blown refetch: delete
    _odds(con, "B3", "dk", 9000, SEED, market="BLK")         # blown seed: delete
    for e in ("B1", "B2", "B3"):
        _line(con, e, 1.5, market="BLK")
    _odds(con, "G1", "ud", 1.0, LIVE, market="BLK")          # clean live: keep
    _odds(con, "G2", "sl", 1.2, REFETCH, market="BLK")       # clean refetch: keep
    _line(con, "G1", 1.5, market="BLK")
    _line(con, "G2", 1.5, market="BLK")
    _odds(con, "DG", "dk", 2.0, SEED, market="BLK")          # degenerate seed group:
    _odds(con, "DG", "fd", 2.0, SEED, market="BLK")          #   magnitude-clean -> keep
    _line(con, "DG", 2.5, market="BLK")
    _odds(con, "O1", "ud", 9999, LIVE)                        # other cell (FTM): untouched


def test_blown_all_layers_deletes_blown_across_layers_keeps_clean(con):
    _blown_layer_corpus(con)

    assert count_blown(con, "NBA", "BLK") == 3
    assert delete_blown(con, "NBA", "BLK") == 3

    remaining = set(con.execute("SELECT entity FROM odds WHERE market = 'BLK'").fetchall())
    assert remaining == {("G1",), ("G2",), ("DG",)}  # clean live/refetch + degenerate kept


def test_blown_all_layers_leaves_other_cell(con):
    _blown_layer_corpus(con)
    delete_blown(con, "NBA", "BLK")
    assert con.execute("SELECT COUNT(*) FROM odds WHERE market = 'FTM'").fetchone()[0] == 1
