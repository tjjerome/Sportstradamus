"""One-shot in-place migration: add ``observed_at`` to existing archive tables.

Adds ``observed_at TIMESTAMP NOT NULL`` to ``odds`` and ``lines`` and
backfills every existing row to ``game_date`` midnight (the sentinel that
the time-series rework treats as "single legacy observation"). Sorts on
disk so zone-map pruning over the new column is effective.

Idempotent: refuses to run twice (checks for the column first).

Run with::

    poetry run python -m sportstradamus.scripts.add_observed_at_to_archive
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import duckdb

_DEFAULT_DB_PATH = Path("archive/archive.duckdb")


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table],
        ).fetchall()
    }


def _has_observed_at(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return "observed_at" in _table_columns(con, table)


def _migrate(con: duckdb.DuckDBPyConnection) -> None:
    """Run the ALTER + backfill + recreate-sorted dance for both tables.

    If the table already has a ``sample_ts`` column from prior in-progress
    work, its non-NULL values are carried over into ``observed_at`` instead
    of being lost to the game-date midnight sentinel.
    """
    odds_cols = _table_columns(con, "odds")
    lines_cols = _table_columns(con, "lines")

    # 1. ALTER + backfill (NULLABLE intermediate state).
    con.execute("ALTER TABLE odds ADD COLUMN IF NOT EXISTS observed_at TIMESTAMP")
    con.execute("ALTER TABLE lines ADD COLUMN IF NOT EXISTS observed_at TIMESTAMP")
    if "sample_ts" in odds_cols:
        # Carry over real observation timestamps before falling back to midnight.
        con.execute(
            "UPDATE odds SET observed_at = sample_ts "
            "WHERE observed_at IS NULL AND sample_ts IS NOT NULL"
        )
    if "sample_ts" in lines_cols:
        con.execute(
            "UPDATE lines SET observed_at = sample_ts "
            "WHERE observed_at IS NULL AND sample_ts IS NOT NULL"
        )
    con.execute(
        "UPDATE odds SET observed_at = CAST(game_date AS TIMESTAMP) WHERE observed_at IS NULL"
    )
    con.execute(
        "UPDATE lines SET observed_at = CAST(game_date AS TIMESTAMP) WHERE observed_at IS NULL"
    )

    # 2. Recreate odds with NOT NULL constraint, sorted on disk.
    con.execute(
        """
        CREATE TABLE odds_sorted (
            league       TEXT NOT NULL,
            market       TEXT NOT NULL,
            game_date    DATE NOT NULL,
            entity       TEXT NOT NULL,
            book         TEXT NOT NULL,
            ev           DOUBLE,
            observed_at  TIMESTAMP NOT NULL
        )
        """
    )
    con.execute(
        "INSERT INTO odds_sorted "
        "SELECT league, market, game_date, entity, book, ev, observed_at FROM odds "
        "ORDER BY league, market, game_date, entity, book, observed_at"
    )
    con.execute("DROP TABLE odds")
    con.execute("ALTER TABLE odds_sorted RENAME TO odds")

    # 3. Same for lines.
    con.execute(
        """
        CREATE TABLE lines_sorted (
            league       TEXT NOT NULL,
            market       TEXT NOT NULL,
            game_date    DATE NOT NULL,
            entity       TEXT NOT NULL,
            line         DOUBLE NOT NULL,
            observed_at  TIMESTAMP NOT NULL
        )
        """
    )
    con.execute(
        "INSERT INTO lines_sorted "
        "SELECT league, market, game_date, entity, line, observed_at FROM lines "
        "ORDER BY league, market, game_date, entity, line, observed_at"
    )
    con.execute("DROP TABLE lines")
    con.execute("ALTER TABLE lines_sorted RENAME TO lines")

    con.execute("ANALYZE")
    con.execute("CHECKPOINT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=str(_DEFAULT_DB_PATH),
        help=f"Target DuckDB file (default: {_DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the .bak copy before mutating the DB.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.is_file():
        sys.exit(f"DB file not found: {db_path}")

    # Quick read-only probe to detect whether the migration has already run.
    probe = duckdb.connect(str(db_path), read_only=True)
    odds_done = _has_observed_at(probe, "odds")
    lines_done = _has_observed_at(probe, "lines")
    probe.close()
    if odds_done and lines_done:
        print(f"{db_path}: observed_at already present in odds + lines — nothing to do.")
        return

    if not args.no_backup:
        backup = db_path.with_suffix(f".duckdb.bak-{int(time.time())}")
        shutil.copy2(db_path, backup)
        print(f"Backed up {db_path} -> {backup}")

    con = duckdb.connect(str(db_path))
    try:
        _migrate(con)
    finally:
        con.close()

    final_size = db_path.stat().st_size / (1024 * 1024)
    print(f"Migration complete. Final DB size: {final_size:.1f} MB at {db_path}")


if __name__ == "__main__":
    main()
