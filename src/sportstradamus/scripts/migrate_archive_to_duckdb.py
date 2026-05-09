"""One-shot migration: klepto HDF archive -> DuckDB single-file archive.

Walks every existing ``archive/<LEAGUE>/`` klepto tree, flattens the four-level
``league -> market -> date -> entity -> {EV, Lines}`` structure into the new
``odds`` and ``lines`` tables, and writes them into ``archive/archive.duckdb``
via DuckDB's bulk Appender API.

The legacy klepto directories are left in place — delete them manually after
validating the new database. Run with::

    poetry run python -m sportstradamus.scripts.migrate_archive_to_duckdb
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
from klepto.archives import hdfdir_archive
from tqdm import tqdm

from sportstradamus.helpers.text import remove_accents

_TEAM_ONLY_MARKETS = frozenset({"Moneyline", "Totals", "1st 1 innings"})

# Match helpers/archive.py: no PK. The PK index alone bloats the DB ~10x for
# this row count; zone-map pruning on naturally sorted data is fast enough.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS odds (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    book         TEXT NOT NULL,
    ev           DOUBLE,
    observed_at  TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS lines (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    line         DOUBLE NOT NULL,
    observed_at  TIMESTAMP NOT NULL
);
"""


def _parse_date(s: str) -> datetime.date | None:
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _is_skip_entity(entity: str) -> bool:
    return " + " in entity or " vs. " in entity


def _iter_league(archive_root: Path, league: str, cutoff: datetime.date):
    """Yield (odds_rows, lines_rows) batches for one league.

    Legacy klepto rows have no observation timestamp; we emit
    ``observed_at = game_date`` midnight as a single-point sentinel. The
    time-series read path treats this as one observation at the start of
    game day so post-rework queries with ``at >= midnight`` still resolve.
    """
    league_dir = archive_root / league
    if not league_dir.is_dir():
        return
    a = hdfdir_archive(str(league_dir), {}, protocol=-1)
    a.load()

    odds_rows: list[tuple] = []
    lines_rows: list[tuple] = []

    for market, dates in a.items():
        if not isinstance(dates, dict):
            continue
        for date_str, entities in dates.items():
            d = _parse_date(date_str)
            if d is None or d < cutoff or not isinstance(entities, dict):
                continue
            sentinel_ts = datetime.datetime.combine(d, datetime.time.min)
            for entity, value in entities.items():
                if _is_skip_entity(entity):
                    continue
                clean_entity = remove_accents(entity)
                if not isinstance(value, dict):
                    continue

                if market in _TEAM_ONLY_MARKETS:
                    # Team-only markets store {book: ev} directly.
                    for book, ev in value.items():
                        if book == "Line":
                            continue
                        try:
                            odds_rows.append(
                                (
                                    league,
                                    market,
                                    d,
                                    clean_entity,
                                    str(book),
                                    float(ev),
                                    sentinel_ts,
                                )
                            )
                        except (TypeError, ValueError):
                            continue
                else:
                    # Player markets store {"EV": {book: ev}, "Lines": [float, ...]}.
                    ev_dict = value.get("EV") or {}
                    if isinstance(ev_dict, dict):
                        for book, ev in ev_dict.items():
                            if book == "Line":
                                continue
                            try:
                                odds_rows.append(
                                    (
                                        league,
                                        market,
                                        d,
                                        clean_entity,
                                        str(book),
                                        float(ev),
                                        sentinel_ts,
                                    )
                                )
                            except (TypeError, ValueError):
                                continue
                    seen_lines: set[float] = set()
                    for line in value.get("Lines") or []:
                        try:
                            f = float(line)
                        except (TypeError, ValueError):
                            continue
                        if not f or f in seen_lines:
                            continue
                        seen_lines.add(f)
                        lines_rows.append((league, market, d, clean_entity, f, sentinel_ts))

    return odds_rows, lines_rows


_ODDS_COLS = ["league", "market", "game_date", "entity", "book", "ev", "observed_at"]
_LINES_COLS = ["league", "market", "game_date", "entity", "line", "observed_at"]


def _bulk_insert_odds(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> int:
    """Bulk-insert deduped odds rows via a registered DataFrame."""
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=_ODDS_COLS).drop_duplicates(  # noqa: F841 — DuckDB DF replacement
        subset=["league", "market", "game_date", "entity", "book", "observed_at"], keep="last"
    )
    before = con.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
    con.execute("INSERT INTO odds SELECT * FROM df")
    after = con.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
    return after - before


def _bulk_insert_lines(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> int:
    """Bulk-insert deduped line rows via a registered DataFrame."""
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=_LINES_COLS).drop_duplicates()  # noqa: F841 — DuckDB DF replacement
    before = con.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
    con.execute("INSERT INTO lines SELECT * FROM df")
    after = con.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
    return after - before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", default="archive", help="Source klepto archive directory")
    parser.add_argument("--db-path", default="archive/archive.duckdb", help="Target DuckDB file")
    parser.add_argument(
        "--cutoff-years",
        type=int,
        default=4,
        help="Drop dates older than this many years (matches legacy clean_archive default)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the target DuckDB file before inserting (default: append)",
    )
    args = parser.parse_args()

    archive_root = Path(args.archive_root)
    db_path = Path(args.db_path)
    cutoff = (datetime.datetime.today() - datetime.timedelta(days=365 * args.cutoff_years)).date()

    if not archive_root.is_dir():
        sys.exit(f"archive root not found: {archive_root}")

    if args.fresh and db_path.is_file():
        # Move the old DB aside rather than deleting outright.
        backup = db_path.with_suffix(f".duckdb.bak-{int(time.time())}")
        shutil.move(str(db_path), str(backup))
        print(f"Moved existing {db_path} -> {backup}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_SCHEMA_DDL)

    existing = con.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
    if existing and not args.fresh:
        sys.exit(
            f"refusing to migrate into a populated DB ({existing:,} rows present); "
            "rerun with --fresh to back up the existing DB and start over"
        )

    leagues = sorted(f.name for f in os.scandir(archive_root) if f.is_dir())
    print(f"Migrating {len(leagues)} leagues from {archive_root} -> {db_path}")
    print(f"Cutoff date: {cutoff} ({args.cutoff_years} years)")

    totals = Counter()
    for league in tqdm(leagues, desc="leagues"):
        try:
            result = _iter_league(archive_root, league, cutoff)
        except Exception as exc:
            tqdm.write(f"  !! {league}: {exc!r} — skipped")
            continue
        if result is None:
            continue
        odds_rows, lines_rows = result
        n_odds = _bulk_insert_odds(con, odds_rows)
        n_lines = _bulk_insert_lines(con, lines_rows)
        totals["odds"] += n_odds
        totals["lines"] += n_lines
        tqdm.write(f"  {league}: +{n_odds} odds, +{n_lines} lines")

    # Sort the tables on disk so zone-map pruning is effective at read time.
    print("Sorting and compacting...")
    con.execute(
        "CREATE TABLE odds_sorted AS SELECT * FROM odds "
        "ORDER BY league, market, game_date, entity, book, observed_at"
    )
    con.execute("DROP TABLE odds")
    con.execute("ALTER TABLE odds_sorted RENAME TO odds")
    con.execute(
        "CREATE TABLE lines_sorted AS SELECT * FROM lines "
        "ORDER BY league, market, game_date, entity, line, observed_at"
    )
    con.execute("DROP TABLE lines")
    con.execute("ALTER TABLE lines_sorted RENAME TO lines")
    con.execute("ANALYZE")
    con.execute("CHECKPOINT")
    con.close()

    final_size = db_path.stat().st_size / (1024 * 1024)
    print()
    print(f"Inserted {totals['odds']:,} odds rows, {totals['lines']:,} lines rows")
    print(f"Final DB size: {final_size:.1f} MB at {db_path}")


if __name__ == "__main__":
    main()
