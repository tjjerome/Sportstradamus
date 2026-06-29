#!/usr/bin/env python3
"""Losslessly merge one DuckDB odds archive into another.

The archive (``archive/archive.duckdb``) is append-only time-series: the same
``(league, market, game_date, entity, book)`` key recurs with different
``observed_at`` timestamps and readers pick the latest. Two copies of the
archive — e.g. the production server's live one and a dev machine's — drift
apart, each holding observations the other lacks. The lossless reconciliation
is a *set-union of full rows*: keep every row present in either database,
collapsing only bit-identical duplicates. DuckDB's ``UNION`` does exactly that;
re-sorting in the same statement restores the on-disk order zone-map pruning
relies on (same table-rebuild as ``migrate_archive_to_duckdb``).

The odds union dedups on the observation identity (not the full row) so the WS1 shape-free
quote columns (``under_prob``/``line``, derived from ``ev``) survive a merge of a migrated
target with a not-yet-migrated source — see ``_odds_select``.

The source is attached read-only and never modified. The target is rebuilt in
place (a timestamped ``.bak-<epoch>`` is written first) unless ``--output``
directs the union to a fresh file instead.

Usage
-----
    poetry run merge-archives --source /tmp/prod_archive_snapshot.duckdb --dry-run
    poetry run merge-archives --source /tmp/prod_archive_snapshot.duckdb
    poetry run merge-archives --source other.duckdb --output merged.duckdb
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import click
import duckdb

# The observation identity. WS1 added shape-free quote columns (under_prob, line) to odds;
# those are derived from ev and are NOT part of the identity, so the union dedups on this key
# and reconciles the quote columns separately (see _odds_select).
_ODDS_KEY = "league, market, game_date, entity, book, ev, observed_at"
_LINES_COLS = "league, market, game_date, entity, line, observed_at"
# ev is omitted from the sort: equal-ev rows with different observed_at are distinct
# observations and sort order only needs the key fields.
_ODDS_SORT = "league, market, game_date, entity, book, observed_at"

# Same env-var override and default the Archive singleton honours.
_DEFAULT_TARGET = os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", "archive/archive.duckdb")

_ARCHIVE_LOCK_FILE = "/tmp/sportstradamus-archive.lock"


def _connect(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Translate a DuckDB lock collision into an actionable operator message."""
    try:
        return duckdb.connect(str(path), read_only=read_only)
    except duckdb.IOException as exc:
        if "Could not set lock" not in str(exc):
            raise
        raise RuntimeError(
            f"archive {path} is locked by another process (a running cron?). "
            "Run during a quiet window or while holding scripts/run_job.sh's "
            f"archive flock ({_ARCHIVE_LOCK_FILE})."
        ) from exc


def _require_tables(con: duckdb.DuckDBPyConnection, alias: str) -> None:
    """Fail loud if a database is missing the odds/lines tables."""
    prefix = f"{alias}." if alias else ""
    label = "source" if alias else "target"
    for table in ("odds", "lines"):
        try:
            con.execute(f"SELECT 1 FROM {prefix}{table} LIMIT 1")
        except duckdb.CatalogException as exc:
            raise RuntimeError(
                f"{label} archive missing '{table}' table — not a valid odds archive."
            ) from exc


def _has_shapefree(con: duckdb.DuckDBPyConnection, table_ref: str) -> bool:
    """Whether ``table_ref`` (``odds`` or ``src.odds``) carries the WS1 quote columns."""
    cols = {row[0] for row in con.execute(f"DESCRIBE {table_ref}").fetchall()}
    return "under_prob" in cols and "line" in cols


def _odds_select(con: duckdb.DuckDBPyConnection) -> str:
    """Build the odds union-select, preserving the shape-free (under_prob, line) quotes.

    Output schema follows the target: a migrated dev archive carries the quote columns, a
    not-yet-migrated prod source still has the 7-column schema. When the target has them, dedup
    on the observation identity and keep the populated (non-NULL) quote via ``MAX`` — so a sync
    never drops the dev backfill, and prod's ev-only rows merge in with NULL quotes (a reader
    falls back to ev for those). When the target predates WS1, fall back to the plain union.
    """
    if not _has_shapefree(con, "odds"):
        return f"SELECT {_ODDS_KEY} FROM odds UNION SELECT {_ODDS_KEY} FROM src.odds"
    src_quote = (
        "under_prob, line"
        if _has_shapefree(con, "src.odds")
        else "CAST(NULL AS DOUBLE) AS under_prob, CAST(NULL AS DOUBLE) AS line"
    )
    union = (
        f"SELECT {_ODDS_KEY}, under_prob, line FROM odds "
        f"UNION ALL SELECT {_ODDS_KEY}, {src_quote} FROM src.odds"
    )
    return (
        f"SELECT {_ODDS_KEY}, MAX(under_prob) AS under_prob, MAX(line) AS line "
        f"FROM ({union}) AS u GROUP BY {_ODDS_KEY}"
    )


def _merge_table(
    con: duckdb.DuckDBPyConnection, table: str, select_sql: str, sort: str, dry_run: bool
) -> dict[str, int]:
    """Perform (or simulate) the union-rebuild from ``select_sql`` and return row counts."""
    target_before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    source_count = con.execute(f"SELECT COUNT(*) FROM src.{table}").fetchone()[0]

    if dry_run:
        merged = con.execute(f"SELECT COUNT(*) FROM ({select_sql}) AS u").fetchone()[0]
    else:
        con.execute(f"CREATE TABLE {table}_merged AS {select_sql} ORDER BY {sort}")
        con.execute(f"DROP TABLE {table}")
        con.execute(f"ALTER TABLE {table}_merged RENAME TO {table}")
        merged = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    return {
        "target_before": target_before,
        "source": source_count,
        "merged": merged,
        "added": merged - target_before,
        "shared": target_before + source_count - merged,
    }


def merge_archives(
    source: str | Path,
    target: str | Path,
    output: str | Path | None = None,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, dict[str, int]]:
    """Merge the ``source`` archive into ``target`` as a lossless union of rows.

    ``source`` is opened read-only and left untouched. With ``output`` set the
    union is written to that new file and ``target`` is also left untouched;
    otherwise ``target`` is rebuilt in place (backed up first unless
    ``backup=False``). Returns ``{table: {target_before, source, merged, added,
    shared}}``.
    """
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(f"source archive not found: {source}")
    if not target.is_file():
        raise FileNotFoundError(f"target archive not found: {target}")
    if source.resolve() == target.resolve():
        raise ValueError("source and target are the same file; nothing to merge")

    if output is not None:
        output = Path(output)
        if output.resolve() in (source.resolve(), target.resolve()):
            raise ValueError("--output must differ from both --source and --target")

    if not dry_run:
        if output is not None:
            shutil.copy2(target, output)
        elif backup:
            backup_path = target.with_name(target.name + f".bak-{int(time.time())}")
            shutil.copy2(target, backup_path)

    writable = target if (output is None or dry_run) else output
    con = _connect(writable, read_only=dry_run)
    try:
        _require_tables(con, "")
        safe_source = str(source).replace("'", "''")
        con.execute(f"ATTACH '{safe_source}' AS src (READ_ONLY)")
        _require_tables(con, "src")

        lines_select = f"SELECT {_LINES_COLS} FROM lines UNION SELECT {_LINES_COLS} FROM src.lines"
        report = {
            "odds": _merge_table(con, "odds", _odds_select(con), _ODDS_SORT, dry_run),
            "lines": _merge_table(con, "lines", lines_select, _LINES_COLS, dry_run),
        }

        if not dry_run:
            con.execute("ANALYZE")
            con.execute("CHECKPOINT")
        con.execute("DETACH src")
    finally:
        con.close()
    return report


@click.command()
@click.option(
    "--source",
    required=True,
    type=click.Path(path_type=Path),
    help="Archive to merge FROM (opened read-only, never modified).",
)
@click.option(
    "--target",
    default=_DEFAULT_TARGET,
    type=click.Path(path_type=Path),
    help="Archive to merge INTO, rebuilt in place as the union "
    "(default: $SPORTSTRADAMUS_ARCHIVE_DB or archive/archive.duckdb).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Write the union to a NEW file instead of mutating --target.",
)
@click.option("--dry-run", is_flag=True, help="Report counts only; write nothing.")
@click.option(
    "--no-backup",
    is_flag=True,
    help="Skip the timestamped backup of --target before an in-place merge.",
)
def main(source: Path, target: Path, output: Path | None, dry_run: bool, no_backup: bool) -> None:
    """Merge the SOURCE odds archive into TARGET as a lossless set-union of rows."""
    report = merge_archives(source, target, output, dry_run=dry_run, backup=not no_backup)
    if dry_run:
        click.echo("DRY RUN — no changes written.")
    for table, r in report.items():
        click.echo(
            f"{table + ':':7} target {r['target_before']:,} + source {r['source']:,} "
            f"-> merged {r['merged']:,} (added {r['added']:,}; shared {r['shared']:,})"
        )
    if not dry_run:
        click.echo(f"Merged into {output if output is not None else target}.")


if __name__ == "__main__":
    main()
