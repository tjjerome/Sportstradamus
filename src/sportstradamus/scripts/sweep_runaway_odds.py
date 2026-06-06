#!/usr/bin/env python3
"""Sweep magnitude-blown ``ev`` rows from the archive in one global pass.

The ``get_ev`` clamp fix bounds every new book-EV write at
``SN_MAX_MEAN_FACTOR × line``, but the archive still carries the historical
runaway rows the old unbounded inversion wrote — the ZINB ``under ≤ gate``
runaway on the DFS write path and the SkewNormal ``mean → ∞`` asymptote — with
means up to the millions across every poisoned cell.
:mod:`sportstradamus.scripts.delete_corrupt_seed` strips those per named cell;
this driver sweeps EVERY cell at once for the one-shot post-deploy cleanup of a
live archive (dev or prod), reusing that module's blown-row predicate.

It is **dry-run by default** — it prints the blown-row count and the worst cells
and writes nothing. ``--apply`` deletes them (a timestamped ``.bak-<epoch>`` is
written first unless ``--no-backup``). The CLI holds ``scripts/run_job.sh``'s
shared archive flock so the sweep serializes against a live cron write.

Only sweep AFTER the ``get_ev`` fix is deployed: a ``confer`` running the old
encode repopulates runaway rows faster than you can delete them.

    poetry run sweep-runaway-odds                                            # dry run
    poetry run sweep-runaway-odds --apply                                    # back up + delete
    poetry run python -m sportstradamus.scripts.sweep_runaway_odds --apply   # on prod
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import time
from pathlib import Path

import click
import duckdb

from sportstradamus.scripts.delete_corrupt_seed import (
    BLOWN_ABS_CEILING,
    BLOWN_LINE_FACTOR,
    BLOWN_PREDICATE,
)

# Same env override + default the Archive singleton and merge-archives honour.
_DEFAULT_ARCHIVE = os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", "archive/archive.duckdb")

# The lock scripts/run_job.sh wraps every cron in; holding it serializes this
# sweep against a live confer/prophecize/close-lines archive write.
_ARCHIVE_LOCK_FILE = "/tmp/sportstradamus-archive.lock"
# Match run_job.sh's `flock -w 900`: wait up to 15 min for a running cron, then fail.
_FLOCK_TIMEOUT_S = 900
_FLOCK_POLL_S = 2

# Enough rows to identify the worst offenders without a wall of output.
_WORST_CELLS_SHOWN = 15


@contextlib.contextmanager
def _archive_flock(timeout_s: float = _FLOCK_TIMEOUT_S):
    """Hold ``run_job.sh``'s archive flock, waiting up to ``timeout_s`` for a cron."""
    fd = os.open(_ARCHIVE_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"archive flock {_ARCHIVE_LOCK_FILE} still held after "
                        f"{timeout_s:.0f}s (a long cron?); retry in a quiet window."
                    ) from None
                time.sleep(_FLOCK_POLL_S)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _connect(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open the archive, translating a DuckDB lock collision into operator guidance."""
    try:
        return duckdb.connect(str(path), read_only=read_only)
    except duckdb.IOException as exc:
        if "Could not set lock" not in str(exc):
            raise
        raise RuntimeError(
            f"archive {path} is locked despite the flock — is the dashboard or a "
            "stray reader holding it?"
        ) from exc


def _count_blown(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute(f"SELECT COUNT(*) FROM odds WHERE {BLOWN_PREDICATE}").fetchone()[0]


def sweep_runaway_odds(archive: str | Path, *, apply: bool = False, backup: bool = True) -> dict:
    """Count (and, with ``apply``, delete) every magnitude-blown ``ev`` row.

    Caller owns concurrency (the CLI holds the archive flock). With ``apply`` a
    timestamped ``.bak-<epoch>`` is written first (unless ``backup=False``), the
    blown rows are deleted in one pass and the DB is checkpointed. Returns
    ``{total_rows, blown, remaining, worst, backup}``.
    """
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")

    backup_path = None
    if apply and backup:
        backup_path = archive.with_name(archive.name + f".bak-{int(time.time())}")
        shutil.copy2(archive, backup_path)

    con = _connect(archive, read_only=not apply)
    try:
        total_rows = con.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
        blown = _count_blown(con)
        worst = con.execute(
            f"SELECT league, market, COUNT(*) AS n FROM odds WHERE {BLOWN_PREDICATE} "
            f"GROUP BY league, market ORDER BY n DESC LIMIT {_WORST_CELLS_SHOWN}"
        ).fetchall()
        remaining = blown
        if apply and blown:
            con.execute(f"DELETE FROM odds WHERE {BLOWN_PREDICATE}")
            remaining = _count_blown(con)
            con.execute("CHECKPOINT")
    finally:
        con.close()

    return {
        "total_rows": total_rows,
        "blown": blown,
        "remaining": remaining,
        "worst": worst,
        "backup": backup_path,
    }


def _print_report(report: dict, archive: Path, apply: bool) -> None:
    click.echo(f"archive: {archive}")
    click.echo(
        f"runaway rows (ev > {BLOWN_ABS_CEILING:.0f} OR ev > {BLOWN_LINE_FACTOR:.0f}×max positive line): "
        f"{report['blown']:,} of {report['total_rows']:,}"
    )
    for league, market, n in report["worst"]:
        click.echo(f"  {league:6} {market:24} {n:>10,}")
    if not apply:
        click.echo("DRY RUN — nothing written; pass --apply to back up + delete.")
        return
    if report["backup"] is not None:
        click.echo(f"backed up -> {report['backup']}")
    click.echo(f"deleted {report['blown']:,} rows ({report['remaining']:,} remaining).")


@click.command()
@click.option(
    "--archive",
    default=_DEFAULT_ARCHIVE,
    type=click.Path(path_type=Path),
    help="Archive to sweep in place (default: $SPORTSTRADAMUS_ARCHIVE_DB or archive/archive.duckdb).",
)
@click.option("--apply", is_flag=True, help="Delete the blown rows (default: dry run, count only).")
@click.option(
    "--no-backup", is_flag=True, help="Skip the timestamped backup before an --apply delete."
)
def main(archive: Path, apply: bool, no_backup: bool) -> None:
    """Sweep every magnitude-blown ev row from the archive (dry run unless --apply).

    Deploy the get_ev fix BEFORE sweeping prod — an old-encode confer repopulates
    runaway rows otherwise.
    """
    with _archive_flock():
        report = sweep_runaway_odds(archive, apply=apply, backup=not no_backup)
    _print_report(report, archive, apply)


if __name__ == "__main__":
    main()
