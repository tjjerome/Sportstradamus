#!/usr/bin/env python3
"""Repair both classes of poisoned book ``ev`` in the archive, in one global pass.

Two encoder bugs left ``ev`` values that no distribution produced, and both need
sweeping across every cell after their fix deploys:

* **Magnitude-blown** — the old unbounded inversion (the ZINB ``under ≤ gate``
  runaway on the DFS write path, the SkewNormal ``mean → ∞`` asymptote) wrote
  means up to the millions. The ``get_ev`` clamp fix bounds new writes at
  ``SN_MAX_MEAN_FACTOR × line``.
* **Gate-refuted** — a sharp quote whose de-vigged under-prob sits at or below
  the calibrated zero-inflation gate has no mean solution at all, so the encoder
  stored ``get_ev``'s clamp ceiling as if it were one.

Both are repaired the same way, because in both the *quote* is sound and only the
mean derived from it is not: where the row still carries its shape-free
``(under_prob, line)`` the **ev is nulled** and the row is kept. The consensus
reader None-skips a book with no ev, and the training join re-derives the mean
from the quote anyway. Only a row left with nothing usable — no quote and no
trustworthy ev — is **deleted**. That distinction is worth the branch: on the dev
archive 34,266 of 34,280 blown rows are real 2025–26 quotes from real books
(fanduel, betrivers, draftkings) on live cells, and deleting them would discard
book evidence that only paid historical API calls could buy back.

:mod:`sportstradamus.scripts.delete_corrupt_seed` strips blown rows per named
cell; this driver sweeps EVERY cell at once for the one-shot post-deploy cleanup
of a live archive (dev or prod), reusing that module's blown-row predicate.

It is **dry-run by default** — it prints both counts and the worst cells and
writes nothing. ``--apply`` repairs them (a timestamped ``.bak-<epoch>`` is
written first unless ``--no-backup``). The CLI holds ``scripts/run_job.sh``'s
shared archive flock so the sweep serializes against a live cron write.

Only sweep AFTER both encoder fixes are deployed to the box you are sweeping: a
``confer`` running the old encode repopulates poisoned rows faster than you can
repair them. Prod tracks ``devel``, so check what ``devel`` actually carries.

    poetry run sweep-runaway-odds                                            # dry run
    poetry run sweep-runaway-odds --apply                                    # back up + repair
    poetry run python -m sportstradamus.scripts.sweep_runaway_odds --apply   # on prod
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import click
import duckdb

from sportstradamus.helpers.locks import ARCHIVE_LOCK_FILE, file_lock
from sportstradamus.scripts.delete_corrupt_seed import (
    BLOWN_ABS_CEILING,
    BLOWN_LINE_FACTOR,
    BLOWN_PREDICATE,
)

# Same env override + default the Archive singleton and merge-archives honour.
_DEFAULT_ARCHIVE = os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", "archive/archive.duckdb")

# Match run_job.sh's `flock -w 900`: wait up to 15 min for a running cron, then fail.
_FLOCK_TIMEOUT_S = 900

# Enough rows to identify the worst offenders without a wall of output.
_WORST_CELLS_SHOWN = 15

# Same tolerance training_quotes.archive_ev_is_runaway uses to spare the ceiling from the
# runaway test, so the two agree on exactly which rows sit *at* the clamp.
_CLAMP_RTOL = 1e-12

# A clamped quote: ev sitting exactly on get_ev's ceiling for the row's own line. Keys on the
# odds row's own shape-free line — the line get_ev was actually called with — not the
# lines-table MAX that BLOWN_PREDICATE has to fall back on. Excluding blown rows keeps the two
# classes disjoint, so the dry run's counts sum to the real total. Underdog and Sleeper are in
# scope: Archive.add_dfs used to treat its ceiling as a deliberate placeholder, but it clamps
# on every unboosted pick against a high-zero-rate cell and rode into both the served blend
# and the fitted book weights, so it now stores NULL like the sportsbook writer.
_CLAMPED_PREDICATE = f"""
    ev IS NOT NULL AND line IS NOT NULL AND line > 0
    AND abs(ev - {BLOWN_LINE_FACTOR} * line) <= {_CLAMP_RTOL} * {BLOWN_LINE_FACTOR} * line
    AND NOT ({BLOWN_PREDICATE})
"""

# What the training join actually reads on the authentic path, so a row that has it survives
# a poisoned ev intact.
_SHAPE_FREE = "under_prob IS NOT NULL AND line IS NOT NULL"
_UNUSABLE_EV = f"(({BLOWN_PREDICATE}) OR ({_CLAMPED_PREDICATE}))"
_NULL_EV_PREDICATE = f"{_UNUSABLE_EV} AND {_SHAPE_FREE}"
_DELETE_PREDICATE = f"{_UNUSABLE_EV} AND NOT ({_SHAPE_FREE})"


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


def _count(con: duckdb.DuckDBPyConnection, predicate: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM odds WHERE {predicate}").fetchone()[0]


def _worst_cells(con: duckdb.DuckDBPyConnection, predicate: str) -> list[tuple[str, str, int]]:
    return con.execute(
        f"SELECT league, market, COUNT(*) AS n FROM odds WHERE {predicate} "
        f"GROUP BY league, market ORDER BY n DESC LIMIT {_WORST_CELLS_SHOWN}"
    ).fetchall()


def sweep_runaway_odds(archive: str | Path, *, apply: bool = False, backup: bool = True) -> dict:
    """Count (and, with ``apply``, repair) both classes of poisoned ``ev`` row.

    A row keeping its shape-free ``(under_prob, line)`` quote loses only the ``ev``;
    one left with nothing usable is deleted. Caller owns concurrency (the CLI holds the
    archive flock). With ``apply`` a timestamped ``.bak-<epoch>`` is written first
    (unless ``backup=False``) and the DB is checkpointed. Returns ``{total_rows, blown,
    clamped, nulled, deleted, remaining, remaining_clamped, worst, worst_clamped, backup}``.
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
        blown, clamped = _count(con, BLOWN_PREDICATE), _count(con, _CLAMPED_PREDICATE)
        nulled, deleted = _count(con, _NULL_EV_PREDICATE), _count(con, _DELETE_PREDICATE)
        worst = _worst_cells(con, BLOWN_PREDICATE)
        worst_clamped = _worst_cells(con, _CLAMPED_PREDICATE)
        remaining, remaining_clamped = blown, clamped
        if apply and (blown or clamped):
            con.execute(f"DELETE FROM odds WHERE {_DELETE_PREDICATE}")
            con.execute(f"UPDATE odds SET ev = NULL WHERE {_NULL_EV_PREDICATE}")
            remaining = _count(con, BLOWN_PREDICATE)
            remaining_clamped = _count(con, _CLAMPED_PREDICATE)
            con.execute("CHECKPOINT")
    finally:
        con.close()

    return {
        "total_rows": total_rows,
        "blown": blown,
        "clamped": clamped,
        "nulled": nulled,
        "deleted": deleted,
        "remaining": remaining,
        "remaining_clamped": remaining_clamped,
        "worst": worst,
        "worst_clamped": worst_clamped,
        "backup": backup_path,
    }


def _print_cells(worst: list[tuple[str, str, int]]) -> None:
    for league, market, n in worst:
        click.echo(f"  {league:6} {market:24} {n:>10,}")


def _print_report(report: dict, archive: Path, apply: bool) -> None:
    click.echo(f"archive: {archive}")
    click.echo(
        f"runaway rows (ev > {BLOWN_ABS_CEILING:.0f} OR ev > {BLOWN_LINE_FACTOR:.0f}×max positive line): "
        f"{report['blown']:,} of {report['total_rows']:,}"
    )
    _print_cells(report["worst"])
    click.echo(
        f"gate-refuted rows (ev = {BLOWN_LINE_FACTOR:.0f}×line, non-DFS book): "
        f"{report['clamped']:,} of {report['total_rows']:,}"
    )
    _print_cells(report["worst_clamped"])
    click.echo(
        f"repair plan: null the ev on {report['nulled']:,} rows (quotes kept), "
        f"delete {report['deleted']:,} with no quote left."
    )
    if not apply:
        click.echo("DRY RUN — nothing written; pass --apply to back up + repair.")
        return
    if report["backup"] is not None:
        click.echo(f"backed up -> {report['backup']}")
    click.echo(
        f"done; {report['remaining']:,} runaway and {report['remaining_clamped']:,} "
        "gate-refuted rows remain."
    )


@click.command()
@click.option(
    "--archive",
    default=_DEFAULT_ARCHIVE,
    type=click.Path(path_type=Path),
    help="Archive to sweep in place (default: $SPORTSTRADAMUS_ARCHIVE_DB or archive/archive.duckdb).",
)
@click.option(
    "--apply", is_flag=True, help="Repair the poisoned rows (default: dry run, counts only)."
)
@click.option(
    "--no-backup", is_flag=True, help="Skip the timestamped backup before an --apply repair."
)
def main(archive: Path, apply: bool, no_backup: bool) -> None:
    """Repair every poisoned ev row in the archive (dry run unless --apply).

    Deletes magnitude-blown rows; nulls the ev on gate-refuted ones, keeping their
    quote. Deploy the encoder fixes to the box you are sweeping BEFORE sweeping it —
    an old-encode confer repopulates the rows otherwise.
    """
    with file_lock(
        ARCHIVE_LOCK_FILE,
        timeout_s=_FLOCK_TIMEOUT_S,
        label="sweep-runaway-odds",
        contention_hint="A long cron? Retry in a quiet window.",
    ):
        report = sweep_runaway_odds(archive, apply=apply, backup=not no_backup)
    _print_report(report, archive, apply)


if __name__ == "__main__":
    main()
