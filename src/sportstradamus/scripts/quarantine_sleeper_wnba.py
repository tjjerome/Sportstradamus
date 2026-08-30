#!/usr/bin/env python3
"""Quarantine the poisoned Sleeper WNBA quotes in the archive, one shot.

Sleeper changed its WNBA product around Aug 3, 2026: discounted/moved lines
sitting 5-9 points off the real books started arriving, and the ingestion path
kept archiving them as fair 50/50 quotes (``under_prob = 0.5``, ``ev ~= line``).
Those rows poisoned the weighted book consensus — the fitted book weights ended
up trusting Sleeper 0.626 on WNBA PRA. The ingestion fix lands separately; this
script repairs the archive rows the old ingest already wrote, on BOTH the prod
and dev boxes (syncs spread the poison to each).

The repair nulls ``ev`` and ``under_prob`` on the matching rows so every reader
skips Sleeper there: the consensus ``_weighted_book_ev`` NaN-masks a null ev,
and ``get_training_book_quotes`` treats a null ``under_prob`` as no native
quote. The rows' ``line`` column and the separate ``lines`` table are
deliberately untouched — they are the record of what Sleeper actually offered.

Dry-run by default: prints per-market affected-row counts and writes nothing.
``--apply`` requires an explicit ``--until`` (set it to the ingestion-fix
deploy time so rows the fixed encoder writes afterwards survive), writes a
timestamped ``.bak`` copy of the duckdb first (unless ``--no-backup``), then
nulls the rows. The CLI holds ``scripts/run_job.sh``'s shared archive flock
itself — never wrap it in an outer flock. Only apply AFTER the ingestion fix is
deployed to the box you are repairing: an old-ingest ``confer`` repopulates the
rows otherwise.

    poetry run python -m sportstradamus.scripts.quarantine_sleeper_wnba
    poetry run python -m sportstradamus.scripts.quarantine_sleeper_wnba \
        --apply --until "2026-08-29 18:00:00"

Follow an apply with ``python -m sportstradamus.scripts.refit_book_weights
--league WNBA`` so the fitted weights stop encoding the poison.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import click

from sportstradamus.helpers.locks import ARCHIVE_LOCK_FILE, file_lock
from sportstradamus.scripts.sweep_runaway_odds import _connect

# Same env override + default the Archive singleton and merge-archives honour.
_DEFAULT_ARCHIVE = os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", "archive/archive.duckdb")

# Match run_job.sh's `flock -w 900`: wait up to 15 min for a running cron, then fail.
_FLOCK_TIMEOUT_S = 900

# First game date Sleeper's discounted-line WNBA product showed up in the feed.
_POISON_START = "2026-08-03"

_SCOPE = "league = 'WNBA' AND book = 'Sleeper'"
# The `?` is the --until bound: rows the fixed ingest observes later must survive.
_QUARANTINE_PREDICATE = f"{_SCOPE} AND game_date >= DATE '{_POISON_START}' AND observed_at <= ?"


def quarantine_sleeper_wnba(
    archive: str | Path, *, until: datetime, apply: bool = False, backup: bool = True
) -> dict:
    """Count (and, with ``apply``, null) the poisoned Sleeper WNBA odds rows.

    Nulls ``ev`` and ``under_prob`` on rows matching the quarantine predicate;
    their ``line`` and the ``lines`` table are kept. Caller owns concurrency
    (the CLI holds the archive flock). With ``apply`` a timestamped
    ``.bak-<epoch>`` is written first (unless ``backup=False``) and the DB is
    checkpointed. Returns ``{total_sleeper_wnba, affected, per_market, updated,
    backup}``.
    """
    db = Path(archive)
    if not db.is_file():
        raise FileNotFoundError(f"archive not found: {db}")

    backup_path = None
    if apply and backup:
        backup_path = db.with_name(f"{db.name}.bak-{int(time.time())}")
        shutil.copy2(db, backup_path)

    con = _connect(db, read_only=not apply)
    try:
        total = con.execute(f"SELECT COUNT(*) FROM odds WHERE {_SCOPE}").fetchone()[0]
        affected = con.execute(
            f"SELECT COUNT(*) FROM odds WHERE {_QUARANTINE_PREDICATE}", [until]
        ).fetchone()[0]
        per_market = con.execute(
            f"SELECT market, COUNT(*) AS n FROM odds WHERE {_QUARANTINE_PREDICATE} "
            "GROUP BY market ORDER BY n DESC, market",
            [until],
        ).fetchall()
        updated = 0
        if apply and affected:
            updated = con.execute(
                f"UPDATE odds SET ev = NULL, under_prob = NULL WHERE {_QUARANTINE_PREDICATE}",
                [until],
            ).fetchone()[0]
            con.execute("CHECKPOINT")
    finally:
        con.close()

    return {
        "total_sleeper_wnba": total,
        "affected": affected,
        "per_market": per_market,
        "updated": updated,
        "backup": backup_path,
    }


def _print_report(report: dict, archive: Path, apply: bool, until: datetime) -> None:
    click.echo(f"archive: {archive}")
    click.echo(
        f"quarantine scope: WNBA Sleeper odds, game_date >= {_POISON_START}, "
        f"observed_at <= {until:%Y-%m-%d %H:%M:%S}"
    )
    click.echo(
        f"affected: {report['affected']:,} of {report['total_sleeper_wnba']:,} "
        "WNBA Sleeper rows — ev + under_prob -> NULL, line and lines table kept"
    )
    for market, n in report["per_market"]:
        click.echo(f"  {market:24} {n:>10,}")
    if not apply:
        click.echo("DRY RUN — nothing written; pass --apply --until <fix-deploy-ts> to repair.")
        return
    if report["backup"] is not None:
        click.echo(f"backed up -> {report['backup']}")
    click.echo(f"done; nulled ev/under_prob on {report['updated']:,} rows.")


@click.command()
@click.option(
    "--archive",
    default=_DEFAULT_ARCHIVE,
    type=click.Path(path_type=Path),
    help="Archive to repair in place (default: $SPORTSTRADAMUS_ARCHIVE_DB or archive/archive.duckdb).",
)
@click.option(
    "--until",
    type=click.DateTime(),
    default=None,
    help="Quarantine rows observed at or before this timestamp — the ingestion-fix "
    "deploy time. Required with --apply; a dry run defaults it to now.",
)
@click.option(
    "--apply", is_flag=True, help="Null the poisoned rows (default: dry run, counts only)."
)
@click.option("--no-backup", is_flag=True, help="Skip the timestamped .bak copy before --apply.")
def main(archive: Path, until: datetime | None, apply: bool, no_backup: bool) -> None:
    """Null ev/under_prob on poisoned Sleeper WNBA odds rows (dry run unless --apply).

    Deploy the Sleeper ingestion fix to the box you are repairing BEFORE
    applying, and pass its deploy time as --until — an old-ingest confer
    repopulates the rows otherwise. Follow up with
    `python -m sportstradamus.scripts.refit_book_weights --league WNBA`.
    """
    if apply and until is None:
        raise click.UsageError(
            "--apply needs an explicit --until (the ingestion-fix deploy time) so "
            "rows the fixed ingest writes afterwards survive."
        )
    if until is None:
        until = datetime.now()
    with file_lock(
        ARCHIVE_LOCK_FILE,
        timeout_s=_FLOCK_TIMEOUT_S,
        label="quarantine-sleeper-wnba",
        contention_hint="A long cron? Retry in a quiet window.",
    ):
        report = quarantine_sleeper_wnba(archive, until=until, apply=apply, backup=not no_backup)
    _print_report(report, archive, apply, until)


if __name__ == "__main__":
    main()
