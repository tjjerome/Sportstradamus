"""Date-stamped ``run`` / ``verify`` builders for cumulative collector sources.

Cleaning the Glass and Baseball Savant serve *already-cumulative* season-to-date
tables with no per-week grain, so their snapshots are keyed by capture date
(``{season}/{YYYY-MM-DD}/``) rather than NFL week. These builders expose
``--season`` / ``--date`` instead of the week/mode surface in
:mod:`collectors.commands`, and reuse the same source-neutral
:func:`collectors.runner.run_specs` drive loop (with ``week=0`` as an unused
label). Backfill is intentionally absent — historical as-of backfill is a
documented follow-on that needs date-ranged upstream queries.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import click

from sportstradamus.collectors import dispatch, runner
from sportstradamus.collectors._options import CATALOG_OPTION, LOG_LEVEL_OPTION, load_or_empty
from sportstradamus.collectors.catalog import load_catalog
from sportstradamus.collectors.report import build_url, existing_parquet_rows
from sportstradamus.helpers.logging import get_logger

if TYPE_CHECKING:
    from sportstradamus.collectors.cli import Source

_DATE_OPTION = click.option(
    "--date",
    "capture_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Capture-date stamp (YYYY-MM-DD) for the snapshot dir. Defaults to today.",
)


def _resolve(source: Source, season, capture_date) -> tuple[int, str]:
    """Return ``(season, iso_date)`` — season via the source, date defaulting to today."""
    resolved_season = source.default_context(season, None)["season"]
    stamp = (capture_date.date() if capture_date else date.today()).isoformat()
    return resolved_season, stamp


def build_run_dated(source: Source):
    @click.command("run")
    @click.option("--season", type=int, default=None, help="Season (year). Defaults to current.")
    @_DATE_OPTION
    @click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
    @click.option("--dry-run", is_flag=True, help="Print resolved URLs and parquet paths only.")
    @click.option(
        "--refetch",
        is_flag=True,
        help="Re-download even if a non-empty parquet already exists for this "
        "(season, date) cell. Default is to skip existing non-empty files.",
    )
    @CATALOG_OPTION
    @LOG_LEVEL_OPTION
    def run(season, capture_date, only, dry_run, refetch, catalog_path, log_level) -> None:
        """Snapshot each catalog endpoint's current season-to-date table to disk.

        Writes one parquet per tool under
        ``{player,team}_data/{LEAGUE}/{source}/{season}/{date}/``. Re-runs on the
        same date overwrite; a later date writes a fresh dated snapshot.
        """
        log = get_logger(source.name)
        log.setLevel(log_level)
        season, stamp = _resolve(source, season, capture_date)
        log.info(source.name, extra={"season": season, "date": stamp, "dry_run": dry_run})
        specs = load_or_empty(catalog_path, source.catalog_path, source.name)
        if specs is None:
            return
        if only:
            specs = dispatch.filter_by_name(specs, only)
        if dry_run:
            _echo_dry_run(source, specs, season=season, stamp=stamp, refetch=refetch)
            return
        client = source.client()
        runner.run_specs(
            specs,
            fetch_one=lambda spec: dispatch.dispatch_capturing_errors(
                source, client, spec, season=season, week=0, mode=None, use_cache=True, log=log
            ),
            request_body_for=lambda spec: None,
            path_for=lambda spec: source.path_for(spec, season=season, date=stamp),
            transform=source.transform,
            report_prefix=source.report_prefix,
            desc=source.name,
            command="run",
            extra={"season": season, "date": stamp, "refetch": refetch},
            log=log,
            season=season,
            week=0,
            refetch=refetch,
        )

    return run


def build_verify_dated(source: Source):
    @click.command("verify")
    @click.option("--season", type=int, default=None, help="Season. Defaults to current.")
    @_DATE_OPTION
    @click.option("--only", multiple=True, help="Only check these endpoint names (repeatable).")
    @CATALOG_OPTION
    def verify(season, capture_date, only, catalog_path) -> None:
        """Confirm each tool's parquet landed with rows for the season and date.

        Exits non-zero if any expected snapshot is missing or empty so cron can
        detect a botched capture without parsing stdout.
        """
        season, stamp = _resolve(source, season, capture_date)
        specs = load_catalog(catalog_path or source.catalog_path)
        if not specs:
            click.echo("Catalog is empty. Nothing to verify.", err=True)
            return
        if only:
            specs = dispatch.filter_by_name(specs, only)
        ok = fail = 0
        for spec in specs:
            try:
                path = source.path_for(spec, season=season, date=stamp)
            except ValueError as exc:
                fail += 1
                click.echo(f"FAIL {spec.name}: unrouted ({exc})")
                continue
            rows = existing_parquet_rows(path)
            if rows > 0:
                ok += 1
                click.echo(f"OK   {spec.name} ({rows} rows)")
            else:
                fail += 1
                click.echo(f"FAIL {spec.name}: no parquet / 0 rows at {path}")
        click.echo("")
        click.echo(f"Summary: {ok} ok, {fail} fail (season={season}, date={stamp}).")
        if fail:
            raise click.ClickException(f"{fail} spec(s) failed verification.")

    return verify


def _echo_dry_run(source: Source, specs, *, season: int, stamp: str, refetch: bool) -> None:
    for spec in specs:
        url = build_url(spec.url, spec.render_params(season=season))
        try:
            path = source.path_for(spec, season=season, date=stamp)
        except ValueError as exc:
            click.echo(f"{spec.name}: {spec.method} {url} -> <unrouted: {exc}>")
            continue
        skip = "  [SKIP]" if not refetch and existing_parquet_rows(path) > 0 else ""
        click.echo(f"{spec.name}: {spec.method} {url} -> {path}{skip}")
