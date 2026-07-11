"""The per-source subcommand builders (``run`` / ``backfill`` / ``list`` /
``verify`` / ``refresh-auth``).

Each ``build_*`` returns a ``click.Command`` bound to one source's closures;
:func:`collectors.cli.build_source_cli` assembles them into a group. The
commands delegate the generic fetch/parse/write and report work to
:mod:`collectors.runner`, handing it the source's closures.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from sportstradamus.collectors import dispatch, runner
from sportstradamus.collectors._options import (
    BACKFILL_REQUEST_PAUSE_S,
    BACKFILL_WEEK_PAUSE_S,
    CATALOG_OPTION,
    LOG_LEVEL_OPTION,
    TOKEN_PREVIEW_CHARS,
    load_or_empty,
    mode_option,
)
from sportstradamus.collectors.auth import extract_auth_updates, update_keys
from sportstradamus.collectors.catalog import load_catalog
from sportstradamus.helpers.logging import get_logger

if TYPE_CHECKING:
    from sportstradamus.collectors.cli import Source


def build_run(source: Source):
    @click.command("run")
    @click.option("--season", type=int, default=None, help="Season (year). Defaults to current.")
    @click.option("--week", type=int, default=None, help="Week. Defaults to inferred.")
    @click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
    @mode_option(source.modes, source.default_mode)
    @click.option("--dry-run", is_flag=True, help="Print resolved URLs and parquet paths only.")
    @click.option(
        "--no-cache",
        "no_cache",
        is_flag=True,
        help="Flip ``useCache`` to ``False`` on every v2 body — bypasses the "
        "server-side result cache. Use when a previous broken-body run may "
        "have cached an empty result under the same query hash.",
    )
    @click.option(
        "--refetch",
        is_flag=True,
        help="Re-download even if a non-empty parquet already exists for this "
        "cell. Default is to skip existing non-empty files; zero-row parquets "
        "are always re-fetched.",
    )
    @CATALOG_OPTION
    @LOG_LEVEL_OPTION
    def run(season, week, only, dry_run, no_cache, refetch, catalog_path, log_level, **kw) -> None:
        """Walk the catalog, fetch every endpoint, write one parquet per tool.

        Defaults to the most recently completed period. On a 401 mid-batch in
        an interactive terminal, prompts for a fresh curl and resumes.
        """
        log = get_logger(source.name)
        log.setLevel(log_level)
        context = source.default_context(season, week)
        season, week = context["season"], context["week"]
        mode = kw.get("mode", source.default_mode)
        log.info(
            source.name, extra={"season": season, "week": week, "mode": mode, "dry_run": dry_run}
        )
        specs = load_or_empty(catalog_path, source.catalog_path, source.name)
        if specs is None:
            return
        if only:
            specs = dispatch.filter_by_name(specs, only)
        if dry_run:
            dispatch.echo_dry_run(
                specs, source, season=season, week=week, mode=mode, refetch=refetch
            )
            return
        use_cache = not no_cache
        client = source.client()
        runner.run_specs(
            specs,
            fetch_one=lambda spec: dispatch.dispatch_capturing_errors(
                source,
                client,
                spec,
                season=season,
                week=week,
                mode=mode,
                use_cache=use_cache,
                log=log,
            ),
            request_body_for=lambda spec: source.render_request_body(
                spec, season=season, week=week, mode=mode, use_cache=use_cache
            ),
            path_for=lambda spec: source.path_for(spec, season=season, week=week, mode=mode),
            transform=source.transform,
            report_prefix=source.report_prefix,
            desc=source.name,
            command="run",
            extra={
                "season": season,
                "week": week,
                "mode": mode,
                "use_cache": use_cache,
                "refetch": refetch,
            },
            log=log,
            season=season,
            week=week,
            refetch=refetch,
        )

    return run


def build_backfill(source: Source):
    @click.command("backfill")
    @click.option(
        "--start-season", type=int, required=True, help="First season to fetch (inclusive)."
    )
    @click.option("--end-season", type=int, required=True, help="Last season to fetch (inclusive).")
    @click.option(
        "--start-week", type=int, default=1, show_default=True, help="First week of each season."
    )
    @click.option(
        "--end-week",
        type=int,
        default=source.backfill_end_week,
        show_default=True,
        help="Last week of each season.",
    )
    @click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
    @click.option(
        "--dry-run", is_flag=True, help="Print the total call count and exit without fetching."
    )
    @click.option(
        "--request-pause-min",
        type=float,
        default=BACKFILL_REQUEST_PAUSE_S[0],
        show_default=True,
        help="Min seconds (random uniform) between endpoints in the same week.",
    )
    @click.option(
        "--request-pause-max",
        type=float,
        default=BACKFILL_REQUEST_PAUSE_S[1],
        show_default=True,
        help="Max seconds (random uniform) between endpoints in the same week.",
    )
    @click.option(
        "--week-pause-min",
        type=float,
        default=BACKFILL_WEEK_PAUSE_S[0],
        show_default=True,
        help="Min seconds (random uniform) when transitioning to a new week.",
    )
    @click.option(
        "--week-pause-max",
        type=float,
        default=BACKFILL_WEEK_PAUSE_S[1],
        show_default=True,
        help="Max seconds (random uniform) when transitioning to a new week.",
    )
    @mode_option(source.modes, source.default_mode)
    @click.option(
        "--no-cache",
        "no_cache",
        is_flag=True,
        help="Same as ``run --no-cache`` — bypass the server-side cache.",
    )
    @click.option(
        "--refetch",
        is_flag=True,
        help="Re-download even if a non-empty parquet already exists for a given "
        "cell. Default is to skip — useful for resuming a backfill that stopped "
        "midway. Zero-row parquets are always re-fetched.",
    )
    @CATALOG_OPTION
    @LOG_LEVEL_OPTION
    def backfill(
        start_season,
        end_season,
        start_week,
        end_week,
        only,
        dry_run,
        request_pause_min,
        request_pause_max,
        week_pause_min,
        week_pause_max,
        no_cache,
        refetch,
        catalog_path,
        log_level,
        **kw,
    ) -> None:
        """Iterate (season, week) pairs and write per-tool parquets for each.

        Designed for an overnight one-time grab. Pacing is conservative by
        default (short pause between endpoints, longer on a week transition);
        override via the ``--request-pause-*`` / ``--week-pause-*`` flags.
        """
        log = get_logger(source.name)
        log.setLevel(log_level)
        mode = kw.get("mode", source.default_mode)
        specs = load_or_empty(catalog_path, source.catalog_path, source.name)
        if specs is None:
            return
        if only:
            specs = dispatch.filter_by_name(specs, only)
        seasons = list(range(start_season, end_season + 1))
        weeks = list(range(start_week, end_week + 1))
        if dry_run:
            click.echo(
                f"Would fetch {len(seasons) * len(weeks) * len(specs)} "
                f"({len(specs)} tools x {len(weeks)} weeks x {len(seasons)} seasons)."
            )
            return
        use_cache = not no_cache
        # Drive pacing from this loop, not the client, so the week-transition
        # pause is visible alongside the per-spec pause.
        client = source.client(inter_request_sleep_s=0)
        runner.backfill_specs(
            specs,
            seasons=seasons,
            weeks=weeks,
            make_fetch_one=lambda spec, s, w: dispatch.dispatch_capturing_errors(
                source, client, spec, season=s, week=w, mode=mode, use_cache=use_cache, log=log
            ),
            request_body_for=lambda spec, s, w: source.render_request_body(
                spec, season=s, week=w, mode=mode, use_cache=use_cache
            ),
            path_for_cell=lambda spec, s, w: source.path_for(spec, season=s, week=w, mode=mode),
            would_skip=lambda spec, s, w: dispatch.would_skip(
                source, spec, season=s, week=w, mode=mode, refetch=refetch
            ),
            transform=source.transform,
            report_prefix=source.report_prefix,
            desc=f"{source.name}-backfill",
            extra={
                "seasons": seasons,
                "weeks": weeks,
                "mode": mode,
                "use_cache": use_cache,
                "refetch": refetch,
            },
            log=log,
            refetch=refetch,
            request_range=(request_pause_min, request_pause_max),
            week_range=(week_pause_min, week_pause_max),
        )

    return backfill


def build_list(source: Source):
    @click.command("list")
    @CATALOG_OPTION
    def list_endpoints(catalog_path) -> None:
        """Print the registered endpoint catalog."""
        specs = load_catalog(catalog_path or source.catalog_path)
        if not specs:
            click.echo("(empty catalog)")
            return
        for spec in specs:
            cadence = "weekly" if spec.weekly else "season"
            click.echo(f"{spec.name:30s} {spec.method:5s} {cadence:7s} {spec.url}")

    return list_endpoints


def build_verify(source: Source):
    @click.command("verify")
    @click.option("--season", type=int, default=None, help="Season. Defaults to inferred current.")
    @click.option("--week", type=int, default=None, help="Week. Defaults to inferred.")
    @mode_option(source.modes, source.default_mode)
    @click.option("--only", multiple=True, help="Only check these endpoint names (repeatable).")
    @CATALOG_OPTION
    def verify(season, week, only, catalog_path, **kw) -> None:
        """Spot-check downloaded parquets against the requested season / week / mode.

        Loads the parquet that ``run`` should have written and confirms its
        rows carry the expected season/week values. Exits non-zero if any spec
        yielded an error-severity issue so cron / scripts can detect a botched
        download without parsing stdout.
        """
        context = source.default_context(season, week)
        season, week = context["season"], context["week"]
        mode = kw.get("mode", source.default_mode)
        specs = load_catalog(catalog_path or source.catalog_path)
        if not specs:
            click.echo("Catalog is empty. Nothing to verify.", err=True)
            return
        if only:
            specs = dispatch.filter_by_name(specs, only)
        results = source.verify_fn(specs, season=season, week=week, mode=mode)
        dispatch.echo_verify(results, season=season, week=week, mode=mode)

    return verify


def build_refresh_auth(source: Source):
    @click.command("refresh-auth")
    @click.argument("curl_input", type=click.File("r"))
    @click.option(
        "--keys-path",
        type=click.Path(path_type=Path),
        default=None,
        help="Override keys.json path (default: bundled creds/keys.json).",
    )
    def refresh_auth(curl_input, keys_path) -> None:
        """Update auth fields in creds/keys.json from a fresh curl command."""
        try:
            extracted = extract_auth_updates(curl_input.read(), source.auth_fields)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if not extracted:
            raise click.ClickException(
                "No Authorization / Cookie / User-Agent headers found in input. "
                "Did you paste a curl command from DevTools?"
            )
        path = update_keys(extracted, path=keys_path)
        for key, value in extracted.items():
            preview = value[:TOKEN_PREVIEW_CHARS] + (
                "..." if len(value) > TOKEN_PREVIEW_CHARS else ""
            )
            click.echo(f"Updated {key} ({len(value)} chars): {preview}")
        click.echo(f"Wrote {path}")

    return refresh_auth
