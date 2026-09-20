"""``fp-fetch season`` — one request per tool for a whole season.

The API's ``splits=week`` switch answers a multi-week window with one row
per entity-week instead of one aggregated row, and the values match what
the per-week requests return (only the ``row_id`` / ``rank`` order and an
extra ``split_pct`` column differ). One request per tool therefore replaces
18 regular-season or 4 postseason requests, which is what makes a
multi-season re-pull fit inside the account's daily request budget
(``docs/fantasypoints.md`` § Historical backfill). ``lineup-combos/ol``
ignores the switch and answers with its aggregated rows; a tool like that
falls back to the per-week path, so the parquets on disk are the same
either way.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial

import click
from tqdm import tqdm

from sportstradamus.collectors import dispatch, runner
from sportstradamus.collectors._options import CATALOG_OPTION, LOG_LEVEL_OPTION, load_or_empty
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.fantasypoints.source import (
    FP_SOURCE,
    NFL_REGULAR_SEASON_WEEKS,
    POSTSEASON_ROUNDS,
    split_period_params,
)
from sportstradamus.collectors.fantasypoints.transform import (
    extract_rows,
    parquet_path_for_spec,
    parse_table_response,
    write_parquet,
)
from sportstradamus.collectors.report import (
    RESULT_EMPTY,
    RESULT_FETCH_FAILED,
    RESULT_OK,
    RESULT_SKIPPED,
    RunResult,
    build_url,
    existing_parquet_rows,
)
from sportstradamus.collectors.transport import CookieClient
from sportstradamus.helpers.logging import get_logger

# Postseason splits are labelled by round; the parquet tree numbers them
# 1-4 (folders week_19..22 under ``--mode postseason``).
_POSTSEASON_ROUND_LABELS = ("WC", "DV", "CC", "SB")
_SPLIT_WEEK_COL = "split_week"
# Only a split response carries it; dropped so season-pulled and
# week-pulled parquets share one schema.
_SPLIT_ONLY_COLS = ["split_pct"]


@click.command("season")
@click.option("--season", type=int, required=True, help="Season (year).")
@click.option(
    "--mode",
    type=click.Choice(["weekly", "postseason"]),
    default="weekly",
    show_default=True,
    help="Regular-season weeks 1-18, or the four postseason rounds.",
)
@click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
@click.option(
    "--refetch",
    is_flag=True,
    help="Re-download a tool even when every week of the season is on disk. "
    "Default is to skip those; a tool with any missing or zero-row week is "
    "re-pulled whole.",
)
@CATALOG_OPTION
@LOG_LEVEL_OPTION
def fetch_season(season, mode, only, refetch, catalog_path, log_level) -> None:
    """Fetch every tool for a whole season, one request per tool."""
    log = get_logger(FP_SOURCE.name)
    log.setLevel(log_level)
    specs = load_or_empty(catalog_path, FP_SOURCE.catalog_path, FP_SOURCE.name)
    if specs is None:
        return
    if only:
        specs = dispatch.filter_by_name(specs, only)
    weeks = range(1, (POSTSEASON_ROUNDS if mode == "postseason" else NFL_REGULAR_SEASON_WEEKS) + 1)
    client = FP_SOURCE.client()
    walk = (
        result
        for spec in tqdm(specs, desc=f"{FP_SOURCE.name}-season", unit="tool")
        for result in _fetch_season(
            spec, client, season=season, mode=mode, weeks=weeks, refetch=refetch, log=log
        )
    )
    runner.collect_results(
        walk,
        total=len(specs) * len(weeks),
        unit="season calls",
        report_prefix=FP_SOURCE.report_prefix,
        command="season",
        extra={"season": season, "mode": mode, "refetch": refetch},
        log=log,
    )


def _fetch_season(
    spec: EndpointSpec,
    client: CookieClient,
    *,
    season: int,
    mode: str,
    weeks: range,
    refetch: bool,
    log: logging.Logger,
) -> list[RunResult]:
    """Write one parquet per week of ``season`` for ``spec``; one outcome per week."""
    paths = {
        week: parquet_path_for_spec(spec, season=season, week=week, mode=mode) for week in weeks
    }
    on_disk = {week: existing_parquet_rows(path) for week, path in paths.items()}
    url = build_url(
        spec.url,
        {
            **spec.render_params(season=season, week=weeks[0]),
            **split_period_params(season=season, mode=mode),
        },
    )
    make_result = partial(RunResult, name=spec.name, url=url, method="GET", season=season)
    if not refetch and all(on_disk.values()):
        click.echo(f"  {spec.name}: skip (every week on disk)", err=True)
        return [
            make_result(status=RESULT_SKIPPED, week=week, rows=rows, path=str(paths[week]))
            for week, rows in on_disk.items()
        ]
    body, err = dispatch.dispatch_capturing_errors(
        FP_SOURCE,
        client,
        spec,
        season=season,
        week=weeks[0],
        mode=mode,
        use_cache=True,
        window="season",
        log=log,
    )
    if err is not None:
        click.echo(f"  {spec.name}: {err['error_message']}", err=True)
        return [make_result(status=RESULT_FETCH_FAILED, week=0, **err)]
    by_week = _rows_by_week(body, mode)
    if by_week is None:
        click.echo(
            f"  {spec.name}: no per-week split in the response; fetching week by week", err=True
        )
        return [
            runner.fetch_and_write_one(
                spec,
                path_for=lambda s, w=week: paths[w],
                fetch_one=lambda s, w=week: dispatch.dispatch_capturing_errors(
                    FP_SOURCE, client, s, season=season, week=w, mode=mode, use_cache=True, log=log
                ),
                transform=parse_table_response,
                log=log,
                season=season,
                week=week,
                refetch=refetch,
            )
            for week in weeks
        ]
    results = []
    for week, path in paths.items():
        df = parse_table_response(by_week.get(week, []), spec=spec, season=season, week=week)
        write_parquet(df.drop(columns=_SPLIT_ONLY_COLS, errors="ignore"), path)
        status = RESULT_EMPTY if df.empty else RESULT_OK
        results.append(make_result(status=status, week=week, rows=len(df), path=str(path)))
    click.echo(
        f"  {spec.name}: wrote {sum(r.rows for r in results)} rows across {len(weeks)} weeks",
        err=True,
    )
    return results


def _rows_by_week(payload: object, mode: str) -> dict[int, list[dict]] | None:
    """Group split rows by folder week; ``None`` when the tool answered without splits.

    An empty body is a season with no rows for this tool, not a missing
    split, so it comes back as an empty mapping and every week gets an
    empty parquet without a per-week re-pull.
    """
    rows = extract_rows(payload)
    if rows and not any(row.get(_SPLIT_WEEK_COL) for row in rows):
        return None
    by_week: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        label = row[_SPLIT_WEEK_COL]
        week = _POSTSEASON_ROUND_LABELS.index(label) + 1 if mode == "postseason" else int(label)
        by_week[week].append(row)
    return by_week
