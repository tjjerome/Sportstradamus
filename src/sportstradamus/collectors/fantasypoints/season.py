"""``fp-fetch season`` — one request per tool for a whole season.

The API's ``splits=week`` switch answers a multi-week window with one row
per entity-week instead of one aggregated row, and the columns the stats
layer reads match what the per-week requests return. It takes the regular
season and the postseason rounds together, so one request per tool
replaces 22, which is what makes a multi-season re-pull fit inside the
account's daily request budget (``docs/fantasypoints.md`` § Historical
backfill).

Two things keep the parquets usable in place of per-week pulls. A response
is capped at a per-tool row count (1,500 on the receiving tools, up to
3,700 on the snap tools) and, past the cap, the lowest-volume players drop
out of every week asked for; the API flags such a response in a header,
and the window is then halved until each half fits. And
``lineup-combos/ol`` ignores the switch and answers with its aggregated
rows; a tool like that goes week by week.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

import click
from tqdm import tqdm

from sportstradamus.collectors import dispatch, runner
from sportstradamus.collectors._options import CATALOG_OPTION, LOG_LEVEL_OPTION, load_or_empty
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.fantasypoints.source import (
    FP_SOURCE,
    NFL_REGULAR_SEASON_WEEKS,
    POSTSEASON_ROUNDS,
    Mode,
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

# Set on a response the API cut off at its row cap (``x-result-cap``);
# absent on a complete one, and kept on a server-cache hit.
_TRUNCATED_HEADER = "x-result-truncated"
# A season on disk: folders week_01..18, then week_19..22 for the postseason
# rounds, whose split rows the API labels by round instead of numbering.
_FOLDER_WEEKS = list(range(1, NFL_REGULAR_SEASON_WEEKS + POSTSEASON_ROUNDS + 1))
_POSTSEASON_ROUND_LABELS = ("WC", "DV", "CC", "SB")
_SPLIT_WEEK_COL = "split_week"
# A week's share of its window's total: present only on split responses and
# a function of the window asked for, so it never reaches the parquet.
_WINDOW_SHARE_COLS = ["split_pct", "split_rte_pct"]


@click.command("season")
@click.option("--season", type=int, required=True, help="Season (year).")
@click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
@click.option(
    "--refetch",
    is_flag=True,
    help="Re-download every week even when it is on disk. Default is to ask "
    "only for the weeks that are missing or hold zero rows.",
)
@CATALOG_OPTION
@LOG_LEVEL_OPTION
def fetch_season(season, only, refetch, catalog_path, log_level) -> None:
    """Fetch every tool for a whole season, one request per tool.

    Covers weeks 1-18 and the four postseason rounds (folders week_19..22).
    """
    log = get_logger(FP_SOURCE.name)
    log.setLevel(log_level)
    specs = load_or_empty(catalog_path, FP_SOURCE.catalog_path, FP_SOURCE.name)
    if specs is None:
        return
    if only:
        specs = dispatch.filter_by_name(specs, only)
    client = FP_SOURCE.client()
    walk = (
        result
        for spec in tqdm(specs, desc=f"{FP_SOURCE.name}-season", unit="tool")
        for result in _fetch_season(spec, client, season=season, refetch=refetch, log=log)
    )
    runner.collect_results(
        walk,
        total=len(specs) * len(_FOLDER_WEEKS),
        unit="season calls",
        report_prefix=FP_SOURCE.report_prefix,
        command="season",
        extra={"season": season, "refetch": refetch},
        log=log,
    )


def _mode_and_week(folder_week: int) -> tuple[Mode, int]:
    """Return the per-week command's ``(mode, week)`` for one folder week."""
    if folder_week > NFL_REGULAR_SEASON_WEEKS:
        return "postseason", folder_week - NFL_REGULAR_SEASON_WEEKS
    return "weekly", folder_week


def _fetch_season(
    spec: EndpointSpec,
    client: CookieClient,
    *,
    season: int,
    refetch: bool,
    log: logging.Logger,
) -> list[RunResult]:
    """Write one parquet per folder week of ``season`` for ``spec``; one outcome per week."""
    paths = {}
    for folder_week in _FOLDER_WEEKS:
        mode, week = _mode_and_week(folder_week)
        paths[folder_week] = parquet_path_for_spec(spec, season=season, week=week, mode=mode)
    on_disk = {folder_week: existing_parquet_rows(path) for folder_week, path in paths.items()}
    wanted = [w for w in _FOLDER_WEEKS if refetch or not on_disk[w]]
    results = [
        RunResult(
            name=spec.name,
            url=spec.url,
            method="GET",
            status=RESULT_SKIPPED,
            season=season,
            week=folder_week,
            rows=on_disk[folder_week],
            path=str(paths[folder_week]),
        )
        for folder_week in _FOLDER_WEEKS
        if folder_week not in wanted
    ]
    if wanted:
        results += _fetch_window(spec, client, season=season, weeks=wanted, paths=paths, log=log)
    wrote = sum(r.rows for r in results if r.status == RESULT_OK)
    click.echo(f"  {spec.name}: wrote {wrote} rows across {len(wanted)} weeks", err=True)
    return results


def _fetch_window(
    spec: EndpointSpec,
    client: CookieClient,
    *,
    season: int,
    weeks: list[int],
    paths: dict[int, Path],
    log: logging.Logger,
) -> list[RunResult]:
    """One split request for folder ``weeks``, halved for as long as the API truncates it."""
    url = build_url(
        spec.url,
        {
            **spec.render_params(season=season, week=weeks[0]),
            **split_period_params(season=season, folder_weeks=weeks),
        },
    )
    make_result = partial(RunResult, name=spec.name, url=url, method="GET", season=season)
    body, err = dispatch.dispatch_capturing_errors(
        FP_SOURCE, client, spec, season=season, week=weeks[0], split_weeks=weeks, log=log
    )
    if err is not None:
        click.echo(f"  {spec.name}: {err['error_message']}", err=True)
        return [make_result(status=RESULT_FETCH_FAILED, week=week, **err) for week in weeks]
    if client.last_response_headers.get(_TRUNCATED_HEADER) and len(weeks) > 1:
        click.echo(
            f"  {spec.name}: weeks {weeks[0]}-{weeks[-1]} hit the row cap; halving", err=True
        )
        half = len(weeks) // 2
        return [
            result
            for part in (weeks[:half], weeks[half:])
            for result in _fetch_window(
                spec, client, season=season, weeks=part, paths=paths, log=log
            )
        ]
    by_week = _rows_by_week(body)
    if by_week is None:
        click.echo(
            f"  {spec.name}: no per-week split in the response; fetching week by week", err=True
        )
        return [
            _fetch_one_week(spec, client, season=season, folder_week=w, path=paths[w], log=log)
            for w in weeks
        ]
    results = []
    for folder_week in weeks:
        df = parse_table_response(
            by_week.get(folder_week, []),
            spec=spec,
            season=season,
            week=_mode_and_week(folder_week)[1],
        )
        write_parquet(df.drop(columns=_WINDOW_SHARE_COLS, errors="ignore"), paths[folder_week])
        status = RESULT_EMPTY if df.empty else RESULT_OK
        results.append(
            make_result(status=status, week=folder_week, rows=len(df), path=str(paths[folder_week]))
        )
    return results


def _fetch_one_week(
    spec: EndpointSpec,
    client: CookieClient,
    *,
    season: int,
    folder_week: int,
    path: Path,
    log: logging.Logger,
) -> RunResult:
    """The per-week command's request for one folder week, on the same client.

    Forces ``refetch=True``: every folder week reaching this fallback was
    already chosen by ``_fetch_season``'s ``wanted`` filter, so this must not
    let ``fetch_and_write_one``'s own on-disk skip-check re-decide and drop a
    ``--refetch`` week that already has rows.
    """
    mode, week = _mode_and_week(folder_week)
    return runner.fetch_and_write_one(
        spec,
        path_for=lambda _spec: path,
        fetch_one=lambda s: dispatch.dispatch_capturing_errors(
            FP_SOURCE, client, s, season=season, week=week, mode=mode, log=log
        ),
        transform=parse_table_response,
        log=log,
        season=season,
        week=week,
        refetch=True,
    )


def _rows_by_week(payload: object) -> dict[int, list[dict]] | None:
    """Group split rows by folder week; ``None`` when the tool answered without splits.

    An empty body is a window with no rows for this tool, not a missing
    split, so it comes back as an empty mapping and every week gets an
    empty parquet without a per-week re-pull.
    """
    rows = extract_rows(payload)
    if rows and not any(row.get(_SPLIT_WEEK_COL) for row in rows):
        return None
    by_week: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        label = row[_SPLIT_WEEK_COL]
        if label in _POSTSEASON_ROUND_LABELS:
            folder_week = NFL_REGULAR_SEASON_WEEKS + _POSTSEASON_ROUND_LABELS.index(label) + 1
        else:
            folder_week = int(label)
        by_week[folder_week].append(row)
    return by_week
