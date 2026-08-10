"""Generic fetch -> parse -> write loop for collector sources.

The per-endpoint driver (:func:`fetch_and_write_one`) and the run /
backfill loop bodies (:func:`run_specs`, :func:`backfill_specs`) are
source-neutral: they take closures a source binds to its own context
(``path_for``, ``fetch_one``, ``transform``) and know nothing about
season / week / date rendering. Each endpoint's outcome is captured in a
:class:`~sportstradamus.collectors.report.RunResult` and dumped to a
transient JSON report so failures can be diagnosed without re-running.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import pandas as pd
from tqdm import tqdm

from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.report import (
    RESULT_EMPTY,
    RESULT_FETCH_FAILED,
    RESULT_OK,
    RESULT_ROUTING_FAILED,
    RESULT_SKIPPED,
    RunResult,
    build_url,
    existing_parquet_rows,
    summarize,
    truncate_for_preview,
)

PathFor = Callable[[EndpointSpec], Path]
FetchOne = Callable[[EndpointSpec], "tuple[Any | None, dict[str, Any] | None]"]
Transform = Callable[[Any], pd.DataFrame]


def fetch_and_write_one(
    spec: EndpointSpec,
    *,
    path_for: PathFor,
    fetch_one: FetchOne,
    transform: Transform,
    log: logging.Logger,
    season: int,
    week: int,
    period: str | None = None,
    request_body: dict | list | None = None,
    refetch: bool = False,
) -> RunResult:
    """Route, fetch, parse to DataFrame, write parquet, return the outcome.

    Any failure (routing, fetch, decode) is captured into the returned
    :class:`RunResult` so the outer loop can keep going and the report can
    show exactly what broke per spec.

    When ``refetch=False`` (the default), a non-empty parquet already on
    disk for this cell short-circuits the network call — the file is treated
    as already-fetched and the result is recorded as ``RESULT_SKIPPED``.
    Zero-row parquets are treated as failed downloads and re-fetched anyway.
    Pass ``refetch=True`` to force a re-download.
    """
    # style: allow-complexity — per-endpoint orchestrator: route -> fetch ->
    # parse -> write, capturing every failure (routing, fetch, decode) into the
    # returned RunResult so the outer loop keeps going. Residual CC is the
    # sequential stages plus the cache/skip/refetch guards.
    base_result = RunResult(
        name=spec.name,
        url=build_url(spec.url, spec.render_params(season=season, week=week)),
        method=spec.method.upper(),
        season=season,
        week=week,
        period=period,
        status=RESULT_FETCH_FAILED,
        request_body=request_body,
    )
    try:
        target = path_for(spec)
    except ValueError as exc:
        log.error("routing failed", extra={"endpoint": spec.name, "error": str(exc)})
        click.echo(f"  {spec.name}: {exc}", err=True)
        base_result.status = RESULT_ROUTING_FAILED
        base_result.error_class = exc.__class__.__name__
        base_result.error_message = str(exc)
        return base_result
    if not refetch:
        existing_rows = existing_parquet_rows(target)
        if existing_rows > 0:
            base_result.path = str(target)
            base_result.rows = existing_rows
            base_result.status = RESULT_SKIPPED
            click.echo(f"  {spec.name}: skip ({existing_rows} rows on disk)", err=True)
            log.info(
                "skip (already on disk)",
                extra={"endpoint": spec.name, "path": str(target), "rows": existing_rows},
            )
            return base_result
    body, err = fetch_one(spec)
    if err is not None:
        click.echo(f"  {spec.name}: {err['error_message']}", err=True)
        base_result.status = RESULT_FETCH_FAILED
        base_result.error_class = err.get("error_class")
        base_result.error_message = err.get("error_message")
        base_result.http_status = err.get("http_status")
        base_result.response_preview = err.get("response_preview")
        return base_result
    df = transform(body)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    base_result.path = str(target)
    base_result.rows = len(df)
    base_result.status = RESULT_EMPTY if df.empty else RESULT_OK
    if df.empty:
        # Capture a truncated preview of the response so the user can diff
        # against a working browser curl. The empty-rows case ("filters too
        # restrictive" vs. "different response shape" vs. "cached empty from
        # a previous bad query") is invisible without this.
        base_result.response_preview = truncate_for_preview(body)
        log.warning(
            "empty response — writing empty parquet anyway",
            extra={"endpoint": spec.name, "season": season, "week": week, "path": str(target)},
        )
    click.echo(f"  {spec.name}: wrote {len(df)} rows -> {target}", err=True)
    log.info(
        "wrote parquet",
        extra={"endpoint": spec.name, "path": str(target), "rows": len(df)},
    )
    return base_result


def run_specs(
    specs: list[EndpointSpec],
    *,
    fetch_one: FetchOne,
    request_body_for: Callable[[EndpointSpec], dict | list | None],
    path_for: PathFor,
    transform: Transform,
    report_prefix: str,
    desc: str,
    command: str,
    extra: dict[str, Any],
    log: logging.Logger,
    season: int,
    week: int,
    refetch: bool,
) -> None:
    """Walk ``specs`` once, writing each parquet, then dump the report.

    ``fetch_one(spec) -> (body, err_dict|None)`` is the source's dispatch,
    already bound to this run's context. Echoes the ok/skip/empty/failed
    summary to stderr and raises :class:`click.ClickException` if any spec
    failed so cron surfaces a non-zero exit.
    """
    results: list[RunResult] = []
    for spec in tqdm(specs, desc=desc, unit="endpoint"):
        results.append(
            fetch_and_write_one(
                spec,
                path_for=path_for,
                fetch_one=fetch_one,
                transform=transform,
                log=log,
                season=season,
                week=week,
                request_body=request_body_for(spec),
                refetch=refetch,
            )
        )
    report_path, failures = summarize(
        results, report_prefix=report_prefix, command=command, extra=extra
    )
    if failures:
        raise click.ClickException(f"{len(failures)} spec(s) failed — see report at {report_path}")


def backfill_specs(
    specs: list[EndpointSpec],
    *,
    seasons: list[int],
    weeks: list[int],
    make_fetch_one: Callable[..., tuple[Any | None, dict[str, Any] | None]],
    request_body_for: Callable[..., dict | list | None],
    path_for_cell: Callable[[EndpointSpec, int, int], Path],
    would_skip: Callable[[EndpointSpec, int, int], bool],
    transform: Transform,
    report_prefix: str,
    desc: str,
    extra: dict[str, Any],
    log: logging.Logger,
    refetch: bool,
    request_range: tuple[float, float],
    week_range: tuple[float, float],
) -> None:
    """Iterate (season x week x spec), pacing between calls, then dump the report.

    ``make_fetch_one`` and ``request_body_for`` are called per (spec, season,
    week) cell. Pacing is conservative: a short pause between endpoints in the
    same week, a longer one on a week transition; cached cells skip the pause
    so resuming a half-finished backfill is near-instant.
    """
    total = len(seasons) * len(weeks) * len(specs)
    results: list[RunResult] = []
    prev_week_key: tuple[int, int] | None = None
    with tqdm(total=total, desc=desc, unit="call") as bar:
        for season in seasons:
            for week in weeks:
                prev_week_key = _backfill_week(
                    specs,
                    season,
                    week,
                    prev_week_key,
                    results,
                    bar,
                    make_fetch_one=make_fetch_one,
                    request_body_for=request_body_for,
                    path_for_cell=path_for_cell,
                    would_skip=would_skip,
                    transform=transform,
                    log=log,
                    refetch=refetch,
                    request_range=request_range,
                    week_range=week_range,
                )
    report_path, failures = summarize(
        results, report_prefix=report_prefix, command="backfill", extra=extra
    )
    if failures:
        raise click.ClickException(
            f"{len(failures)} of {len(results)} backfill calls failed — see {report_path}"
        )


def _backfill_week(
    specs,
    season,
    week,
    prev_week_key,
    results,
    bar,
    *,
    make_fetch_one,
    request_body_for,
    path_for_cell,
    would_skip,
    transform,
    log,
    refetch,
    request_range,
    week_range,
):
    """Fetch every spec for one (season, week); return the updated prev_week_key.

    The pacing pause fires only for specs that aren't skipped, so a fully-cached
    week resumes near-instantly. Appends each outcome to ``results`` and ticks ``bar``.
    """
    week_key = (season, week)
    for spec in specs:
        if not would_skip(spec, season, week):
            _backfill_pause(
                prev_week_key,
                week_key,
                request_range=request_range,
                week_range=week_range,
                log=log,
            )
            prev_week_key = week_key
        results.append(
            fetch_and_write_one(
                spec,
                path_for=lambda s: path_for_cell(s, season, week),
                fetch_one=lambda s: make_fetch_one(s, season, week),
                transform=transform,
                log=log,
                season=season,
                week=week,
                request_body=request_body_for(spec, season, week),
                refetch=refetch,
            )
        )
        bar.update(1)
    return prev_week_key


def _backfill_pause(
    prev_week_key: tuple[int, int] | None,
    next_week_key: tuple[int, int],
    *,
    request_range: tuple[float, float],
    week_range: tuple[float, float],
    log: logging.Logger,
) -> None:
    """Sleep before the next backfill call: long pause on week change, short otherwise.

    No-op on the very first iteration (``prev_week_key is None``) — the client
    itself adds no internal pause in backfill mode, so the first call goes
    through immediately.
    """
    if prev_week_key is None:
        return
    if next_week_key != prev_week_key:
        pause = random.uniform(*week_range)
        season, week = next_week_key
        click.echo(
            f"\n[pausing {pause:.0f}s before {season} week {week:02d}]",
            err=True,
        )
        log.info(
            "week transition pause",
            extra={"pause_s": round(pause, 1), "next_season": season, "next_week": week},
        )
    else:
        pause = random.uniform(*request_range)
        log.debug("request pause", extra={"pause_s": round(pause, 1)})
    time.sleep(pause)
