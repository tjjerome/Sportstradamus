"""``fp-fetch`` CLI — snapshot every registered Fantasy Points tool.

Six subcommands:

* ``fp-fetch run`` — fetch all (or ``--only``) catalog endpoints for a
  given week/season, parse the response tables, write one parquet per
  tool under ``data/{player,team}_data/NFL/{season}/``. Defaults to
  the most recently completed week. On a 401 mid-batch in an
  interactive terminal, prompts for a fresh curl and resumes.
* ``fp-fetch backfill`` — iterate (season × week) pairs and run the
  same fetch+parse+write for each cell. One-time historical grab.
* ``fp-fetch discover`` — auto-populate the catalog from FP's tool
  registry.
* ``fp-fetch list`` — prints the registered catalog entries.
* ``fp-fetch import-curl`` — registers a new endpoint from a
  DevTools-copied curl command. Handles both GET and POST curls,
  including ``--data-raw`` JSON bodies.
* ``fp-fetch refresh-auth`` — updates the Authorization / Cookie /
  User-Agent fields in ``creds/keys.json`` from a fresh curl command
  (or stdin), so token rotation is a one-paste operation.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import logging
import random
import re
import shlex
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import click
from tqdm import tqdm

from sportstradamus import creds
from sportstradamus.fantasypoints.catalog import (
    CATALOG_PATH,
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.fantasypoints.client import (
    FantasyPointsAuthError,
    FantasyPointsClient,
)
from sportstradamus.fantasypoints.discover import (
    REGISTRY_BODY,
    REGISTRY_URL,
    expand_registry,
)
from sportstradamus.fantasypoints.transform import (
    parquet_path_for_spec,
    parse_table_response,
    write_parquet,
)
from sportstradamus.helpers.logging import get_logger

# NFL regular season is 18 weeks. The default-week inference clamps to
# this range; the user can always pass --week explicitly.
NFL_REGULAR_SEASON_WEEKS = 18

# Backfill pacing — overnight job, much more conservative than `run`'s
# 2 s default. Randomised inside each range to avoid a regular drumbeat.
# Between endpoints in the same week: 10–20 s. Between weeks (a longer
# pause that mimics a human switching tools and scrolling): 30–90 s.
_BACKFILL_REQUEST_PAUSE_S: tuple[float, float] = (10.0, 20.0)
_BACKFILL_WEEK_PAUSE_S: tuple[float, float] = (30.0, 90.0)

# A "NFL season" labelled by year Y starts in September of year Y.
# Before July we're still in the prior year's playoff/offseason tail.
_SEASON_FLIP_MONTH = 7

# NFL Week 1 starts Thursday after Labor Day; a Tuesday cutoff is a
# close-enough boundary for the default. Tuesdays have weekday() == 1.
_TUESDAY = 1

# Headers that should never be persisted in the catalog — Authorization
# and Cookie come from creds/keys.json, host/connection/length are
# per-request, UA is set by the client, accept-encoding is set by
# requests itself (we can only decompress gzip + deflate without extra
# packages; letting the catalog ask for br/zstd produces undecoded
# binary bodies that fail json parsing).
_STRIPPED_CURL_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "user-agent",
        "authority",
        ":authority",
        "host",
        "connection",
        "content-length",
        "alt-used",
        "te",
        "accept-encoding",
    }
)

# Curl option tokens that take a value but whose value we never need.
_CURL_SKIP_VALUE_FLAGS = frozenset({"-b", "--cookie", "-A", "--user-agent", "-e", "--referer"})

# Backslash-newline continuation in shell here-doc style curls.
_LINE_CONTINUATION_RE = re.compile(r"\\\s*\r?\n")

# Mapping from curl header name (lowercase) to the keys.json field
# that `refresh-auth` writes when it sees that header.
_AUTH_HEADER_TO_KEY = {
    "authorization": "fantasypoints_authorization",
    "cookie": "fantasypoints_cookie",
    "user-agent": "fantasypoints_user_agent",
}

# Number of leading characters to echo when previewing a token value —
# enough to confirm it changed without leaking the full secret.
_TOKEN_PREVIEW_CHARS = 24

# One initial attempt plus one retry after an interactive auth-refresh.
# Cron runs never see the second attempt (stdin is not a TTY), so the
# effective retry count in production is 1.
_AUTH_MAX_ATTEMPTS = 2


@click.group()
def fp_fetch():
    """Snapshot Fantasy Points Data Suite tools to disk."""


@fp_fetch.command("run")
@click.option("--season", type=int, default=None, help="NFL season (year). Defaults to current.")
@click.option("--week", type=int, default=None, help="NFL week (1-18). Defaults to inferred.")
@click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
@click.option("--dry-run", is_flag=True, help="Print resolved URLs and parquet paths only.")
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override catalog path (default: bundled config).",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
)
def run(season, week, only, dry_run, catalog_path, log_level) -> None:
    """Walk the catalog, fetch every endpoint, write one parquet per tool.

    Player-context entries land in
    ``data/player_data/NFL/{season}/<tool>_week_NN.parquet``; team /
    opponent entries land in ``data/team_data/NFL/{season}/`` (with
    ``_opp`` suffix for opponent context). Re-runs overwrite.
    """
    log = get_logger("fp-fetch")
    log.setLevel(log_level)
    season = season or _default_season()
    week = week or _default_week(season)
    log.info(
        "fp-fetch run",
        extra={"season": season, "week": week, "dry_run": dry_run},
    )
    specs = load_catalog(catalog_path)
    if not specs:
        click.echo(
            "Catalog is empty. Run `fp-fetch discover` or "
            "`fp-fetch import-curl` first (see docs/fantasypoints.md).",
            err=True,
        )
        return
    if only:
        specs = _filter_by_name(specs, only)
    if dry_run:
        for spec in specs:
            url = _build_url(spec.url, spec.render_params(season=season, week=week))
            try:
                path = parquet_path_for_spec(spec, season=season, week=week)
            except ValueError as exc:
                path = f"<unrouted: {exc}>"
            click.echo(f"{spec.name}: {spec.method} {url} -> {path}")
        return
    client = FantasyPointsClient()
    results: list[_RunResult] = []
    for spec in tqdm(specs, desc="fp-fetch", unit="endpoint"):
        results.append(_fetch_and_write_one(spec, client, season=season, week=week, log=log))
    report_path = _write_run_report(
        results,
        command="run",
        extra={"season": season, "week": week},
    )
    failures = [r for r in results if r.status in (_RESULT_FETCH_FAILED, _RESULT_ROUTING_FAILED)]
    click.echo("", err=True)
    click.echo(f"Report: {report_path}", err=True)
    click.echo(
        f"Summary: {sum(1 for r in results if r.status == _RESULT_OK)} ok, "
        f"{sum(1 for r in results if r.status == _RESULT_EMPTY)} empty, "
        f"{len(failures)} failed.",
        err=True,
    )
    if failures:
        raise click.ClickException(f"{len(failures)} spec(s) failed — see report at {report_path}")


@fp_fetch.command("backfill")
@click.option("--start-season", type=int, required=True, help="First season to fetch (inclusive).")
@click.option("--end-season", type=int, required=True, help="Last season to fetch (inclusive).")
@click.option(
    "--start-week",
    type=int,
    default=1,
    show_default=True,
    help="First week of each season (1-18).",
)
@click.option(
    "--end-week",
    type=int,
    default=NFL_REGULAR_SEASON_WEEKS,
    show_default=True,
    help="Last week of each season (1-18).",
)
@click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the total call count and exit without fetching.",
)
@click.option(
    "--request-pause-min",
    type=float,
    default=_BACKFILL_REQUEST_PAUSE_S[0],
    show_default=True,
    help="Min seconds (random uniform) between endpoints in the same week.",
)
@click.option(
    "--request-pause-max",
    type=float,
    default=_BACKFILL_REQUEST_PAUSE_S[1],
    show_default=True,
    help="Max seconds (random uniform) between endpoints in the same week.",
)
@click.option(
    "--week-pause-min",
    type=float,
    default=_BACKFILL_WEEK_PAUSE_S[0],
    show_default=True,
    help="Min seconds (random uniform) when transitioning to a new week.",
)
@click.option(
    "--week-pause-max",
    type=float,
    default=_BACKFILL_WEEK_PAUSE_S[1],
    show_default=True,
    help="Max seconds (random uniform) when transitioning to a new week.",
)
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
)
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
    catalog_path,
    log_level,
) -> None:
    """Iterate (season × week) and write per-tool parquets for each.

    Designed for an overnight one-time grab. Pacing is conservative
    by default: 10-20 s random pause between endpoints in the same
    week, 30-90 s when transitioning to a new week. Override via the
    ``--request-pause-{min,max}`` / ``--week-pause-{min,max}`` flags
    if you need slower (or faster) traffic.

    With ~45 catalog entries × 18 weeks × N seasons at the defaults,
    plan for roughly ~3 hours per season.
    """
    log = get_logger("fp-fetch")
    log.setLevel(log_level)
    specs = load_catalog(catalog_path)
    if not specs:
        click.echo("Catalog is empty. Run `fp-fetch discover` first.", err=True)
        return
    if only:
        specs = _filter_by_name(specs, only)
    seasons = list(range(start_season, end_season + 1))
    weeks = list(range(start_week, end_week + 1))
    total = len(seasons) * len(weeks) * len(specs)
    if dry_run:
        click.echo(
            f"Would fetch {total} ({len(specs)} tools x "
            f"{len(weeks)} weeks x {len(seasons)} seasons)."
        )
        return
    # Drive pacing from this orchestrator instead of the client so the
    # week-transition pause is visible alongside the per-spec pause.
    client = FantasyPointsClient(inter_request_sleep_s=0)
    results: list[_RunResult] = []
    request_range = (request_pause_min, request_pause_max)
    week_range = (week_pause_min, week_pause_max)
    prev_week_key: tuple[int, int] | None = None
    with tqdm(total=total, desc="fp-backfill", unit="call") as bar:
        for season in seasons:
            for week in weeks:
                week_key = (season, week)
                for spec in specs:
                    _backfill_pause(
                        prev_week_key,
                        week_key,
                        request_range=request_range,
                        week_range=week_range,
                        log=log,
                    )
                    results.append(
                        _fetch_and_write_one(spec, client, season=season, week=week, log=log)
                    )
                    prev_week_key = week_key
                    bar.update(1)
    report_path = _write_run_report(
        results,
        command="backfill",
        extra={"seasons": list(seasons), "weeks": list(weeks)},
    )
    failures = [r for r in results if r.status in (_RESULT_FETCH_FAILED, _RESULT_ROUTING_FAILED)]
    click.echo("", err=True)
    click.echo(f"Report: {report_path}", err=True)
    click.echo(
        f"Summary: {sum(1 for r in results if r.status == _RESULT_OK)} ok, "
        f"{sum(1 for r in results if r.status == _RESULT_EMPTY)} empty, "
        f"{len(failures)} failed.",
        err=True,
    )
    if failures:
        raise click.ClickException(
            f"{len(failures)} of {len(results)} backfill calls failed — see {report_path}"
        )


@fp_fetch.command("list")
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
def list_endpoints(catalog_path) -> None:
    """Print the registered endpoint catalog."""
    specs = load_catalog(catalog_path)
    if not specs:
        click.echo("(empty catalog)")
        return
    for spec in specs:
        cadence = "weekly" if spec.weekly else "season"
        click.echo(f"{spec.name:30s} {spec.method:5s} {cadence:7s} {spec.url}")


@fp_fetch.command("discover")
@click.option("--league", default="nfl", show_default=True, help="League segment in tool URLs.")
@click.option(
    "--include-private",
    is_flag=True,
    help="Include tools flagged isPrivate=true (debug/VIP tables).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print discovered specs without writing to the catalog.",
)
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
def discover(league, include_private, dry_run, catalog_path) -> None:
    """Auto-populate the catalog from Fantasy Points' tool registry.

    Hits ``POST /v2/ds/all/tools``, walks the published-tool list,
    and adds one catalog entry per (tool, context) pair. Existing
    entries are preserved — re-run safely when FP adds new tools.
    """
    client = FantasyPointsClient()
    registry = client.post(REGISTRY_URL, json_body=REGISTRY_BODY)
    existing = load_catalog(catalog_path)
    existing_names = {s.name for s in existing}
    new_specs = expand_registry(
        registry,
        league=league,
        existing_names=existing_names,
        include_private=include_private,
    )
    if not new_specs:
        click.echo(f"No new endpoints (catalog already has {len(existing)} entries).")
        return
    if dry_run:
        for spec in new_specs:
            click.echo(f"  + {spec.name:40s} {spec.method:5s} {spec.url}")
        click.echo(f"Would add {len(new_specs)} new endpoints.")
        return
    save_catalog(existing + new_specs, catalog_path)
    click.echo(
        f"Added {len(new_specs)} new endpoints "
        f"(catalog now has {len(existing) + len(new_specs)} entries)."
    )


@fp_fetch.command("import-curl")
@click.argument(
    "curl_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--name", required=True, help="Catalog name for this endpoint.")
@click.option("--output-subdir", required=True, help="Relative output path (no extension).")
@click.option(
    "--response-format",
    type=click.Choice(["json", "csv", "html"]),
    default="json",
)
@click.option("--weekly/--season-long", default=True)
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
def import_curl(curl_file, name, output_subdir, response_format, weekly, catalog_path) -> None:
    """Register a new endpoint from a DevTools-copied curl command."""
    spec = parse_curl_to_spec(
        curl_file.read_text(),
        name=name,
        output_subdir=output_subdir,
        response_format=response_format,
        weekly=weekly,
    )
    existing = load_catalog(catalog_path)
    if any(s.name == spec.name for s in existing):
        raise click.ClickException(f"Endpoint named {spec.name!r} already exists.")
    existing.append(spec)
    save_catalog(existing, catalog_path)
    body_note = " + body" if spec.json_body else ""
    click.echo(f"Registered {spec.name} -> {spec.method} {spec.url}{body_note}")


@fp_fetch.command("refresh-auth")
@click.argument("curl_input", type=click.File("r"))
@click.option(
    "--keys-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override keys.json path (default: bundled creds/keys.json).",
)
def refresh_auth(curl_input, keys_path) -> None:
    """Update auth fields in creds/keys.json from a fresh curl command.

    Capture any logged-in XHR in DevTools (Network → right-click →
    Copy as cURL), save to a file, then run::

        poetry run fp-fetch refresh-auth /tmp/fresh.curl

    Or pipe from clipboard::

        pbpaste | poetry run fp-fetch refresh-auth -

    The command extracts the ``Authorization`` (and optionally
    ``Cookie`` / ``User-Agent``) header values and writes them to
    ``creds/keys.json``, preserving every other field.
    """
    extracted = _extract_auth_from_curl(curl_input.read())
    if not extracted:
        raise click.ClickException(
            "No Authorization / Cookie / User-Agent headers found in input. "
            "Did you paste a curl command from DevTools?"
        )
    path = _update_keys(extracted, path=keys_path)
    for key, value in extracted.items():
        preview = value[:_TOKEN_PREVIEW_CHARS] + (
            "..." if len(value) > _TOKEN_PREVIEW_CHARS else ""
        )
        click.echo(f"Updated {key} ({len(value)} chars): {preview}")
    click.echo(f"Wrote {path}")


def _extract_auth_from_curl(curl_text: str) -> dict[str, str]:
    """Pull auth-related header values out of a DevTools-copied curl.

    Args:
        curl_text: Raw text of the curl command.

    Returns:
        A dict keyed by ``creds/keys.json`` field name
        (``fantasypoints_authorization``, ``fantasypoints_cookie``,
        ``fantasypoints_user_agent``) for whichever headers were
        present. Empty dict if none were found.
    """
    cleaned = _LINE_CONTINUATION_RE.sub(" ", curl_text.strip())
    tokens = shlex.split(cleaned)
    if not tokens or tokens[0] != "curl":
        raise click.ClickException("Input does not look like a curl command.")
    parsed = _parse_curl_tokens(tokens[1:])
    out: dict[str, str] = {}
    for key, value in parsed["headers"].items():
        target = _AUTH_HEADER_TO_KEY.get(key.lower())
        if target and value:
            out[target] = value
    return out


def _update_keys(updates: dict[str, str], *, path: Path | None = None) -> Path:
    """Merge ``updates`` into ``creds/keys.json`` atomically.

    Preserves every other field; creates the file if absent.

    Args:
        updates: Dict of keys.json fields to overwrite.
        path: Override target path (default: bundled creds/keys.json).

    Returns:
        The path that was written.
    """
    target = path or Path(str(pkg_resources.files(creds) / "keys.json"))
    existing: dict[str, str] = {}
    if target.is_file():
        with target.open() as f:
            existing = json.load(f)
    existing.update(updates)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    tmp.replace(target)
    return target


def parse_curl_to_spec(
    curl_text: str,
    *,
    name: str,
    output_subdir: str,
    response_format: str = "json",
    weekly: bool = True,
) -> EndpointSpec:
    """Parse a ``curl '...' ...`` invocation into an :class:`EndpointSpec`.

    Accepts the format produced by Chromium / Firefox DevTools'
    "Copy as cURL (bash)" action. Handles both GET (default) and POST
    (``-X POST`` + ``--data-raw '{...}'``) calls. ``Authorization``,
    ``Cookie``, and ``User-Agent`` headers are stripped (those come
    from ``creds/keys.json``); the URL query string is split out into
    ``params``; the POST body is parsed as JSON into ``json_body``.

    Args:
        curl_text: Raw text of the curl command.
        name: Catalog name to register.
        output_subdir: Relative output path, without extension.
        response_format: ``json``, ``csv``, or ``html``.
        weekly: Whether this endpoint refreshes weekly.

    Returns:
        A new :class:`EndpointSpec` ready to append to the catalog.
    """
    cleaned = _LINE_CONTINUATION_RE.sub(" ", curl_text.strip())
    tokens = shlex.split(cleaned)
    if not tokens or tokens[0] != "curl":
        raise ValueError("Input does not look like a curl command.")
    parsed = _parse_curl_tokens(tokens[1:])
    if parsed["url"] is None:
        raise ValueError("No URL found in curl command.")
    filtered_headers = {
        k: v for k, v in parsed["headers"].items() if k.lower() not in _STRIPPED_CURL_HEADERS
    }
    url_parts = urlsplit(parsed["url"])
    params = dict(parse_qsl(url_parts.query, keep_blank_values=True))
    bare_url = urlunsplit((url_parts.scheme, url_parts.netloc, url_parts.path, "", ""))
    raw_body = parsed["body"]
    try:
        json_body = json.loads(raw_body) if raw_body is not None else None
    except json.JSONDecodeError:
        json_body = None
    method = (parsed["method"] or ("POST" if json_body is not None else "GET")).upper()
    return EndpointSpec(
        name=name,
        url=bare_url,
        method=method,
        params=params or None,
        json_body=json_body,
        extra_headers=filtered_headers or None,
        response_format=response_format,
        output_subdir=output_subdir,
        weekly=weekly,
    )


def _parse_curl_tokens(tokens: list[str]) -> dict:
    """Walk curl arg tokens; collect URL, method, headers, body."""
    url: str | None = None
    method: str | None = None
    body: str | None = None
    headers: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-H", "--header"):
            i += 1
            if i < len(tokens):
                key, _, value = tokens[i].partition(":")
                headers[key.strip()] = value.strip()
        elif tok in ("-X", "--request"):
            i += 1
            if i < len(tokens):
                method = tokens[i]
        elif tok in ("-d", "--data", "--data-raw", "--data-binary", "--data-ascii"):
            i += 1
            if i < len(tokens):
                body = tokens[i]
        elif tok in _CURL_SKIP_VALUE_FLAGS:
            i += 1  # Value irrelevant — UA/cookie come from keys.json.
        elif tok.startswith("-"):
            pass  # Flag without value (--compressed, --location, ...).
        elif url is None:
            url = tok
        i += 1
    return {"url": url, "method": method, "headers": headers, "body": body}


def _filter_by_name(specs: list[EndpointSpec], names: tuple[str, ...]) -> list[EndpointSpec]:
    wanted = set(names)
    filtered = [s for s in specs if s.name in wanted]
    missing = wanted - {s.name for s in filtered}
    if missing:
        raise click.ClickException(f"Unknown endpoint name(s): {sorted(missing)}")
    return filtered


# Status values for `_RunResult.status` — also the dict keys in the
# report summary. ``ok`` = wrote rows; ``empty`` = wrote a zero-row
# parquet; ``fetch_failed`` = HTTP error / decode error / etc;
# ``routing_failed`` = parquet_path_for_spec couldn't resolve a
# context for the catalog entry.
_RESULT_OK = "ok"
_RESULT_EMPTY = "empty"
_RESULT_FETCH_FAILED = "fetch_failed"
_RESULT_ROUTING_FAILED = "routing_failed"

# Body preview length in the failure report. Long enough to spot
# HTML error pages, JSON error envelopes, or compressed binary blobs
# at a glance; short enough not to inflate the report file when a
# tool returns a huge body that happens to be unparseable.
_REPORT_PREVIEW_CHARS = 500


@dataclass
class _RunResult:
    """One spec's outcome from a single ``run`` / ``backfill`` iteration.

    Serialised verbatim into the report file so the human (or me) has
    everything needed to diagnose failing specs without re-running.
    """

    name: str
    url: str
    method: str
    status: str
    season: int
    week: int
    rows: int = 0
    path: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    response_preview: str | None = None
    request_body: dict | list | None = None


def _write_run_report(
    results: list[_RunResult],
    *,
    command: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Dump the per-spec results to ``/tmp/fp_fetch_report_<ts>.json``.

    The report is overwriteable, transient debug output — it lives
    under tempdir so users (and CI) don't accidentally commit it.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(tempfile.gettempdir()) / f"fp_fetch_report_{timestamp}.json"
    summary = {
        "total": len(results),
        _RESULT_OK: sum(1 for r in results if r.status == _RESULT_OK),
        _RESULT_EMPTY: sum(1 for r in results if r.status == _RESULT_EMPTY),
        _RESULT_FETCH_FAILED: sum(1 for r in results if r.status == _RESULT_FETCH_FAILED),
        _RESULT_ROUTING_FAILED: sum(1 for r in results if r.status == _RESULT_ROUTING_FAILED),
    }
    payload: dict[str, Any] = {
        "command": command,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    # ``extra`` is namespaced under "context" so callers can't silently
    # clobber the top-level "command" / "summary" / "results" fields
    # by passing a conflicting key.
    if extra:
        payload["context"] = extra
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _fetch_and_write_one(
    spec: EndpointSpec,
    client: FantasyPointsClient,
    *,
    season: int,
    week: int,
    log: logging.Logger,
) -> _RunResult:
    """Fetch one endpoint, parse to DataFrame, write parquet, return outcome.

    Any failure (routing, fetch, decode) is captured into the returned
    :class:`_RunResult` so the outer loop can keep going and the
    report can show exactly what broke per spec.
    """
    method = spec.method.upper()
    rendered_url = _build_url(spec.url, spec.render_params(season=season, week=week))
    base_result = _RunResult(
        name=spec.name,
        url=rendered_url,
        method=method,
        season=season,
        week=week,
        status=_RESULT_FETCH_FAILED,
        request_body=spec.render_json_body(season=season, week=week),
    )
    try:
        target = parquet_path_for_spec(spec, season=season, week=week)
    except ValueError as exc:
        log.error("routing failed", extra={"endpoint": spec.name, "error": str(exc)})
        click.echo(f"  {spec.name}: {exc}", err=True)
        base_result.status = _RESULT_ROUTING_FAILED
        base_result.error_class = exc.__class__.__name__
        base_result.error_message = str(exc)
        return base_result
    body, err = _dispatch_capturing_errors(spec, client, season=season, week=week, log=log)
    if err is not None:
        click.echo(f"  {spec.name}: {err['error_message']}", err=True)
        base_result.status = _RESULT_FETCH_FAILED
        base_result.error_class = err.get("error_class")
        base_result.error_message = err.get("error_message")
        base_result.http_status = err.get("http_status")
        base_result.response_preview = err.get("response_preview")
        return base_result
    df = parse_table_response(body)
    write_parquet(df, target)
    base_result.path = str(target)
    base_result.rows = len(df)
    base_result.status = _RESULT_EMPTY if df.empty else _RESULT_OK
    if df.empty:
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


def _backfill_pause(
    prev_week_key: tuple[int, int] | None,
    next_week_key: tuple[int, int],
    *,
    request_range: tuple[float, float],
    week_range: tuple[float, float],
    log: logging.Logger,
) -> None:
    """Sleep before the next backfill call: long pause on week change, short otherwise.

    No-op on the very first iteration (``prev_week_key is None``) — the
    client itself adds no internal pause in backfill mode, so the first
    call goes through immediately.
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


def _dispatch_capturing_errors(
    spec: EndpointSpec,
    client: FantasyPointsClient,
    *,
    season: int,
    week: int,
    log: logging.Logger,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Dispatch with auth-refresh; return ``(body, None)`` or ``(None, err_dict)``.

    On 401/403 in a TTY, prompts for a fresh curl and retries once.
    In non-TTY contexts (cron) the auth failure is returned as an
    error so the caller can record it in the report and continue —
    the calling command surfaces the aggregate failure as a non-zero
    exit so run_job.sh still pings ``/fail``.
    """
    for attempt in range(1, _AUTH_MAX_ATTEMPTS + 1):
        try:
            return _dispatch(client, spec, season=season, week=week), None
        except FantasyPointsAuthError as exc:
            log.warning(
                "auth error",
                extra={"endpoint": spec.name, "attempt": attempt, "error": str(exc)},
            )
            if attempt == _AUTH_MAX_ATTEMPTS or not _refresh_auth_interactively(client):
                return None, _exc_to_err_dict(exc)
        except Exception as exc:
            log.error("fetch failed", extra={"endpoint": spec.name, "error": str(exc)})
            return None, _exc_to_err_dict(exc)
    return None, {"error_class": "Unknown", "error_message": "retries exhausted"}


def _exc_to_err_dict(exc: BaseException) -> dict[str, Any]:
    """Convert an exception into the JSON-friendly fields the report needs."""
    info: dict[str, Any] = {
        "error_class": exc.__class__.__name__,
        "error_message": str(exc),
    }
    response = getattr(exc, "response", None)
    if response is not None:
        info["http_status"] = getattr(response, "status_code", None)
        text = getattr(response, "text", None)
        if isinstance(text, str):
            info["response_preview"] = text[:_REPORT_PREVIEW_CHARS]
    return info


def _refresh_auth_interactively(client: FantasyPointsClient) -> bool:
    """Prompt for a fresh curl on stdin; update keys + client in place.

    Returns ``False`` (without prompting) when stdin is not a TTY, so
    cron runs fail fast instead of hanging on input that will never
    arrive.
    """
    if not sys.stdin.isatty():
        return False
    click.echo("", err=True)
    click.echo("=" * 60, err=True)
    click.echo("Authorization expired.", err=True)
    click.echo("Paste a fresh DevTools curl below, then send EOF (Ctrl+D):", err=True)
    click.echo("=" * 60, err=True)
    try:
        curl_text = sys.stdin.read()
    except KeyboardInterrupt:
        click.echo("Aborted.", err=True)
        return False
    if not curl_text.strip():
        click.echo("Empty input. Aborting.", err=True)
        return False
    try:
        updates = _extract_auth_from_curl(curl_text)
    except click.ClickException as exc:
        click.echo(f"Failed to parse curl: {exc.message}", err=True)
        return False
    if not updates.get("fantasypoints_authorization"):
        click.echo("No Authorization header found in pasted curl.", err=True)
        return False
    _update_keys(updates)
    client.refresh_credentials(
        authorization=updates.get("fantasypoints_authorization"),
        cookie=updates.get("fantasypoints_cookie"),
        user_agent=updates.get("fantasypoints_user_agent"),
    )
    click.echo("Auth refreshed. Resuming...", err=True)
    return True


def _dispatch(
    client: FantasyPointsClient,
    spec: EndpointSpec,
    *,
    season: int,
    week: int,
) -> dict | list | str | bytes:
    """Call the right verb on the client for one endpoint spec."""
    params = spec.render_params(season=season, week=week)
    # Map catalog response_format to the client's accept token.
    if spec.response_format == "json":
        accept = "json"
    elif spec.response_format in ("csv", "html"):
        accept = "text"
    else:
        accept = "bytes"
    method = spec.method.upper()
    if method == "POST":
        return client.post(
            spec.url,
            json_body=spec.render_json_body(season=season, week=week),
            params=params or None,
            headers=spec.extra_headers,
            accept=accept,
        )
    if method == "GET":
        return client.get(
            spec.url,
            params=params or None,
            headers=spec.extra_headers,
            accept=accept,
        )
    raise click.ClickException(f"Unsupported method {method!r} for endpoint {spec.name!r}")


def _build_url(url: str, params: dict[str, str]) -> str:
    parsed = urlsplit(url)
    query = urlencode(params) if params else ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def _default_season() -> int:
    today = date.today()
    return today.year if today.month >= _SEASON_FLIP_MONTH else today.year - 1


def _default_week(season: int) -> int:
    today = date.today()
    season_start = date(season, 9, 1)
    while season_start.weekday() != _TUESDAY:
        season_start += timedelta(days=1)
    delta_days = (today - season_start).days
    return max(1, min(NFL_REGULAR_SEASON_WEEKS, (delta_days // 7) + 1))
