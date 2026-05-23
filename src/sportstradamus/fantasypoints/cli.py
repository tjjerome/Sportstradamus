"""``fp-fetch`` CLI — snapshot every registered Fantasy Points tool.

Three subcommands:

* ``fp-fetch run`` — walks the catalog and writes per-tool snapshots.
* ``fp-fetch list`` — prints the registered catalog entries.
* ``fp-fetch import-curl`` — registers a new endpoint from a
  DevTools-copied curl command. Handles both GET and POST curls,
  including ``--data-raw`` JSON bodies.

The runner resolves ``--season`` / ``--week`` (defaults inferred from
the current date) and substitutes them into each endpoint's ``params``
and ``json_body`` templates before calling :class:`FantasyPointsClient`.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import logging
import re
import shlex
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import click
from tqdm import tqdm

from sportstradamus import data
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
from sportstradamus.helpers.logging import get_logger

# Snapshots live alongside the rest of the runtime data tree and are
# gitignored. Co-located with helpers/io.py's _RUNTIME_DIR convention.
OUTPUT_BASE = Path(str(pkg_resources.files(data) / "fantasypoints"))

# NFL regular season is 18 weeks. The default-week inference clamps to
# this range; the user can always pass --week explicitly.
NFL_REGULAR_SEASON_WEEKS = 18

# A "NFL season" labelled by year Y starts in September of year Y.
# Before July we're still in the prior year's playoff/offseason tail.
_SEASON_FLIP_MONTH = 7

# NFL Week 1 starts Thursday after Labor Day; a Tuesday cutoff is a
# close-enough boundary for the default. Tuesdays have weekday() == 1.
_TUESDAY = 1

# Headers that should never be persisted in the catalog — Authorization
# and Cookie come from creds/keys.json, host/connection/length are
# per-request, UA is set by the client.
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
    }
)

# Curl option tokens whose next argument is a value we want to capture
# rather than skip outright.
_CURL_VALUE_FLAGS = frozenset(
    {
        "-H",
        "--header",
        "-X",
        "--request",
        "-d",
        "--data",
        "--data-raw",
        "--data-binary",
        "--data-ascii",
        "-b",
        "--cookie",
        "-A",
        "--user-agent",
        "-e",
        "--referer",
    }
)

# Curl option tokens that take a value but whose value we never need.
_CURL_SKIP_VALUE_FLAGS = frozenset({"-b", "--cookie", "-A", "--user-agent", "-e", "--referer"})

# Backslash-newline continuation in shell here-doc style curls.
_LINE_CONTINUATION_RE = re.compile(r"\\\s*\r?\n")


@click.group()
def fp_fetch():
    """Snapshot Fantasy Points Data Suite tools to disk."""


@fp_fetch.command("run")
@click.option("--season", type=int, default=None, help="NFL season (year). Defaults to current.")
@click.option("--week", type=int, default=None, help="NFL week (1-18). Defaults to inferred.")
@click.option("--only", multiple=True, help="Only fetch these endpoint names (repeatable).")
@click.option("--dry-run", is_flag=True, help="Print resolved URLs without fetching.")
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override catalog path (default: bundled config).",
)
@click.option(
    "--output",
    "output_base",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Override output base directory.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
)
def run(season, week, only, dry_run, catalog_path, output_base, log_level) -> None:
    """Walk the catalog and write per-tool snapshots for the given week."""
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
            "Catalog is empty. Register endpoints with `fp-fetch import-curl` "
            "(see docs/fantasypoints.md).",
            err=True,
        )
        return
    if only:
        specs = _filter_by_name(specs, only)
    out_base = output_base or OUTPUT_BASE
    if dry_run:
        for spec in specs:
            url = _build_url(spec.url, spec.render_params(season=season, week=week))
            path = spec.output_path(base=out_base, season=season, week=week)
            click.echo(f"{spec.name}: {spec.method} {url} -> {path}")
        return
    client = FantasyPointsClient()
    failures = _fetch_all(specs, client, season=season, week=week, base=out_base, log=log)
    if failures:
        raise click.ClickException(f"Failed to fetch: {failures}")


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
    json_body = _decode_body(parsed["body"])
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


def _decode_body(body: str | None):
    """Parse a curl body string as JSON. Return ``None`` if not JSON."""
    if body is None:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _filter_by_name(specs: list[EndpointSpec], names: tuple[str, ...]) -> list[EndpointSpec]:
    wanted = set(names)
    filtered = [s for s in specs if s.name in wanted]
    missing = wanted - {s.name for s in filtered}
    if missing:
        raise click.ClickException(f"Unknown endpoint name(s): {sorted(missing)}")
    return filtered


def _fetch_all(
    specs: list[EndpointSpec],
    client: FantasyPointsClient,
    *,
    season: int,
    week: int,
    base: Path,
    log: logging.Logger,
) -> list[str]:
    failures: list[str] = []
    for spec in tqdm(specs, desc="fp-fetch", unit="endpoint"):
        target = spec.output_path(base=base, season=season, week=week)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            body = _dispatch(client, spec, season=season, week=week)
        except FantasyPointsAuthError as exc:
            log.error("auth failed", extra={"endpoint": spec.name, "error": str(exc)})
            click.echo(str(exc), err=True)
            raise click.ClickException(
                "Authorization token expired. Refresh creds/keys.json and rerun."
            ) from exc
        except Exception as exc:
            log.error("fetch failed", extra={"endpoint": spec.name, "error": str(exc)})
            failures.append(spec.name)
            continue
        _write_payload(target, body, spec.response_format)
        log.info("wrote snapshot", extra={"endpoint": spec.name, "path": str(target)})
    return failures


def _dispatch(
    client: FantasyPointsClient,
    spec: EndpointSpec,
    *,
    season: int,
    week: int,
) -> dict | list | str | bytes:
    """Call the right verb on the client for one endpoint spec."""
    params = spec.render_params(season=season, week=week)
    accept = _client_accept(spec.response_format)
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


def _client_accept(response_format: str) -> str:
    if response_format == "json":
        return "json"
    if response_format in ("csv", "html"):
        return "text"
    return "bytes"


def _build_url(url: str, params: dict[str, str]) -> str:
    parsed = urlsplit(url)
    query = urlencode(params) if params else ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def _write_payload(path: Path, body: dict | list | str | bytes, fmt: str) -> None:
    if fmt == "json":
        with path.open("w") as f:
            json.dump(body, f, indent=2, default=str)
        return
    if fmt in ("csv", "html"):
        text = body if isinstance(body, str) else body.decode()
        with path.open("w") as f:
            f.write(text)
        return
    raw = body if isinstance(body, bytes) else body.encode()
    with path.open("wb") as f:
        f.write(raw)


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
