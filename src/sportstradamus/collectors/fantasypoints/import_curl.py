"""``fp-fetch import-curl`` — register an endpoint from a DevTools curl.

Parses a ``curl '...' ...`` invocation (Chromium / Firefox "Copy as cURL
(bash)") into an :class:`EndpointSpec` and appends or replaces it in the
catalog. Handles GET and POST (``-X POST`` + ``--data-raw '{...}'``); the
curl-token parsing itself is shared with the generic ``refresh-auth`` flow
via :mod:`sportstradamus.collectors.auth`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import click

from sportstradamus.collectors.auth import (
    _STRIPPED_CURL_HEADERS,
    parse_curl_tokens,
    tokenize_curl,
)
from sportstradamus.collectors.catalog import EndpointSpec, load_catalog, save_catalog
from sportstradamus.collectors.fantasypoints.source import CATALOG_PATH

# Sentinel string :mod:`body_substitute` recognises and rewrites to the
# integer season at call time. Mirrors the constant in :mod:`discover` —
# kept duplicated here (rather than imported) so ``import-curl`` doesn't
# depend on the discover module.
_SEASON_INT_SENTINEL = "__SEASON_INT__"


@click.command("import-curl")
@click.argument(
    "curl_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--name", required=True, help="Catalog name for this endpoint.")
@click.option(
    "--output-subdir",
    default=None,
    help="Relative output path (no extension). Required for new entries; "
    "with ``--replace`` it defaults to the existing entry's value.",
)
@click.option(
    "--response-format",
    type=click.Choice(["json", "csv", "html"]),
    default="json",
)
@click.option("--weekly/--season-long", default=True)
@click.option(
    "--replace",
    is_flag=True,
    help="Overwrite an existing entry of this name instead of erroring. "
    "Use to patch a discover-generated body with the SPA's tool-specific "
    "filters (position, qualifier, etc.) from a working DevTools curl.",
)
@click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
def import_curl(
    curl_file, name, output_subdir, response_format, weekly, replace, catalog_path
) -> None:
    """Register a new endpoint from a DevTools-copied curl command.

    With ``--replace``, overwrite an existing catalog entry instead of
    appending a new one — useful when a tool needs per-tool filters
    (position + filterResult qualifier) the SPA injects per tool but
    discover can't infer from the registry. The pasted body's literal
    ``game.season`` value is rewritten to the integer sentinel so the entry
    stays usable across seasons; the ``weeks`` block doesn't need
    sentinelisation because ``body_substitute`` overrides it per call based
    on ``--mode``.
    """
    path = catalog_path or CATALOG_PATH
    existing = load_catalog(path)
    existing_by_name = {s.name: i for i, s in enumerate(existing)}
    if name in existing_by_name and not replace:
        raise click.ClickException(
            f"Endpoint named {name!r} already exists. Pass --replace to overwrite."
        )
    if not replace and name not in existing_by_name and not output_subdir:
        raise click.ClickException("--output-subdir is required when registering a new entry.")
    effective_subdir = output_subdir or existing[existing_by_name[name]].output_subdir
    spec = parse_curl_to_spec(
        curl_file.read_text(),
        name=name,
        output_subdir=effective_subdir,
        response_format=response_format,
        weekly=weekly,
    )
    _sentinelize_season(spec.json_body)
    if name in existing_by_name:
        existing[existing_by_name[name]] = spec
        verb = "Replaced"
    else:
        existing.append(spec)
        verb = "Registered"
    save_catalog(existing, path)
    body_note = " + body" if spec.json_body else ""
    click.echo(f"{verb} {spec.name} -> {spec.method} {spec.url}{body_note}")


def parse_curl_to_spec(
    curl_text: str,
    *,
    name: str,
    output_subdir: str,
    response_format: str = "json",
    weekly: bool = True,
) -> EndpointSpec:
    """Parse a ``curl '...' ...`` invocation into an :class:`EndpointSpec`.

    Accepts the format produced by Chromium / Firefox DevTools' "Copy as
    cURL (bash)" action. Handles both GET (default) and POST (``-X POST`` +
    ``--data-raw '{...}'``) calls. ``Authorization``, ``Cookie``, and
    ``User-Agent`` headers are stripped (those come from ``creds/keys.json``);
    the URL query string is split out into ``params``; the POST body is
    parsed as JSON into ``json_body``.

    Args:
        curl_text: Raw text of the curl command.
        name: Catalog name to register.
        output_subdir: Relative output path, without extension.
        response_format: ``json``, ``csv``, or ``html``.
        weekly: Whether this endpoint refreshes weekly.

    Returns:
        A new :class:`EndpointSpec` ready to append to the catalog.
    """
    parsed = parse_curl_tokens(tokenize_curl(curl_text))
    if parsed["url"] is None:
        raise ValueError("No URL found in curl command.")
    filtered_headers = {
        k: v for k, v in parsed["headers"].items() if k.lower() not in _STRIPPED_CURL_HEADERS
    }
    url_parts = urlsplit(parsed["url"])
    params = dict(parse_qsl(url_parts.query, keep_blank_values=True))
    bare_url = urlunsplit((url_parts.scheme, url_parts.netloc, url_parts.path, "", ""))
    json_body, method = _curl_body_and_method(parsed["body"], parsed["method"])
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


def _curl_body_and_method(raw_body: str | None, declared_method: str | None) -> tuple:
    """Parse the curl body as JSON (None on failure) and resolve the HTTP method.

    Method precedence: an explicit ``-X`` wins; otherwise POST when a body is
    present, else GET. Returns ``(json_body, method)``.
    """
    try:
        json_body = json.loads(raw_body) if raw_body is not None else None
    except json.JSONDecodeError:
        json_body = None
    method = (declared_method or ("POST" if json_body is not None else "GET")).upper()
    return json_body, method


def _sentinelize_season(body: Any) -> None:
    """Rewrite a literal ``context.filterMatch.game.season.eq`` int to the sentinel.

    The curl the user pastes was captured for one specific season; if we
    stored that literal in the catalog the entry would be locked to that
    season forever. Walking just the canonical path keeps the rewrite
    surgical — other ``season`` fields elsewhere in the body (if any) stay
    as the user captured them.
    """
    if not isinstance(body, dict):
        return
    ctx = body.get("context")
    if not isinstance(ctx, dict):
        return
    filter_match = ctx.get("filterMatch")
    if not isinstance(filter_match, dict):
        return
    season_node = filter_match.get("game.season")
    if isinstance(season_node, dict) and isinstance(season_node.get("eq"), int):
        season_node["eq"] = _SEASON_INT_SENTINEL
