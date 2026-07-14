"""``fp-fetch`` CLI — snapshot every registered Fantasy Points tool.

The standard ``run`` / ``backfill`` / ``verify`` / ``list`` / ``refresh-auth``
subcommands come from the generic collector builder
(:func:`sportstradamus.collectors.cli.build_source_cli`) wired to
:data:`FP_SOURCE`. Two FP-only commands are attached here:

* ``fp-fetch discover`` — auto-populate the catalog from FP's tool registry.
* ``fp-fetch import-curl`` — register an endpoint from a DevTools curl.

See ``docs/fantasypoints.md`` for the end-user runbook (capturing a fresh
cookie, adding endpoints, refresh schedule).
"""

from __future__ import annotations

from pathlib import Path

import click

from sportstradamus.collectors.catalog import load_catalog, save_catalog
from sportstradamus.collectors.cli import build_source_cli
from sportstradamus.collectors.fantasypoints.discover import (
    REGISTRY_BODY,
    REGISTRY_URL,
    expand_registry,
)
from sportstradamus.collectors.fantasypoints.import_curl import import_curl
from sportstradamus.collectors.fantasypoints.source import CATALOG_PATH, FP_SOURCE

fp_fetch = build_source_cli(FP_SOURCE)


@click.command("discover")
@click.option("--league", default="nfl", show_default=True, help="League segment in tool URLs.")
@click.option(
    "--include-private",
    is_flag=True,
    help="Include tools flagged isPrivate=true (debug/VIP tables).",
)
@click.option(
    "--replace",
    is_flag=True,
    help="Discard the existing catalog and rewrite from scratch. Use when the body "
    "schema changes (catalogs from before the v2 ``/values`` rewrite need this).",
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
def discover(league, include_private, replace, dry_run, catalog_path) -> None:
    """Auto-populate the catalog from Fantasy Points' tool registry.

    Hits ``POST /v2/ds/all/tools``, walks the published-tool list, and adds
    one catalog entry per (tool, context) pair. By default existing entries
    are preserved — re-run safely when FP adds new tools. Pass ``--replace``
    to discard the existing catalog first (use this when the body schema or
    URL pattern changes).
    """
    path = catalog_path or CATALOG_PATH
    client = FP_SOURCE.client()
    registry = client.post(REGISTRY_URL, json_body=REGISTRY_BODY)
    existing = [] if replace else load_catalog(path)
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
        verb = "Would replace catalog with" if replace else "Would add"
        click.echo(f"{verb} {len(new_specs)} endpoints.")
        return
    save_catalog(existing + new_specs, path)
    verb = "Wrote" if replace else "Added"
    click.echo(
        f"{verb} {len(new_specs)} endpoints "
        f"(catalog now has {len(existing) + len(new_specs)} entries)."
    )


fp_fetch.add_command(discover)
fp_fetch.add_command(import_curl)
