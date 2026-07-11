"""Shared click option constants and small catalog helpers for command builders.

Kept free of any :class:`~sportstradamus.collectors.cli.Source` dependency so
both :mod:`collectors.cli` and :mod:`collectors.commands` can import it without
an import cycle.
"""

from __future__ import annotations

from pathlib import Path

import click

from sportstradamus.collectors.catalog import EndpointSpec, load_catalog

# Number of leading characters to echo when previewing a token value —
# enough to confirm it changed without leaking the full secret.
TOKEN_PREVIEW_CHARS = 24

# Backfill pacing defaults — overnight job, more conservative than ``run``'s
# 2 s default. Randomised inside each range to avoid a regular drumbeat.
# Between endpoints in the same week: 2-8 s. Between weeks: 8-28 s.
BACKFILL_REQUEST_PAUSE_S: tuple[float, float] = (2.0, 8.0)
BACKFILL_WEEK_PAUSE_S: tuple[float, float] = (8.0, 28.0)

CATALOG_OPTION = click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override catalog path (default: bundled config).",
)
LOG_LEVEL_OPTION = click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
)


def mode_option(modes: tuple[str, ...] | None, default_mode: str | None):
    """A ``--mode`` decorator when the source has modes, else a no-op decorator."""
    if not modes:
        return lambda f: f
    return click.option(
        "--mode",
        type=click.Choice(list(modes)),
        default=default_mode,
        show_default=True,
        help="Aggregation mode for this fetch.",
    )


def load_or_empty(
    catalog_path: Path | None, default_catalog: Path, source_name: str
) -> list[EndpointSpec] | None:
    """Load the catalog; echo a hint and return ``None`` when it is empty."""
    specs = load_catalog(catalog_path or default_catalog)
    if not specs:
        click.echo(
            f"Catalog is empty. Register endpoints first (see {source_name} docs).",
            err=True,
        )
        return None
    return specs
