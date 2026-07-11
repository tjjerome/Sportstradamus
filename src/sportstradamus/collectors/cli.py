"""Assemble a click command group for one collector source.

:func:`build_source_cli` turns a :class:`Source` — plain data plus the
source-specific closures — into a ``click.Group`` with the standard ``run`` /
``list`` / ``verify`` subcommands (plus ``backfill`` and ``refresh-auth`` when
the source supports them). The per-command builders live in
:mod:`collectors.commands`; source-only commands (e.g. a ``discover`` /
``import-curl``) are attached by the source after building.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from sportstradamus.collectors import commands, commands_dated
from sportstradamus.collectors.auth import AuthFields, ResolvedAuth, read_auth


@dataclass
class Source:
    """Everything :func:`build_source_cli` needs to wire one source.

    Callables receive the resolved CLI context (``season`` / ``week`` /
    ``mode`` / ``use_cache``) as keyword arguments; each closure reads the
    ones it needs.
    """

    name: str
    help: str
    catalog_path: Path
    make_client: Callable[[ResolvedAuth, float | None], Any]
    default_context: Callable[[int | None, int | None], dict[str, Any]]
    path_for: Callable[..., Path]
    dispatch: Callable[..., Any]
    transform: Callable[[Any], Any]
    report_prefix: str
    # Week-source-only (Fantasy Points): the request-body renderer for the run
    # report, and the parquet spot-check verifier. Date sources leave these
    # ``None`` — their run has no request body and their ``verify`` is generic.
    render_request_body: Callable[..., Any] | None = None
    verify_fn: Callable[..., dict[str, list]] | None = None
    auth_fields: AuthFields | None = None
    env_prefix: str | None = None
    modes: tuple[str, ...] | None = None
    default_mode: str | None = None
    has_backfill: bool = False
    backfill_end_week: int | None = None
    # "week" = NFL-style integer week + modes (Fantasy Points). "date" =
    # cumulative season-to-date sources stamped by capture date (Cleaning the
    # Glass, Baseball Savant); these skip --week / --mode / --backfill.
    period_kind: str = "week"

    def client(self, *, inter_request_sleep_s: float | None = None) -> Any:
        resolved = (
            read_auth(self.auth_fields, env_prefix=self.env_prefix)
            if self.auth_fields is not None
            else ResolvedAuth(None, None, None)
        )
        return self.make_client(resolved, inter_request_sleep_s)


def build_source_cli(source: Source) -> click.Group:
    """Return a ``click.Group`` wiring ``source`` to the generic runner."""

    @click.group(name=source.name, help=source.help)
    def group() -> None:
        pass

    if source.period_kind == "date":
        group.add_command(commands_dated.build_run_dated(source), "run")
        group.add_command(commands_dated.build_verify_dated(source), "verify")
    else:
        group.add_command(commands.build_run(source), "run")
        group.add_command(commands.build_verify(source), "verify")
        if source.has_backfill:
            group.add_command(commands.build_backfill(source), "backfill")
    group.add_command(commands.build_list(source), "list")
    if source.auth_fields is not None:
        group.add_command(commands.build_refresh_auth(source), "refresh-auth")
    return group
