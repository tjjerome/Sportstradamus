"""``fp-fetch`` CLI — snapshot every registered Fantasy Points tool.

The standard ``run`` / ``backfill`` / ``verify`` / ``list`` / ``refresh-auth``
subcommands come from the generic collector builder
(:func:`sportstradamus.collectors.cli.build_source_cli`) wired to
:data:`FP_SOURCE`. Two FP-only commands are attached here:

* ``fp-fetch import-curl`` — register an endpoint from a DevTools curl.
* ``fp-fetch login`` — mint a fresh session cookie from stored credentials.

See ``docs/fantasypoints.md`` for the end-user runbook (capturing a fresh
cookie, adding endpoints, refresh schedule).
"""

from __future__ import annotations

import click

from sportstradamus.collectors.cli import build_source_cli
from sportstradamus.collectors.fantasypoints.import_curl import import_curl
from sportstradamus.collectors.fantasypoints.session import SessionRenewalError, renew_session
from sportstradamus.collectors.fantasypoints.source import FP_SOURCE

fp_fetch = build_source_cli(FP_SOURCE)


@click.command("login")
def login() -> None:
    """Mint a fresh session cookie from the stored credentials.

    ``run`` does this by itself the moment a call comes back 401, so this
    exists to prove the stored email and password work — right after you set
    them, rather than a week later when the cookie lapses mid-cron.
    """
    try:
        cookie = renew_session()
    except SessionRenewalError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Session renewed and written to keys.json ({len(cookie)} chars).")


fp_fetch.add_command(import_curl)
fp_fetch.add_command(login)
