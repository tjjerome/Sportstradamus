"""``fp-fetch`` CLI — snapshot every registered Fantasy Points tool.

The standard ``run`` / ``backfill`` / ``verify`` / ``list`` / ``refresh-auth``
subcommands come from the generic collector builder
(:func:`sportstradamus.collectors.cli.build_source_cli`) wired to
:data:`FP_SOURCE`. One FP-only command is attached here:

* ``fp-fetch import-curl`` — register an endpoint from a DevTools curl.

See ``docs/fantasypoints.md`` for the end-user runbook (capturing a fresh
cookie, adding endpoints, refresh schedule).
"""

from __future__ import annotations

from sportstradamus.collectors.cli import build_source_cli
from sportstradamus.collectors.fantasypoints.import_curl import import_curl
from sportstradamus.collectors.fantasypoints.source import FP_SOURCE

fp_fetch = build_source_cli(FP_SOURCE)

fp_fetch.add_command(import_curl)
