"""``ctg-fetch`` CLI — snapshot Cleaning the Glass NBA tables to disk.

Subcommands: ``run`` (capture each catalog endpoint's current season-to-date
table, stamped by today's date), ``verify``, ``list``, ``refresh-auth`` (paste a
fresh logged-in DevTools curl to rotate the session cookie). See
``docs/cleaningtheglass.md``.
"""

from __future__ import annotations

from sportstradamus.collectors.cleaningtheglass.source import CTG_SOURCE
from sportstradamus.collectors.cli import build_source_cli

ctg_fetch = build_source_cli(CTG_SOURCE)
