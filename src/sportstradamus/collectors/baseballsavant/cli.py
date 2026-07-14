"""``savant-fetch`` CLI — snapshot Baseball Savant (Statcast) MLB leaderboards.

Subcommands: ``run`` (capture each catalog endpoint's current season-to-date
leaderboard, stamped by today's date), ``verify``, ``list``. Public source — no
auth, no ``refresh-auth``. See ``docs/baseballsavant.md``.
"""

from __future__ import annotations

from sportstradamus.collectors.baseballsavant.source import SAVANT_SOURCE
from sportstradamus.collectors.cli import build_source_cli

savant_fetch = build_source_cli(SAVANT_SOURCE)
