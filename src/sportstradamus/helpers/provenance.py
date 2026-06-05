"""Code-version provenance for stamping training artifacts."""

from __future__ import annotations

import subprocess


def git_sha() -> str:
    """Return the short git SHA of the current HEAD, or ``"unknown"`` outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
