#!/usr/bin/env python3
"""PreToolUse[Task] hook: snapshot HEAD/branch/dirty state before a subagent
runs, for posttask-diff.py to compare against."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys


def project_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=project_dir()
    ).stdout.strip()


def main() -> None:
    payload = json.load(sys.stdin)
    session_id = payload.get("session_id", "unknown")
    state_dir = project_dir() / ".claude" / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status": git("status", "--porcelain"),
    }
    (state_dir / f"pretask-{session_id}.json").write_text(json.dumps(snapshot))


if __name__ == "__main__":
    main()
