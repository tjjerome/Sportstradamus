#!/usr/bin/env python3
"""PostToolUse[Task] hook: warn the main agent if a subagent moved HEAD or
changed files it was not handed (shared-checkout drift)."""
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


def current() -> dict:
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status": git("status", "--porcelain"),
    }


def drift_message(before: dict, after: dict) -> str | None:
    lines: list[str] = []
    if before.get("head") != after.get("head") or before.get("branch") != after.get("branch"):
        lines.append(
            "Subagent moved HEAD: {}@{} -> {}@{}. Recover with: git checkout {}".format(
                before.get("branch"),
                (before.get("head") or "")[:8],
                after.get("branch"),
                (after.get("head") or "")[:8],
                before.get("branch"),
            )
        )
    before_files = {ln[3:] for ln in before.get("status", "").splitlines()}
    after_files = {ln[3:] for ln in after.get("status", "").splitlines()}
    new_files = sorted(after_files - before_files)
    if new_files:
        lines.append(
            "Subagent changed files not dirty before it ran (verify in-scope, "
            "revert with `git checkout HEAD -- <file>` if not): " + ", ".join(new_files)
        )
    return "\n".join(lines) if lines else None


def main() -> None:
    payload = json.load(sys.stdin)
    session_id = payload.get("session_id", "unknown")
    snap_path = project_dir() / ".claude" / ".state" / f"pretask-{session_id}.json"
    if not snap_path.exists():
        return
    before = json.loads(snap_path.read_text())
    message = drift_message(before, current())
    if message:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "[subagent-drift-check] " + message,
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
