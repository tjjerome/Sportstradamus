#!/usr/bin/env python3
"""PreToolUse[Bash] hook: ask to confirm a git push when the integration suite
has not passed since the last .py edit."""
from __future__ import annotations

import json
import os
import pathlib
import sys


def is_git_push(command: str) -> bool:
    parts = command.split()
    return "git" in parts and "push" in parts


def decide(is_push: bool, dirty: float | None, green: float | None) -> str:
    if not is_push or dirty is None:
        return "allow"
    if green is None or dirty > green:
        return "ask"
    return "allow"


def _mtime(path: pathlib.Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def main() -> None:
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")
    state = pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / ".claude" / ".state"
    decision = decide(
        is_git_push(command),
        _mtime(state / "code_dirty"),
        _mtime(state / "integration_green"),
    )
    if decision == "ask":
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                        "permissionDecisionReason": (
                            "Integration suite has not passed since your last code "
                            "edit. Run `poetry run pytest -m integration -n0 && touch "
                            '"$CLAUDE_PROJECT_DIR/.claude/.state/integration_green"` '
                            "before pushing, or confirm to push anyway."
                        ),
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
