#!/usr/bin/env python3
"""PostToolUse[Edit|Write] hook: mark the tree code-dirty when a .py changes,
so the push-gate knows the integration suite is stale."""
from __future__ import annotations

import json
import os
import pathlib
import sys


def main() -> None:
    payload = json.load(sys.stdin)
    path = payload.get("tool_input", {}).get("file_path", "")
    if path.endswith(".py"):
        state = pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / ".claude" / ".state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "code_dirty").touch()


if __name__ == "__main__":
    main()
