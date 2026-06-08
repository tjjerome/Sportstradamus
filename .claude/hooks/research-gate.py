#!/usr/bin/env python3
"""PreToolUse[Edit|Write] hook: block a research-gated edit (a shipped: flip in
stat_meta.json, or a distribution-family file) when no research brief and no
waiver exist. The main agent then dispatches research-analyst or writes a waiver."""
from __future__ import annotations

import fnmatch
import glob
import json
import os
import pathlib
import sys


def project_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))


def gated_globs() -> list[str]:
    cfg = project_dir() / ".claude" / "research_gated.txt"
    if not cfg.exists():
        return []
    return [
        ln.strip()
        for ln in cfg.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def is_shipped_flip(tool_input: dict) -> bool:
    if not tool_input.get("file_path", "").endswith("stat_meta.json"):
        return False
    blob = (
        tool_input.get("old_string", "")
        + tool_input.get("new_string", "")
        + tool_input.get("content", "")
    )
    return "shipped" in blob


def is_gated(tool_input: dict, globs: list[str]) -> bool:
    if is_shipped_flip(tool_input):
        return True
    path = tool_input.get("file_path", "")
    return any(fnmatch.fnmatch(path, g) for g in globs)


def brief_or_waiver_exists() -> bool:
    waiver = project_dir() / ".claude" / ".state" / "research_waiver"
    return bool(glob.glob("/tmp/researcher_*.md")) or waiver.exists()


def main() -> None:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})
    if is_gated(tool_input, gated_globs()) and not brief_or_waiver_exists():
        sys.stderr.write(
            "research-gated change ({}). Dispatch the research-analyst subagent "
            "(it writes /tmp/researcher_<topic>.md), OR write a one-line "
            "justification to .claude/.state/research_waiver, then retry this edit.".format(
                tool_input.get("file_path", "?")
            )
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
