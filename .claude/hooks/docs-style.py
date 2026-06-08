#!/usr/bin/env python3
"""PostToolUse[Edit|Write|MultiEdit] hook: nudge on documentation drift.

Fires a one-line, non-blocking reminder of docs/STYLE_GUIDE.md §16 when an edit to
a Markdown file adds a dated changelog/build-log block (a dated `###`+ heading or a
dated bold label) or a large dated block of prose -- the two shapes that turned
operation_ship_75.md into a contradictory 1,000-line accretion. The edit has
already applied; exit 2 only surfaces the reminder (the repo's PostToolUse feedback
convention) so the next edit consolidates instead of piling on. A non-`.md` path, a
plain one-line changelog entry, or an edit without that smell exits 0 silently.
"""
from __future__ import annotations

import json
import re
import sys

# The build-log smell: a dated `###`+ subsection heading or a dated **bold label**,
# e.g. "#### Lever 1 -- BUILT 2026-06-06" or "**Updated 2026-06-07 ...**". A plain
# one-line changelog entry ("- 2026-06-07 fixed X", no bold) does NOT match -- that
# is the format §16 wants. Top-level `#`/`##` headers (e.g. a dated "## Current
# standings") are exempt: they date the doc's current state, not a build-log entry.
_DATED_LABEL = re.compile(
    r"^\s*(?:#{3,6}\s|[-*]\s+\*\*|\*\*).*\b\d{4}-\d{2}-\d{2}\b",
    re.MULTILINE,
)
_ANY_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_BIG_ADD_LINES = 60


def added_text(tool_input: dict) -> str:
    parts = [tool_input.get("new_string", ""), tool_input.get("content", "")]
    parts += [e.get("new_string", "") for e in tool_input.get("edits", [])]
    return "\n".join(p for p in parts if p)


def reasons(text: str) -> list[str]:
    out = []
    if _DATED_LABEL.search(text):
        out.append("a dated changelog/build-log block")
    if text.count("\n") + 1 > _BIG_ADD_LINES and _ANY_DATE.search(text):
        out.append(f"a >{_BIG_ADD_LINES}-line dated block")
    return out


def main() -> int:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})
    path = tool_input.get("file_path", "")
    if not path.endswith(".md"):
        return 0
    why = reasons(added_text(tool_input))
    if not why:
        return 0
    sys.stderr.write(
        f"[docs-style] {path}: this edit adds {' and '.join(why)}. Living docs "
        "describe CURRENT state, not history. Keep any changelog SHORT -- one-line "
        "caveman entries, newest-first, cap the recent few; older history lives in "
        "git, never a dated narrative block in the body. When a fact changes, fix "
        "or delete the OLD statement instead of layering a new one beside it; give "
        "each fact ONE canonical home and cross-ref it. See docs/STYLE_GUIDE.md §16.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
