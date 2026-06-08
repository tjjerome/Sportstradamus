#!/usr/bin/env python3
"""PostToolUse[Edit|Write|MultiEdit] hook: nudge on dashboard "AI-slop" design tells.

Fires a one-line, non-blocking reminder of DESIGN.md when an edit to a dashboard source
(`src/sportstradamus/dashboard*.py`, `src/sportstradamus/pages/*.py`) or
`.streamlit/config.toml` *adds* one of the named tells the design system bans: emoji used as
iconography, an overused font (Inter/Roboto/Arial/Helvetica/Open Sans/Lato), the default
Streamlit red `#FF4B4B`, a purple/violet gradient, or `st.balloons()`/`st.snow()`. The edit
has already applied; exit 2 only surfaces the reminder (the repo's PostToolUse feedback
convention, same as docs-style.py) so the next edit corrects course. Anything out of scope, or
an edit without a tell, exits 0 silently. The hard gate is tests/golden/test_design_tokens.py.
"""
from __future__ import annotations

import json
import re
import sys

# Pictographic emoji / dingbats used as iconography. Deliberately EXCLUDES the arrows
# (U+2190-21FF), technical (U+2300-23FF) and CJK-punctuation (U+3000-303F) blocks: the
# directional glyphs ↑ ↓ → ⇒ are legitimate stat-table craft ("↑ higher is better",
# sort-direction annotations), not the emoji-as-icon tell. Em-dash/middot/×/≥ are likewise
# outside these ranges and never trip it.
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff"
    "\U0000fe00-\U0000fe0f\U0001f1e6-\U0001f1ff]"
)
_BANNED_FONT = re.compile(r"\b(Inter|Roboto|Arial|Helvetica|Open Sans|Lato)\b")
_DEFAULT_RED = re.compile(r"#ff4b4b", re.IGNORECASE)
_PURPLE_GRADIENT = re.compile(
    r"(?:linear|radial|conic)-gradient[^;)]*?(?:purple|violet|"
    r"#(?:6366f1|8b5cf6|7c3aed|a855f7|635bff|9333ea|6d28d9))",
    re.IGNORECASE,
)
_PARTY = re.compile(r"st\.(balloons|snow)\s*\(")


def in_scope(path: str) -> bool:
    p = path.replace("\\", "/")
    if p.endswith(".streamlit/config.toml"):
        return True
    base = p.rsplit("/", 1)[-1]
    return base.endswith(".py") and (
        "/sportstradamus/pages/" in p or base.startswith("dashboard")
    )


def added_text(tool_input: dict) -> str:
    parts = [tool_input.get("new_string", ""), tool_input.get("content", "")]
    parts += [e.get("new_string", "") for e in tool_input.get("edits", [])]
    return "\n".join(p for p in parts if p)


def reasons(text: str) -> list[str]:
    out = []
    if _EMOJI.search(text):
        out.append("emoji used as an icon (use a :material/icon: shortcode)")
    if _BANNED_FONT.search(text):
        out.append("an overused font (Inter/Roboto/Arial/...) — IBM Plex only")
    if _DEFAULT_RED.search(text):
        out.append("the default Streamlit red #FF4B4B")
    if _PURPLE_GRADIENT.search(text):
        out.append("a purple/violet gradient")
    if _PARTY.search(text):
        out.append("st.balloons()/st.snow()")
    return out


def main() -> int:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})
    path = tool_input.get("file_path", "")
    if not in_scope(path):
        return 0
    why = reasons(added_text(tool_input))
    if not why:
        return 0
    sys.stderr.write(
        f"[design-lint] {path}: this edit adds {'; '.join(why)}. These are the "
        "AI-generated tells DESIGN.md bans. Theme via .streamlit/config.toml (the FIXED "
        "tokens), use Material Symbols not emoji, and keep red for negative values only. "
        "See DESIGN.md §5-6.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
