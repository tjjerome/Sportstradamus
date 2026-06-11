"""Pin the DESIGN.md design system: committed visual identity, no AI-generated tells.

The dashboard's visual identity is FIXED in ``.streamlit/config.toml`` and documented in
``DESIGN.md``. This gate fails if the tokens drift back toward the Streamlit defaults that read
as "AI-generated" — the default red ``#FF4B4B``, an overused font (Inter/Roboto/Arial/...), a
purple/violet gradient, ``st.balloons``/``st.snow``, or emoji used as iconography in the
dashboard sources. The live, non-blocking counterpart is ``.claude/hooks/design-lint.py``.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "sportstradamus"
_CONFIG = _REPO / ".streamlit" / "config.toml"

# Kept in lockstep with .claude/hooks/design-lint.py. Excludes arrows (U+2190-21FF),
# technical (U+2300-23FF) and CJK punctuation (U+3000-303F): ↑ ↓ → ⇒ are legitimate
# stat-direction glyphs, not the emoji-as-icon tell.
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


def _theme() -> dict:
    with _CONFIG.open("rb") as fh:
        return tomllib.load(fh)["theme"]


def _dashboard_files() -> list[Path]:
    files = sorted((_SRC / "dashboard").rglob("*.py"))
    return [f for f in files if f.name != "__init__.py"]


def test_config_toml_carries_fixed_identity() -> None:
    t = _theme()
    assert t["base"] == "dark"
    assert t["primaryColor"] == "#2E6BE6"
    assert _DEFAULT_RED.search(t["primaryColor"]) is None, "primary reverted to Streamlit red"
    assert "IBM Plex Sans" in t["font"]
    assert len(t["chartSequentialColors"]) == 10, "sequential ramp must be exactly 10 colors"
    for key in ("font", "headingFont", "codeFont"):
        assert _BANNED_FONT.search(t[key]) is None, f"{key} uses an overused font"


def test_design_md_present_with_rules() -> None:
    text = (_REPO / "DESIGN.md").read_text(encoding="utf-8")
    for needle in ("Design tokens (FIXED", "## 6. NEVER", "FIXED vs FLEXIBLE"):
        assert needle in text, f"DESIGN.md missing section: {needle!r}"


def test_design_md_celestial_layer() -> None:
    """The approved celestial amendment: gold token, display-only faces, ambient-art rules."""
    text = (_REPO / "DESIGN.md").read_text(encoding="utf-8")
    for needle in (
        "#C9A227",
        "Cinzel",
        "Cormorant Garamond",
        "Never for data, numerals",
        "no AI-generated images",
        "constellation",
    ):
        assert needle in text, f"DESIGN.md missing celestial-layer rule: {needle!r}"
    theme_py = _SRC / "dashboard" / "theme.py"
    assert "#C9A227" in theme_py.read_text(encoding="utf-8"), (
        "dashboard/theme.py gold constant drifted from DESIGN.md #C9A227"
    )


def test_no_emoji_in_dashboard_sources() -> None:
    offenders = []
    for path in _dashboard_files():
        if _EMOJI.search(path.name):
            offenders.append(f"{path.name} (filename)")
        if _EMOJI.search(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name} (contents)")
    assert not offenders, (
        "Emoji used as iconography in dashboard sources — use Material Symbols "
        f"(:material/icon:) or semantic colored text instead. Offenders: {offenders}"
    )


def test_no_default_red_or_party_effects() -> None:
    offenders = []
    for path in [*_dashboard_files(), _CONFIG]:
        text = path.read_text(encoding="utf-8")
        if _DEFAULT_RED.search(text):
            offenders.append(f"{path.name}: default red #FF4B4B")
        if _PARTY.search(text):
            offenders.append(f"{path.name}: st.balloons/st.snow")
        if _PURPLE_GRADIENT.search(text):
            offenders.append(f"{path.name}: purple/violet gradient")
    assert not offenders, f"DESIGN.md NEVER rules violated: {offenders}"


def test_design_lint_hook_registered() -> None:
    settings = json.loads((_REPO / ".claude" / "settings.json").read_text(encoding="utf-8"))
    commands = [
        h.get("command", "")
        for block in settings["hooks"]["PostToolUse"]
        for h in block.get("hooks", [])
    ]
    assert any("design-lint.py" in c for c in commands), (
        "design-lint.py hook is not registered in .claude/settings.json PostToolUse"
    )
