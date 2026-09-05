"""Pins for the constellation component's hand-authored frontend.

``build/main.js`` has no build step, no bundler and no JS test runner, so nothing
between the author and the browser reads it. A call to a function that does not
exist is therefore a runtime ``ReferenceError`` on the user's machine, silent in
every gate we run — which is exactly how ``fadeInWholeMap()`` shipped and threw on
every "Look wider" toggle for two months, taking the frame-height post down with
it. These read the file as text and catch that class.
"""

from __future__ import annotations

import pathlib
import re

MAIN_JS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/sportstradamus/dashboard/components/constellation_component/build/main.js"
)

# Statement keywords that a "(" follows without meaning a call.
_KEYWORDS = frozenset(
    {"if", "for", "while", "switch", "catch", "function", "return", "typeof"}
    | {"new", "delete", "void", "in", "of", "do", "else", "case"}
)
# Browser and language globals main.js actually calls. Derived from the file, and
# deliberately short: anything else it calls, it has to define.
_GLOBALS = frozenset({"Number", "String", "clearTimeout", "requestAnimationFrame", "setTimeout"})


def _source() -> str:
    text = MAIN_JS.read_text()
    return re.sub(r"(?m)//.*$", "", re.sub(r"/\*.*?\*/", " ", text, flags=re.S))


def test_every_function_main_js_calls_is_one_it_defines():
    source = _source()
    defined = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", source))
    defined |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", source))
    called = {
        match.group(1)
        for match in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", source)
        if match.group(1) not in _KEYWORDS
    }
    assert called - defined - _GLOBALS == set()


def test_the_frame_height_is_posted_before_anything_that_can_throw():
    """The phone sizes the component's iframe from ``setFrameHeight``. Posting it
    after the lens animation is what clipped the grown sky off the bottom when the
    animation threw, so the post has to come first in ``render``."""
    body = _source().split("function render(")[1].split("\n  }")[0]
    fade = re.search(r"\bfade\w*\(", body)
    assert fade is not None
    assert body.index("setFrameHeight(") < fade.start()
