"""Pins for the hand-authored frontends of the constellation and astrolabe components.

``build/main.js`` has no build step, no bundler and no JS test runner, so nothing
between the author and the browser reads it. A call to a function that does not
exist is therefore a runtime ``ReferenceError`` on the user's machine, silent in
every gate we run — which is exactly how ``fadeInWholeMap()`` shipped and threw on
every "Look wider" toggle for two months, taking the frame-height post down with
it. These read the files as text and catch that class.
"""

from __future__ import annotations

import pathlib
import re

import pytest

COMPONENTS = pathlib.Path(__file__).resolve().parents[2] / "src/sportstradamus/dashboard/components"
CONSTELLATION_JS = COMPONENTS / "constellation_component/build/main.js"
ASTROLABE_JS = COMPONENTS / "astrolabe_component/build/main.js"

# Statement keywords that a "(" follows without meaning a call.
_KEYWORDS = frozenset(
    {"if", "for", "while", "switch", "catch", "function", "return", "typeof"}
    | {"new", "delete", "void", "in", "of", "do", "else", "case"}
)
# Browser and language globals the two files actually call. Derived from the files, and
# deliberately short: anything else they call, they have to define.
_GLOBALS = frozenset({"Number", "String", "clearTimeout", "requestAnimationFrame", "setTimeout"})
# One left-to-right pass, so a "//" or "rotate(" inside a string stays string, and a quote
# inside a comment stays comment.
_NOT_CODE = re.compile(r"""'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|/\*.*?\*/|//[^\n]*""", re.S)


def _code(main_js: pathlib.Path) -> str:
    """The file with every string literal emptied and every comment blanked."""
    return _NOT_CODE.sub(
        lambda match: '""' if match.group()[0] in "'\"" else " ", main_js.read_text()
    )


@pytest.mark.parametrize(
    "main_js", [CONSTELLATION_JS, ASTROLABE_JS], ids=["constellation", "astrolabe"]
)
def test_every_function_main_js_calls_is_one_it_defines(main_js):
    code = _code(main_js)
    defined = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", code))
    defined |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", code))
    called = {
        match.group(1)
        for match in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", code)
        if match.group(1) not in _KEYWORDS
    }
    assert called - defined - _GLOBALS == set()


def test_the_frame_height_is_posted_before_anything_that_can_throw():
    """The phone sizes the component's iframe from ``setFrameHeight``. Posting it
    after the lens animation is what clipped the grown sky off the bottom when the
    animation threw, so the post has to come first in ``render``: ahead of every
    plotly call, the advance and draw that make them, and any fade or recede."""
    body = _code(CONSTELLATION_JS).split("function render(")[1].split("\n  }")[0]
    risky = re.search(r"\bPlotly\.|\b(?:advance|draw|fade|recede)\w*\(", body)
    assert risky is not None
    assert body.index("setFrameHeight(") < risky.start()
