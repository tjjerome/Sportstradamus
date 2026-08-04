"""Pin viewport.is_mobile()'s resolution order: force-key > query param > UA sniff.

Runs each check through AppTest (st.session_state / st.context need a script
runtime). AppTest requests carry no User-Agent header, so the UA fallback
resolves False — which is also why every existing desktop-path test stays on
the desktop branch untouched.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

_SCRIPT = """
import streamlit as st
from sportstradamus.dashboard.viewport import is_mobile

st.write(f"mobile={is_mobile()}")
"""


def _run(force: bool | None = None, query_m: str | None = None) -> str:
    at = AppTest.from_string(_SCRIPT, default_timeout=10)
    if force is not None:
        at.session_state["_force_mobile"] = force
    if query_m is not None:
        at.query_params["m"] = query_m
    at.run()
    assert not at.exception
    return at.markdown[0].value


def test_defaults_to_desktop_without_headers():
    assert _run() == "mobile=False"


def test_force_key_wins():
    assert _run(force=True) == "mobile=True"


def test_query_param_overrides_ua():
    assert _run(query_m="1") == "mobile=True"
    assert _run(query_m="0") == "mobile=False"


def test_force_key_beats_query_param():
    assert _run(force=False, query_m="1") == "mobile=False"
