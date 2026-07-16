"""Mobile detection — one User-Agent sniff per session (Phase M spec §2.1).

Resolution order: the ``_force_mobile`` session key (tests), a ``?m=1`` /
``?m=0`` query param (manual override, made sticky in session state), then the
request's User-Agent — the ``"Mobi"`` substring every phone browser carries
(iPad reports a desktop UA and deliberately takes the desktop path). The
verdict is cached in session state: a phone never becomes a desktop
mid-session, and a desktop window dragged narrow staying on the desktop path
is by design. The 767px breakpoint twin for the CSS layer lives in
``theme.MOBILE_MAX_PX``; this module never measures pixels.
"""

from __future__ import annotations

import streamlit as st

_CACHED = "_is_mobile"
FORCE_KEY = "_force_mobile"


def is_mobile() -> bool:
    """True when this session should render the phone experience."""
    ss = st.session_state
    if FORCE_KEY in ss:
        return bool(ss[FORCE_KEY])
    param = st.query_params.get("m")
    if param in ("0", "1"):
        ss[_CACHED] = param == "1"
    if _CACHED not in ss:
        agent = (st.context.headers or {}).get("User-Agent", "")
        ss[_CACHED] = "Mobi" in agent
    return ss[_CACHED]
