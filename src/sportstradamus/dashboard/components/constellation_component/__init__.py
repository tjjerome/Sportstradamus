"""Bidirectional constellation component — client-side hover, optimistic star clicks to Python.

``st.plotly_chart`` has no hover event and ``st.components.v1.html`` can't return a value, so
the slip editor's faint-on-hover edges, rich hover card and instant star toggles need a
declared component. The served ``build/`` is **hand-authored static files** (no JS build
toolchain — vanilla ES6 runs untranspiled; plotly.js + fonts load from CDN); ``build/main.js``
speaks the Streamlit postMessage protocol directly. The frontend renders the figure JSON,
raises a star's incident edges and shows a hover card on hover (no rerun), and paints a
clicked star lit or dim at once, before Python has seen the click. It re-sends each painted
intent until Python acknowledges it (see :func:`render_constellation`). Import-safe: no
Archive, no network at import.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

_BUILD_DIR = Path(__file__).parent / "build"
_component = components.declare_component("constellation", path=str(_BUILD_DIR))


def render_constellation(
    fig: go.Figure,
    *,
    key: str,
    ack: int,
    sparks: dict[str, str],
    moves: dict[str, str],
    shots: dict[str, str],
    bans: dict[str, str],
    on_change: Callable[[], object],
    mobile: bool = False,
) -> None:
    """Draw the hover-card carrier, then the star map; ``on_change`` applies the map's value.

    The value is ``{"seq": int, "lit": {star_key: bool, ...}, "detail": star_key or null}``
    with ``Player|Market|Bet`` star keys:

    * ``seq`` rises on every send (the JS takes ``max(seq + 1, Date.now())``, so it keeps
      rising across an iframe remount);
    * ``lit`` holds every intent the frontend has painted but Python hasn't acknowledged,
      so the one value Streamlit keeps when it coalesces reruns still carries them all;
    * ``detail`` is the star whose **Full detail** button was pressed.

    ``ack`` is the last ``seq`` Python applied (0 before any click); the frontend drops
    intents at or below it and repaints the rest over each render. ``bans`` maps a star's
    key to its "won't pair" row (no key, no row). ``mobile`` switches the frontend to its
    touch flow (docked tap card, no hover).

    ``sparks`` maps a star's key to its card's last-five markup, ``moves`` to its
    line-movement row (no key, no row), and ``shots`` a player's display name to their
    headshot data URI. These ~100-400 KB are the same on every click, yet inside the
    component args they were re-sent with every figure. So they ride in a hidden ``st.html``
    carrier ahead of the map instead: Streamlit's forward-message cache sends an unchanged
    element of 10 KB or more as a hash reference rather than its bytes
    (``runtime/forward_msg_cache.py``; the hash ignores element position). Every ``<`` goes
    out as a JSON unicode escape: Streamlit's DOMPurify pass deletes a script whose text holds
    anything tag-like, which the spark SVG markup is, and a raw ``</`` would close the tag
    besides. ``st.html`` isn't iframed, so the script sets its global on the app window;
    theme.py keeps the carrier out of the layout, and ``main.js`` reads
    ``window.parent.__cstCards`` when a card opens.
    """
    cards = json.dumps({"sparks": sparks, "moves": moves, "shots": shots}).replace("<", "\\u003c")
    with st.container(key=f"{key}_cards"):
        st.html(f"<script>window.__cstCards = {cards};</script>", unsafe_allow_javascript=True)
    _component(
        figure_json=fig.to_json(),
        ack=ack,
        mobile=mobile,
        bans=bans,
        key=key,
        default=None,
        on_change=on_change,
    )
