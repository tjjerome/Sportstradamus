"""Astrolabe readout component — animated slip dials, no callback.

Streamlit has no built-in way to animate a metric from one value to the next, so the slip
builders' flat metric row is replaced by this declared component: three orbiting dials
(win/EV/Kelly) plus the lift arc between the two win dots, swept together from wherever
they stand toward each new payload, so an incidental rerun with the same numbers holds
still. The component's key keeps one iframe across reruns. The served ``build/`` is **hand-authored
static files** (no JS build toolchain, same pattern as ``constellation_component``);
``build/main.js`` speaks the Streamlit postMessage protocol directly. Pure readout — no
star/candidate lives here, so unlike the constellation there is no click/detail callback
back to Python. Import-safe: no Archive, no network at import.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_BUILD_DIR = Path(__file__).parent / "build"
_component = components.declare_component("astrolabe", path=str(_BUILD_DIR))


def render_astrolabe(payload: dict, *, key: str) -> None:
    """Render the astrolabe dials for one slip's ``astrolabe_payload`` dict.

    Below two legs the builder passes ``{"legs": n}`` instead, which draws the rest pose: dials at
    zero, readouts blank, the leg count shown. ``payload`` is plain JSON-primitive values
    (floats/ints/strings, a nested ``crowns`` dict) — Streamlit's component-argument marshalling
    JSON-encodes it for us, so unlike ``render_constellation``'s ``figure_json`` there is nothing to
    pre-serialize here. The component has no return value: selection happens on the constellation's
    own stars, not on this readout.
    """
    _component(payload=payload, key=key, default=None)
