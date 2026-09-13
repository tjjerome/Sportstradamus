"""Bidirectional constellation component — client-side hover (edges + card), clicks to Python.

``st.plotly_chart`` has no hover event and ``st.components.v1.html`` can't return a value, so
the slip editor's faint-on-hover edges and rich hover card need a declared component. The served
``build/`` is **hand-authored static files** (no JS build toolchain — vanilla ES6 runs
untranspiled; plotly.js + fonts load from CDN); ``build/main.js`` speaks the Streamlit
postMessage protocol directly. The frontend renders the figure JSON, dims a star's incident
edges + shows a hover card on hover (no rerun), and on a star click or the card's **Full detail**
button calls back with ``{action, key, nonce}`` — the nonce makes a repeat click a fresh value so
Streamlit reruns. Import-safe: no Archive, no network at import.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit.components.v1 as components

_BUILD_DIR = Path(__file__).parent / "build"
_component = components.declare_component("constellation", path=str(_BUILD_DIR))


def render_constellation(
    fig: go.Figure,
    *,
    key: str,
    sparks: dict[str, str],
    moves: dict[str, str],
    shots: dict[str, str],
    bans: dict[str, str],
    mobile: bool = False,
) -> dict | None:
    """Render the star map; return the last ``{action, key, nonce}`` the user fired.

    ``action`` is ``"click"`` (toggle the star's leg) or ``"detail"`` (open the offer
    dialog); ``key`` is the star's ``Player|Market|Bet``. ``None`` until the user acts.
    The caller dedups by ``nonce`` — a repeat click re-sends the same value.
    ``mobile`` switches the frontend to its touch flow (docked tap card, no hover).

    ``sparks`` maps a star's key to its hover card's last-five markup, and ``moves`` to its
    line-movement row; a key missing from ``moves`` draws no row at all. ``shots`` maps a
    player's display name to their headshot data URI, keyed by name rather than star key so
    one face serves all of that player's legs. ``bans`` maps a star's key to the text of its
    "won't pair" row; a key missing from ``bans`` draws no row either. All four travel beside
    the figure rather than inside ``customdata`` because four traces carry customdata
    (active, candidate, deep, wider) — a field there would ship the same payload four times
    and renumber every reader of the positional card fields.
    """
    return _component(
        figure_json=fig.to_json(),
        mobile=mobile,
        sparks=sparks,
        moves=moves,
        shots=shots,
        bans=bans,
        key=key,
        default=None,
    )
