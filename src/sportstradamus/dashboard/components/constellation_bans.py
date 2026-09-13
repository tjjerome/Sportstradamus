"""The "won't pair" mark — an orange × over a star the platform won't pair with the slip.

Every channel a star already carries is spoken for (fill = team, size = edge, brightness =
in the slip, border = the Over/Under call, gold = correlation), so the mark rides on top as
an overlay trace of its own: orange is the warning intent, and an × says "no" without color.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import plotly.graph_objects as go

from sportstradamus.dashboard.theme import ORANGE

# The × sits inside its star, so the star's own points still show past its arms.
BAN_MARK_SCALE = 0.7
# A 7 px lens star would scale to a 5 px speck; 8 px still reads as a cross.
BAN_MARK_MIN = 8
# Heavier than the 1.5 px Over/Under border, so the × never reads as one more outline.
BAN_MARK_WIDTH = 2.0


def add_ban_marks(fig: go.Figure, banned: Collection[str]) -> None:
    """Cross each star whose key is in ``banned``, one overlay per star trace, drawn on top.

    An overlay is named ``<host>_banned``, which is how ``main.js`` fades a lens's marks in
    with its stars. It copies the host's opacity at whatever level the host sets it, so a
    lens star's × stays as dim as the star (DESIGN §4a). No ``customdata`` and a skipped
    hover keep it inert: hover and click fall through to the star underneath.
    """
    stars = [trace for trace in fig.data if trace.customdata is not None]
    for trace in stars:
        hit = [i for i, card in enumerate(trace.customdata) if card[0] in banned]
        if not hit:
            continue
        star_size = np.asarray(_at(trace.marker.size, hit))
        fig.add_trace(
            go.Scatter(
                x=_at(trace.x, hit),
                y=_at(trace.y, hit),
                mode="markers",
                name=f"{trace.name}_banned",
                marker={
                    "symbol": "x-thin",
                    # .tolist(): plotly 6 ships a numpy array as base64, which the
                    # component's plotly.js 2.27 cannot decode.
                    "size": np.maximum(BAN_MARK_SCALE * star_size, BAN_MARK_MIN).tolist(),
                    "opacity": _at(trace.marker.opacity, hit),
                    "line": {"color": ORANGE, "width": BAN_MARK_WIDTH},
                },
                opacity=trace.opacity,
                hoverinfo="skip",
            )
        )


def _at(value, hit: list[int]):
    """``value`` at the ``hit`` points; a scalar or unset attribute already covers them all."""
    return value if np.ndim(value) == 0 else [value[i] for i in hit]
