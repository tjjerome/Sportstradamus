"""The *look wider* lens — other games' best legs, in the open sky around the map.

An optional overlay on ``constellation.py``'s figure that obeys one rule: a lens
may add stars, never move the ones already drawn. *Look wider* leaves the map
alone apart from a slight recede and fills the open sky around it with other
games' best legs, scattered in per-game clusters and coloured by team.

Placement is ``constellation_spacing.settle``, with everything already on screen
passed as ``fixed`` — which is what makes "revealing a lens never moves a star" a
property of the geometry rather than a convention. The one choice that looks
arbitrary — where a sky cluster sits along its band — is an md5 of the key it
belongs to (the seeding ``constellation_shapes.assign_templates`` uses) and never
``hash()``, whose ``str`` ordering ``PYTHONHASHSEED`` randomizes between runs.

The phone starts with no room beside the map: even receded, the constellation
spans nearly the whole width, so its sky opens in y from the outset
(``_SKY_EXTRA_Y_MOBILE``). A deep enough tier walks the desktop into the same
corner, and both take the same way out — grow in y until a band fits, and grow
the figure with it, which is what keeps px-per-unit, and so every clearance
already solved against it, unchanged.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_spacing import (
    _FRAME_INSET,
    X_RANGE,
    Y_RANGE,
    settle,
)
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.theme import GRAY, team_colors

# Both lenses draw at one flat size: under the main floor (14 desktop / 22 phone)
# so a lens star can never outrank a real one, over the engraving's FILLER_SIZE 6
# so it never reads as decoration. The phone value stays a tappable target — a
# missed tap lands on a neighbour's card, not on nothing.
WIDER_STAR_SIZE = 10
WIDER_STAR_SIZE_MOBILE = 16

WIDER_GAMES = 6  # 6 games x <= 6 legs fills the open bands without crowding them
_WIDER_ALPHA = 0.75  # dimmer than an active star, so the sky reads as background
# Clear air between a sky star and the map's outermost glyph: a sky star that
# grazes the constellation reads as one of its own.
_WIDER_MARGIN_PX = 48
# The most a game's legs scatter from their slot and still read as one group. A
# band with more games than room tightens it rather than letting them collide.
_WIDER_CLUSTER_PX = 30
# Slot jitter as a fraction of the room a slot has spare. Under a half, two
# neighbours that jitter toward each other still clear.
_WIDER_JITTER = 0.35
# A game label's ink box as a multiple of its font px: plotly centres the text on
# its y, so half a line reaches past the drop on either side.
_LABEL_LINE_PX = 1.25
_LABEL_DROP_PX = 12  # clear air under a cluster's lowest star: attached, not touching
# The focus recedes only a little — the owner asked for room, not a shrunken map.
_WIDER_SCALE = 0.8
# y-units added above and below on the phone (~129 px a band), the one viewport
# with no side band left after the recede.
_SKY_EXTRA_Y_MOBILE = 1.0


def wider_positions(
    groups: Sequence[tuple[str, list[dict]]],
    occupied: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    px: tuple[float, float],
    *,
    size: float,
    sky_y: float,
    label_px: float,
) -> tuple[dict[str, tuple[float, float]], list[tuple[str, float, float]]]:
    """Scatter other games' legs through the open sky, one cluster per game.

    The sky is the inset frame minus the constellation's own footprint, cut into
    at most four bands; the games deal round-robin into whatever survives, largest
    band first. Each game takes a slot along its band's long axis, jittered so the
    row doesn't read as a scale, and its legs scatter around that slot inside what
    the slot has room for. ``settle`` then does the same job it does
    for the map — nearest free cell, everything drawn ``fixed`` — with the
    footprint excluded, which is what makes "never inside the constellation" hold
    even where a gap between two main stars is the nearest free air.

    Args:
        groups: ``(game key, rows)`` per other game, best game first.
        occupied: every star already drawn, in data units.
        sizes: marker px for every ``occupied`` key.
        px: rendered css px per data unit, ``(x, y)``.
        size: the flat marker px a sky star draws at.
        sky_y: the y half-range the sky spans — past the frame on the phone.
        label_px: the game label's font px; a band too thin to hold one is not sky.

    Returns:
        ``(key -> position, [(game, label x, label y)])``, all in data units.
    """
    footprint = _footprint(occupied, sizes, px)
    bands = sorted(
        _sky_bands(footprint, px, sky_y=sky_y, label_px=label_px),
        key=lambda band: (band[2] - band[0]) * (band[3] - band[1]),
        reverse=True,
    )[: len(groups)]
    if not bands:
        return {}, []
    anchors: dict[str, tuple[float, float]] = {}
    for index, band in enumerate(bands):
        anchors |= _scatter_band(
            band, groups[index :: len(bands)], px, size=size, label_px=label_px
        )
    placed = settle(
        anchors,
        {**sizes, **dict.fromkeys(anchors, size)},
        px,
        fixed=occupied,
        frame=(X_RANGE, sky_y),
        exclude=footprint,
    )
    lift = _label_lift(size, label_px) / px[1]
    frame_y = sky_y * _FRAME_INSET
    labels = []
    for game, rows in groups:
        cluster = [corr_key(row) for row in rows if corr_key(row) in placed]
        if not cluster:
            continue
        below = min(placed[key][1] for key in cluster) - lift
        labels.append(
            (
                game,
                sum(placed[key][0] for key in cluster) / len(cluster),
                below if below >= -frame_y else max(placed[key][1] for key in cluster) + lift,
            )
        )
    return placed, labels


def _label_lift(size: float, label_px: float) -> float:
    """How far a game label's ink reaches past the star it hangs under, in px.

    Both ends of the sky read this: ``_scatter_band`` reserves it in the stride it
    gives each game along a vertical band, and ``wider_positions`` spends exactly
    it placing the label. They have to be the same number or a label lands on the
    group below.
    """
    return size / 2 + _LABEL_DROP_PX + _LABEL_LINE_PX * label_px / 2


def _footprint(
    occupied: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    px: tuple[float, float],
) -> tuple[float, float, float, float]:
    """The px rectangle the drawn map fills, grown by ``_WIDER_MARGIN_PX`` of clear air."""
    reach = {key: sizes[key] / 2 + _WIDER_MARGIN_PX for key in occupied}
    return (
        min(x * px[0] - reach[key] for key, (x, _) in occupied.items()),
        min(y * px[1] - reach[key] for key, (_, y) in occupied.items()),
        max(x * px[0] + reach[key] for key, (x, _) in occupied.items()),
        max(y * px[1] + reach[key] for key, (_, y) in occupied.items()),
    )


def _sky_bands(
    footprint: tuple[float, float, float, float],
    px: tuple[float, float],
    *,
    sky_y: float,
    label_px: float,
) -> list[tuple[float, float, float, float]]:
    """The open px rectangles around the footprint: left, right, above, below.

    A band too thin to hold a cluster and its label is a gutter, not sky, and is
    dropped — which is how the desktop ends up with only its two side bands and
    the phone, whose map spans the width, with only the two it grew in y.
    """
    left, bottom = -X_RANGE * _FRAME_INSET * px[0], -sky_y * _FRAME_INSET * px[1]
    right, top = -left, -bottom
    x0, y0, x1, y1 = footprint
    room = 2 * _WIDER_CLUSTER_PX + label_px
    return [
        band
        for band in (
            (left, bottom, x0, top),
            (x1, bottom, right, top),
            (left, y1, right, top),
            (left, bottom, right, y0),
        )
        if min(band[2] - band[0], band[3] - band[1]) >= room
    ]


def _scatter_band(
    band: tuple[float, float, float, float],
    games: Sequence[tuple[str, list[dict]]],
    px: tuple[float, float],
    *,
    size: float,
    label_px: float,
) -> dict[str, tuple[float, float]]:
    """One band's games as jittered clusters, spread along its long axis.

    A vertical band stacks its games one over another, so a cluster has to own the
    strip its label hangs into as well or the label lands on the group below; along
    a horizontal band the labels sit side by side and only the clusters compete.
    Each game gets an equal ``stride`` of the band's long axis and its legs scatter
    inside whatever that stride leaves once the label is reserved, so a crowded
    band draws tighter groups rather than overlapping ones; the jitter is a
    fraction of what is still spare, which on a full band is nothing. From four
    games up the strip is consumed exactly (stride equals pitch), so a ``settle``
    nudge past the anchor radius eats into the neighbour's reservation — the
    residual graze in a crammed band.
    """
    x0, y0, x1, y1 = band
    along_x = (x1 - x0) >= (y1 - y0)
    span, start = (x1 - x0, x0) if along_x else (y1 - y0, y0)
    across = (y0 + y1) / 2 if along_x else (x0 + x1) / 2
    ink = 0.0 if along_x else _label_lift(size, label_px)
    reach = min(_WIDER_CLUSTER_PX, max((span / len(games) - size - ink) / 2, 0.0))
    stride = 2 * reach + size + ink
    lo = start + reach + size / 2 + ink
    hi = max(lo, start + span - reach - size / 2)
    pitch = (hi - lo) / max(len(games) - 1, 1)
    play = _WIDER_JITTER * max(pitch - stride, 0.0)
    anchors: dict[str, tuple[float, float]] = {}
    for slot, (game, rows) in enumerate(games):
        rng = random.Random(int(hashlib.md5(game.encode()).hexdigest(), 16))
        seat = lo + slot * pitch if len(games) > 1 else (lo + hi) / 2
        along = min(max(seat + rng.uniform(-1, 1) * play, lo), hi)
        center = (along, across) if along_x else (across, along)
        for row in rows:
            radius, angle = rng.uniform(0, reach), rng.uniform(0, 2 * math.pi)
            anchors[corr_key(row)] = (
                (center[0] + radius * math.cos(angle)) / px[0],
                (center[1] + radius * math.sin(angle)) / px[1],
            )
    return anchors


def _grown_sky(
    occupied: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    px: tuple[float, float],
    *,
    sky_y: float,
    label_px: float,
) -> float:
    """``sky_y``, opened in y far enough that the map leaves at least one band.

    A deep enough tier closes every band the sky had — immediately on the phone,
    whose map already spans the width, and on the desktop once the tier spreads
    past the side inset. Either way the layer would otherwise draw nothing at all,
    with no feedback, so both viewports take the phone's own way out and grow.
    """
    box = _footprint(occupied, sizes, px)
    if _sky_bands(box, px, sky_y=sky_y, label_px=label_px):
        return sky_y
    clear = max(box[3], -box[1]) + 2 * _WIDER_CLUSTER_PX + label_px
    return max(sky_y, clear / (px[1] * _FRAME_INSET))


def add_wider_layer(
    fig: go.Figure,
    groups: Sequence[tuple[str, list[dict]]],
    node_info: Mapping[str, dict],
    occupied: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    px: tuple[float, float],
    *,
    size: float,
    label_size: int,
    sky_y: float,
) -> None:
    """The wider lens: other games' best legs out in the sky, one label per game.

    ``_grown_sky`` says how far the sky has to open; whenever that reaches past the
    frame the figure grows by exactly the added y-range, so the px-per-unit the map
    was spaced against survives the reshape.
    """
    sky_y = _grown_sky(occupied, sizes, px, sky_y=sky_y, label_px=label_size)
    pos, labels = wider_positions(
        groups, occupied, sizes, px, size=size, sky_y=sky_y, label_px=label_size
    )
    if not pos:
        return
    if sky_y > Y_RANGE:
        fig.update_layout(
            height=fig.layout.height + 2 * (sky_y - Y_RANGE) * px[1],
            yaxis_range=[-sky_y, sky_y],
        )
    rows = [row for _, group in groups for row in group if corr_key(row) in pos]
    keys = [corr_key(row) for row in rows]
    fig.add_trace(
        go.Scatter(
            x=[pos[key][0] for key in keys],
            y=[pos[key][1] for key in keys],
            mode="markers",
            name="wider",
            marker={
                "symbol": "star",
                "size": [size] * len(keys),
                "color": [team_colors(str(row["League"]), str(row["Team"]))[0] for row in rows],
            },
            opacity=_WIDER_ALPHA,
            customdata=[[key, *node_info[key]["card"], 0] for key in keys],
            hovertext=[node_info[key]["hover"] for key in keys],
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x for _, x, _ in labels],
            y=[y for _, _, y in labels],
            mode="text",
            name="wider_labels",
            text=[game for game, _, _ in labels],
            textfont={"color": GRAY, "size": label_size},
            hoverinfo="skip",
        )
    )
