"""The *look wider* lens — other games' best legs, in the open sky around the map.

An optional overlay on ``constellation.py``'s figure that obeys one rule: a lens
may add stars, never move the ones already drawn. *Look wider* leaves the map
alone apart from a slight recede and fills the open sky around it with other
games' best legs, scattered in per-game clusters and coloured by team.

Placement is ``constellation_spacing.settle``, with everything already on screen
passed as ``fixed`` — which is what makes "revealing a lens never moves a star" a
property of the geometry rather than a convention. Where in its band a cluster
lands, and how far it spreads, is a seeded dart throw (:func:`_throw_cluster`)
rather than a seat dealt along the band: an even pitch reads as six games each
assigned the one place it is allowed to be, which is the opposite of a sky. The
seed is an md5 of the game key (the seeding ``constellation_shapes.assign_templates``
uses) and never ``hash()``, whose ``str`` ordering ``PYTHONHASHSEED`` randomizes
between runs.

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
    _CELL_PX,
    _CHAR_WIDTH_EM,
    _FRAME_INSET,
    _STAR_GAP_PX,
    X_RANGE,
    Y_RANGE,
    _box,
    _clash,
    settle,
)
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.theme import GRAY, team_colors

# The sky draws at one flat size: under the main floor (14 desktop / 22 phone)
# so a sky star can never outrank a real one, over the engraving's FILLER_SIZE 6
# so it never reads as decoration. The phone value stays a tappable target — a
# missed tap lands on a neighbour's card, not on nothing.
WIDER_STAR_SIZE = 10
WIDER_STAR_SIZE_MOBILE = 16

WIDER_GAMES = 6  # 6 games x <= 6 legs fills the open bands without crowding them
_WIDER_ALPHA = 0.75  # dimmer than an active star, so the sky reads as background
# Clear air between a sky star and the map's outermost glyph: a sky star that
# grazes the constellation reads as one of its own.
_WIDER_MARGIN_PX = 48
_WIDER_REACH_MIN = 16  # the tightest cluster radius that still reads as a group, not a knot
_WIDER_REACH_MAX = 44  # the loosest that still reads as one game
_WIDER_DARTS = 64  # seeded throws per cluster; the second half throw at the tight reach
# A settle nudge in a crammed disc carries a sky star about two lattice cells. A
# label box lies outside settle's occupancy, so it is padded by this much or a
# nudged star lands on the neighbour's name.
_SKY_NUDGE_PX = 12
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
    band first. Where a game lands inside its band, and how wide it opens, is its
    own seeded throw (:func:`_throw_cluster`), taken best game first so a weaker
    one joining the sky only ever fits itself around what is already there.
    ``settle`` then does the same job it does for the map — nearest free cell,
    everything drawn ``fixed`` — with the footprint excluded, which is what makes
    "never inside the constellation" hold even where a gap between two main stars
    is the nearest free air.

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
    members = max((len(rows) for _, rows in groups), default=0)
    bands = sorted(
        _sky_bands(footprint, px, sky_y=sky_y, size=size, label_px=label_px, members=members),
        key=lambda band: (band[2] - band[0]) * (band[3] - band[1]),
        reverse=True,
    )[: len(groups)]
    if not bands:
        return {}, []
    frame = (
        -X_RANGE * _FRAME_INSET * px[0],
        -sky_y * _FRAME_INSET * px[1],
        X_RANGE * _FRAME_INSET * px[0],
        sky_y * _FRAME_INSET * px[1],
    )
    # A label may use the map's margin — text beside the outermost star reads as a
    # name, where a sky star there reads as one of the map's own.
    blocks = (footprint, _footprint(occupied, sizes, px, margin=0.0))
    rects: list[tuple[float, float, float, float]] = []
    discs: list[tuple[float, float, float]] = []
    anchors: dict[str, tuple[float, float]] = {}
    labels: list[tuple[str, float, float]] = []
    for index, (game, rows) in enumerate(groups):
        points, (label_x, label_y) = _throw_cluster(
            game,
            len(rows),
            bands[index % len(bands)],
            blocks,
            rects,
            discs,
            size=size,
            label_px=label_px,
            frame=frame,
        )
        anchors |= {
            corr_key(row): (x / px[0], y / px[1]) for row, (x, y) in zip(rows, points, strict=True)
        }
        labels.append((game, label_x / px[0], label_y / px[1]))
    placed = settle(
        anchors,
        {**sizes, **dict.fromkeys(anchors, size)},
        px,
        fixed=occupied,
        frame=(X_RANGE, sky_y),
        exclude=footprint,
    )
    return placed, labels


def _throw_cluster(
    game: str,
    members: int,
    band: tuple[float, float, float, float],
    blocks: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
    rects: list[tuple[float, float, float, float]],
    discs: list[tuple[float, float, float]],
    *,
    size: float,
    label_px: float,
    frame: tuple[float, float, float, float],
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """One game's cluster: seeded throws into its band, the first that clears.

    A throw is a centre anywhere in the band and a ``reach`` its legs scatter
    within, uniformly over that disc. It clears when the disc misses the map's
    footprint and the ink box its label hangs in misses the map's stars (the two
    ``blocks``), and both miss every label in ``rects``, every disc already
    thrown, and the frame. Half the throws take a random reach and half the
    tightest one the legs fit in, so a band with room reads loose while a crammed
    one still has throws that can squeeze. The label hangs under the lowest leg,
    or over the highest where the frame leaves no room below, and is drawn
    exactly where it was reserved: ``settle`` may still nudge a leg, but never
    the name.

    Reserves the throw it takes (appending to ``rects`` and ``discs``) and returns
    the legs' px positions in row order plus the label's px position.
    """
    rng = random.Random(int(hashlib.md5(game.encode()).hexdigest(), 16))
    lift = _label_lift(size, label_px)
    pad = _label_pad(size)
    width = len(game) * _CHAR_WIDTH_EM * label_px + 2 * pad
    height = _LABEL_LINE_PX * label_px + 2 * pad
    lo = _tight_reach(members, size)
    hi = max(lo, _WIDER_REACH_MAX)
    spread = [(math.sqrt(rng.random()), rng.uniform(0, math.tau)) for _ in range(members)]
    best = None
    for throw in range(_WIDER_DARTS):
        reach = rng.uniform(lo, hi) if 2 * throw < _WIDER_DARTS else lo
        room = reach + size
        x0, x1, y0, y1 = band[0] + room, band[2] - room, band[1] + room, band[3] - room
        x = rng.uniform(x0, x1) if x1 > x0 else (band[0] + band[2]) / 2
        y = rng.uniform(y0, y1) if y1 > y0 else (band[1] + band[3]) / 2
        points = [(x + reach * r * math.cos(a), y + reach * r * math.sin(a)) for r, a in spread]
        label_y = min(py for _, py in points) - lift
        if label_y - height / 2 < frame[1]:
            label_y = max(py for _, py in points) + lift
        label = _box(x, label_y, width, height)
        clash = _clash((x, y, room), label, blocks, rects, discs, frame)
        if best is None or clash < best[0]:
            best = (clash, (x, y, room), label, points, label_y)
        if not clash:
            break
    # A band can be crammed past what any throw clears — six games and one narrow
    # strip. Drawing the least-overlapping throw keeps every game in the sky,
    # which is the promise the lens makes.
    _, disc, label, points, label_y = best
    discs.append(disc)
    rects.append(label)
    return points, (disc[0], label_y)


def _label_lift(size: float, label_px: float) -> float:
    """How far a game label's ink reaches past the star it hangs under, in px."""
    return size / 2 + _LABEL_DROP_PX + _LABEL_LINE_PX * label_px / 2


def _label_pad(size: float) -> float:
    """Clear air reserved around a label's ink: a glyph's radius plus the nudge."""
    return size / 2 + _SKY_NUDGE_PX


def _tight_reach(members: int, size: float) -> float:
    """The tightest disc ``members`` glyphs fit in with their clear air, in px.

    One glyph's worth of area per member, so the legs rarely collide and a settle
    nudge stays inside the reach the throw reserved.
    """
    return max(_WIDER_REACH_MIN, math.sqrt(members) * (size + _STAR_GAP_PX) / 2)


def _band_room(members: int, size: float, label_px: float) -> tuple[float, float]:
    """Px a band must span to be sky: ``(across a side band, across a top or bottom one)``.

    A side band has the map's whole height for the label to hang into, so it
    only has to be as wide as the tightest disc; a band above or below the map
    stacks the label under the disc, so it has to be that tall.
    """
    disc = 2 * (_tight_reach(members, size) + size)
    hang = _label_lift(size, label_px) + _LABEL_LINE_PX * label_px / 2 + _label_pad(size)
    return disc, disc + hang


def _footprint(
    occupied: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    px: tuple[float, float],
    *,
    margin: float = _WIDER_MARGIN_PX,
) -> tuple[float, float, float, float]:
    """The px rectangle the drawn map fills, grown by ``margin`` of clear air."""
    reach = {key: sizes[key] / 2 + margin for key in occupied}
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
    size: float,
    label_px: float,
    members: int,
) -> list[tuple[float, float, float, float]]:
    """The open px rectangles around the footprint: left, right, above, below.

    A band too thin to hold the tightest cluster (and, above or below the map,
    the label hanging under it — :func:`_band_room`) is a gutter, not sky, and
    is dropped — which is how the desktop ends up with only its two side bands
    and the phone, whose map spans the width, with only the two it grew in y.
    """
    left, bottom = -X_RANGE * _FRAME_INSET * px[0], -sky_y * _FRAME_INSET * px[1]
    right, top = -left, -bottom
    x0, y0, x1, y1 = footprint
    beside, stacked = _band_room(members, size, label_px)
    return [
        band
        for band, room in (
            ((left, bottom, x0, top), beside),
            ((x1, bottom, right, top), beside),
            ((left, y1, right, top), stacked),
            ((left, bottom, right, y0), stacked),
        )
        if min(band[2] - band[0], band[3] - band[1]) >= room
    ]


def _grown_sky(
    occupied: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    px: tuple[float, float],
    *,
    sky_y: float,
    size: float,
    label_px: float,
    members: int,
) -> float:
    """``sky_y``, opened in y until the sky above and below the map both hold.

    The desktop's sky is its two side bands; while one of those survives the
    figure keeps its height. A deep enough tier closes both — immediately on the
    phone, whose map already spans the width — and the sky the map has left is
    the pair above and below it. The map's footprint is rarely symmetric in y,
    so one of that pair fits before the other, and a lens that takes the first
    piles every game into half a sky: both have to fit, and the figure grows by
    exactly what that takes.
    """
    box = _footprint(occupied, sizes, px)
    bands = _sky_bands(box, px, sky_y=sky_y, size=size, label_px=label_px, members=members)
    if any(band[2] <= box[0] or band[0] >= box[2] for band in bands):
        return sky_y
    # A lattice cell of play past the gutter test: the sky opens the same amount
    # above and below, and the band that lands an ulp short would be dropped.
    clear = max(box[3], -box[1]) + _band_room(members, size, label_px)[1] + _CELL_PX
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
    sky_y = _grown_sky(
        occupied,
        sizes,
        px,
        sky_y=sky_y,
        size=size,
        label_px=label_size,
        members=max((len(rows) for _, rows in groups), default=0),
    )
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
