"""The wider lens's sky geometry — where around the map there is room for other games.

``constellation_wider`` decides what the lens draws; this module answers where it
may draw it, in the css px ``constellation_spacing`` measures in: the rectangle the
drawn map fills (:func:`footprint`), the open bands beside, above and below it that
can hold a cluster and its label (:func:`sky_bands`), how far the sky has to open
in y when none can (:func:`grown_sky`), and the seeded dart throw that lands one
game's cluster and its label inside a band (:func:`throw_cluster`). The seed is an
md5 of the game key and never ``hash()``, whose ``str`` ordering ``PYTHONHASHSEED``
randomizes between runs.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping

from sportstradamus.dashboard.components.constellation_spacing import (
    _CELL_PX,
    _CHAR_WIDTH_EM,
    _FRAME_INSET,
    _STAR_GAP_PX,
    X_RANGE,
    _box,
    _clash,
)

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


def throw_cluster(
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


def footprint(
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


def sky_bands(
    box: tuple[float, float, float, float],
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
    x0, y0, x1, y1 = box
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


def grown_sky(
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
    box = footprint(occupied, sizes, px)
    bands = sky_bands(box, px, sky_y=sky_y, size=size, label_px=label_px, members=members)
    if any(band[2] <= box[0] or band[0] >= box[2] for band in bands):
        return sky_y
    # A lattice cell of play past the gutter test: the sky opens the same amount
    # above and below, and the band that lands an ulp short would be dropped.
    clear = max(box[3], -box[1]) + _band_room(members, size, label_px)[1] + _CELL_PX
    return max(sky_y, clear / (px[1] * _FRAME_INSET))
