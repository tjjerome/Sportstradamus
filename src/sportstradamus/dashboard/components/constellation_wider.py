"""The *look wider* lens — other games' best legs, in the open sky around the map.

An optional overlay on ``constellation.py``'s figure that obeys one rule: a lens
may add stars, never move the ones already drawn. *Look wider* leaves the map
alone apart from a slight recede and fills the open sky around it with other
games' best legs, scattered in per-game clusters and coloured by team.

Placement is ``constellation_spacing.settle``, with everything already on screen
passed as ``fixed`` — which is what makes "revealing a lens never moves a star" a
property of the geometry rather than a convention. Where the sky is, and where in
its band a cluster lands and how far it spreads, is ``constellation_sky``'s: a
seeded dart throw (:func:`constellation_sky.throw_cluster`) rather than a seat
dealt along the band, because an even pitch reads as six games each assigned the
one place it is allowed to be, which is the opposite of a sky.

The phone starts with no room beside the map: even receded, the constellation
spans nearly the whole width, so its sky opens in y from the outset
(``SKY_EXTRA_Y_MOBILE``). A deep enough tier walks the desktop into the same
corner, and both take the same way out — grow in y until a band fits, and grow
the figure with it, which is what keeps px-per-unit, and so every clearance
already solved against it, unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_sky import (
    footprint,
    grown_sky,
    sky_bands,
    throw_cluster,
)
from sportstradamus.dashboard.components.constellation_spacing import (
    _FRAME_INSET,
    X_RANGE,
    Y_RANGE,
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
# The focus recedes only a little — the owner asked for room, not a shrunken map.
WIDER_SCALE = 0.8
# y-units added above and below on the phone (~129 px a band), the one viewport
# with no side band left after the recede.
SKY_EXTRA_Y_MOBILE = 1.0


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
    own seeded throw (:func:`constellation_sky.throw_cluster`), taken best game
    first so a weaker one joining the sky only ever fits itself around what is
    already there. ``settle`` then does the same job it does for the map — nearest
    free cell, everything drawn ``fixed`` — with the footprint excluded, which is
    what makes "never inside the constellation" hold even where a gap between two
    main stars is the nearest free air.

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
    box = footprint(occupied, sizes, px)
    members = max((len(rows) for _, rows in groups), default=0)
    bands = sorted(
        sky_bands(box, px, sky_y=sky_y, size=size, label_px=label_px, members=members),
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
    blocks = (box, footprint(occupied, sizes, px, margin=0.0))
    rects: list[tuple[float, float, float, float]] = []
    discs: list[tuple[float, float, float]] = []
    anchors: dict[str, tuple[float, float]] = {}
    labels: list[tuple[str, float, float]] = []
    for index, (game, rows) in enumerate(groups):
        points, (label_x, label_y) = throw_cluster(
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
        exclude=box,
    )
    return placed, labels


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

    ``constellation_sky.grown_sky`` says how far the sky has to open; whenever that
    reaches past the frame the figure grows by exactly the added y-range, so the
    px-per-unit the map was spaced against survives the reshape.
    """
    sky_y = grown_sky(
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
