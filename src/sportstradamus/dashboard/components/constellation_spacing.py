"""What the constellation shows and how close its stars may sit.

A busy game offers far more model-liked legs than the map can hold — a hundred
stars on a template's dozen vertices is a blob, not a reading. So the map draws a
bounded default set (:func:`default_stars`), the rest waits behind *look deeper*,
and whatever ends up drawn goes through one placement primitive
(:func:`settle`): the highest-priority stars keep their wanted position exactly
and only a star that would collide moves, to the nearest free cell of a fixed
lattice on its own team's half. Captions follow the same economy —
:func:`caption_positions` gives one only to the slip's stars and the biggest few
candidates, and only where the text box clears every glyph and every caption
already placed.

Distances are in screen px, not data units: the frame is far wider than it is
tall and flips shape between viewports, so two stars that read as separated on a
laptop can overlap on a phone at identical data coordinates. Everything here is
deterministic — a fixed lattice, ordered iteration, ties broken by first index —
because a golden pin on a star's position has to hold across processes.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Container, Mapping, Sequence

import numpy as np

from sportstradamus.leg_schema import is_model_liked, leg_field

X_RANGE, Y_RANGE = 1.6, 1.4  # the figure's locked axis ranges (constellation._blank_figure)

# Data units to css px, per viewport: the desktop plot box is ~980x360 over the
# 3.2 x 2.8 unit frame, the phone ~358 wide at the same height — the aspect
# inversion SHAPE_SCALE_MOBILE is tuned for.
PX_PER_UNIT = (306.0, 128.6)
PX_PER_UNIT_MOBILE = (112.0, 128.6)

_CELL_PX = 6  # lattice pitch: a nudged star lands within 6 px of ideal, invisible at star scale
_STAR_GAP_PX = 6  # clear air between two glyphs' bounding circles
# Cells stay 12% inside each axis range — clear of the frame and the Cinzel team
# tags. The lattice therefore caps at |y| <= 1.232 while a desktop template vertex
# reaches 1.297, so a crowded edge star settles inward, never along the frame.
_FRAME_INSET = 0.88

DEFAULT_STARS = 12  # ~ a template's vertex count (5-13, median 10), so the cut fills the shape
MIN_PER_TEAM = 4  # both halves populated — the both-teams parlay rule, made visual
MAX_PER_PLAYER = 2  # one hot player's five markets must not own the map

CAPTION_TOP_K = 5  # captions for the slip plus the five biggest candidates; the rest hover
_CHAR_WIDTH_EM = 0.6  # IBM Plex Sans average advance — plotly cannot measure text server-side
_LINE_HEIGHT_EM = 1.25  # a caption is one line — plotly's single-line box at font_px


def settle(
    anchors: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    px: tuple[float, float],
    *,
    fixed: Mapping[str, tuple[float, float]] | None = None,
    side: Mapping[str, float] | None = None,
    frame: tuple[float, float] = (X_RANGE, Y_RANGE),
    exclude: tuple[float, float, float, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Nearest free lattice cell to each anchor, given the stars already down.

    Args:
        anchors: key -> wanted position in data units. Iteration order is
            priority: an anchor with room keeps its position to the float, so
            passing the biggest stars first leaves them exactly on their vertices.
        sizes: marker px for every anchor and every ``fixed`` key.
        px: rendered css px per data unit, ``(x, y)`` — the whole solve runs in px
            because the frame's aspect (and so what "too close" means) flips with the
            viewport. A caller that rescales positions after the solve folds that
            factor in here, or the clearance it enforces is not the one drawn.
        fixed: stars that occupy space, never move, and are not returned.
        side: key -> -1 / +1 / 0, the half of the team axis a star may occupy.
            x = 0 is legal for both, mirroring ``constellation_layout._clamp_to_side``.
        frame: the figure's axis ranges; the lattice insets itself inside them.
        exclude: px rectangle ``(x0, y0, x1, y1)`` no star's centre may enter.

    Returns:
        key -> position in data units, for the ``anchors`` keys only.
    """
    # style: allow-complexity — one clear-air field built in one pass; the branches
    # are the placement rules themselves, not steps a helper could take away.
    grid = [
        np.arange(-limit * _FRAME_INSET * scale, limit * _FRAME_INSET * scale, _CELL_PX)
        for limit, scale in zip(frame, px, strict=True)
    ]
    mesh = np.meshgrid(*grid)
    cells_x, cells_y = mesh[0].ravel(), mesh[1].ravel()
    clear = np.full(cells_x.shape, np.inf)  # px a newcomer's edge may still spread here
    if exclude is not None:
        x0, y0, x1, y1 = exclude
        inside = (cells_x >= x0) & (cells_x <= x1) & (cells_y >= y0) & (cells_y <= y1)
        clear[inside] = -np.inf
    occupied: list[tuple[float, float, float]] = []

    def occupy(x: float, y: float, size: float) -> None:
        np.minimum(clear, np.hypot(cells_x - x, cells_y - y) - size / 2 - _STAR_GAP_PX, out=clear)
        occupied.append((x, y, size))

    for key, (x, y) in (fixed or {}).items():
        occupy(x * px[0], y * px[1], sizes[key])

    placed: dict[str, tuple[float, float]] = {}
    for key, anchor in anchors.items():
        size = sizes[key]
        x, y = anchor[0] * px[0], anchor[1] * px[1]
        room = min(
            (math.dist((x, y), (ox, oy)) - other / 2 - _STAR_GAP_PX for ox, oy, other in occupied),
            default=math.inf,
        )
        banned = (
            exclude is not None and exclude[0] <= x <= exclude[2] and exclude[1] <= y <= exclude[3]
        )
        spot = anchor
        if room < size / 2 or banned:
            free = clear >= size / 2
            if side and side.get(key):
                free &= cells_x * side[key] >= 0
            reach = np.where(free, np.hypot(cells_x - x, cells_y - y), np.inf)
            nearest = int(np.argmin(reach))  # first index wins, so ties break deterministically
            if math.isfinite(reach[nearest]):  # nowhere legal to go: the anchor stands
                x, y = float(cells_x[nearest]), float(cells_y[nearest])
                spot = (x / px[0], y / px[1])
        placed[key] = spot
        occupy(x, y, size)
    return placed


def default_stars(
    universe: Mapping[str, Mapping],
    teams: Sequence[str],
    *,
    cap: int = DEFAULT_STARS,
    per_team: int = MIN_PER_TEAM,
    per_player: int = MAX_PER_PLAYER,
) -> list[str]:
    """The game's strongest model-liked legs — the stars drawn before *look deeper*.

    Ranked by Kelly edge, but two floors bend the ranking so the map still reads
    as a matchup: each team fills ``per_team`` slots before the open ranking
    spends the rest, and no player carries more than ``per_player`` legs. Takes
    the defining rows (canonical legs or raw offer rows alike, via ``leg_field``)
    so the slate classifier can call it without building the figure's node info.

    ``cap`` bounds the open ranking, not the result: the team floors are filled
    first and keep what they took, so ``per_team`` times the number of teams
    overshoots ``cap`` when it is set above it (the shipped 4 + 4 sits under 12).
    """
    ranked = sorted(
        (key for key, leg in universe.items() if is_model_liked(leg)),
        key=lambda key: (-float(leg_field(universe[key], "kelly")), key),
    )
    chosen: list[str] = []
    legs_of_player: Counter[str] = Counter()

    def fill(candidates: list[str], limit: int) -> None:
        for key in candidates:
            if len(chosen) >= limit:
                return
            player = str(leg_field(universe[key], "player"))
            if key not in chosen and legs_of_player[player] < per_player:
                chosen.append(key)
                legs_of_player[player] += 1

    for team in teams:
        own = [key for key in ranked if leg_field(universe[key], "team") == team]
        fill(own, len(chosen) + per_team)
    fill(ranked, cap)
    return sorted(chosen)


def caption_positions(
    keys: Sequence[str],
    pos: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    labels: Mapping[str, str],
    active: Container[str],
    px: tuple[float, float],
    *,
    font_px: int,
    top_k: int = CAPTION_TOP_K,
) -> dict[str, str]:
    """Key -> plotly ``textposition`` for the stars that get a caption; absent = none.

    The slip's stars come first, then the ``top_k`` biggest candidates — a
    caption marks importance, never selection (that stays fill + opacity, DESIGN
    §4a). Each tries above the star then below it, and takes the first placement
    whose text box clears every glyph and every caption already accepted.
    """
    glyphs = [_box(pos[k][0] * px[0], pos[k][1] * px[1], sizes[k], sizes[k]) for k in keys]
    by_size = sorted(keys, key=lambda k: (-sizes[k], k))
    order = [k for k in by_size if k in active] + [k for k in by_size if k not in active][:top_k]

    boxes = list(glyphs)
    captions: dict[str, str] = {}
    for key in order:
        x, y = pos[key][0] * px[0], pos[key][1] * px[1]
        height = _LINE_HEIGHT_EM * font_px
        offset = sizes[key] / 2 + height / 2
        for placement, direction in (("top center", 1.0), ("bottom center", -1.0)):
            box = _box(
                x, y + direction * offset, len(labels[key]) * _CHAR_WIDTH_EM * font_px, height
            )
            if not any(_overlaps(box, other) for other in boxes):
                captions[key] = placement
                boxes.append(box)
                break
    return captions


def _box(x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def _overlaps(box: tuple[float, ...], other: tuple[float, ...]) -> bool:
    return box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]
