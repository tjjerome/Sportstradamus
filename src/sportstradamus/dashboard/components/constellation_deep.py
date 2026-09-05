"""The *look deeper* lens — the game's remaining legs, drawn inside the map.

An optional overlay on ``constellation.py``'s figure that obeys one rule: a lens
may add stars, never move the ones already drawn. A click grows only the picked
star's own glyph, which may nudge the neighbours it touches, and nothing
animates. *Look deeper* fades the game's remaining legs in as small stars
**inside** the constellation, each settled beside the main star it correlates
with (or into its own team's open space) with its ties drawn, so the map gains
detail instead of a second ring around it.

A deep star lands *near* the main star it belongs to, never on it: seeded polar
throws into the ring outside that main's clear air, bounded inside half the gap
to the next main so the tie still reads. ``constellation_spacing.settle`` takes
the throws it did not use as candidates, so only a crowded star falls back to the
lattice — placing every star on the lattice drew a ring of identical dots around
each main, which is a grid, not a sky. Everything already on screen goes to
``settle`` as ``fixed``, which is what makes "revealing a lens never moves a
star" a property of the geometry rather than a convention. Size and opacity both
carry the star's Kelly edge, so the tier ranks itself in a band that starts over
the engraving and stops under the main map's floor.

Every draw — the scatter and the main star an untied deep star borrows — is an
md5 of the key it belongs to (the seeding ``constellation_shapes.assign_templates``
uses) and never ``hash()``, whose ``str`` ordering ``PYTHONHASHSEED`` randomizes
between runs. Keying it on the star rather than on its rank is the second half of
the rule: a running counter re-deals every star behind the one you just clicked.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_spacing import (
    _FRAME_INSET,
    _STAR_GAP_PX,
    X_RANGE,
    Y_RANGE,
    settle,
)

# The model-passed tier: a flat cool gray, distinct from both a team color and
# GRAY (the unknown-team fallback and the label color), so it reads as "the model
# passes on this" rather than as one more desaturated candidate.
_DEEP_COLOR = "#5f6b80"
# Over the engraving's FILLER_SIZE (6) so a lens star never reads as decoration,
# under the main map's floor (_SIZE_MIN 14) so it never outranks a real star.
DEEP_SIZE_MIN, DEEP_SIZE_MAX = 7, 13
# A thumb needs ~12 px; 20 stays under the 22 px mobile main floor.
DEEP_SIZE_MIN_MOBILE, DEEP_SIZE_MAX_MOBILE = 12, 20
DEEP_ALPHA_MIN = 0.25  # the model-passed floor: present, never a distraction
_SIDE_FALLBACK_X = 0.6  # where an untied deep star heads when its half holds no main star
# How far past clear air a deep star may drift from its main: its neighbourhood,
# never halfway to the next one.
DEEP_SCATTER_PX = 28
DEEP_DARTS = 12  # seeded throws before the lattice fallback
# The strongest ties are the ones that placed the star; a 12-way fan off a 10 px
# star is not a reading, and a whole tier's fans are a thousand traces of payload.
DEEP_EDGES_PER_STAR = 2


def deep_tier(node_info: Mapping[str, dict], keys: Sequence[str]) -> list[str]:
    """Every known leg that is not a main star, strongest first.

    One sort key gives both readings the deeper lens holds — the model-liked legs
    the default cut left behind, then the model-passed ones — with no tier
    bookkeeping: edge orders them and the sign says which is which.
    """
    main = set(keys)
    return sorted(
        (key for key in node_info if key not in main),
        key=lambda key: (-node_info[key]["edge"], key),
    )


def deep_positions(
    tier: Sequence[str],
    main_pos: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    edges: Sequence[tuple[str, str, float]],
    node_team: Mapping[str, str | None],
    teams: Sequence[str],
    px: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Place the deeper lens's stars inside the map, beside what they correlate with.

    A tied star targets the |rho|-weighted centroid of the main stars it is tied
    to, so a single tie puts the target *on* that star. An untied star borrows one
    of its own half's main stars instead, which spreads the field through the
    constellation rather than piling one blob per side; with no main star to
    borrow it falls back to its half's midpoint. The star then scatters off that
    target by seeded throw (:func:`_scatter_darts`) and hands ``settle`` the
    throws it did not use, so a star with room keeps a float position of its own
    and only a crowded one takes a lattice cell.

    Args:
        tier: the deep keys — iteration order is placement priority, and the
            caller puts its promoted keys first because a promoted star is lit
            with the lens shut, so its place must not depend on the tier growing
            around it.
        main_pos: the drawn map in data units, passed to ``settle`` as ``fixed`` so
            no main star can be pushed by a lens.
        sizes: marker px for every tier key and every main key.
        edges: signed ``(a, b, rho)`` ties over the main stars and the tier together.
        node_team: team code per tier key.
        teams: the matchup's two codes, sorted — index 0 owns the left half.
        px: rendered css px per data unit, ``(x, y)``.

    Returns:
        key -> position in data units, for the ``tier`` keys only.
    """
    rest = set(tier)
    ties: defaultdict[str, list[tuple[float, str]]] = defaultdict(list)
    for node_a, node_b, rho in edges:
        for one, other in ((node_a, node_b), (node_b, node_a)):
            if one in rest and other in main_pos:
                ties[one].append((abs(rho), other))
    side = {key: _half(node_team.get(key), teams) for key in tier}
    mains_px = {key: (x * px[0], y * px[1]) for key, (x, y) in main_pos.items()}
    anchors: dict[str, tuple[float, float]] = {}
    candidates: dict[str, list[tuple[float, float]]] = {}
    for key in tier:
        target = (
            _tie_target(ties[key], main_pos)
            if ties[key]
            else _open_target(key, main_pos, side[key])
        )
        darts = _scatter_darts(key, target, mains_px, sizes, side[key], px)
        anchors[key] = darts[0] if darts else target
        candidates[key] = darts[1:]
    return settle(anchors, sizes, px, fixed=main_pos, side=side, candidates=candidates)


def _scatter_darts(
    key: str,
    target: tuple[float, float],
    mains_px: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    side: float,
    px: tuple[float, float],
) -> list[tuple[float, float]]:
    """Seeded throws around ``target``, into the ring a deep star may occupy.

    The band starts outside the clear air of ``target``'s nearest main star and
    reaches ``DEEP_SCATTER_PX`` further out, but never past half the gap to the
    next main: inside that half a star is always nearer its own main, which is
    what keeps "beside its tie" — and the borrowed-main reading of an untied star
    — true whatever the throw does. A throw that would leave the frame or cross
    onto the other team's half is dropped rather than clamped, since a clamp
    piles stars along the line it clamps to.

    ``mains_px`` is ``main_pos`` in px; the band is measured there because the
    frame's aspect flips with the viewport.
    """
    size = sizes[key]
    at = (target[0] * px[0], target[1] * px[1])
    home = min(mains_px, key=lambda main: math.dist(at, mains_px[main]), default=None)
    if home is None:
        r_lo, cap = size / 2 + _STAR_GAP_PX, math.inf
    else:
        r_lo = (sizes[home] + size) / 2 + _STAR_GAP_PX
        cap = min(
            (
                math.dist(mains_px[home], mains_px[other]) / 2 - size / 2
                for other in mains_px
                if other != home
            ),
            default=math.inf,
        )
    r_hi = max(r_lo, min(r_lo + DEEP_SCATTER_PX, cap))

    rng = random.Random(int(hashlib.md5(key.encode()).hexdigest(), 16))
    darts = []
    for _ in range(DEEP_DARTS):
        angle, radius = rng.uniform(0, math.tau), rng.uniform(r_lo, r_hi)
        x = target[0] + radius * math.cos(angle) / px[0]
        y = target[1] + radius * math.sin(angle) / px[1]
        if x * side >= 0 and abs(x) <= X_RANGE * _FRAME_INSET and abs(y) <= Y_RANGE * _FRAME_INSET:
            darts.append((x, y))
    return darts


def _tie_target(
    ties: list[tuple[float, str]], main_pos: Mapping[str, tuple[float, float]]
) -> tuple[float, float]:
    weight = sum(rho for rho, _ in ties)
    return (
        sum(rho * main_pos[key][0] for rho, key in ties) / weight,
        sum(rho * main_pos[key][1] for rho, key in ties) / weight,
    )


def _open_target(
    key: str, main_pos: Mapping[str, tuple[float, float]], side: float
) -> tuple[float, float]:
    """The main star an untied deep star borrows, drawn from its own half by key.

    The draw has to be a property of the key alone. A running rank would re-deal
    every star behind the one that leaves the sequence — which is what promoting a
    star does — and a click is not animated, so those stars would teleport.
    """
    half = [main for main in sorted(main_pos) if side == 0 or main_pos[main][0] * side >= 0]
    if not half:
        return (side * _SIDE_FALLBACK_X, 0.0)
    return main_pos[half[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(half)]]


def _half(team: str | None, teams: Sequence[str]) -> float:
    """-1 / +1 for the half a team owns; 0 for a team that is neither side."""
    return teams.index(team) * 2.0 - 1.0 if team in teams else 0.0


def add_deep_trace(
    fig: go.Figure,
    keys: Sequence[str],
    pos: Mapping[str, tuple[float, float]],
    node_info: Mapping[str, dict],
    *,
    colors: Sequence[str],
    alphas: Sequence[float],
    sizes: Mapping[str, float],
) -> None:
    """The deeper lens's own stars, as the one fade-able trace named ``deep``.

    Everything arrives per point because the tier carries two readings and a
    ranking inside each: a model-liked leg the cut left behind wears the candidate
    look, a model-passed one the cool gray of the lens itself, and edge sets how
    big and how bright either is.
    """
    if not keys:
        return
    fig.add_trace(
        go.Scatter(
            x=[pos[key][0] for key in keys],
            y=[pos[key][1] for key in keys],
            mode="markers",
            name="deep",
            marker={
                "symbol": "star",
                "size": [sizes[key] for key in keys],
                "color": list(colors),
                "opacity": list(alphas),
            },
            customdata=[[key, *node_info[key]["card"], 0] for key in keys],
            hovertext=[node_info[key]["hover"] for key in keys],
            hoverinfo="none",
        )
    )
