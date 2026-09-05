"""The *look deeper* lens — the game's remaining legs, drawn inside the map.

An optional overlay on ``constellation.py``'s figure that obeys one rule: a lens
may add stars, never move the ones already drawn. A click grows only the picked
star's own glyph, which may nudge the neighbours it touches, and nothing
animates. *Look deeper* fades the game's remaining legs in as small stars
**inside** the constellation, each settled beside the main star it correlates
with (or into its own team's open space) with its ties drawn, so the map gains
detail instead of a second ring around it.

Placement is ``constellation_spacing.settle``, with everything already on screen
passed as ``fixed`` — which is what makes "revealing a lens never moves a star" a
property of the geometry rather than a convention. The one choice that looks
arbitrary — the main star an untied deep star borrows — is an md5 of the key it
belongs to (the seeding ``constellation_shapes.assign_templates`` uses) and never
``hash()``, whose ``str`` ordering ``PYTHONHASHSEED`` randomizes between runs.
Keying it on the star rather than on its rank is the second half of the rule: a
running counter re-deals every star behind the one you just clicked.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_spacing import settle

# The model-passed tier: a flat cool gray, distinct from both a team color and
# GRAY (the unknown-team fallback and the label color), so it reads as "the model
# passes on this" rather than as one more desaturated candidate.
_DEEP_COLOR = "#5f6b80"
_DEEP_ALPHA = 0.35
_SIDE_FALLBACK_X = 0.6  # where an untied deep star heads when its half holds no main star
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
    to, so a single tie puts the target *on* that star and ``settle`` only has to
    find the cell next to it. An untied star borrows one of its own half's main
    stars instead, which spreads the field through the constellation rather than
    piling one blob per side; with no main star to borrow it falls back to its
    half's midpoint.

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
    targets = {
        key: _tie_target(ties[key], main_pos)
        if ties[key]
        else _open_target(key, main_pos, side[key])
        for key in tier
    }
    return settle(targets, sizes, px, fixed=main_pos, side=side)


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
    size: float,
) -> None:
    """The deeper lens's own stars, as the one fade-able trace named ``deep``.

    Colour and opacity arrive per point because the tier carries two readings at
    one size: a model-liked leg the cut left behind wears the candidate look, a
    model-passed one the cool gray of the lens itself.
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
                "size": [size] * len(keys),
                "color": list(colors),
                "opacity": list(alphas),
            },
            customdata=[[key, *node_info[key]["card"], 0] for key in keys],
            hovertext=[node_info[key]["hover"] for key in keys],
            hoverinfo="none",
        )
    )
