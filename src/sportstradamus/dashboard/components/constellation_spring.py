"""The shapeless layout — where a game's stars sit when it was dealt no template.

A team-anchored force-directed solve: each team's most-connected leg is pinned to
its side, and a |ρ|-weighted networkx spring places the rest, warm-started in a
left-right basin under a fixed seed so the same game always draws the same map.
:func:`star_positions` is the figure's one entry point — a game with a template
goes to ``constellation_slate.template_positions`` instead, and everything about
that path (supernodes, vertex assignment, the engraving) lives there.
"""

from __future__ import annotations

import networkx as nx

from sportstradamus.dashboard.components.constellation_slate import template_positions

# Force-directed layout: each team's hub is pinned to its side, edges place the rest.
_LAYOUT_SEED = 17
_LAYOUT_TARGET = 1.1  # rescale radius — keeps stars + their top labels inside the frame
_ANCHOR_X = (-1.0, 1.0)  # team[0] pinned left, team[1] pinned right
_WARM_INSET = 0.6  # free nodes warm-start inside the anchors so springs can move them


def star_positions(
    nodes: list[str],
    node_team: dict[str, str | None],
    teams: list[str],
    edges: list[tuple[str, str, float]],
    template: dict | None,
    scale: tuple[float, float],
) -> tuple[dict[str, tuple[float, float]], list[int]]:
    """Star positions, plus the template vertices no star filled.

    Without a template this is the spring layout and an empty filler list — the
    original behavior, untouched; with one, ``constellation_slate`` places the
    stars on the shape.
    """
    if template is None:
        return _layout(nodes, node_team, teams, edges), []
    return template_positions(nodes, node_team, teams, edges, template, scale)


def _layout(
    nodes: list[str],
    node_team: dict[str, str | None],
    teams: list[str],
    edges: list[tuple[str, str, float]],
) -> dict[str, tuple[float, float]]:
    """Team-anchored force-directed positions.

    Each team's most-connected node is pinned to its side; a weighted spring layout
    (|ρ| edges) then places the rest, so cross-team correlations pull stars toward
    the centre and an unrepresented team leaves an empty half. Deterministic: a
    team-biased warm start over every node plus a fixed seed (no random init).
    """
    if not nodes:
        return {}  # nx.spring_layout cannot warm-start an empty graph
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)
    anchors = _anchors(graph, node_team, teams)
    init = _initial_positions(nodes, node_team, teams) | anchors
    pos = nx.spring_layout(
        graph, pos=init, fixed=list(anchors) or None, weight="weight", seed=_LAYOUT_SEED
    )
    return _rescale(pos)


def _anchors(
    graph: nx.Graph, node_team: dict[str, str | None], teams: list[str]
) -> dict[str, tuple[float, float]]:
    """Pin each team's most-connected node to its side (left / right)."""
    strength = graph.degree(weight="weight")
    pinned: dict[str, tuple[float, float]] = {}
    for team, x in zip(teams, _ANCHOR_X, strict=False):
        members = [n for n in graph.nodes if node_team.get(n) == team]
        if members:
            pinned[max(members, key=lambda n: (strength[n], n))] = (x, 0.0)
    return pinned


def _initial_positions(
    nodes: list[str], node_team: dict[str, str | None], teams: list[str]
) -> dict[str, tuple[float, float]]:
    """Team-biased warm start: team[0] left, team[1] right, the rest centre.

    Seeds the spring layout in a left-right basin (and makes it deterministic)
    instead of leaving free nodes to a seed-dependent tangle.
    """
    side = dict(zip(teams, _ANCHOR_X, strict=False))
    init: dict[str, tuple[float, float]] = {}
    for x in (*_ANCHOR_X, 0.0):
        members = [n for n in sorted(nodes) if side.get(node_team.get(n), 0.0) == x]
        for n, y in zip(members, _spread(len(members)), strict=True):
            init[n] = (x * _WARM_INSET, y * _WARM_INSET)
    return init


def _spread(n: int) -> list[float]:
    """``n`` y-coordinates evenly spaced top→bottom in [-1, 1] (centre when n == 1)."""
    if n <= 1:
        return [0.0] * n
    return [1.0 - 2.0 * i / (n - 1) for i in range(n)]


def _rescale(pos: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Center on the bbox midpoint, then uniformly scale to fill the frame (radius ``_LAYOUT_TARGET``).

    Centering first keeps a one-sided game (all stars on one team, or a lopsided web)
    from piling against a frame edge — scaling about the raw origin leaves an off-center
    cloud off-center. A normal two-team game is already ~symmetric about its anchors, so
    the shift is near-zero there.
    """
    if not pos:
        return {}
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    centered = {k: (x - cx, y - cy) for k, (x, y) in pos.items()}
    span = max((max(abs(x), abs(y)) for x, y in centered.values()), default=0.0)
    if span == 0.0:
        return dict.fromkeys(pos, (0.0, 0.0))
    factor = _LAYOUT_TARGET / span
    return {k: (x * factor, y * factor) for k, (x, y) in centered.items()}
