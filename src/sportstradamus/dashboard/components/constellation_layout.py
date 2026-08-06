"""How a game's correlation graph becomes star positions on a shape template.

The catalog in :mod:`constellation_shapes` says what the shapes *are*; this
module decides which one a game gets and where its legs land on it. Two steps
happen here, both pure functions over plain data so they test without a browser:

* **Clustering.** Three legs on one player, wired to each other at ρ 0.7, are one
  idea, not three. They collapse into a supernode that occupies a single vertex
  and gets exploded back into a tight knot at render time — otherwise a single
  hot player eats half a template and the shape stops reading.
* **Classification.** A game whose legs hang off one player looks nothing like a
  game that stacks two lineups against each other. Naming that difference is what
  lets the assigner deal a radial template to the first and a two-lobed one to
  the second.

Every threshold arrives as an argument (production passes
:func:`constellation_shapes.tuning`, tests pass explicit values), so retuning the
engine is a JSON edit and a browser rerun.
"""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx

# Fewest supernodes that can carry a shape at all. Below this the cascade answers
# "generic" before measuring anything — three points describe a triangle, not an
# object, and every metric below is noise at n=2.
_MIN_CLASSIFIABLE_NODES = 4

# Knot geometry. A 2-member cluster rings its vertex at _EXPLODE_R0; each further
# member widens it by _EXPLODE_DR up to _EXPLODE_RMAX, past which the knot stops
# reading as one bright point and starts reading as separate stars. The ceiling
# also keeps members apart at the 22px mobile touch floor.
_EXPLODE_R0, _EXPLODE_DR, _EXPLODE_RMAX = 0.05, 0.015, 0.11

# How far an overflowed star drifts from its side's centroid. Small enough to stay
# clear of the axis, large enough that two overflows don't stack.
_OVERFLOW_RADIUS = 0.18
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

_ZERO_READINGS = {
    "n": 0,
    "cross_share": 0.0,
    "top_share": 0.0,
    "mean_degree": 0.0,
    "diameter_frac": 0.0,
    "density": 0.0,
}


def cluster_players(
    nodes: list[str],
    edges: list[tuple[str, str, float]],
    rho_min: float,
) -> dict[str, list[str]]:
    """Map supernode key -> its member leg keys, sorted.

    Leg keys are ``"Player|Market|Bet"``. Same-player legs merge along pairwise
    ``|rho| >= rho_min`` connected components; different players never merge, however
    correlated, because two players stacking is exactly the structure the star map
    exists to show. Singletons pass through under their own key.
    """
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    for node_a, node_b, rho in edges:
        if abs(rho) >= rho_min and node_a.split("|")[0] == node_b.split("|")[0]:
            graph.add_edge(node_a, node_b)
    clusters = {}
    for component in nx.connected_components(graph):
        members = sorted(component)
        clusters["+".join(members)] = members
    return clusters


def collapse_edges(
    edges: list[tuple[str, str, float]],
    clusters: dict[str, list[str]],
) -> list[tuple[str, str, float]]:
    """Rewrite ``edges`` onto supernodes, keeping the strongest member pair.

    Averaging instead would bury the adjacency signal clustering exists to
    expose. Edges that land inside one supernode disappear — they are the knot.
    """
    supernode_of = {member: key for key, members in clusters.items() for member in members}
    strongest: dict[tuple[str, str], float] = {}
    for node_a, node_b, rho in edges:
        pair = tuple(sorted((supernode_of[node_a], supernode_of[node_b])))
        if pair[0] != pair[1]:
            strongest[pair] = max(strongest.get(pair, 0.0), abs(rho))
    return [(a, b, weight) for (a, b), weight in sorted(strongest.items())]


def _graph_readings(
    supernodes: list[str],
    node_team: dict[str, str | None],
    edges: list[tuple[str, str, float]],
) -> dict:
    """The six raw metrics behind a topology verdict.

    Surfaced in the Games tuning expander so thresholds get set against observed
    values rather than guesses.
    """
    count = len(supernodes)
    if not edges or count < 2:
        return {**_ZERO_READINGS, "n": count}

    graph = nx.Graph()
    graph.add_nodes_from(sorted(supernodes))
    incident = dict.fromkeys(supernodes, 0.0)
    total = cross = 0.0
    for node_a, node_b, weight in edges:
        graph.add_edge(node_a, node_b)
        incident[node_a] += weight
        incident[node_b] += weight
        total += weight
        if node_team.get(node_a) != node_team.get(node_b):
            cross += weight

    diameter = max(nx.diameter(graph.subgraph(part)) for part in nx.connected_components(graph))
    return {
        "n": count,
        "cross_share": cross / total,
        "top_share": max(incident.values()) / (2 * total),
        "mean_degree": 2 * graph.number_of_edges() / count,
        "diameter_frac": diameter / count,
        "density": 2 * graph.number_of_edges() / (count * (count - 1)),
    }


def topology_class(
    supernodes: list[str],
    node_team: dict[str, str | None],
    edges: list[tuple[str, str, float]],
    cfg: dict,
) -> tuple[str, dict]:
    """Classify a collapsed game graph as ``hub``/``chain``/``twin``/``mesh``/``generic``.

    First match wins, and the order is deliberate: twin before hub, because a
    two-cluster graph usually *contains* a local hub per side and the two-lobe read
    is the truer one; hub before chain, because a star trivially satisfies low mean
    degree; mesh last, its being the densest signature and the hardest to fake.
    ``generic`` is the no-match answer — the game keeps the spring layout.
    """
    readings = _graph_readings(supernodes, node_team, edges)
    if readings["n"] < _MIN_CLASSIFIABLE_NODES or not edges:
        return "generic", readings

    two_teams = len({node_team.get(node) for node in supernodes}) > 1
    if two_teams and readings["cross_share"] < cfg["twin_cross_share"]:
        return "twin", readings
    if readings["top_share"] >= cfg["hub_top_share"]:
        return "hub", readings
    if (
        readings["mean_degree"] <= cfg["chain_mean_degree"]
        and readings["diameter_frac"] >= cfg["chain_diameter_frac"]
    ):
        return "chain", readings
    if readings["density"] >= cfg["mesh_density"]:
        return "mesh", readings
    return "generic", readings


def _clamp_to_side(x: float, reference: float) -> float:
    """Hold ``x`` on ``reference``'s side of the team axis.

    A star that drifts across x=0 reads as the other team's, which is the one
    thing the layout is not allowed to lie about. Stars whose reference sits on
    the axis (a centre vertex) are free to move either way.
    """
    if reference < 0:
        return min(x, 0.0)
    if reference > 0:
        return max(x, 0.0)
    return x


def _scatter(nodes: list[str], center: tuple[float, float]) -> dict[str, tuple[float, float]]:
    """Golden-angle spiral around ``center`` for supernodes their side had no vertex for.

    Overflow is not an error — a template is never stretched to fit a busy game
    (it has to stay visibly the same shape across games), so the extras orbit
    their own side instead.
    """
    placed = {}
    for index, node in enumerate(nodes):
        radius = _OVERFLOW_RADIUS * math.sqrt((index + 1) / (len(nodes) + 1))
        angle = index * _GOLDEN_ANGLE
        x = center[0] + radius * math.cos(angle)
        placed[node] = (_clamp_to_side(x, center[0]), center[1] + radius * math.sin(angle))
    return placed


def _vertex_pools(vertices: dict[int, dict], teams: list[str]) -> dict[str | None, list[int]]:
    """Vertex ids each side may draw from, most prominent first.

    Centre vertices sit in every pool rather than being reserved: a template's
    spine is often its most important star (a dartboard's bullseye is prominence
    1), so withholding it from both teams would leave the shape headless. Sides
    draw from a shared used-set, so a contested centre goes to whoever picks
    first — and the caller lets the scarcer team pick first.
    """
    by_side = defaultdict(list)
    for vid in sorted(vertices, key=lambda v: (vertices[v]["prominence"], v)):
        by_side[vertices[vid]["side"]].append(vid)
    pools: dict[str | None, list[int]] = {
        side: by_side[side] + by_side["C"] for side in ("L", "R")[: len(teams)]
    }
    pools[None] = by_side["C"] + by_side["L"] + by_side["R"]
    return pools


def _placement_cost(
    candidate_xy: tuple[float, float],
    node: str,
    rho: dict[frozenset, float],
    positions: dict[str, tuple[float, float]],
) -> float:
    """Σ |ρ| · distance from ``node`` to every star already placed, at a candidate vertex."""
    return sum(
        rho.get(frozenset((node, other)), 0.0) * math.dist(candidate_xy, place)
        for other, place in positions.items()
    )


def assign_stars(
    supernodes: list[str],
    node_team: dict[str, str | None],
    teams: list[str],
    edges: list[tuple[str, str, float]],
    template: dict,
) -> tuple[dict[str, tuple[float, float]], list[int]]:
    """Lay a game's supernodes onto a template's vertices.

    Returns ``(positions, filler_vertex_ids)`` — every supernode placed, plus the
    vertices nobody filled, which the renderer draws as faint stars so the shape
    still reads when a game is thinner than its template. Loose by design: a
    single-team game leaves the far side entirely to fillers, and that empty half
    is the truth about the game.

    Placement is greedy over weighted degree. The busiest supernode on a side goes
    first onto that side's most prominent free vertex, then each next one takes the
    free vertex minimising ``Σ |ρ| · distance`` to what is already placed, so
    strongly-tied stars end up near each other and their gold edge stays short.
    Prominence breaks ties, which means it decides outright for an uncorrelated
    game. No randomness anywhere — the same game lays out the same way every
    rerun.
    """
    vertices = {vertex["id"]: vertex for vertex in template["vertices"]}
    xy = {vid: (vertex["x"], vertex["y"]) for vid, vertex in vertices.items()}
    side_of_team = dict(zip(teams, ("L", "R"), strict=False))

    strength: dict[str, float] = dict.fromkeys(supernodes, 0.0)
    rho: dict[frozenset, float] = {}
    for node_a, node_b, weight in edges:
        strength[node_a] += abs(weight)
        strength[node_b] += abs(weight)
        rho[frozenset((node_a, node_b))] = abs(weight)

    members: dict[str | None, list[str]] = defaultdict(list)
    for node in sorted(supernodes):
        members[side_of_team.get(node_team.get(node))].append(node)

    pools = _vertex_pools(vertices, teams)
    positions: dict[str, tuple[float, float]] = {}
    used: set[int] = set()
    # Scarcest side first, so a contested centre vertex goes where it is needed most.
    for side in sorted(members, key=lambda s: (len(pools.get(s, [])) - len(members[s]), s or "")):
        overflow = []
        for node in sorted(members[side], key=lambda n: (-strength[n], n)):
            free = [vid for vid in pools.get(side, []) if vid not in used]
            if not free:
                overflow.append(node)
                continue
            best = min(
                free,
                key=lambda vid: (
                    _placement_cost(xy[vid], node, rho, positions),
                    vertices[vid]["prominence"],
                    vid,
                ),
            )
            used.add(best)
            positions[node] = xy[best]
        positions |= _scatter(overflow, _side_centroid(xy, vertices, side))

    return positions, sorted(set(vertices) - used)


def _side_centroid(
    xy: dict[int, tuple[float, float]], vertices: dict[int, dict], side: str | None
) -> tuple[float, float]:
    """Where a side's overflow orbits: the mean of its own vertices, or the origin."""
    own = [xy[vid] for vid, vertex in vertices.items() if vertex["side"] == side]
    if not own:
        return (0.0, 0.0)
    return (sum(x for x, _ in own) / len(own), sum(y for _, y in own) / len(own))


def explode_clusters(
    positions: dict[str, tuple[float, float]],
    clusters: dict[str, list[str]],
) -> dict[str, tuple[float, float]]:
    """Expand each supernode back into its member legs.

    A singleton sits exactly on its vertex. Two or more ring it, starting at -90°
    and equally spaced, tight enough to read as one bright point while every star
    stays its own leg you can tap. A knot beside the team axis clamps rather than
    spilling a leg onto the wrong team's half.
    """
    exploded = {}
    for key, members in clusters.items():
        origin = positions[key]
        if len(members) == 1:
            exploded[members[0]] = origin
            continue
        radius = min(_EXPLODE_R0 + _EXPLODE_DR * (len(members) - 2), _EXPLODE_RMAX)
        for index, member in enumerate(sorted(members)):
            angle = -math.pi / 2 + 2 * math.pi * index / len(members)
            exploded[member] = (
                _clamp_to_side(origin[0] + radius * math.cos(angle), origin[0]),
                origin[1] + radius * math.sin(angle),
            )
    return exploded
