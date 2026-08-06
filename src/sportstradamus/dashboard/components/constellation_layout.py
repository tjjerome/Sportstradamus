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

import networkx as nx

# Fewest supernodes that can carry a shape at all. Below this the cascade answers
# "generic" before measuring anything — three points describe a triangle, not an
# object, and every metric below is noise at n=2.
_MIN_CLASSIFIABLE_NODES = 4

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
