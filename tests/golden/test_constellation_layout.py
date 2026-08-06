"""Golden pins for the constellation layout engine — clustering and classification.

``cluster_players`` collapses one player's tightly-correlated legs into a single
supernode so a three-leg knot occupies one vertex instead of eating three, and
``topology_class`` reads the collapsed graph's shape so a game gets dealt a
template that matches how its legs actually hang together.

Every threshold arrives as an argument. These fixtures run at the shipped tuning
values, and the sensitivity pin proves a JSON edit really does move the verdict
with no code involved — the whole premise of the D6 cockpit.
"""

from __future__ import annotations

import itertools

import pytest

from sportstradamus.dashboard.components import constellation_shapes as cs
from sportstradamus.dashboard.components.constellation_layout import (
    cluster_players,
    collapse_edges,
    topology_class,
)

TUNING = cs.tuning()

_W = 0.5  # uniform edge weight; these fixtures test shape, not weighting


def _legs(prefix: str, count: int) -> list[str]:
    """``count`` distinct players, one leg each, so nothing clusters."""
    return [f"{prefix}{i}|PTS|Over" for i in range(count)]


def _star():
    """1 center + 6 spokes. Spokes straddle both teams so the cross-team share
    stays well above the twin threshold — otherwise twin would fire first and the
    fixture would prove nothing about the hub branch."""
    nodes = _legs("P", 7)
    edges = [(nodes[0], nodes[i], _W) for i in range(1, 7)]
    teams = {node: ("AAA" if i <= 3 else "BBB") for i, node in enumerate(nodes)}
    return nodes, edges, teams


def _path():
    """A 6-node path with teams alternating along it, again to keep twin out."""
    nodes = _legs("P", 6)
    edges = [(nodes[i], nodes[i + 1], _W) for i in range(5)]
    teams = {node: ("AAA" if i % 2 == 0 else "BBB") for i, node in enumerate(nodes)}
    return nodes, edges, teams


def _two_lobes():
    """Two 4-cliques, one per team, joined by a single weak bridge."""
    left, right = _legs("L", 4), _legs("R", 4)
    edges = [(a, b, 0.6) for a, b in itertools.combinations(left, 2)]
    edges += [(a, b, 0.6) for a, b in itertools.combinations(right, 2)]
    edges.append((left[0], right[0], 0.1))
    teams = {node: "AAA" for node in left} | {node: "BBB" for node in right}
    return left + right, edges, teams


def _clique(count: int = 6):
    nodes = _legs("P", count)
    edges = [(a, b, _W) for a, b in itertools.combinations(nodes, 2)]
    teams = {node: ("AAA" if i % 2 == 0 else "BBB") for i, node in enumerate(nodes)}
    return nodes, edges, teams


def _sparse_web():
    """6-cycle plus two chords: dense enough to read mesh at 0.45, sparse enough
    that a plausible in-bounds retune (0.60) can push it back off."""
    nodes = _legs("P", 6)
    ring = [(nodes[i], nodes[(i + 1) % 6], _W) for i in range(6)]
    edges = ring + [(nodes[0], nodes[3], _W), (nodes[1], nodes[4], _W)]
    teams = {node: ("AAA" if i % 2 == 0 else "BBB") for i, node in enumerate(nodes)}
    return nodes, edges, teams


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        (_star, "hub"),
        (_path, "chain"),
        (_two_lobes, "twin"),
        (_clique, "mesh"),
        (_sparse_web, "mesh"),
    ],
)
def test_topology_cascade_at_shipped_tuning(fixture, expected):
    nodes, edges, teams = fixture()
    label, _ = topology_class(nodes, teams, edges, TUNING)
    assert label == expected


def test_a_graph_too_small_to_have_a_shape_is_generic():
    nodes = _legs("P", 2)
    teams = {nodes[0]: "AAA", nodes[1]: "BBB"}
    label, readings = topology_class(nodes, teams, [(nodes[0], nodes[1], 0.9)], TUNING)
    assert label == "generic"
    assert readings["n"] == 2


def test_nodes_with_no_edges_are_generic():
    nodes = _legs("P", 6)
    teams = dict.fromkeys(nodes, "AAA")
    label, readings = topology_class(nodes, teams, [], TUNING)
    assert label == "generic"
    assert readings["density"] == 0.0 and readings["mean_degree"] == 0.0


def test_readings_match_hand_computed_metrics_for_the_star():
    nodes, edges, teams = _star()
    _, readings = topology_class(nodes, teams, edges, TUNING)
    assert set(readings) == {
        "n",
        "cross_share",
        "top_share",
        "mean_degree",
        "diameter_frac",
        "density",
    }
    assert readings["n"] == 7
    # 3 of the 6 spokes sit on the far team.
    assert readings["cross_share"] == pytest.approx(0.5)
    # The center carries 6 of the 12 endpoint-weights.
    assert readings["top_share"] == pytest.approx(0.5)
    assert readings["mean_degree"] == pytest.approx(12 / 7)
    assert readings["diameter_frac"] == pytest.approx(2 / 7)
    assert readings["density"] == pytest.approx(12 / 42)


def test_a_tuning_edit_moves_the_verdict_with_no_code_change():
    """The cockpit's premise: raise mesh_density past a game's reading and that
    game stops being a mesh."""
    nodes, edges, teams = _sparse_web()
    _, readings = topology_class(nodes, teams, edges, TUNING)
    assert readings["density"] == pytest.approx(16 / 30)

    retuned = {**TUNING, "mesh_density": 0.60}
    assert topology_class(nodes, teams, edges, retuned)[0] == "generic"


def test_a_lower_hub_bar_promotes_a_graph_that_missed_it():
    nodes, edges, teams = _sparse_web()
    retuned = {**TUNING, "hub_top_share": 0.15}
    assert topology_class(nodes, teams, edges, retuned)[0] == "hub"


def test_one_players_tight_legs_collapse_to_a_single_supernode():
    legs = ["A|PTS|Over", "A|PRA|Over", "A|AST|Over"]
    edges = [(legs[0], legs[1], 0.7), (legs[1], legs[2], 0.6), (legs[0], legs[2], 0.65)]
    clusters = cluster_players(legs, edges, TUNING["cluster_rho"])
    assert clusters == {"A|AST|Over+A|PRA|Over+A|PTS|Over": sorted(legs)}


def test_a_players_loosely_related_legs_stay_apart():
    legs = ["A|PTS|Over", "A|AST|Over"]
    clusters = cluster_players(legs, [(legs[0], legs[1], 0.1)], TUNING["cluster_rho"])
    assert clusters == {leg: [leg] for leg in legs}


def test_two_players_never_merge_however_correlated():
    """The supernode exists to stop one player eating three vertices. Two players
    stacking is exactly the structure the star map is supposed to show."""
    legs = ["A|PTS|Over", "B|REB|Over"]
    clusters = cluster_players(legs, [(legs[0], legs[1], 0.95)], TUNING["cluster_rho"])
    assert clusters == {leg: [leg] for leg in legs}


def test_a_partially_tight_player_splits_into_two_supernodes():
    legs = ["A|PTS|Over", "A|PRA|Over", "A|BLK|Over"]
    edges = [(legs[0], legs[1], 0.8), (legs[0], legs[2], 0.05), (legs[1], legs[2], 0.02)]
    clusters = cluster_players(legs, edges, TUNING["cluster_rho"])
    assert clusters == {
        "A|PRA|Over+A|PTS|Over": ["A|PRA|Over", "A|PTS|Over"],
        "A|BLK|Over": ["A|BLK|Over"],
    }


def test_collapsed_edge_takes_the_strongest_member_pair():
    """Averaging would bury the adjacency signal clustering exists to expose."""
    legs = ["A|PTS|Over", "A|PRA|Over", "B|REB|Over"]
    edges = [
        (legs[0], legs[1], 0.9),  # inside the A knot, disappears on collapse
        (legs[0], legs[2], 0.2),
        (legs[1], legs[2], 0.55),
    ]
    clusters = cluster_players(legs, edges, TUNING["cluster_rho"])
    collapsed = collapse_edges(edges, clusters)
    assert collapsed == [("A|PRA|Over+A|PTS|Over", "B|REB|Over", 0.55)]


def test_clustering_and_classification_are_deterministic():
    nodes, edges, teams = _two_lobes()
    shuffled = list(reversed(edges))
    first = cluster_players(nodes, edges, TUNING["cluster_rho"])
    assert first == cluster_players(list(reversed(nodes)), shuffled, TUNING["cluster_rho"])
    assert collapse_edges(edges, first) == collapse_edges(shuffled, first)
    assert topology_class(nodes, teams, edges, TUNING) == topology_class(
        list(reversed(nodes)), teams, shuffled, TUNING
    )
