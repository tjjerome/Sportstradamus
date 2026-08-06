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
import math

import pytest

from sportstradamus.dashboard.components import constellation_shapes as cs
from sportstradamus.dashboard.components.constellation_layout import (
    _EXPLODE_DR,
    _EXPLODE_R0,
    assign_stars,
    cluster_players,
    collapse_edges,
    explode_clusters,
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
    teams = dict.fromkeys(left, "AAA") | dict.fromkeys(right, "BBB")
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
    edges = [*ring, (nodes[0], nodes[3], _W), (nodes[1], nodes[4], _W)]
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


def _template(vertices):
    """A bare template — assign_stars reads geometry only, never the decoration."""
    return {"vertices": vertices, "min_nodes": 2}


def _vertex(vid, x, y, side, prominence):
    return {"id": vid, "x": x, "y": y, "side": side, "prominence": prominence}


_MIRRORED = _template(
    [
        _vertex(0, -0.9, 0.9, "L", 1),
        _vertex(1, -0.9, 0.0, "L", 3),
        _vertex(2, -0.9, -0.9, "L", 5),
        _vertex(3, 0.9, 0.9, "R", 2),
        _vertex(4, 0.9, 0.0, "R", 4),
        _vertex(5, 0.9, -0.9, "R", 6),
    ]
)

# One left vertex against three right ones at deliberately unequal distances, so
# a correlation pull and a prominence ranking disagree about where to put B.
_LOPSIDED = _template(
    [
        _vertex(0, -0.9, 0.0, "L", 1),
        _vertex(1, 0.1, 0.0, "R", 3),
        _vertex(2, 0.9, 0.9, "R", 1),
        _vertex(3, 0.9, -0.9, "R", 2),
    ]
)


def _two_by_two():
    nodes = ["A1|PTS|Over", "A2|PTS|Over", "B1|PTS|Over", "B2|PTS|Over"]
    teams = {nodes[0]: "AAA", nodes[1]: "AAA", nodes[2]: "BBB", nodes[3]: "BBB"}
    edges = [(nodes[0], nodes[2], 0.8), (nodes[1], nodes[3], 0.2)]
    return nodes, teams, edges


def test_a_star_never_lands_on_the_other_teams_half():
    """The one thing the layout is not allowed to lie about."""
    nodes, teams, edges = _two_by_two()
    positions, _ = assign_stars(nodes, teams, ["AAA", "BBB"], edges, _MIRRORED)
    for node, (x, _) in positions.items():
        assert (x < 0) == (teams[node] == "AAA"), node


def test_placement_is_deterministic_whatever_order_the_inputs_arrive_in():
    nodes, teams, edges = _two_by_two()
    first = assign_stars(nodes, teams, ["AAA", "BBB"], edges, _MIRRORED)
    assert first == assign_stars(nodes, teams, ["AAA", "BBB"], edges, _MIRRORED)
    assert first == assign_stars(
        list(reversed(nodes)), teams, ["AAA", "BBB"], list(reversed(edges)), _MIRRORED
    )


def test_a_correlated_star_sits_near_its_partner_even_over_a_showier_vertex():
    """Adjacency beats prominence: the gold edge should stay short."""
    nodes = ["A|PTS|Over", "B|PTS|Over"]
    teams = {nodes[0]: "AAA", nodes[1]: "BBB"}
    positions, _ = assign_stars(nodes, teams, ["AAA", "BBB"], [(*nodes, 0.8)], _LOPSIDED)
    assert positions["B|PTS|Over"] == (0.1, 0.0)  # nearest, and the least prominent


def test_prominence_decides_when_nothing_is_correlated():
    nodes = ["A|PTS|Over", "B|PTS|Over"]
    teams = {nodes[0]: "AAA", nodes[1]: "BBB"}
    positions, _ = assign_stars(nodes, teams, ["AAA", "BBB"], [], _LOPSIDED)
    assert positions["B|PTS|Over"] == (0.9, 0.9)  # the most prominent right vertex


def test_a_single_team_game_leaves_the_far_half_to_fillers():
    """ARI/ARI is real in this data. The empty half is the truth about the game."""
    nodes = ["A1|PTS|Over", "A2|PTS|Over"]
    teams = dict.fromkeys(nodes, "ARI")
    positions, fillers = assign_stars(nodes, teams, ["ARI"], [], _MIRRORED)
    assert all(x < 0 for x, _ in positions.values())
    assert {3, 4, 5} <= set(fillers)


def test_overflow_orbits_its_own_side_instead_of_stretching_the_template():
    nodes = [f"A{i}|PTS|Over" for i in range(5)]
    teams = dict.fromkeys(nodes, "AAA")
    positions, _ = assign_stars(nodes, teams, ["AAA", "BBB"], [], _MIRRORED)
    assert len(positions) == 5
    assert all(x <= 0.0 for x, _ in positions.values()), "overflow never crosses the axis"


def test_unfilled_vertices_come_back_as_fillers_so_the_shape_still_reads():
    nodes = ["A|PTS|Over", "B|PTS|Over"]
    teams = {nodes[0]: "AAA", nodes[1]: "BBB"}
    positions, fillers = assign_stars(nodes, teams, ["AAA", "BBB"], [], _MIRRORED)
    assert len(fillers) == 4
    assert set(fillers).isdisjoint(
        vertex["id"]
        for vertex in _MIRRORED["vertices"]
        if (vertex["x"], vertex["y"]) in positions.values()
    )


def test_a_lone_leg_sits_exactly_on_its_vertex():
    exploded = explode_clusters({"A|PTS|Over": (0.4, -0.2)}, {"A|PTS|Over": ["A|PTS|Over"]})
    assert exploded == {"A|PTS|Over": (0.4, -0.2)}


def test_a_knot_rings_its_vertex_tightly_enough_to_read_as_one_point():
    members = ["A|AST|Over", "A|PRA|Over", "A|PTS|Over"]
    exploded = explode_clusters({"+".join(members): (0.5, 0.5)}, {"+".join(members): members})
    radius = _EXPLODE_R0 + _EXPLODE_DR  # 3 members: one step past the 2-member ring
    assert len(exploded) == 3
    for point in exploded.values():
        assert math.dist(point, (0.5, 0.5)) == pytest.approx(radius)
    # Ring starts at -90 degrees, members in sorted order.
    assert exploded[members[0]] == pytest.approx((0.5, 0.5 - radius))


def test_a_knot_beside_the_axis_clamps_rather_than_spilling_onto_the_wrong_team():
    members = ["A|AST|Over", "A|PRA|Over", "A|PTS|Over"]
    key = "+".join(members)
    exploded = explode_clusters({key: (-0.02, 0.0)}, {key: members})
    assert all(x <= 0.0 for x, _ in exploded.values())
    assert min(x for x, _ in exploded.values()) < -0.02, (
        "the knot still opens up, it just can't cross"
    )


def test_clustering_and_classification_are_deterministic():
    nodes, edges, teams = _two_lobes()
    shuffled = list(reversed(edges))
    first = cluster_players(nodes, edges, TUNING["cluster_rho"])
    assert first == cluster_players(list(reversed(nodes)), shuffled, TUNING["cluster_rho"])
    assert collapse_edges(edges, first) == collapse_edges(shuffled, first)
    assert topology_class(nodes, teams, edges, TUNING) == topology_class(
        list(reversed(nodes)), teams, shuffled, TUNING
    )
