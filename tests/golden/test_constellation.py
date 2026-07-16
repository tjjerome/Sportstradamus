"""Pins for the dashboard constellation — the DESIGN §4a static team-anchored star map.

``constellation_figure`` is pure (networkx force layout + plotly draw, no Streamlit).
The node set is the game's **model-liked** legs (Kelly ``K`` > 0), fixed per game —
selecting a leg never moves a star, it only lights it up. A slip leg renders active
(full team color, opacity 1, labelled); a candidate renders desaturated + dim,
labelled like the rest. Star size ∝ Kelly edge. Each team's hub anchors to its side. Every tie is a
gold edge (width/opacity ∝ |ρ|, dashed when ρ < 0) drawn as a faint base web that
brightens when both its stars are in the slip. These assert that grammar, the static layout,
the size/selection encodings, the hover, the per-edge endpoint ``meta``, and the
click-key + card-field ``customdata`` the in-app editor and its JS card rely on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.dashboard.components.constellation import (
    _DEEP_ALPHA,
    _EDGE_BASE_ALPHA,
    _INACTIVE_ALPHA,
    _LABEL_FONT_SIZE_MOBILE,
    _SIZE_MAX,
    _SIZE_MIN_MOBILE,
    _WIDER_SCALE,
    constellation_figure,
)
from sportstradamus.dashboard.theme import GOLD, GRAY, team_colors

_TEAMS = ("NYK", "SAS")  # sorted -> NYK anchors left (-x), SAS right (+x); real NBA codes


def _row(i: int, key: str, k: float) -> dict:
    player, market, bet = key.split("|")
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": 10.5,
        "Game": "NYK/SAS",
        "League": "NBA",
        "Team": _TEAMS[i % 2],
        "Kelly": k,
        "Win Prob": 0.6,
        "Boost": 1.5,
    }


def _slip(*keys: str) -> list[dict]:
    return [_row(i, key, 0.3) for i, key in enumerate(keys)]


def _pool(*specs) -> pd.DataFrame:
    rows = [
        _row(i, s[0], s[1]) if isinstance(s, tuple) else _row(i, s, 0.2)
        for i, s in enumerate(specs)
    ]
    return pd.DataFrame(rows)


def _corr(*triples: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"League": "NBA", "Game": "NYK/SAS", "leg_a": a, "leg_b": b, "rho": r}
            for a, b, r in triples
        ]
    )


def _trace(fig, name: str):
    return next((t for t in fig.data if getattr(t, "name", None) == name), None)


def _node_traces(fig) -> list:
    return [t for t in fig.data if t.mode and "markers" in t.mode]


def _edge_traces(fig) -> list:
    return [t for t in fig.data if t.mode == "lines"]


def _shown_keys(fig) -> set:
    return {cd[0] for t in _node_traces(fig) for cd in t.customdata}


def _node_pos(fig) -> dict:
    pos = {}
    for t in _node_traces(fig):
        for cd, x, y in zip(t.customdata, t.x, t.y, strict=True):
            pos[cd[0]] = (round(float(x), 5), round(float(y), 5))
    return pos


def _sizes(fig) -> dict:
    out = {}
    for t in _node_traces(fig):
        for cd, s in zip(t.customdata, t.marker.size, strict=True):
            out[cd[0]] = s
    return out


def test_universe_is_only_k_positive_legs():
    legs = _slip("A|PTS|Over")
    pool = _pool(
        ("A|PTS|Over", 0.4), ("B|REB|Under", 0.1), ("C|AST|Over", -0.2), ("D|PTS|Under", 0.0)
    )
    fig = constellation_figure(legs, _corr(("A|PTS|Over", "B|REB|Under", 0.3)), pool)
    assert _shown_keys(fig) == {"A|PTS|Over", "B|REB|Under"}  # C (<0) and D (==0) dropped


def test_active_full_color_candidate_desaturated_and_dim():
    legs = _slip("A|PTS|Over")  # NYK, active
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))  # B = SAS candidate
    fig = constellation_figure(legs, _corr(("A|PTS|Over", "B|REB|Under", 0.3)), pool)
    active, cand = _trace(fig, "active"), _trace(fig, "candidate")
    nyk_primary = team_colors("NBA", "NYK")[0]
    assert active.marker.symbol == "star"
    assert active.marker.opacity == 1.0
    assert list(active.marker.color) == [nyk_primary]  # full NYK fill
    assert cand.marker.opacity == _INACTIVE_ALPHA
    assert cand.marker.color[0] != team_colors("NBA", "SAS")[0]  # blended toward gray, not raw


def test_two_team_fixture_resolves_two_distinct_team_primaries():
    legs = _slip("A|PTS|Over", "B|PTS|Over")  # A = NYK, B = SAS, both active
    pool = _pool(("A|PTS|Over", 0.4), ("B|PTS|Over", 0.3))
    fig = constellation_figure(legs, _corr(("A|PTS|Over", "B|PTS|Over", 0.5)), pool)
    active = _trace(fig, "active")
    colors = dict(zip((cd[0] for cd in active.customdata), active.marker.color, strict=True))
    assert colors["A|PTS|Over"] == team_colors("NBA", "NYK")[0]
    assert colors["B|PTS|Over"] == team_colors("NBA", "SAS")[0]
    assert colors["A|PTS|Over"] != colors["B|PTS|Over"]


def test_unknown_team_falls_back_to_gray():
    # A team code absent from team_assets.json for the league resolves to GRAY,
    # never a KeyError (theme.team_colors' own contract). "ZZZ" must be one of the
    # matchup's own two sides (derived from Game, not the row's Team field) so
    # team_colors("NBA", "ZZZ") is actually invoked, not just missed by the node
    # dict's own .get(..., GRAY) fallback.
    legs = [{**_row(0, "A|PTS|Over", 0.3), "Game": "NYK/ZZZ", "Team": "ZZZ"}]
    pool = pd.DataFrame([{**_row(0, "A|PTS|Over", 0.4), "Game": "NYK/ZZZ", "Team": "ZZZ"}])
    active = _trace(constellation_figure(legs, _corr(), pool), "active")
    assert active.marker.color[0] == GRAY


def test_no_node_marker_color_equals_gold():
    # Team fills are NEVER gold (DESIGN §4a) — gold is reserved for correlation edges.
    legs = _slip("A|PTS|Over", "B|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.3), ("B|PTS|Over", 0.2))
    fig = constellation_figure(legs, _corr(("A|PTS|Over", "B|PTS|Over", 0.5)), pool)
    for trace in _node_traces(fig):
        assert all(c != GOLD for c in trace.marker.color)


def test_active_and_candidate_stars_both_carry_labels():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))
    fig = constellation_figure(legs, _corr(("A|PTS|Over", "B|REB|Under", 0.3)), pool)
    assert list(_trace(fig, "active").text) == ["A PTS o10.5"]
    assert list(_trace(fig, "candidate").text) == ["B REB u10.5"]  # candidates now labelled too


def test_star_size_scales_with_edge():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.1))
    sizes = _sizes(constellation_figure(legs, _corr(("A|PTS|Over", "B|REB|Under", 0.3)), pool))
    assert sizes["A|PTS|Over"] > sizes["B|REB|Under"]  # higher K → bigger star
    assert sizes["A|PTS|Over"] == _SIZE_MAX  # the game's strongest leg maxes out


def test_layout_is_static_under_selection():
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))
    corr = _corr(("A|PTS|Over", "B|REB|Under", 0.5))
    one = _node_pos(constellation_figure(_slip("A|PTS|Over"), corr, pool))
    both = _node_pos(constellation_figure(_slip("A|PTS|Over", "B|REB|Under"), corr, pool))
    assert one == both  # selecting B moves no star — the map is fixed per game


def test_nodes_carry_leg_key_customdata():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4))
    assert [
        cd[0] for cd in _trace(constellation_figure(legs, _corr(), pool), "active").customdata
    ] == ["A|PTS|Over"]


def test_node_customdata_carries_card_fields():
    # customdata = [key, player, market, bet, line, win, boost, kelly] — the JS card
    # reads the fields after the key; the click path still reads customdata[0].
    fig = constellation_figure(_slip("A|PTS|Over"), _corr(), _pool(("A|PTS|Over", 0.4)))
    cd = _trace(fig, "active").customdata[0]
    assert cd[0] == "A|PTS|Over"
    assert [cd[1], cd[2], cd[3]] == ["A", "PTS", "Over"]
    assert float(cd[4]) == 10.5  # line
    assert 0.0 <= float(cd[5]) <= 1.0  # win prob
    assert float(cd[6]) > 0  # boost


def test_team_hubs_anchor_to_opposite_sides():
    legs = _slip("A|PTS|Over", "B|PTS|Over")  # A = NYK, B = SAS
    pool = _pool(("A|PTS|Over", 0.4), ("B|PTS|Over", 0.3))
    pos = _node_pos(constellation_figure(legs, _corr(("A|PTS|Over", "B|PTS|Over", 0.5)), pool))
    assert pos["A|PTS|Over"][0] < 0 < pos["B|PTS|Over"][0]  # NYK left, SAS right


def test_candidate_layout_anchors_from_pool_without_a_slip():
    # The map's game is read from the pool, so the field anchors each team to its
    # side before any leg is picked (and stays static once one is).
    pool = _pool(("A|PTS|Over", 0.4), ("B|PTS|Over", 0.3))  # A = NYK, B = SAS
    pos = _node_pos(constellation_figure([], _corr(("A|PTS|Over", "B|PTS|Over", 0.5)), pool))
    assert pos["A|PTS|Over"][0] < 0 < pos["B|PTS|Over"][0]


def test_layout_bbox_is_centered_on_origin():
    # A one-sided game (every leg on NYK, no SAS) piles the cloud against one anchor;
    # _rescale re-centers on the bbox midpoint so it never mats against a frame edge.
    rows = [
        {**_row(0, k, 0.3), "Team": "NYK"}
        for k in ("A|PTS|Over", "B|REB|Over", "C|AST|Over", "D|STL|Over")
    ]
    corr = _corr(
        ("A|PTS|Over", "B|REB|Over", 0.5),
        ("B|REB|Over", "C|AST|Over", 0.4),
        ("C|AST|Over", "D|STL|Over", 0.3),
    )
    pos = _node_pos(constellation_figure([], corr, pd.DataFrame(rows)))
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    assert abs((min(xs) + max(xs)) / 2) < 0.05
    assert abs((min(ys) + max(ys)) / 2) < 0.05


def _edge_by_pair(fig) -> dict:
    return {frozenset(e.meta): e for e in _edge_traces(fig)}


def test_all_ties_drawn_with_endpoint_meta_and_dashed_negative():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.3), ("C|AST|Over", 0.3))
    corr = _corr(
        ("A|PTS|Over", "B|REB|Under", -0.4),  # negative → dashed
        ("A|PTS|Over", "C|AST|Over", 0.1),
        ("B|REB|Under", "C|AST|Over", 0.2),
        ("A|PTS|Over", "C|AST|Over", 0.1),  # dupe ignored by the rho map
    )
    edges = _edge_by_pair(constellation_figure(legs, corr, pool))
    assert set(edges) == {  # every |rho|>=floor pair, not a strongest-tie backbone
        frozenset(("A|PTS|Over", "B|REB|Under")),
        frozenset(("A|PTS|Over", "C|AST|Over")),
        frozenset(("B|REB|Under", "C|AST|Over")),
    }
    assert edges[frozenset(("A|PTS|Over", "B|REB|Under"))].line.dash == "dot"  # negative
    assert edges[frozenset(("A|PTS|Over", "C|AST|Over"))].line.dash == "solid"  # positive


def test_slip_edges_brighten_over_the_faint_base_web():
    # Every tie is drawn at a faint base alpha (the base web); a tie whose BOTH endpoints
    # are in the slip brightens above it. A + B are in the slip, C is a candidate: A-B
    # brightens; A-C and B-C stay at the faint base.
    legs = _slip("A|PTS|Over", "B|REB|Under")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.3), ("C|AST|Over", 0.3))
    corr = _corr(
        ("A|PTS|Over", "B|REB|Under", 0.4),
        ("A|PTS|Over", "C|AST|Over", 0.3),
        ("B|REB|Under", "C|AST|Over", 0.5),
    )
    edges = _edge_by_pair(constellation_figure(legs, corr, pool))
    assert (
        edges[frozenset(("A|PTS|Over", "B|REB|Under"))].opacity > _EDGE_BASE_ALPHA
    )  # both in slip
    assert edges[frozenset(("A|PTS|Over", "C|AST|Over"))].opacity == _EDGE_BASE_ALPHA  # candidate
    assert edges[frozenset(("B|REB|Under", "C|AST|Over"))].opacity == _EDGE_BASE_ALPHA  # candidate


def test_single_slip_leg_leaves_only_the_faint_base_web():
    # One leg in the slip → no pair is fully in the slip → no edge brightens; every tie
    # sits at the faint base alpha (bright ties preview only on hover, client-side).
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.3))
    edges = _edge_traces(
        constellation_figure(legs, _corr(("A|PTS|Over", "B|REB|Under", 0.5)), pool)
    )
    assert edges and all(e.opacity == _EDGE_BASE_ALPHA for e in edges)


def test_empty_slip_shows_the_faint_base_web():
    # With nothing in the slip no edge brightens — but the whole web is still drawn at the
    # faint base alpha so the correlation structure reads before any pick.
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.3))
    edges = _edge_traces(constellation_figure([], _corr(("A|PTS|Over", "B|REB|Under", 0.5)), pool))
    assert edges  # the tie is present as a (base-web) trace
    assert all(e.opacity == _EDGE_BASE_ALPHA for e in edges)


def test_layout_is_deterministic():
    legs = _slip("A|PTS|Over", "B|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|PTS|Over", 0.3))
    corr = _corr(("A|PTS|Over", "B|PTS|Over", 0.3))
    assert _node_pos(constellation_figure(legs, corr, pool)) == _node_pos(
        constellation_figure(legs, corr, pool)
    )


def test_hover_shows_win_prob_boost_and_kelly():
    fig = constellation_figure(_slip("A|PTS|Over"), _corr(), _pool(("A|PTS|Over", 0.4)))
    hover = _trace(fig, "active").hovertext[0]
    assert "PTS" in hover and "Win" in hover and "Kelly" in hover


def test_active_leg_below_k_floor_still_shows():
    # An active leg whose offer slipped to K ≤ 0 keeps its star — a slip never loses a leg.
    fig = constellation_figure([_row(0, "A|PTS|Over", -0.1)], None, _pool(("A|PTS|Over", -0.1)))
    assert "A|PTS|Over" in _shown_keys(fig)


def test_no_legs_no_pool_is_blank():
    fig = constellation_figure([], None, None)
    assert _shown_keys(fig) == set()
    assert _edge_traces(fig) == []


def test_deep_pool_none_is_byte_stable_with_no_deep_trace():
    # Lenses off: no "deep" trace at all — the explicit regression guarantee.
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))
    fig = constellation_figure(_slip("A|PTS|Over"), _corr(), pool)
    assert _trace(fig, "deep") is None


def test_deep_pool_adds_dim_unconnected_stars():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))  # model-liked candidate pool
    deep_pool = _pool(("D|PTS|Under", -0.1), ("E|AST|Over", -0.3))  # model-passed
    fig = constellation_figure(legs, _corr(), pool, deep_pool=deep_pool)
    deep = _trace(fig, "deep")
    assert deep is not None
    assert {cd[0] for cd in deep.customdata} == {"D|PTS|Under", "E|AST|Over"}
    assert deep.marker.opacity == _DEEP_ALPHA
    assert deep.marker.color[0] not in (GRAY, GOLD)
    # No edge trace names either deep star as an endpoint.
    deep_keys = {cd[0] for cd in deep.customdata}
    for edge in _edge_traces(fig):
        assert not (set(edge.meta) & deep_keys)


def test_deep_pool_excludes_legs_already_in_slip():
    legs = _slip("D|PTS|Under")
    pool = _pool(("A|PTS|Over", 0.4))
    deep_pool = _pool(("D|PTS|Under", -0.1), ("E|AST|Over", -0.3))
    fig = constellation_figure(legs, _corr(), pool, deep_pool=deep_pool)
    deep = _trace(fig, "deep")
    assert {cd[0] for cd in deep.customdata} == {"E|AST|Over"}  # D is already active, not deep


def test_deep_pool_does_not_move_existing_stars():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))
    corr = _corr(("A|PTS|Over", "B|REB|Under", 0.5))
    deep_pool = _pool(("D|PTS|Under", -0.1))
    without = _node_pos(constellation_figure(legs, corr, pool))
    with_deep = _node_pos(constellation_figure(legs, corr, pool, deep_pool=deep_pool))
    for key, xy in without.items():
        assert with_deep[key] == xy


def test_deep_pool_star_sizes_are_uniform_not_kelly_scaled():
    deep_pool = _pool(("D|PTS|Under", -0.1), ("E|AST|Over", -5.0))
    fig = constellation_figure([], _corr(), _pool(("A|PTS|Over", 0.4)), deep_pool=deep_pool)
    deep = _trace(fig, "deep")
    assert len(set(deep.marker.size)) == 1  # flat size, unlike the Kelly-scaled active/candidate


def test_deep_pool_positions_are_deterministic_across_calls():
    deep_pool = _pool(("D|PTS|Under", -0.1), ("E|AST|Over", -0.3), ("F|STL|Under", -0.2))
    pool = _pool(("A|PTS|Over", 0.4))
    one = _trace(constellation_figure([], _corr(), pool, deep_pool=deep_pool), "deep")
    two = _trace(constellation_figure([], _corr(), pool, deep_pool=deep_pool), "deep")
    assert list(one.x) == list(two.x)
    assert list(one.y) == list(two.y)


def test_wider_groups_none_is_byte_stable_with_no_wider_trace():
    pool = _pool(("A|PTS|Over", 0.4))
    fig = constellation_figure(_slip("A|PTS|Over"), _corr(), pool)
    assert _trace(fig, "wider") is None


def _wider_row(player, market, bet, game, team, league, kelly):
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": 8.5,
        "Game": game,
        "League": league,
        "Team": team,
        "Kelly": kelly,
        "Win Prob": 0.55,
        "Boost": 1.4,
    }


def test_wider_groups_scales_focus_layout():
    legs = _slip("A|PTS|Over", "B|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|PTS|Over", 0.3))
    corr = _corr(("A|PTS|Over", "B|PTS|Over", 0.5))
    plain = _node_pos(constellation_figure(legs, corr, pool))
    wider = _node_pos(constellation_figure(legs, corr, pool, wider_groups=[]))
    for key, (x, y) in plain.items():
        wx, wy = wider[key]
        assert wx == pytest.approx(x * _WIDER_SCALE, rel=1e-6)
        assert wy == pytest.approx(y * _WIDER_SCALE, rel=1e-6)


def test_wider_groups_renders_other_games_legs_with_customdata():
    pool = _pool(("A|PTS|Over", 0.4))
    groups = [
        ("NYK/SAS", [_wider_row("Brunson", "PTS", "Over", "NYK/SAS", "NYK", "NBA", 0.5)]),
        ("MIA/ORL", [_wider_row("Herro", "3PM", "Over", "MIA/ORL", "MIA", "NBA", 0.3)]),
    ]
    fig = constellation_figure([], _corr(), pool, wider_groups=groups)
    wider = _trace(fig, "wider")
    assert wider is not None
    assert {cd[0] for cd in wider.customdata} == {"Brunson|PTS|Over", "Herro|3PM|Over"}


def test_wider_groups_have_no_edges():
    pool = _pool(("A|PTS|Over", 0.4))
    groups = [("NYK/SAS", [_wider_row("Brunson", "PTS", "Over", "NYK/SAS", "NYK", "NBA", 0.5)])]
    fig = constellation_figure([], _corr(), pool, wider_groups=groups)
    wider_keys = {cd[0] for cd in _trace(fig, "wider").customdata}
    for edge in _edge_traces(fig):
        assert not (set(edge.meta) & wider_keys)


def test_both_lenses_together_produce_sane_nonoverlapping_geometry():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))
    deep_pool = _pool(("D|PTS|Under", -0.1))
    groups = [("NYK/SAS", [_wider_row("Brunson", "PTS", "Over", "NYK/SAS", "NYK", "NBA", 0.5)])]
    fig = constellation_figure(legs, _corr(), pool, deep_pool=deep_pool, wider_groups=groups)
    deep_trace, wider_trace = _trace(fig, "deep"), _trace(fig, "wider")
    deep_r = max((x**2 + y**2) ** 0.5 for x, y in zip(deep_trace.x, deep_trace.y, strict=True))
    wider_r = max((x**2 + y**2) ** 0.5 for x, y in zip(wider_trace.x, wider_trace.y, strict=True))
    assert wider_r > deep_r  # wider ring sits clearly outside the deep ring


def test_mobile_figure_raises_size_floor_and_flags_slip_membership():
    """mobile=True lifts every star to the touch floor and appends the in-slip
    flag to customdata; mobile=False stays byte-identical to the no-arg figure."""
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2), ("C|AST|Over", 0.1))
    corr = _corr(("A|PTS|Over", "B|REB|Under", 0.3))
    baseline = constellation_figure(legs, corr, pool)
    desktop = constellation_figure(legs, corr, pool, mobile=False)
    assert desktop.to_json() == baseline.to_json()

    fig = constellation_figure(legs, corr, pool, mobile=True)
    node_traces = _node_traces(fig)
    assert node_traces
    for t in node_traces:
        assert min(t.marker.size) >= _SIZE_MIN_MOBILE
        assert t.textfont.size == _LABEL_FONT_SIZE_MOBILE
        for cd in t.customdata:
            assert len(cd) == 9
            assert cd[8] in (0, 1)
    active = _trace(fig, "active")
    assert all(cd[8] == 1 for cd in active.customdata)
    candidate = _trace(fig, "candidate")
    assert all(cd[8] == 0 for cd in candidate.customdata)
