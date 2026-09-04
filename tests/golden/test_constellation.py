"""Pins for the dashboard constellation — the DESIGN §4a static, shape-dealt star map.

``constellation_figure`` is pure (networkx force layout + plotly draw, no Streamlit).
The node set is the game's strongest **model-liked** legs (Kelly ``K`` > 0), capped and
fixed per game — selecting a leg never moves a star, it only lights it up, and a slip
leg beyond the cut is promoted into the spot the *deeper* lens would have given it. A
slip leg renders active (full team color, opacity 1, labelled); a candidate renders
desaturated + dim, labelled like the rest. Star size ∝ Kelly edge. Each team's hub anchors to its side — on the
template the game was dealt when it has one, on the spring solve when it doesn't. Every tie is a
gold edge (width/opacity ∝ |ρ|, dashed when ρ < 0) drawn as a faint base web that
brightens when both its stars are in the slip. These assert that grammar, the static layout,
the size/selection encodings, the hover, the per-edge endpoint ``meta``, the
click-key + card-field ``customdata`` the in-app editor and its JS card rely on, and
the Phase D decoration layer's subordination to all of it.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd
import pytest

from sportstradamus.dashboard.components.constellation import (
    _EDGE_BASE_ALPHA,
    _INACTIVE_ALPHA,
    _LABEL_FONT_SIZE_MOBILE,
    _SIZE_MAX,
    _SIZE_MIN,
    _SIZE_MIN_MOBILE,
    constellation_figure,
)
from sportstradamus.dashboard.components.constellation_lenses import (
    _DEEP_ALPHA,
    _DEEP_COLOR,
    _WIDER_SCALE,
    WIDER_GAMES,
)
from sportstradamus.dashboard.components.constellation_shapes import shape_catalog
from sportstradamus.dashboard.components.constellation_slate import (
    DECORATION,
    FILLER_SIZE,
    SHAPE_SCALE,
    SHAPE_SCALE_MOBILE,
    SILHOUETTE_ALPHA,
    scale_path,
)
from sportstradamus.dashboard.components.constellation_spacing import (
    _STAR_GAP_PX,
    DEFAULT_STARS,
    PX_PER_UNIT,
    PX_PER_UNIT_MOBILE,
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
    # The Phase D decoration layer draws markers and lines too, but carries no
    # customdata, so every helper below that zips against it has to skip it.
    return [t for t in fig.data if t.mode and "markers" in t.mode and t.name != DECORATION]


def _edge_traces(fig) -> list:
    return [t for t in fig.data if t.mode == "lines" and t.name != DECORATION]


def _decoration_traces(fig) -> list:
    return [t for t in fig.data if t.name == DECORATION]


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


def _ladder(n: int) -> pd.DataFrame:
    """``n`` one-leg players, teams alternating, Kelly strictly descending."""
    return _pool(*[(f"P{i:02d}|PTS|Over", 0.9 - i * 0.05) for i in range(n)])


def test_layout_is_static_under_selection():
    # The picked leg ranks 15th, outside the default cut, so it joins the map by
    # promotion — the one case that could re-solve the layout if the node set were
    # ever allowed to depend on the selection.
    pool = _ladder(15)
    corr = _corr(("P00|PTS|Over", "P01|PTS|Over", 0.5))
    before = _node_pos(constellation_figure([], corr, pool))
    after = _node_pos(constellation_figure([pool.to_dict("records")[14]], corr, pool))
    assert len(before) == DEFAULT_STARS
    assert all(after[key] == xy for key, xy in before.items())


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


def test_deep_stars_take_their_ties_to_main_stars():
    """A deep star arrives with its correlations, under the name the JS fades."""
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))  # model-liked candidate pool
    deep_pool = _pool(("D|PTS|Under", -0.1))  # model-passed
    fig = constellation_figure(
        legs, _corr(("A|PTS|Over", "D|PTS|Under", 0.5)), pool, deep_pool=deep_pool
    )
    tie = next(e for e in _edge_traces(fig) if set(e.meta) == {"A|PTS|Over", "D|PTS|Under"})
    assert tie.name == "deep_edge"  # a lens trace, so it fades in and out with its star
    assert tie.opacity == _EDGE_BASE_ALPHA  # only a both-ends-in-slip tie brightens


def test_deep_tier_colors_split_liked_from_passed():
    """One size, two readings: a liked leg the cut left behind is a candidate, just
    smaller; only the model-passed tier wears the lens's own gray."""
    fig = constellation_figure([], None, _ladder(13), deep_pool=_pool(("D|PTS|Under", -0.1)))
    deep = _trace(fig, "deep")
    assert [cd[0] for cd in deep.customdata] == ["P12|PTS|Over", "D|PTS|Under"]
    assert list(deep.marker.opacity) == [_INACTIVE_ALPHA, _DEEP_ALPHA]
    assert deep.marker.color[1] == _DEEP_COLOR
    assert all(color not in (GRAY, GOLD) for color in deep.marker.color)


def test_an_in_slip_deep_leg_burns_as_a_promoted_star():
    """A slip never loses a leg to the cut: picking a model-passed star promotes it
    out of the tier and lights it at main size, lens or no lens."""
    legs = [_row(0, "D|PTS|Under", -0.1)]
    pool = _pool(("A|PTS|Over", 0.4))
    deep_pool = _pool(("D|PTS|Under", -0.1), ("E|AST|Over", -0.3))
    fig = constellation_figure(legs, _corr(), pool, deep_pool=deep_pool)
    assert {cd[0] for cd in _trace(fig, "deep").customdata} == {"E|AST|Over"}
    active = _trace(fig, "active")
    assert [cd[0] for cd in active.customdata] == ["D|PTS|Under"]
    assert active.marker.size[0] >= _SIZE_MIN


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
    assert deep.marker.size[0] < _SIZE_MIN  # and under the floor, so it never outranks a real star


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


def _sky(n: int, *, per_game: int = 4) -> list[tuple[str, list[dict]]]:
    matchups = [
        ("MIA/ORL", "MIA"),
        ("BOS/PHI", "BOS"),
        ("LAL/GSW", "LAL"),
        ("DAL/HOU", "DAL"),
        ("DEN/PHX", "DEN"),
        ("MIL/CHI", "MIL"),
        ("ATL/CLE", "ATL"),
    ][:n]
    return [
        (
            game,
            [
                _wider_row(f"{team}{i}", "3PM", "Over", game, team, "NBA", 0.3)
                for i in range(per_game)
            ],
        )
        for game, team in matchups
    ]


def test_wider_lens_recedes_the_focus_and_keeps_clearance():
    """The map recedes to make room — it is never re-laid-out — and the spacing pass
    re-solves at the tighter scale, so nothing collides once the sky is in."""
    pool = _ladder(13)
    groups = _sky(1, per_game=1)
    for mobile, px in ((False, PX_PER_UNIT), (True, PX_PER_UNIT_MOBILE)):
        plain = constellation_figure([], None, pool, shape=_HOURGLASS, mobile=mobile)
        wider = constellation_figure(
            [], None, pool, shape=_HOURGLASS, mobile=mobile, wider_groups=groups
        )
        before, after, sizes = _node_pos(plain), _node_pos(wider), _sizes(wider)
        for key, (x, y) in before.items():
            drift = math.hypot(
                (after[key][0] - x * _WIDER_SCALE) * px[0],
                (after[key][1] - y * _WIDER_SCALE) * px[1],
            )
            # Only the spacing pass moves a star off the exact recede, and never by
            # more than its own radius — the drawing stays the same drawing.
            assert drift <= _SIZE_MAX / 2, (key, mobile, drift)
        for one, other in itertools.combinations(sorted(before), 2):
            apart = math.hypot(
                (after[one][0] - after[other][0]) * px[0],
                (after[one][1] - after[other][1]) * px[1],
            )
            assert apart >= (sizes[one] + sizes[other]) / 2 + _STAR_GAP_PX - 1e-9, (one, other)
        # The phone has no side band even receded, so its sky grows in y instead.
        assert (wider.layout.height > plain.layout.height) is mobile


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


def test_wider_stars_never_enter_the_constellations_footprint():
    """Both lenses at once: the sky stays outside the whole drawn map, deep stars
    included, and keeps a glyph's clear air from every one of them."""
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Under", 0.2))
    deep_pool = _pool(("D|PTS|Under", -0.1), ("E|AST|Over", -0.3))
    fig = constellation_figure(legs, _corr(), pool, deep_pool=deep_pool, wider_groups=_sky(2))
    sky = {cd[0] for cd in _trace(fig, "wider").customdata}
    pos, sizes = _node_pos(fig), _sizes(fig)
    drawn = [(pos[key], sizes[key]) for key in pos if key not in sky]
    box = (
        min(x * PX_PER_UNIT[0] - size / 2 for (x, _), size in drawn),
        min(y * PX_PER_UNIT[1] - size / 2 for (_, y), size in drawn),
        max(x * PX_PER_UNIT[0] + size / 2 for (x, _), size in drawn),
        max(y * PX_PER_UNIT[1] + size / 2 for (_, y), size in drawn),
    )
    for key in sky:
        (x, y), size = pos[key], sizes[key]
        at = (x * PX_PER_UNIT[0], y * PX_PER_UNIT[1])
        assert not (box[0] <= at[0] <= box[2] and box[1] <= at[1] <= box[3]), key
        for (ox, oy), other in drawn:
            apart = math.hypot(at[0] - ox * PX_PER_UNIT[0], at[1] - oy * PX_PER_UNIT[1])
            assert apart >= (size + other) / 2 + _STAR_GAP_PX - 1e-9, key


def test_wider_stars_are_not_a_ring():
    """The owner's one hard no. A ring holds one radius and leaves no empty sector."""
    groups = _sky(WIDER_GAMES)
    for mobile in (False, True):
        wider = _trace(
            constellation_figure([], None, _ladder(13), wider_groups=groups, mobile=mobile),
            "wider",
        )
        radii = [math.hypot(float(x), float(y)) for x, y in zip(wider.x, wider.y, strict=True)]
        assert max(radii) - min(radii) >= 0.25 * (sum(radii) / len(radii)), mobile
        angles = sorted(
            math.atan2(float(y), float(x)) % math.tau for x, y in zip(wider.x, wider.y, strict=True)
        )
        gaps = [b - a for a, b in itertools.pairwise(angles)] + [angles[0] + math.tau - angles[-1]]
        # Bands leave whole sectors of the sky empty; an even ring leaves none.
        assert max(gaps) >= 3 * math.tau / len(angles), (mobile, math.degrees(max(gaps)))


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


# --- Phase D: the constellation-shape layer -------------------------------------

_HOURGLASS = shape_catalog()["templates"]["the-hourglass"]


def _shaped(*, wider=None, mobile=False):
    """A four-leg two-team game rendered onto a real template."""
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Over", 0.3), ("C|AST|Over", 0.2), ("D|PTS|Over", 0.1))
    corr = _corr(("A|PTS|Over", "B|REB|Over", 0.5))
    return constellation_figure(
        legs, corr, pool, shape=_HOURGLASS, wider_groups=wider, mobile=mobile
    )


def test_without_a_template_the_figure_is_exactly_the_spring_map():
    """The shapeless path is today's figure — a game the assigner skips loses nothing."""
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Over", 0.3))
    corr = _corr(("A|PTS|Over", "B|REB|Over", 0.5))
    fig = constellation_figure(legs, corr, pool)
    assert _decoration_traces(fig) == []
    assert fig.layout.shapes == ()


def test_the_silhouette_is_drawn_faintly_beneath_everything():
    shapes = _shaped().layout.shapes
    assert len(shapes) == 1
    assert shapes[0].type == "path"
    assert shapes[0].layer == "below"
    assert shapes[0].line.width == 0
    assert f",{SILHOUETTE_ALPHA})" in shapes[0].fillcolor


def test_the_decoration_layer_is_inert_to_the_pointer():
    """No customdata is what main.js gates click and hover on, so this layer needs
    no guard of its own — but it must stay customdata-free for that to hold."""
    for trace in _decoration_traces(_shaped()):
        assert trace.hoverinfo == "skip"
        assert trace.customdata is None
        assert trace.showlegend is False


def test_no_engraved_stroke_is_ever_gold():
    """Gold means correlation and nothing else (DESIGN §4a)."""
    figure = _shaped()
    assert figure.layout.shapes[0].fillcolor.lower().find(GOLD.lower()) == -1
    for trace in _decoration_traces(figure):
        rendered = f"{trace.line}{trace.marker}".lower()
        assert GOLD.lower() not in rendered
        assert "gold" not in rendered


def test_the_template_does_not_change_how_many_gold_edges_are_drawn():
    legs = _slip("A|PTS|Over")
    pool = _pool(("A|PTS|Over", 0.4), ("B|REB|Over", 0.3))
    corr = _corr(("A|PTS|Over", "B|REB|Over", 0.5))
    plain = constellation_figure(legs, corr, pool)
    shaped = constellation_figure(legs, corr, pool, shape=_HOURGLASS)
    assert len(_edge_traces(shaped)) == len(_edge_traces(plain))
    assert _shown_keys(shaped) == _shown_keys(plain)


def test_stars_land_on_template_vertices():
    frame = {
        (round(v["x"] * SHAPE_SCALE[0], 5), round(v["y"] * SHAPE_SCALE[1], 5))
        for v in _HOURGLASS["vertices"]
    }
    assert set(_node_pos(_shaped()).values()) <= frame


def test_unfilled_vertices_get_filler_stars_too_small_to_read_as_legs():
    fillers = [t for t in _decoration_traces(_shaped()) if t.mode == "markers"]
    assert len(fillers) == 1
    assert fillers[0].marker.size == FILLER_SIZE < _SIZE_MIN
    assert len(fillers[0].x) == len(_HOURGLASS["vertices"]) - len(_node_pos(_shaped()))


def test_the_engraving_shrinks_with_its_stars_under_the_look_wider_lens():
    """Decoration that ignored focus_scale would detach from the map it belongs to."""
    outline = next(t for t in _decoration_traces(_shaped(wider=[])) if t.mode == "lines")
    plain = next(t for t in _decoration_traces(_shaped()) if t.mode == "lines")
    span = max(x for x in outline.x if x is not None)
    plain_span = max(x for x in plain.x if x is not None)
    assert span == pytest.approx(plain_span * _WIDER_SCALE)


def test_a_silhouette_rescales_x_and_y_independently():
    """The frame is far wider than it is tall, so one uniform scale would leave every
    round shape a flat lens — x and y take different corrections."""
    assert scale_path("M -1 0.5 L 1 -0.5 Z", 0.8, 1.32) == "M -0.8 0.66 L 0.8 -0.66 Z"
    # A cubic's control points are coordinates too, and alternate the same way.
    assert scale_path("M 0 0 C 1 1 -1 -1 0.5 0.5", 2.0, 4.0) == "M 0 0 C 2 4 -2 -4 1 2"


def test_the_phone_frame_stretches_the_other_way():
    """Desktop is wide and short, the phone near-square — the correction has to invert
    or a diamond that reads round on a laptop renders as a tall kite in a hand."""
    assert SHAPE_SCALE[0] < SHAPE_SCALE[1] and SHAPE_SCALE_MOBILE[0] > SHAPE_SCALE_MOBILE[1]
    frame = {
        (round(v["x"] * SHAPE_SCALE_MOBILE[0], 5), round(v["y"] * SHAPE_SCALE_MOBILE[1], 5))
        for v in _HOURGLASS["vertices"]
    }
    assert set(_node_pos(_shaped(mobile=True)).values()) <= frame


def test_the_shape_frame_correction_keeps_stars_and_engraving_together():
    """Whatever the correction is, both layers must take it — the whole point is that
    the stars sit on the silhouette."""
    figure = _shaped()
    outline = next(t for t in _decoration_traces(figure) if t.mode == "lines")
    star_xs = [x for x, _ in _node_pos(figure).values()]
    outline_xs = [round(float(x), 5) for x in outline.x if x is not None]
    assert min(outline_xs) <= min(star_xs) and max(star_xs) <= max(outline_xs)


def test_a_shaped_map_is_never_captioned_with_the_shapes_name():
    """The drawing has to carry the name on its own.

    A caption naming the constellation is the tell that the engraving isn't
    reading, so the map annotates only its two team tags whether or not the game
    was dealt a shape.
    """
    plain = constellation_figure(
        _slip("A|PTS|Over"),
        _corr(("A|PTS|Over", "B|REB|Over", 0.5)),
        _pool(("A|PTS|Over", 0.4), ("B|REB|Over", 0.3)),
    )
    for fig in (_shaped(), plain):
        assert len(fig.layout.annotations) == 2  # the two team tags, and nothing else
        assert not [a for a in fig.layout.annotations if _HOURGLASS["label"].upper() in a.text]
