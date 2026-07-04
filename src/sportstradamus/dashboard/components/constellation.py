"""The constellation — an interactive star map of a game's model-liked legs.

The dashboard's signature element (DESIGN.md §4a) and the slip editor's primary
control. Every model-liked leg in the game (Kelly edge ``K`` > 0) is a star, sized
by that edge so the strongest legs read biggest; the layout is **fixed per game**
(team-anchored force-directed) and never moves when you pick legs. A star you've
added to the slip burns at full team color; a candidate you haven't is the same
color desaturated and dimmed — selection is alpha + saturation, not a ring (a gold
ring read as a team color). Stories / Builder / Moon just pre-activate a subset.

Edges are pairwise correlations (gold, width/opacity ∝ |ρ|, dashed when ρ < 0 —
"fights the thesis") and stay **hidden until both their stars are in the slip** — so
the only edges drawn are the correlations among your slip's own legs (an empty or
one-leg slip is a clean field), and hovering any star faint-previews its other ties.
The *layout* still springs on the full correlation web, so a star's placement reflects
every tie. Each team's most-connected leg is pinned to its side, so a cross-matchup leg
floats toward the centre and an unrepresented side leaves its half empty.

The figure is pure (no Streamlit, no Archive): each node carries its
``Player|Market|Bet`` key plus its hover-card fields as ``customdata`` (the key at
index 0 — a plotly click turns into an add/remove), and each edge carries its two
endpoint keys in ``meta`` so the component's JS can dim-in a star's incident ties on
hover. It locks its own axes (no zoom/pan) — the builder hides the modebar. Layout is
force-directed (networkx, always present via torch); plotly draws.
Team fills are an on-token placeholder until ``team_assets.json`` lands (P8).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.theme import GOLD, GRAY
from sportstradamus.leg_schema import is_model_liked, leg_field

# |ρ| floor to draw an edge or weight the layout — the story-menu edge floor.
_MIN_EDGE_RHO = 0.05
# Team fills (chartCategoricalColors[0]/[2]) — placeholder until team_assets.json (P8).
_TEAM_PALETTE = ("#2E6BE6", "#E69F00")

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Force-directed layout: each team's hub is pinned to its side, edges place the rest.
_LAYOUT_SEED = 17
_LAYOUT_TARGET = 1.1  # rescale radius — keeps stars + their top labels inside the frame
_ANCHOR_X = (-1.0, 1.0)  # team[0] pinned left, team[1] pinned right
_WARM_INSET = 0.6  # free nodes warm-start inside the anchors so springs can move them

# Star size scales with the leg's Kelly edge, relative to the game's strongest leg.
_SIZE_MIN = 11
_SIZE_MAX = 30
# A candidate (not-in-slip) star is the team color blended toward gray and dimmed.
_INACTIVE_DESAT = 0.55
_INACTIVE_ALPHA = 0.45

_EDGE_WIDTH_MIN = 1.0
_EDGE_WIDTH_SPAN = 6.0  # width at |ρ|=1 ≈ 7px; weak ties stay hairlines for contrast
_EDGE_ALPHA_MIN = 0.25
_FIG_HEIGHT = 380
_LABEL_FONT_SIZE = 11  # active-star caption — small enough to fit in a dense game


def _last_name(player: str) -> str:
    parts = player.split()
    while len(parts) > 1 and parts[-1].rstrip(".").lower() in _NAME_SUFFIXES:
        parts.pop()
    return parts[-1] if parts else player


def _bet_word(bet) -> str:
    return "Over" if str(bet).lower().startswith("o") else "Under"


def star_label(leg: Mapping) -> str:
    """Compact star caption: ``Lastname MKT o/u Line`` (e.g. ``Brunson PTS o25.5``).

    ``leg`` is a canonical lowercase leg or a raw uppercase ``current_offers``
    row — ``leg_field`` bridges the two shapes (the constellation draws both a
    game's candidate pool and the slip's own legs on one map).
    """
    ou = "o" if _bet_word(leg_field(leg, "bet")) == "Over" else "u"
    return f"{_last_name(str(leg_field(leg, 'player')))} {leg_field(leg, 'market')} {ou}{float(leg_field(leg, 'line')):.10g}"


def _hover_text(leg: Mapping) -> str:
    p = float(leg_field(leg, "win_prob", 0.0) or 0.0)
    boost = float(leg_field(leg, "boost", 1.0) or 1.0)
    k = float(leg_field(leg, "kelly", 0.0) or 0.0)
    head = (
        f"{leg_field(leg, 'player')} — {leg_field(leg, 'market')} "
        f"{_bet_word(leg_field(leg, 'bet'))} {float(leg_field(leg, 'line')):.10g}"
    )
    return f"{head}<br>Win {p:.0%} · {boost:.2f}x · Kelly {k:.0%}"


def _card_fields(leg: Mapping) -> list:
    """Structured fields the hover card reads from a node's ``customdata`` (after the key)."""
    return [
        str(leg_field(leg, "player")),
        str(leg_field(leg, "market")),
        _bet_word(leg_field(leg, "bet")),
        float(leg_field(leg, "line")),
        float(leg_field(leg, "win_prob", 0.0) or 0.0),
        float(leg_field(leg, "boost", 1.0) or 1.0),
        float(leg_field(leg, "kelly", 0.0) or 0.0),
    ]


def _node_info(leg: Mapping) -> dict:
    return {
        "label": star_label(leg),
        "team": leg_field(leg, "team"),
        "edge": float(leg_field(leg, "kelly", 0.0) or 0.0),
        "hover": _hover_text(leg),
        "card": _card_fields(leg),
    }


def constellation_figure(
    slip_legs: Sequence[Mapping],
    corr: pd.DataFrame | None,
    pool: pd.DataFrame | None = None,
) -> go.Figure:
    """Static star map of the game's model-liked legs, the slip's legs lit up.

    ``pool`` is the game's candidate offers — the static universe is its ``K`` > 0
    legs; ``slip_legs`` are the ones currently in the slip (drawn active). ``corr``
    is a ``current_game_corr`` slice. Each node carries its ``Player|Market|Bet`` key
    as customdata for click handling; the layout never depends on the selection.
    """
    fig = _blank_figure()
    info = _universe(pool, slip_legs)
    if not info:
        return fig
    keys = sorted(info)
    active = {corr_key(leg) for leg in slip_legs} & set(keys)
    game = _game_of(pool, slip_legs)
    rho = _rho_map(corr, game)

    teams = _teams_of(game)
    team_color = {team: _TEAM_PALETTE[i % len(_TEAM_PALETTE)] for i, team in enumerate(teams)}
    node_team = {k: info[k]["team"] for k in keys}
    edges = _edges(keys, rho)
    pos = _layout(keys, node_team, teams, [(a, b, abs(r)) for a, b, r in edges])
    sizes = _star_sizes(keys, info)

    for a, b, r in edges:
        _add_edge(fig, a, b, pos[a], pos[b], r, active=active)
    _add_node_trace(
        fig, [k for k in keys if k not in active], pos, info, sizes, team_color, active=False
    )
    _add_node_trace(
        fig, [k for k in keys if k in active], pos, info, sizes, team_color, active=True
    )
    return fig


def _universe(pool: pd.DataFrame | None, slip_legs: Sequence[Mapping]) -> dict[str, dict]:
    """Model-liked legs (Kelly ``K`` > 0) for the game, plus any active slip leg.

    The static node set: it depends on the game's offers, not on which legs are in
    the slip (the slip only decides which stars light up), so the map never
    reshuffles on a pick. An active leg whose offer has slipped below K > 0 is kept
    so a slip never loses one of its own stars.
    """
    info: dict[str, dict] = {}
    if pool is not None and not pool.empty:
        for row in pool.to_dict("records"):
            if is_model_liked(row):
                info.setdefault(corr_key(row), _node_info(row))
    for leg in slip_legs:
        info.setdefault(corr_key(leg), _node_info(leg))
    return info


def _star_sizes(keys: list[str], info: dict[str, dict]) -> dict[str, float]:
    """Per-node star size ∝ Kelly edge, relative to the game's strongest leg."""
    top = max((info[k]["edge"] for k in keys), default=0.0)
    if top <= 0:
        return dict.fromkeys(keys, float(_SIZE_MIN))
    span = _SIZE_MAX - _SIZE_MIN
    return {k: _SIZE_MIN + max(info[k]["edge"], 0.0) / top * span for k in keys}


def _blank_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=_FIG_HEIGHT,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        hovermode="closest",
        dragmode=False,  # no panning — this is a map, not a chart
        xaxis={"visible": False, "fixedrange": True, "range": [-1.6, 1.6]},
        yaxis={"visible": False, "fixedrange": True, "range": [-1.4, 1.4]},
    )
    return fig


def _game_of(pool: pd.DataFrame | None, slip_legs: Sequence[Mapping]) -> str:
    """The matchup this map draws — read from the candidate ``pool`` so the layout is
    static per game and the field renders before any leg is picked; falls back to the
    slip's game when there is no pool (e.g. an active leg whose offer has expired).
    """
    if pool is not None and not pool.empty and "Game" in pool.columns:
        games = pool["Game"].dropna()
        if not games.empty:
            return str(games.iloc[0])
    for leg in slip_legs:
        game = leg.get("game")
        if game:
            return str(game)
    return ""


def _teams_of(game: str) -> list[str]:
    """The matchup's two team codes, sorted — the two anchored sides."""
    return sorted(set(game.split("/"))) if game else []


def _rho_map(corr: pd.DataFrame | None, game: str) -> dict[frozenset, float]:
    """Map ``frozenset(leg_a, leg_b) -> rho`` for the slip's game."""
    if corr is None or corr.empty:
        return {}
    slice_df = corr.loc[corr["Game"] == game] if game else corr
    return {
        frozenset((row.leg_a, row.leg_b)): float(row.rho)
        for row in slice_df.itertuples(index=False)
    }


def _edges(keys: list[str], rho: dict[frozenset, float]) -> list[tuple[str, str, float]]:
    """Every correlation tie among shown nodes (|ρ| ≥ floor), signed and sorted.

    Feeds both the spring *layout* (which pulls on |ρ| so a leg's placement reflects
    all its ties) and the *drawn* edges (one trace each, hidden until both endpoints
    are active). Sorted so the trace order — hence the figure — is deterministic.
    """
    node_set = set(keys)
    out = []
    for pair, r in rho.items():
        if abs(r) < _MIN_EDGE_RHO:
            continue
        a, b = sorted(pair)
        if a in node_set and b in node_set:
            out.append((a, b, r))
    return sorted(out)


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
    """Uniformly scale positions to fill the frame (radius ``_LAYOUT_TARGET``)."""
    span = max((max(abs(x), abs(y)) for x, y in pos.values()), default=0.0)
    if span == 0.0:
        return dict.fromkeys(pos, (0.0, 0.0))
    factor = _LAYOUT_TARGET / span
    return {k: (x * factor, y * factor) for k, (x, y) in pos.items()}


def _add_edge(fig: go.Figure, a: str, b: str, p0, p1, rho: float, *, active: set[str]) -> None:
    """One correlation edge: gold, width/opacity ∝ |ρ|, dashed when ρ < 0.

    Hidden (opacity 0) until **both** endpoints are in the slip, so the only edges
    drawn are the correlations among the slip's own legs (a seeded multi-leg slip
    isn't a hairball of ties out to candidates); ``meta`` carries the endpoint keys
    so the component's JS can faint-preview a star's other ties on hover.
    """
    incident = a in active and b in active
    fig.add_trace(
        go.Scatter(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            mode="lines",
            name="edge",
            line={
                "color": GOLD,
                "width": _EDGE_WIDTH_MIN + abs(rho) * _EDGE_WIDTH_SPAN,
                "dash": "dot" if rho < 0 else "solid",
            },
            opacity=min(1.0, _EDGE_ALPHA_MIN + abs(rho)) if incident else 0.0,
            meta=[a, b],
            hoverinfo="skip",
        )
    )


def _add_node_trace(
    fig: go.Figure,
    keys: list[str],
    pos: dict,
    info: dict[str, dict],
    sizes: dict[str, float],
    team_color: dict[str, str],
    *,
    active: bool,
) -> None:
    """One scatter trace of stars: active = full team color, candidate = desaturated/dim.

    Both active and candidate stars carry their caption; active stars render on top.
    The active/candidate signal is the star's fill color and opacity, not the label.
    """
    if not keys:
        return
    base_colors = [team_color.get(info[k]["team"], GRAY) for k in keys]
    colors = [c if active else _desaturate(c, _INACTIVE_DESAT) for c in base_colors]
    fig.add_trace(
        go.Scatter(
            x=[pos[k][0] for k in keys],
            y=[pos[k][1] for k in keys],
            mode="markers+text",
            name="active" if active else "candidate",
            marker={
                "symbol": "star",
                "size": [sizes[k] for k in keys],
                "color": colors,
                "opacity": 1.0 if active else _INACTIVE_ALPHA,
            },
            text=[info[k]["label"] for k in keys],
            textposition="top center",
            textfont={"color": GRAY, "size": _LABEL_FONT_SIZE},
            customdata=[[k, *info[k]["card"]] for k in keys],
            hovertext=[info[k]["hover"] for k in keys],
            hoverinfo="none",  # the component draws the hover card; suppress the native tooltip
        )
    )


def _desaturate(hex_color: str, amount: float) -> str:
    """Blend ``hex_color`` toward gray by ``amount`` ∈ [0, 1] (1 = full gray)."""
    rgb = _hex_rgb(hex_color)
    gray = _hex_rgb(GRAY)
    mixed = tuple(round(c + (g - c) * amount) for c, g in zip(rgb, gray, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
