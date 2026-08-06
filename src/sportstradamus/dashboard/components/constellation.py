"""The constellation — an interactive star map of a game's model-liked legs.

The dashboard's signature element (DESIGN.md §4a) and the slip editor's primary
control. Every model-liked leg in the game (Kelly edge ``K`` > 0) is a star, sized
by that edge so the strongest legs read biggest; the layout is **fixed per game**
(the sports-object template the game was dealt, or a team-anchored force-directed
solve when it is too thin for one) and never moves when you pick legs. A star you've
added to the slip burns at full team color; a candidate you haven't is the same
color desaturated and dimmed — selection is alpha + saturation, not a ring (a gold
ring read as a team color). Stories / Builder / Moon just pre-activate a subset.

Edges are pairwise correlations (gold, width/opacity ∝ |ρ|, dashed when ρ < 0 —
"fights the thesis"). The whole web is drawn as a **faint base layer** so the
correlation structure reads before any pick; a tie whose **both stars are in the slip**
brightens to full gold — the bright edges are the correlations among your slip's own
legs, sketched over the rest, and hovering any star faint-previews its other ties. The
*layout* still springs on the full correlation web, so a star's placement reflects every
tie. Each team's most-connected leg is pinned to its side, so a cross-matchup leg floats
toward the centre and an unrepresented side leaves its half empty.

Two optional lenses layer onto the same figure (P8 Task C6) instead of living as
separate expanders below the map:

* ``deep_pool`` — "look deeper": this game's model-passed legs (``K`` ≤ 0) as dim,
  unconnected background stars on a ring just outside the normal layout. They carry
  no edges and never join the spring layout, so revealing them can't reshuffle the
  lit constellation.
* ``wider_groups`` — "look wider": the focus layout shrinks toward the centre and
  other games' best legs orbit the edge in per-game clusters, each labelled with its
  game key.

Both lens node sets are positioned by list index around a ring, never by hashing a
key — Python's ``hash()`` on a ``str`` is per-process randomized (``PYTHONHASHSEED``)
unless seeded, so index-based placement is what keeps a golden position pin honest
across separate test runs.

The figure is pure (no Streamlit, no Archive): each node carries its
``Player|Market|Bet`` key plus its hover-card fields as ``customdata`` (the key at
index 0 — a plotly click turns into an add/remove), and each edge carries its two
endpoint keys in ``meta`` so the component's JS can dim-in a star's incident ties on
hover. It locks its own axes (no zoom/pan) — the builder hides the modebar. Positions come
from ``constellation_layout`` when the game carries a template and from networkx's
force solve when it doesn't; plotly draws.
Team fills read ``theme.team_colors(league, team)`` — real per-team primaries from
``team_assets.json``, never gold (gold is the correlation-edge color).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_layout import (
    assign_stars,
    cluster_players,
    collapse_edges,
    explode_clusters,
    topology_class,
)
from sportstradamus.dashboard.components.constellation_shapes import (
    assign_templates,
    shape_catalog,
    tuning,
)
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.theme import GOLD, GRAY, team_colors, team_name
from sportstradamus.leg_schema import is_model_liked, leg_field

# |ρ| floor to draw an edge or weight the layout — the story-menu edge floor.
_MIN_EDGE_RHO = 0.05

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Force-directed layout: each team's hub is pinned to its side, edges place the rest.
_LAYOUT_SEED = 17
_LAYOUT_TARGET = 1.1  # rescale radius — keeps stars + their top labels inside the frame
_ANCHOR_X = (-1.0, 1.0)  # team[0] pinned left, team[1] pinned right
_WARM_INSET = 0.6  # free nodes warm-start inside the anchors so springs can move them

# Star size scales with the leg's Kelly edge, relative to the game's strongest leg.
_SIZE_MIN = 14
_SIZE_MAX = 38
# A candidate (not-in-slip) star keeps most of its team hue — only a light blend toward
# gray plus reduced opacity marks it not-yet-picked. The old 0.55/0.45 crushed dark
# franchise colors to near-gray, so candidates read as a colorless field.
_INACTIVE_DESAT = 0.35
_INACTIVE_ALPHA = 0.60

# "Look deeper" lens (p8-games-lenses.html): the game's model-passed legs as a dim,
# unconnected background field. Flat cool-gray port of the mockup's own literal hex —
# distinct from both a team color and GRAY (reserved for the unknown-team fallback and
# text labels), so a deep star reads as "model passes," not a desaturated candidate.
_DEEP_COLOR = "#5f6b80"
_DEEP_ALPHA = 0.35
# Ring radius just outside the normal layout's own footprint (_LAYOUT_TARGET) so deep
# stars never overlap the lit constellation.
_DEEP_RADIUS = 1.25
# Alternating +/- radius offset (as a fraction of the ring radius, so it co-scales
# with the ring rather than becoming disproportionate once "look wider" shrinks it)
# so the ring isn't mechanically flat.
_RING_WOBBLE_FRAC = 0.04

# "Look wider" lens: the focus constellation shrinks toward the centre and other games'
# best legs (satellite_picker.satellite_groups) orbit an outer ring, grouped by matchup.
_WIDER_SCALE = 0.55
_WIDER_RADIUS = 1.55  # clearly outside _DEEP_RADIUS so "deeper" and "wider" read as
# concentric rings, not one overlapping band, when both lenses are on at once.
_WIDER_LABEL_OFFSET = 0.12  # radial offset of a cluster's game-key label past its dots
_WIDER_CLUSTER_SPAN = 0.06  # small fixed spacing between legs within one game's cluster
_WIDER_ALPHA = 0.75  # dimmer than an active star so periphery legs read as background

_EDGE_WIDTH_MIN = 1.0
_EDGE_WIDTH_SPAN = 6.0  # width at |ρ|=1 ≈ 7px; weak ties stay hairlines for contrast
_EDGE_ALPHA_MIN = 0.25
# The whole correlation web is drawn barely-there at this alpha so the structure reads
# before any pick without drowning the field; a tie whose both endpoints are in the slip
# brightens to full gold (_add_edge), well above this base.
_EDGE_BASE_ALPHA = 0.03
_FIG_HEIGHT = 380
_LABEL_FONT_SIZE = 11  # active-star caption — small enough to fit in a dense game

# Phase M touch floors: a fingertip needs ~22px; the label lifts with it. The Kelly
# ordering (size = edge) survives — the floor compresses the range, never reorders it.
_SIZE_MIN_MOBILE = 22
_LABEL_FONT_SIZE_MOBILE = 13
_ACTIVE_LABEL_COLOR = "#C7CEDA"  # in-slip captions read brighter than gray candidate labels

# Cinzel team tags framing the two sides of the map (docs/mockups/p8-games.html .teamtag).
_TAG_LEFT_COLOR = "#e2909b"  # team[0] tag — warm, left side
_TAG_RIGHT_COLOR = "#8ea6c9"  # team[1] tag — cool, right side

# The Phase D decoration layer: an engraving under the stars, never gold. Gold is
# the correlation-edge color and nothing else, so no engraved stroke can be
# misread as a ρ tie.
_SILHOUETTE_ALPHA = 0.13  # faint intent signal, below the ambient-decoration ceiling
_SILHOUETTE_FILL = f"rgba(95,107,128,{_SILHOUETTE_ALPHA})"  # _DEEP_COLOR #5f6b80
_OUTLINE_COLOR = "rgba(230,233,239,0.22)"  # theme TEXT #E6E9EF
_OUTLINE_GLOW_COLOR = "rgba(230,233,239,0.08)"  # the same stroke, wider and fainter
_FILLER_COLOR = "rgba(138,145,160,0.30)"  # theme GRAY #8A91A0
_OUTLINE_WIDTH, _OUTLINE_GLOW_WIDTH = 1, 4
_FILLER_SIZE = 6  # under _SIZE_MIN, so a filler can never be mistaken for a leg
_DECORATION = "decoration"
_NAMEPLATE_COLOR = "rgba(138,145,160,0.55)"  # theme GRAY #8A91A0, quieter than a team tag

# Templates are authored in a true square; this frame is not one, and which way it
# leans flips with the viewport. Desktop puts 3.2 x-units across ~980px against 2.8 of
# y over _FIG_HEIGHT — a circle renders ~2.4x wide. The phone is near-square at the
# same height and inverts it to ~0.87x, so one constant pair would round the medal
# and turn the diamond into a kite. Each viewport therefore carries its own (x, y),
# sized to fill the frame and land a circle ~1.2x wide: correcting the last of it
# would squeeze the desktop map into a portrait strip and throw away the left-right
# spread the two teams are read by, and ~1.2x is inside "loose".
_SHAPE_SCALE = (0.69, 1.38)
_SHAPE_SCALE_MOBILE = (1.30, 0.98)


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
    *,
    deep_pool: pd.DataFrame | None = None,
    wider_groups: list[tuple[str, list[dict]]] | None = None,
    mobile: bool = False,
    shape: dict | None = None,
) -> go.Figure:
    """Static star map of the game's model-liked legs, the slip's legs lit up.

    ``pool`` is the game's candidate offers — the static universe is its ``K`` > 0
    legs; ``slip_legs`` are the ones currently in the slip (drawn active). ``corr``
    is a ``current_game_corr`` slice. Each node carries its ``Player|Market|Bet`` key
    as customdata for click handling; the layout never depends on the selection.

    ``deep_pool`` (the "look deeper" lens) and ``wider_groups`` (the "look wider"
    lens) are both ``None`` by default, which reproduces today's figure byte-for-byte
    — they are optional overlays, not a change to the base map. See the module
    docstring for what each draws. ``mobile`` lifts star sizes and label fonts to
    touch floors (positions untouched — DESIGN §4a grammar holds on both paths).

    ``shape`` is the constellation template this game was dealt for the night
    (``constellation_shapes.assign_templates``). ``None`` — a game too thin to
    carry one, or a slate with nothing left to deal — reproduces the spring
    layout byte-for-byte, so the shapeless path is exactly today's figure.
    """
    fig = _blank_figure()
    info = _universe(pool, slip_legs)
    if not info:
        return fig
    keys = sorted(info)
    active = {corr_key(leg) for leg in slip_legs} & set(keys)
    game = _pool_field(pool, slip_legs, column="Game", key="game")
    league = _pool_field(pool, slip_legs, column="League", key="league")
    rho = _rho_map(corr, game)

    teams = _teams_of(game)
    _add_team_tags(fig, league, teams)
    team_color = {team: team_colors(league, team)[0] for team in teams}
    node_team = {k: info[k]["team"] for k in keys}
    edges = _edges(keys, rho)
    floor, label_size, shape_scale = (
        (_SIZE_MIN_MOBILE, _LABEL_FONT_SIZE_MOBILE, _SHAPE_SCALE_MOBILE)
        if mobile
        else (_SIZE_MIN, _LABEL_FONT_SIZE, _SHAPE_SCALE)
    )
    pos, fillers = _positions(
        keys, node_team, teams, [(a, b, abs(r)) for a, b, r in edges], shape, shape_scale
    )
    sizes = _star_sizes(keys, info, floor=floor)

    focus_scale = _WIDER_SCALE if wider_groups is not None else 1.0
    pos = {k: (x * focus_scale, y * focus_scale) for k, (x, y) in pos.items()}

    if shape is not None:
        _add_decoration(fig, shape, fillers, shape_scale, focus_scale)
        _add_nameplate(fig, shape["label"])
    if deep_pool is not None:
        _add_deep_trace(fig, deep_pool, slip_legs, radius=_DEEP_RADIUS * focus_scale, floor=floor)
    for a, b, r in edges:
        _add_edge(fig, a, b, pos[a], pos[b], r, active=active)
    _add_node_trace(
        fig,
        [k for k in keys if k not in active],
        pos,
        info,
        sizes,
        team_color,
        active=False,
        label_size=label_size,
    )
    _add_node_trace(
        fig,
        [k for k in keys if k in active],
        pos,
        info,
        sizes,
        team_color,
        active=True,
        label_size=label_size,
    )
    if wider_groups is not None:
        _add_wider_trace(fig, wider_groups, floor=floor)
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


def _ring_positions(keys: list[str], *, radius: float) -> dict[str, tuple[float, float]]:
    """Evenly distribute ``keys`` (already sorted) around a ring by index.

    Never hashes a key — ``hash()`` on a ``str`` is per-process randomized unless
    ``PYTHONHASHSEED`` is pinned, so index order is what keeps this reproducible
    across separate runs (including two golden-test invocations in CI). A small
    alternating radius wobble keeps the ring from reading as mechanically uniform.
    """
    n = len(keys)
    wobble = radius * _RING_WOBBLE_FRAC
    pos = {}
    for i, key in enumerate(keys):
        angle = 2 * math.pi * i / n
        r = radius + (wobble if i % 2 == 0 else -wobble)
        pos[key] = (r * math.cos(angle), r * math.sin(angle))
    return pos


def _add_deep_trace(
    fig: go.Figure,
    deep_pool: pd.DataFrame,
    slip_legs: Sequence[Mapping],
    *,
    radius: float,
    floor: float = _SIZE_MIN,
) -> None:
    """The "look deeper" lens: this game's model-passed legs as dim, unconnected background stars.

    Drawn before any edge or the two node traces, so it sits at the bottom of the
    z-order. Never joins the spring-layout graph — its positions come from
    ``_ring_positions``, entirely independent of ``_layout`` — so revealing it can
    never move an existing star.
    """
    exclude = {corr_key(leg) for leg in slip_legs}
    info = {
        corr_key(row): _node_info(row)
        for row in deep_pool.to_dict("records")
        if not is_model_liked(row) and corr_key(row) not in exclude
    }
    if not info:
        return
    keys = sorted(info)
    pos = _ring_positions(keys, radius=radius)
    fig.add_trace(
        go.Scatter(
            x=[pos[k][0] for k in keys],
            y=[pos[k][1] for k in keys],
            mode="markers",
            name="deep",
            marker={
                "symbol": "star",
                "size": [floor] * len(keys),
                "color": _DEEP_COLOR,
                "opacity": _DEEP_ALPHA,
            },
            customdata=[[k, *info[k]["card"], 0] for k in keys],
            hovertext=[info[k]["hover"] for k in keys],
            hoverinfo="none",
        )
    )


def _add_wider_trace(
    fig: go.Figure, wider_groups: list[tuple[str, list[dict]]], *, floor: float = _SIZE_MIN
) -> None:
    """The "look wider" lens: other games' best legs, clustered by matchup on an outer ring.

    One angular slot per game (evenly spaced by group index — same index-based
    determinism as ``_ring_positions``); legs within one game sit close together near
    that slot. A separate text trace labels each cluster with its game key, offset
    outward so it doesn't overlap the dots. These are real legs (not de-emphasized
    model-passes), so they carry their own team colors like an active star, just
    dimmer to read as periphery.
    """
    n = len(wider_groups)
    if n == 0:
        return
    xs, ys, colors, customdata, hovers = [], [], [], [], []
    label_x, label_y, label_text = [], [], []
    for i, (game, rows) in enumerate(wider_groups):
        angle = 2 * math.pi * i / n
        cx, cy = _WIDER_RADIUS * math.cos(angle), _WIDER_RADIUS * math.sin(angle)
        for j, row in enumerate(rows):
            offset = (j - (len(rows) - 1) / 2) * _WIDER_CLUSTER_SPAN
            xs.append(cx + offset * math.cos(angle + math.pi / 2))
            ys.append(cy + offset * math.sin(angle + math.pi / 2))
            info = _node_info(row)
            colors.append(team_colors(str(row["League"]), str(row["Team"]))[0])
            customdata.append([corr_key(row), *info["card"], 0])
            hovers.append(info["hover"])
        label_r = _WIDER_RADIUS + _WIDER_LABEL_OFFSET
        label_x.append(label_r * math.cos(angle))
        label_y.append(label_r * math.sin(angle))
        label_text.append(game)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="wider",
            marker={"symbol": "star", "size": [floor] * len(xs), "color": colors},
            opacity=_WIDER_ALPHA,
            customdata=customdata,
            hovertext=hovers,
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=label_x,
            y=label_y,
            mode="text",
            name="wider_labels",
            text=label_text,
            textfont={"color": GRAY, "size": _LABEL_FONT_SIZE},
            hoverinfo="skip",
        )
    )


def _star_sizes(
    keys: list[str], info: dict[str, dict], *, floor: float = _SIZE_MIN
) -> dict[str, float]:
    """Per-node star size ∝ Kelly edge, relative to the game's strongest leg."""
    top = max((info[k]["edge"] for k in keys), default=0.0)
    if top <= 0:
        return dict.fromkeys(keys, float(floor))
    span = _SIZE_MAX - floor
    return {k: floor + max(info[k]["edge"], 0.0) / top * span for k in keys}


def _blank_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=_FIG_HEIGHT,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",  # transparent — the page starfield reads through the map
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        hovermode="closest",
        dragmode=False,  # no panning — this is a map, not a chart
        xaxis={"visible": False, "fixedrange": True, "range": [-1.6, 1.6]},
        yaxis={"visible": False, "fixedrange": True, "range": [-1.4, 1.4]},
    )
    return fig


def _add_team_tags(fig: go.Figure, league: str, teams: list[str]) -> None:
    """Cinzel team-name tags framing the two sides (left = ``team[0]``, right = ``team[1]``).

    Ports the mockup's ``.teamtag`` labels; :func:`theme.team_name` gives the full name
    (abbrev fallback). No-op when the matchup isn't two-sided (an unrepresented side or a
    solo/combo game).
    """
    if len(teams) != 2:
        return
    for team, x, anchor, color in (
        (teams[0], 0.0, "left", _TAG_LEFT_COLOR),
        (teams[1], 1.0, "right", _TAG_RIGHT_COLOR),
    ):
        fig.add_annotation(
            text=team_name(league, team).upper(),
            xref="paper",
            yref="paper",
            x=x,
            y=1.0,
            xanchor=anchor,
            yanchor="top",
            showarrow=False,
            font={"family": "Cinzel, serif", "size": 11, "color": color},
        )


def _add_nameplate(fig: go.Figure, label: str) -> None:
    """Name the constellation along the bottom of the map, the way a star chart does.

    Same Cinzel voice as the team tags but quieter and letterspaced, so it reads as
    a caption on the sky rather than a third team. A game the assigner skipped has
    no shape and gets no nameplate — the spring map is not a constellation and
    must not claim to be one.
    """
    fig.add_annotation(
        text=" ".join(label.upper()),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.0,
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        font={"family": "Cinzel, serif", "size": 10, "color": _NAMEPLATE_COLOR},
    )


def _pool_field(
    pool: pd.DataFrame | None, slip_legs: Sequence[Mapping], *, column: str, key: str
) -> str:
    """A single-valued matchup field — read from the candidate ``pool`` so it's
    static per game and renders before any leg is picked; falls back to the slip's
    own legs when there is no pool (e.g. an active leg whose offer has expired).

    ``column`` names the raw ``pool`` column (``"Game"`` / ``"League"``); ``key``
    names the matching canonical-leg key (``"game"`` / ``"league"``). Reads the raw
    pool column directly rather than bridging through ``leg_field``: a single
    ``Game`` key never spans two leagues (team codes don't collide across leagues),
    so the first non-null value is always right, and ``leg_schema._FIELD_TO_OFFER_COL``
    deliberately excludes ``"league"`` (every existing call site holds a canonical
    leg for that field, never a raw offer row).
    """
    if pool is not None and not pool.empty and column in pool.columns:
        values = pool[column].dropna()
        if not values.empty:
            return str(values.iloc[0])
    for leg in slip_legs:
        value = leg.get(key)
        if value:
            return str(value)
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


def _positions(
    nodes: list[str],
    node_team: dict[str, str | None],
    teams: list[str],
    edges: list[tuple[str, str, float]],
    template: dict | None,
    scale: tuple[float, float],
) -> tuple[dict[str, tuple[float, float]], list[int]]:
    """Star positions, plus the template vertices no star filled.

    Without a template this is the spring layout and an empty filler list — the
    original behavior, untouched. With one, a player's tightly-tied legs first
    collapse into a supernode so a three-leg knot claims a single vertex rather
    than eating three, then the supernodes take vertices and explode back into
    their legs. Template coordinates never go through ``_rescale``: the stars
    have to land on the silhouette, which is authored in the same [-1, 1] box,
    and the same template has to stay visibly the same shape from one game to the
    next. The only thing applied to them is ``scale``, the viewport's aspect
    correction, which the decoration layer takes too.
    """
    if template is None:
        return _layout(nodes, node_team, teams, edges), []
    clusters, super_team, collapsed = _supernodes(nodes, node_team, edges)
    placed, fillers = assign_stars(sorted(clusters), super_team, teams, collapsed, template)
    sx, sy = scale
    exploded = explode_clusters(placed, clusters)
    return {key: (x * sx, y * sy) for key, (x, y) in exploded.items()}, fillers


def _supernodes(
    nodes: list[str],
    node_team: dict[str, str | None],
    edges: list[tuple[str, str, float]],
) -> tuple[dict[str, list[str]], dict[str, str | None], list[tuple[str, str, float]]]:
    """Collapse a game's legs into supernodes: ``(clusters, team per cluster, edges)``.

    Both the slate classifier and the per-game layout start here, so they always
    agree about how many nodes a game really has.
    """
    clusters = cluster_players(nodes, edges, tuning()["cluster_rho"])
    return (
        clusters,
        {key: node_team[members[0]] for key, members in clusters.items()},
        collapse_edges(edges, clusters),
    )


class GameShape(NamedTuple):
    """What one game was dealt for the night, and the reading behind it."""

    template: dict | None
    label: str | None
    topology: str
    n_supernodes: int
    readings: dict


def slate_shapes(
    offers: pd.DataFrame, corr: pd.DataFrame | None, league: str, date: str
) -> dict[str, GameShape]:
    """Deal every game on a league's night its own constellation.

    Dealt over the whole league slate rather than the visible platform's slice, so
    a game keeps its shape when the user switches Underdog ↔ Sleeper. Never cached:
    the tuning block is the owner's live surface, and an edit has to reclassify and
    re-deal on the very next rerun.
    """
    cfg = tuning()
    catalog = shape_catalog()["templates"]
    graphs = {}
    for game, group in offers.groupby("Game"):
        info = _universe(group, [])
        keys = sorted(info)
        edges = [(a, b, abs(r)) for a, b, r in _edges(keys, _rho_map(corr, str(game)))]
        clusters, super_team, collapsed = _supernodes(
            keys, {key: info[key]["team"] for key in keys}, edges
        )
        topo, readings = topology_class(sorted(clusters), super_team, collapsed, cfg)
        graphs[str(game)] = (topo, readings, len(clusters))

    dealt = assign_templates(
        league, date, [(game, topo, n) for game, (topo, _, n) in graphs.items()], cfg
    )
    return {
        game: GameShape(
            template=catalog[dealt[game]] if dealt[game] else None,
            label=catalog[dealt[game]]["label"] if dealt[game] else None,
            topology=topo,
            n_supernodes=n,
            readings=readings,
        )
        for game, (topo, readings, n) in graphs.items()
    }


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


def _scale_path(path: str, sx: float, sy: float) -> str:
    """Rescale a silhouette path's coordinates onto the frame.

    Every command the catalog allows (S5) takes strictly alternating x, y pairs,
    which is why ``H``/``V`` are banned there — they would break the alternation
    this walk depends on.
    """
    scaled, axis = [], 0
    for token in path.split():
        if token.isalpha():
            scaled.append(token)
            axis = 0
            continue
        scaled.append(f"{float(token) * (sx if axis % 2 == 0 else sy):g}")
        axis += 1
    return " ".join(scaled)


def _add_decoration(
    fig: go.Figure,
    template: dict,
    fillers: list[int],
    scale: tuple[float, float],
    focus_scale: float,
) -> None:
    """Draw the game's constellation beneath its stars: silhouette, engraved
    outline, and faint stars on the vertices no leg filled.

    Three deliberate constraints. The palette is the cool engraving family and
    never gold, so nothing here can be misread as a correlation edge. Every trace
    is named ``decoration`` with ``hoverinfo="skip"`` and carries no
    ``customdata``, which is what the component's JS gates click and hover on — so
    the layer is inert to the pointer without needing a guard. And it takes both
    the viewport aspect correction and ``focus_scale``, exactly as the stars do,
    or the shape would detach from the map it belongs to under the "look wider"
    lens.
    """
    sx, sy = scale[0] * focus_scale, scale[1] * focus_scale
    fig.add_shape(
        type="path",
        path=_scale_path(template["silhouette"], sx, sy),
        fillcolor=_SILHOUETTE_FILL,
        line={"width": 0},
        layer="below",
    )
    xy = {v["id"]: (v["x"] * sx, v["y"] * sy) for v in template["vertices"]}
    xs, ys = [], []
    for a, b in template["outline"]:
        xs += [xy[a][0], xy[b][0], None]
        ys += [xy[a][1], xy[b][1], None]
    for color, width in (
        (_OUTLINE_GLOW_COLOR, _OUTLINE_GLOW_WIDTH),
        (_OUTLINE_COLOR, _OUTLINE_WIDTH),
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": color, "width": width},
                name=_DECORATION,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if fillers:
        fig.add_trace(
            go.Scatter(
                x=[xy[vid][0] for vid in fillers],
                y=[xy[vid][1] for vid in fillers],
                mode="markers",
                marker={"size": _FILLER_SIZE, "color": _FILLER_COLOR},
                name=_DECORATION,
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _add_edge(fig: go.Figure, a: str, b: str, p0, p1, rho: float, *, active: set[str]) -> None:
    """One correlation edge: gold, width/opacity ∝ |ρ|, dashed when ρ < 0.

    Drawn at a faint base alpha (``_EDGE_BASE_ALPHA``) so the whole web reads as a
    sketch; brightens to full ``|ρ|``-scaled gold only when **both** endpoints are in
    the slip, so the slip's own correlations stand out over the rest. ``meta`` carries
    the endpoint keys so the component's JS can faint-preview a star's other ties on hover.
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
            opacity=min(1.0, _EDGE_ALPHA_MIN + abs(rho)) if incident else _EDGE_BASE_ALPHA,
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
    label_size: int = _LABEL_FONT_SIZE,
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
            textfont={
                "color": _ACTIVE_LABEL_COLOR if active else GRAY,
                "size": label_size,
            },
            customdata=[[k, *info[k]["card"], 1 if active else 0] for k in keys],
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
