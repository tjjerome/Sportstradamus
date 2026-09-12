"""The constellation — an interactive star map of a game's model-liked legs.

The dashboard's signature element (DESIGN.md §4a) and the slip editor's primary
control. The game's strongest model-liked legs are stars — the top
``DEFAULT_STARS`` by edge with both teams represented and at most two per player;
the rest wait behind *look deeper* — each sized by its edge so the strongest read
biggest. The layout is **fixed per game** (the sports-object template the game was
dealt, or a team-anchored force-directed solve when it is too thin for one) and
never moves when you pick legs; ``constellation_spacing.settle`` then nudges only
the stars that would collide, so the biggest keep their vertices exactly. Captions
are sparse for the same reason — the slip's stars plus the biggest few candidates
carry one, everything else reads from its hover card. A star you've added to the
slip burns at full team color; a candidate you haven't is the same color
desaturated and dimmed — selection is alpha + saturation, not a ring (a gold ring
read as a team color). Stories / Builder / Moon just pre-activate a subset.

Edges are pairwise correlations (gold, width/opacity ∝ |ρ|, dashed when ρ < 0 —
"fights the thesis"). The whole web is drawn as a **faint base layer** so the
correlation structure reads before any pick; a tie whose **both stars are in the slip**
brightens to full gold — the bright edges are the correlations among your slip's own
legs, sketched over the rest, and hovering any star faint-previews its other ties. The
*layout* still springs on the full correlation web, so a star's placement reflects every
tie. Each team's most-connected leg is pinned to its side, so a cross-matchup leg floats
toward the centre and an unrepresented side leaves its half empty.

Two optional lenses layer onto the same figure (P8 Task C6) instead of living as
separate expanders below the map; ``constellation_deep`` and ``constellation_wider``
draw one each:

* ``deep_pool`` — "look deeper": the game's remaining legs — the ones the default
  cut left behind, then the model-passed ones — as small stars *inside* the map,
  sized and lit by edge and scattered by seeded draw into the neighbourhood of the
  main star each correlates with, carrying its ties. The main stars are held fixed,
  so revealing the tier can never reshuffle the lit map.
* ``wider_groups`` — "look wider": the map recedes a little and other games' best
  legs fill the open sky around it in per-game clusters, each thrown by seeded
  draw into its band, team-coloured and labelled with their game key — never a
  ring around the edge.

An in-slip leg beyond the default cut is *promoted* rather than laid out: it takes
the position the deeper lens would have given it and burns as a full star with the
lens on or off, so adding one can never re-deal the template.

The figure is pure (no Streamlit, no Archive): each node carries its
``Player|Market|Bet`` key plus its hover-card fields as ``customdata`` (the key at
index 0 — a plotly click turns into an add/remove), and each edge carries its two
endpoint keys in ``meta`` so the component's JS can dim-in a star's incident ties on
hover. It locks its own axes (no zoom/pan) — the builder hides the modebar. Positions
come from ``constellation_slate`` when the game carries a template and from
``constellation_spring``'s force solve when it doesn't; ``constellation_traces`` owns
the node and edge traces and the text they carry, ``constellation_deep_layer`` the
deeper lens's layer. The shape half — which template a night's games are dealt, where
the stars sit on one, and the engraving beneath them — all lives in
``constellation_slate``; this module is the figure's orchestrator.
Team fills read ``theme.team_colors(league, team)`` — real per-team primaries from
``team_assets.json``, never gold (gold is the correlation-edge color).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_deep import (
    DEEP_SIZE_MAX,
    DEEP_SIZE_MAX_MOBILE,
    DEEP_SIZE_MIN,
    DEEP_SIZE_MIN_MOBILE,
)
from sportstradamus.dashboard.components.constellation_deep_layer import add_deep_layer
from sportstradamus.dashboard.components.constellation_slate import (
    SHAPE_SCALE,
    SHAPE_SCALE_MOBILE,
    add_decoration,
    game_edges,
    game_universe,
    rho_map,
    teams_of,
)
from sportstradamus.dashboard.components.constellation_spacing import (
    PX_PER_UNIT,
    PX_PER_UNIT_MOBILE,
    Y_RANGE,
    caption_positions,
    default_stars,
    settle,
)
from sportstradamus.dashboard.components.constellation_spring import star_positions
from sportstradamus.dashboard.components.constellation_traces import (
    LABEL_FONT_SIZE,
    LABEL_FONT_SIZE_MOBILE,
    SIZE_MIN,
    SIZE_MIN_MOBILE,
    add_edge,
    add_node_trace,
    add_team_tags,
    blank_figure,
    edge_scale,
    node_info,
)
from sportstradamus.dashboard.components.constellation_wider import (
    SKY_EXTRA_Y_MOBILE,
    WIDER_GAMES,
    WIDER_SCALE,
    WIDER_STAR_SIZE,
    WIDER_STAR_SIZE_MOBILE,
    add_wider_layer,
)
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.theme import team_colors


def _side_sign(x: float) -> float:
    """-1 / 0 / +1 for ``settle``'s ``side`` — 0 at ``x == 0`` stays legal on either half."""
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0


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
    so the first non-null value is always right.
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
    (``constellation_slate.slate_shapes``). ``None`` — a game too thin to carry
    one, or a slate with nothing left to deal — reproduces the spring layout
    byte-for-byte, so the shapeless path is exactly today's figure. The map is
    never captioned with the shape's name: an engraving a viewer has to be told
    the name of isn't reading, so the drawing has to carry it alone.
    """
    fig = blank_figure()
    universe = game_universe(pool, slip_legs)
    info = {key: node_info(leg) for key, leg in universe.items()}
    if not info:
        return fig
    game = _pool_field(pool, slip_legs, column="Game", key="game")
    league = _pool_field(pool, slip_legs, column="League", key="league")
    teams = teams_of(game)

    active = {corr_key(leg) for leg in slip_legs} & set(info)
    keys = default_stars(universe, teams)
    add_team_tags(fig, league, teams)
    team_color = {team: team_colors(league, team)[0] for team in teams}
    rho = rho_map(corr, game)
    edges = game_edges(keys, rho)
    floor, label_size, shape_scale, px, lens_size, deep_span, sky_y = (
        (
            SIZE_MIN_MOBILE,
            LABEL_FONT_SIZE_MOBILE,
            SHAPE_SCALE_MOBILE,
            PX_PER_UNIT_MOBILE,
            WIDER_STAR_SIZE_MOBILE,
            (DEEP_SIZE_MIN_MOBILE, DEEP_SIZE_MAX_MOBILE),
            Y_RANGE + SKY_EXTRA_Y_MOBILE,
        )
        if mobile
        else (
            SIZE_MIN,
            LABEL_FONT_SIZE,
            SHAPE_SCALE,
            PX_PER_UNIT,
            WIDER_STAR_SIZE,
            (DEEP_SIZE_MIN, DEEP_SIZE_MAX),
            Y_RANGE,
        )
    )
    pos, fillers = star_positions(
        keys,
        {k: info[k]["team"] for k in keys},
        teams,
        [(a, b, abs(r)) for a, b, r in edges],
        shape,
        shape_scale,
    )
    sizes = edge_scale(keys, info, floor=floor)
    focus_scale = WIDER_SCALE if wider_groups is not None else 1.0
    # Biggest first: a top-Kelly star keeps its vertex to the float, and only what
    # would collide with it moves, never across its own team's half of the axis.
    # Spaced against the px the viewer actually gets: "look wider" shrinks positions
    # but not marker sizes, so the solve has to see the shrunk scale, not the base one.
    pos = settle(
        {k: pos[k] for k in sorted(keys, key=lambda k: (-sizes[k], k))},
        sizes,
        (px[0] * focus_scale, px[1] * focus_scale),
        side={k: _side_sign(pos[k][0]) for k in keys},
    )
    pos = {k: (x * focus_scale, y * focus_scale) for k, (x, y) in pos.items()}
    if shape is not None:
        add_decoration(fig, shape, fillers, shape_scale, focus_scale)
    promoted = add_deep_layer(
        fig,
        deep_pool,
        keys,
        info,
        pos,
        sizes,
        active=active,
        edges=edges,
        rho=rho,
        teams=teams,
        team_color=team_color,
        px=px,
        floor=floor,
        deep_span=deep_span,
    )
    keys += promoted
    captions = caption_positions(
        keys,
        pos,
        sizes,
        {k: info[k]["label"] for k in keys},
        active,
        px,
        font_px=label_size,
    )
    for a, b, r in edges:
        add_edge(fig, a, b, pos[a], pos[b], r, active=active)
    add_node_trace(
        fig,
        [k for k in keys if k not in active],
        pos,
        info,
        sizes,
        team_color,
        captions,
        active=False,
        label_size=label_size,
    )
    add_node_trace(
        fig,
        [k for k in keys if k in active],
        pos,
        info,
        sizes,
        team_color,
        captions,
        active=True,
        label_size=label_size,
    )
    if wider_groups is not None:
        groups = wider_groups[:WIDER_GAMES]
        winfo = {corr_key(row): node_info(row) for _, rows in groups for row in rows}
        add_wider_layer(
            fig,
            groups,
            winfo,
            pos,
            sizes,
            px,
            size=lens_size,
            label_size=label_size,
            sky_y=sky_y,
        )
    return fig
