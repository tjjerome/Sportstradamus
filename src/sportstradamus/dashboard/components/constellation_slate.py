"""Everything upstream of drawing a constellation's stars, plus the engraving under them.

One night's offers come in; per game this decides which legs are stars, what the
correlation web among them looks like, which sports-object template the game was
dealt, where each star lands on it, and how the silhouette and outline get drawn.
``constellation.py`` owns the figure itself — the stars, the gold edges, the two
lenses; this owns the shape half. The catalog lives in ``constellation_shapes``
and the graph maths in ``constellation_layout``.

Dealing is per **date**, not per league. A shape is a game's identity for the
night, so no two games showing at once may wear the same one — which means the
deal has to see every league's slate at once. Each game still draws only from
what its own league is eligible for: the general library of unaffiliated objects
that most of the bank is, plus that league's own equipment. Another league's gear
is never on the table, so an NFL night cannot be dealt The Bat.

Deliberately uncached at every level: the catalog's ``tuning`` block is the
owner's live surface, and an edit has to reclassify and re-deal on the very next
browser rerun.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

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
from sportstradamus.dashboard.components.constellation_spacing import default_stars
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.leg_schema import is_model_liked, leg_field

# |ρ| floor to draw an edge or weight the layout — the story-menu edge floor.
_MIN_EDGE_RHO = 0.05

# The Phase D decoration layer: an engraving under the stars, never gold. Gold is
# the correlation-edge color and nothing else, so no engraved stroke can be
# misread as a ρ tie.
SILHOUETTE_ALPHA = 0.13  # faint intent signal, below the ambient-decoration ceiling
_SILHOUETTE_FILL = f"rgba(95,107,128,{SILHOUETTE_ALPHA})"  # constellation._DEEP_COLOR #5f6b80
_OUTLINE_COLOR = "rgba(230,233,239,0.22)"  # theme TEXT #E6E9EF
_OUTLINE_GLOW_COLOR = "rgba(230,233,239,0.08)"  # the same stroke, wider and fainter
_FILLER_COLOR = "rgba(138,145,160,0.30)"  # theme GRAY #8A91A0
_OUTLINE_WIDTH, _OUTLINE_GLOW_WIDTH = 1, 4
FILLER_SIZE = 6  # under the star floor, so a filler can never be mistaken for a leg
DECORATION = "decoration"

# Templates are authored in a true square; the figure's frame is not one, and which
# way it leans flips with the viewport. Desktop puts 3.2 x-units across ~980px against
# 2.8 of y over a 380px figure — a circle renders ~2.4x wide. The phone is near-square
# at the same height and inverts it to ~0.87x, so one constant pair would round the
# medal and turn the diamond into a kite. Each viewport therefore carries its own
# (x, y), sized to fill the frame and land a circle ~1.2x wide: correcting the last of
# it would squeeze the desktop map into a portrait strip and throw away the left-right
# spread the two teams are read by, and ~1.2x is inside "loose".
SHAPE_SCALE = (0.69, 1.38)
SHAPE_SCALE_MOBILE = (1.30, 0.98)


def game_universe(pool: pd.DataFrame | None, slip_legs: Sequence[Mapping]) -> dict[str, Mapping]:
    """``leg key -> the row that defines it``: model-liked legs plus any active slip leg.

    The static node set — it depends on the game's offers, not on which legs are in
    the slip (the slip only decides which stars light up) — so the map never
    reshuffles on a pick. An active leg whose offer has slipped below K > 0 is kept
    so a slip never loses one of its own stars. Returns the defining row rather than
    anything derived from it because the two callers want different things out of it:
    the figure reads a star's whole hover card, the classifier only its team.
    """
    universe: dict[str, Mapping] = {}
    if pool is not None and not pool.empty:
        for row in pool.to_dict("records"):
            if is_model_liked(row):
                universe.setdefault(corr_key(row), row)
    for leg in slip_legs:
        universe.setdefault(corr_key(leg), leg)
    return universe


def teams_of(game: str) -> list[str]:
    """The matchup's two team codes, sorted — the two anchored sides."""
    return sorted(set(game.split("/"))) if game else []


def rho_map(corr: pd.DataFrame | None, game: str) -> dict[frozenset, float]:
    """Map ``frozenset(leg_a, leg_b) -> rho`` for one game's ``current_game_corr`` slice."""
    if corr is None or corr.empty:
        return {}
    slice_df = corr.loc[corr["Game"] == game] if game else corr
    return {
        frozenset((row.leg_a, row.leg_b)): float(row.rho)
        for row in slice_df.itertuples(index=False)
    }


def game_edges(keys: list[str], rho: dict[frozenset, float]) -> list[tuple[str, str, float]]:
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


def supernodes(
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


def template_positions(
    nodes: list[str],
    node_team: dict[str, str | None],
    teams: list[str],
    edges: list[tuple[str, str, float]],
    template: dict,
    scale: tuple[float, float],
) -> tuple[dict[str, tuple[float, float]], list[int]]:
    """Star positions on ``template``, plus the vertices no star filled.

    A player's tightly-tied legs first collapse into a supernode so a three-leg knot
    claims a single vertex rather than eating three, then the supernodes take
    vertices and explode back into their legs. Template coordinates never go through
    the spring layout's ``_rescale``: the stars have to land on the silhouette, which
    is authored in the same [-1, 1] box, and the same template has to stay visibly
    the same shape from one game to the next. The only thing applied to them is
    ``scale``, the viewport's aspect correction, which the engraving takes too.
    """
    clusters, super_team, collapsed = supernodes(nodes, node_team, edges)
    placed, fillers = assign_stars(sorted(clusters), super_team, teams, collapsed, template)
    sx, sy = scale
    exploded = explode_clusters(placed, clusters)
    return {key: (x * sx, y * sy) for key, (x, y) in exploded.items()}, fillers


class GameShape(NamedTuple):
    """What one game was dealt for the night, and the reading behind it."""

    template: dict | None
    label: str | None
    topology: str
    n_supernodes: int
    readings: dict


def slate_shapes(
    offers: pd.DataFrame, corr: pd.DataFrame | None, date: str
) -> dict[str, GameShape]:
    """Deal every game on the night its own constellation, across every league at once.

    Keyed by game, over the whole date rather than one league or one platform, so a
    game keeps its shape when the user switches Underdog ↔ Sleeper or narrows the
    sport filter, and no two games showing at the same time wear the same shape.
    The graph classified is the one actually drawn — the capped default star set,
    not the whole model-liked pool — so the tuning cockpit's readings and
    ``n_supernodes`` describe what a viewer sees.
    """
    cfg = tuning()
    catalog = shape_catalog()["templates"]
    graphs = {}
    for game, group in offers.groupby("Game"):
        universe = game_universe(group, [])
        keys = default_stars(universe, teams_of(str(game)))
        node_team = {key: leg_field(universe[key], "team") for key in keys}
        edges = [(a, b, abs(r)) for a, b, r in game_edges(keys, rho_map(corr, str(game)))]
        clusters, super_team, collapsed = supernodes(keys, node_team, edges)
        topo, readings = topology_class(sorted(clusters), super_team, collapsed, cfg)
        graphs[str(game)] = (str(group["League"].iloc[0]), topo, readings, len(clusters))

    dealt = assign_templates(
        date, [(game, league, topo, n) for game, (league, topo, _, n) in graphs.items()], cfg
    )
    return {
        game: GameShape(
            template=catalog[dealt[game]] if dealt[game] else None,
            label=catalog[dealt[game]]["label"] if dealt[game] else None,
            topology=topo,
            n_supernodes=n,
            readings=readings,
        )
        for game, (_, topo, readings, n) in graphs.items()
    }


def scale_path(path: str, sx: float, sy: float) -> str:
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


def add_decoration(
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
        path=scale_path(template["silhouette"], sx, sy),
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
                name=DECORATION,
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
                marker={"size": FILLER_SIZE, "color": _FILLER_COLOR},
                name=DECORATION,
                hoverinfo="skip",
                showlegend=False,
            )
        )
