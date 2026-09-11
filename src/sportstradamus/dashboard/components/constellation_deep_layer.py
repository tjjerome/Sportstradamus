"""The deeper lens on the figure — what ``constellation_figure`` draws for it.

``constellation_deep`` owns the lens's geometry (where a tier star lands) and its
trace; this module runs that for one figure: it folds the lens's rows into the
node info, promotes the slip's beyond-the-cut legs to full stars whether the lens
is on or not, sizes and places the tier, and brings in its ties — capped per star,
named ``deep_edge`` so ``main.js`` fades them with the stars they belong to.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_deep import (
    _DEEP_COLOR,
    DEEP_ALPHA_MIN,
    DEEP_EDGES_PER_STAR,
    add_deep_trace,
    deep_positions,
    deep_tier,
)
from sportstradamus.dashboard.components.constellation_slate import game_edges
from sportstradamus.dashboard.components.constellation_traces import (
    _INACTIVE_ALPHA,
    _INACTIVE_DESAT,
    add_edge,
    desaturate,
    edge_scale,
    node_info,
)
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.theme import GRAY


def add_deep_layer(
    fig: go.Figure,
    deep_pool: pd.DataFrame | None,
    keys: list[str],
    info: dict[str, dict],
    pos: dict[str, tuple[float, float]],
    sizes: dict[str, float],
    *,
    active: set[str],
    edges: list[tuple[str, str, float]],
    rho: dict[frozenset, float],
    teams: list[str],
    team_color: dict[str, str],
    px: tuple[float, float],
    floor: float,
    deep_span: tuple[float, float],
) -> list[str]:
    """Draw the deeper lens and return the slip legs it promotes to full stars.

    ``info``, ``pos`` and ``sizes`` gain the drawn tier in place. An in-slip leg
    beyond the default cut is promoted whether or not the lens is on, because it
    has to burn in the spot the lens would have given it — which is why ``drawn``
    leads with the promoted keys, whose placement priority must not move when the
    lens grows the tier behind them. The unpicked stars and their ties are gated
    on the lens itself, so a lens-off figure with nothing beyond the cut stays
    byte-identical to today's.

    Recomputing ``edge_scale`` over the promoted legs cannot move a main star's
    size: the top-Kelly leg is always inside the default cut, so the scale's
    denominator is the one ``settle`` already spaced against. ``deep_span`` is the
    lens tier's own size band, scaled over the tier's own strongest leg, so the
    two scales never share a denominator.
    """
    if deep_pool is not None:
        info |= {
            corr_key(row): node_info(row)
            for row in deep_pool.to_dict("records")
            if corr_key(row) not in info
        }
    tier = deep_tier(info, keys)
    if not tier:
        return []
    promoted = [key for key in tier if key in active]
    deep = [key for key in tier if key not in active] if deep_pool is not None else []
    drawn = promoted + deep
    lo, hi = deep_span
    sizes |= edge_scale(deep, info, floor=lo, ceiling=hi) | edge_scale(
        keys + promoted, info, floor=floor
    )
    ties = game_edges(keys + drawn, rho)
    pos |= deep_positions(
        drawn,
        {key: pos[key] for key in keys},
        sizes,
        ties,
        {key: info[key]["team"] for key in drawn},
        teams,
        px,
    )
    alphas = edge_scale(deep, info, floor=DEEP_ALPHA_MIN, ceiling=_INACTIVE_ALPHA)
    add_deep_trace(
        fig,
        deep,
        pos,
        info,
        sizes=sizes,
        # A liked leg the cut left behind is a candidate, just smaller; only the
        # model-passed tier wears the lens's own gray.
        colors=[
            desaturate(team_color.get(info[k]["team"], GRAY), _INACTIVE_DESAT)
            if info[k]["edge"] > 0
            else _DEEP_COLOR
            for k in deep
        ],
        alphas=[alphas[k] for k in deep],
    )
    _add_lens_edges(
        fig,
        ties,
        {(a, b) for a, b, _ in edges},
        pos,
        active=active,
        deep=set(deep),
    )
    return promoted


def _add_lens_edges(
    fig: go.Figure,
    edges: list[tuple[str, str, float]],
    main_pairs: set[tuple[str, str]],
    pos: dict[str, tuple[float, float]],
    *,
    active: set[str],
    deep: set[str],
) -> None:
    """The ties the deeper lens brings in, minus the ones the base web already drew.

    A promoted star is lit with the lens off too, so its ties are permanent
    ``edge`` traces; a deep star's ties carry the ``deep_edge`` name instead, which
    is how ``main.js`` gates the lens animation on them; a deep star keeps only its
    ``DEEP_EDGES_PER_STAR`` strongest.
    """
    kept = _capped_deep_ties(edges, deep)
    for node_a, node_b, tie in edges:
        if (node_a, node_b) in main_pairs:
            continue
        lens = node_a in deep or node_b in deep
        if lens and (node_a, node_b) not in kept:
            continue
        add_edge(
            fig,
            node_a,
            node_b,
            pos[node_a],
            pos[node_b],
            tie,
            active=active,
            name="deep_edge" if lens else "edge",
        )


def _capped_deep_ties(edges: list[tuple[str, str, float]], deep: set[str]) -> set[tuple[str, str]]:
    """Each deep star's strongest ties by |rho|, at most ``DEEP_EDGES_PER_STAR``.

    Deep-to-deep ties never enter the pool — at lens size they are clutter with
    nothing to read against — so they are the pairs the cap drops outright.
    """
    incident: defaultdict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for node_a, node_b, rho in edges:
        if node_a in deep and node_b in deep:
            continue
        for key in (node_a, node_b):
            if key in deep:
                incident[key].append((node_a, node_b, rho))
    return {
        (node_a, node_b)
        for ties in incident.values()
        for node_a, node_b, _ in sorted(ties, key=lambda tie: (-abs(tie[2]), tie[:2]))[
            :DEEP_EDGES_PER_STAR
        ]
    }
