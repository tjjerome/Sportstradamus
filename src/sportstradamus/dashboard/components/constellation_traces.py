"""What a star says and how it is drawn — the constellation's traces and their text.

``constellation.py`` orchestrates the figure; this module owns the pieces it and
the lenses draw with: the caption, hover text and card fields a node carries
(:func:`star_label`, :func:`node_info`), the node and edge traces, the two team
tags, the blank locked frame, and the one edge-proportional scale the main map and
the deeper lens's tier both rank themselves by (:func:`edge_scale`). Nothing here
decides where a star sits — positions arrive from ``constellation_spring``,
``constellation_slate`` and the lens modules.
"""

from __future__ import annotations

from collections.abc import Mapping

import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_spacing import X_RANGE, Y_RANGE
from sportstradamus.dashboard.theme import GOLD, GRAY, team_name
from sportstradamus.helpers import market_display_name
from sportstradamus.leg_schema import leg_field, leg_field_float

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Star size scales with the leg's Kelly edge, relative to the game's strongest leg.
_SIZE_MIN = 14
_SIZE_MAX = 38
# A candidate (not-in-slip) star keeps most of its team hue — only a light blend toward
# gray plus reduced opacity marks it not-yet-picked. The old 0.55/0.45 crushed dark
# franchise colors to near-gray, so candidates read as a colorless field.
_INACTIVE_DESAT = 0.35
_INACTIVE_ALPHA = 0.60

_EDGE_WIDTH_MIN = 1.0
_EDGE_WIDTH_SPAN = 6.0  # width at |ρ|=1 ≈ 7px; weak ties stay hairlines for contrast
_EDGE_ALPHA_MIN = 0.25
# The whole correlation web is drawn barely-there at this alpha so the structure reads
# before any pick without drowning the field; a tie whose both endpoints are in the slip
# brightens to full gold (add_edge), well above this base.
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


def _last_name(player: str) -> str:
    parts = player.split()
    while len(parts) > 1 and parts[-1].rstrip(".").lower() in _NAME_SUFFIXES:
        parts.pop()
    return parts[-1] if parts else player


def _bet_word(bet) -> str:
    return "Over" if str(bet).lower().startswith("o") else "Under"


def _market_name(leg: Mapping) -> str:
    """The Board's name for this leg's market, resolved per leg.

    The *wider* lens can put two leagues' games on one map, and the same slug
    reads differently in each.
    """
    return market_display_name(
        str(leg_field(leg, "league", "") or ""), str(leg_field(leg, "market"))
    )


def star_label(leg: Mapping) -> str:
    """Compact star caption: ``Lastname Market o/u Line`` (e.g. ``Brunson Points o25.5``).

    ``leg`` is a canonical lowercase leg or a raw uppercase ``current_offers``
    row — ``leg_field`` bridges the two shapes (the constellation draws both a
    game's candidate pool and the slip's own legs on one map).
    """
    ou = "o" if _bet_word(leg_field(leg, "bet")) == "Over" else "u"
    return f"{_last_name(str(leg_field(leg, 'player')))} {_market_name(leg)} {ou}{float(leg_field(leg, 'line')):.10g}"


def _hover_text(leg: Mapping) -> str:
    p = leg_field_float(leg, "win_prob")
    boost = leg_field_float(leg, "boost", 1.0)
    k = leg_field_float(leg, "kelly")
    head = (
        f"{leg_field(leg, 'player')} — {_market_name(leg)} "
        f"{_bet_word(leg_field(leg, 'bet'))} {float(leg_field(leg, 'line')):.10g}"
    )
    return f"{head}<br>Win {p:.0%} · {boost:.2f}x · Kelly {k:.0%}"


def _card_fields(leg: Mapping) -> list:
    """Structured fields the hover card reads from a node's ``customdata`` (after the key)."""
    return [
        str(leg_field(leg, "player")),
        _market_name(leg),
        _bet_word(leg_field(leg, "bet")),
        float(leg_field(leg, "line")),
        leg_field_float(leg, "win_prob"),
        leg_field_float(leg, "boost", 1.0),
        leg_field_float(leg, "kelly"),
    ]


def node_info(leg: Mapping) -> dict:
    """Everything a node carries: its caption, team, edge, hover text and card fields."""
    return {
        "label": star_label(leg),
        "team": leg_field(leg, "team"),
        "edge": leg_field_float(leg, "kelly"),
        "hover": _hover_text(leg),
        "card": _card_fields(leg),
    }


def edge_scale(
    keys: list[str], info: dict[str, dict], *, floor: float, ceiling: float = _SIZE_MAX
) -> dict[str, float]:
    """Per-node value in ``[floor, ceiling]`` ∝ Kelly edge, over the strongest of ``keys``.

    Star size on the main map, size *and* opacity on the deeper lens's own tier —
    one scale so the two tiers rank themselves the same way at different volumes.
    A model-passed leg (edge ≤ 0) sits at the floor rather than below it, and a
    set with nothing positive in it is flat there.
    """
    top = max((info[k]["edge"] for k in keys), default=0.0)
    if top <= 0:
        return dict.fromkeys(keys, float(floor))
    span = ceiling - floor
    return {k: floor + max(info[k]["edge"], 0.0) / top * span for k in keys}


def blank_figure() -> go.Figure:
    """The empty locked frame every constellation is drawn into (no zoom, no pan)."""
    fig = go.Figure()
    fig.update_layout(
        height=_FIG_HEIGHT,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",  # transparent — the page starfield reads through the map
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        hovermode="closest",
        dragmode=False,  # no panning — this is a map, not a chart
        xaxis={"visible": False, "fixedrange": True, "range": [-X_RANGE, X_RANGE]},
        yaxis={"visible": False, "fixedrange": True, "range": [-Y_RANGE, Y_RANGE]},
    )
    return fig


def add_team_tags(fig: go.Figure, league: str, teams: list[str]) -> None:
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


def add_edge(
    fig: go.Figure,
    a: str,
    b: str,
    p0,
    p1,
    rho: float,
    *,
    active: set[str],
    name: str = "edge",
) -> None:
    """One correlation edge: gold, width/opacity ∝ |ρ|, dashed when ρ < 0.

    Drawn at a faint base alpha (``_EDGE_BASE_ALPHA``) so the whole web reads as a
    sketch; brightens to full ``|ρ|``-scaled gold only when **both** endpoints are in
    the slip, so the slip's own correlations stand out over the rest. ``meta`` carries
    the endpoint keys so the component's JS can faint-preview a star's other ties on hover.
    ``name`` is ``"deep_edge"`` for a tie one of the deeper lens's own stars owns, which
    is how the JS knows to fade it in and out with them.
    """
    incident = a in active and b in active
    fig.add_trace(
        go.Scatter(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            mode="lines",
            name=name,
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


def add_node_trace(
    fig: go.Figure,
    keys: list[str],
    pos: dict,
    info: dict[str, dict],
    sizes: dict[str, float],
    team_color: dict[str, str],
    captions: Mapping[str, str],
    *,
    active: bool,
    label_size: int = _LABEL_FONT_SIZE,
) -> None:
    """One scatter trace of stars: active = full team color, candidate = desaturated/dim.

    ``captions`` (``constellation_spacing.caption_positions``) decides which stars
    are labelled and where; the rest carry empty text and read from the hover card.
    Both active and candidate stars are eligible and active stars render on top —
    the active/candidate signal is the star's fill color and opacity, never the label.
    """
    if not keys:
        return
    base_colors = [team_color.get(info[k]["team"], GRAY) for k in keys]
    colors = [c if active else desaturate(c, _INACTIVE_DESAT) for c in base_colors]
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
            text=[info[k]["label"] if k in captions else "" for k in keys],
            textposition=[captions.get(k, "top center") for k in keys],
            textfont={
                "color": _ACTIVE_LABEL_COLOR if active else GRAY,
                "size": label_size,
            },
            customdata=[[k, *info[k]["card"], 1 if active else 0] for k in keys],
            hovertext=[info[k]["hover"] for k in keys],
            hoverinfo="none",  # the component draws the hover card; suppress the native tooltip
        )
    )


def desaturate(hex_color: str, amount: float) -> str:
    """Blend ``hex_color`` toward gray by ``amount`` ∈ [0, 1] (1 = full gray)."""
    rgb = _hex_rgb(hex_color)
    gray = _hex_rgb(GRAY)
    mixed = tuple(round(c + (g - c) * amount) for c, g in zip(rgb, gray, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
