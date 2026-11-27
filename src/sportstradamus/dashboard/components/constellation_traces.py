"""What a star says and how it is drawn — the constellation's traces and their text.

``constellation.py`` orchestrates the figure; this module owns the pieces it and
the lenses draw with: the caption, hover text and card fields a node carries
(:func:`star_label`, :func:`node_info`), the node traces and edge records, the two team
tags, the blank locked frame, and the one EV scale the main map and the deeper lens's
tier both rank themselves by (:func:`ev_scale`). Nothing here decides where a star
sits — positions arrive from ``constellation_spring``, ``constellation_slate`` and the
lens modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from sportstradamus.dashboard.components.constellation_spacing import X_RANGE, Y_RANGE
from sportstradamus.dashboard.theme import GRAY, GREEN, RED, team_name
from sportstradamus.helpers import market_display_name, platform_payout
from sportstradamus.leg_schema import is_pick, leg_field, leg_field_float

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Star size scales with the leg's EV, from the lowest to the highest EV drawn with it.
SIZE_MIN = 14
SIZE_MAX = 38
# A candidate (not-in-slip) star keeps most of its team hue — only a light blend toward
# gray plus reduced opacity marks it not yet added. The old 0.55/0.45 crushed dark
# franchise colors to near-gray, so candidates read as a colorless field.
INACTIVE_DESAT = 0.35
INACTIVE_ALPHA = 0.60
# Flat stroke width for the Over/Under border, every tier — matches the module's
# other flat stroke constants (_EDGE_WIDTH_MIN) rather than scaling with star size.
BET_BORDER_WIDTH = 1.5

_EDGE_WIDTH_MIN = 1.0
_EDGE_WIDTH_SPAN = 6.0  # width at |ρ|=1 ≈ 7px; weak ties stay hairlines for contrast
_EDGE_ALPHA_MIN = 0.25
# The whole correlation web is drawn barely-there at this alpha so the structure reads
# before any leg is added without drowning the field; a tie whose both endpoints are in
# the slip brightens to full gold (edge_record), well above this base.
EDGE_BASE_ALPHA = 0.03
FIG_HEIGHT = 380
LABEL_FONT_SIZE = 11  # active-star caption — small enough to fit in a dense game

# Phase M touch floors: a fingertip needs ~22px; the label lifts with it. The EV
# ordering (size = EV) survives — the floor compresses the range, never reorders it.
SIZE_MIN_MOBILE = 22
LABEL_FONT_SIZE_MOBILE = 13
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


def bet_color(bet_word: str) -> str:
    """GREEN for an Over call, RED for Under — never gold (DESIGN.md §4a)."""
    return GREEN if bet_word == "Over" else RED


def bet_marker_line(info: Mapping[str, dict], keys: Sequence[str]) -> dict:
    """Marker ``line`` style for a trace's Over/Under border, keyed off each key's ``card``."""
    return {
        "color": [bet_color(info[k]["card"][2]) for k in keys],
        "width": BET_BORDER_WIDTH,
    }


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
    last_name = _last_name(str(leg_field(leg, "player")))
    return f"{last_name} {_market_name(leg)} {ou}{float(leg_field(leg, 'line')):.10g}"


def node_info(leg: Mapping) -> dict:
    """Everything a node carries: its caption, team, numbers, pick flag, hover text and card.

    ``ev`` is the fitted EV, None on a leg the fit did not price (one side posted);
    ``model_ev`` is the model's own claim, its win chance times the payout. The hover
    leads with the fitted win chance and EV and gives the model's beside them, and
    ``card`` is what the component's hover card reads from ``customdata`` after the
    key: player, market, side, line, fitted win, boost, EV, model win, Model EV.
    """
    ev = leg_field(leg, "ev")
    ev = None if pd.isna(ev) else float(ev)
    fit = None if ev is None else float(leg_field(leg, "fit_prob"))
    win = leg_field_float(leg, "win_prob")
    boost = leg_field_float(leg, "boost", 1.0)
    model_ev = win * float(platform_payout(boost, leg_field(leg, "platform")))
    player = str(leg_field(leg, "player"))
    market = _market_name(leg)
    bet = _bet_word(leg_field(leg, "bet"))
    line = float(leg_field(leg, "line"))
    fitted = (
        f"{boost:.2f}x · no EV" if ev is None else f"Win {fit:.0%} · {boost:.2f}x · EV {ev:.2f}"
    )
    return {
        "label": star_label(leg),
        "team": leg_field(leg, "team"),
        "ev": ev,
        "model_ev": model_ev,
        "pick": is_pick(leg),
        "hover": (
            f"{player} — {market} {bet} {line:.10g}"
            f"<br>{fitted}<br>Model win {win:.0%} · Model EV {model_ev:.2f}"
        ),
        "card": [player, market, bet, line, fit, boost, ev, win, model_ev],
    }


def ev_scale(
    keys: Sequence[str],
    info: Mapping[str, dict],
    *,
    floor: float,
    ceiling: float = SIZE_MAX,
    among: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-node value in ``[floor, ceiling]`` by EV, lowest to highest EV of ``among``.

    Star size on the main map, size *and* opacity on the deeper lens's own tier —
    one scale so the two tiers rank themselves the same way at different volumes.
    EV spans a narrow band, so the scale runs from the lowest EV drawn (``among``,
    ``keys`` by default) to the highest, not from zero. A leg the fit did not price
    sits at the floor, a set of one EV sits at the ceiling, and a leg outside the
    ``among`` range clamps to its end, so sizing a newcomer against the drawn stars
    never resizes them.
    """
    evs = [info[k]["ev"] for k in (keys if among is None else among) if info[k]["ev"] is not None]
    lo, hi = min(evs, default=0.0), max(evs, default=0.0)

    def at(ev: float | None) -> float:
        if ev is None:
            return float(floor)
        share = (ev - lo) / (hi - lo) if hi > lo else float(ev >= hi)
        return floor + min(max(share, 0.0), 1.0) * (ceiling - floor)

    return {k: at(info[k]["ev"]) for k in keys}


def blank_figure() -> go.Figure:
    """The empty locked frame every constellation is drawn into (no zoom, no pan)."""
    fig = go.Figure()
    fig.update_layout(
        height=FIG_HEIGHT,
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


def edge_record(
    a: str,
    b: str,
    p0: tuple[float, float],
    p1: tuple[float, float],
    rho: float,
    *,
    active: set[str],
    lens: bool = False,
) -> dict:
    """One correlation edge as the ``layout.meta`` record ``main.js`` draws as an SVG line.

    Width ∝ |ρ|, dotted when ρ < 0, and no color: the JS strokes every edge in the
    gold design token. ``opacity`` is a faint base alpha (``EDGE_BASE_ALPHA``) so the
    whole web reads as a sketch, and ``lit`` — ``|ρ|``-scaled — only when **both**
    endpoints are in the slip, so the slip's own correlations stand out over the rest.
    Every record carries ``lit`` so a click can brighten a tie before the rerun lands.
    ``a`` and ``b`` let the JS faint-preview a star's other ties on hover; ``lens`` marks
    a tie one of the deeper lens's own stars owns, so it fades in and out with them.
    """
    lit = min(1.0, _EDGE_ALPHA_MIN + abs(rho))
    return {
        "a": a,
        "b": b,
        "x": [p0[0], p1[0]],
        "y": [p0[1], p1[1]],
        "width": _EDGE_WIDTH_MIN + abs(rho) * _EDGE_WIDTH_SPAN,
        "dash": "dot" if rho < 0 else "solid",
        "opacity": lit if a in active and b in active else EDGE_BASE_ALPHA,
        "lit": lit,
        "lens": lens,
    }


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
    label_size: int = LABEL_FONT_SIZE,
) -> None:
    """One scatter trace of stars: active = full team color, candidate = desaturated/dim.

    ``captions`` (``constellation_spacing.caption_positions``) decides which stars
    are labelled and where; the rest carry empty text and read from the hover card.
    Both active and candidate stars are eligible and active stars render on top —
    the active/candidate signal is the star's fill color and opacity, never the label.
    ``meta`` holds each star's fill in the other state, so the component can light or
    dim a clicked star before the rerun redraws it.
    """
    if not keys:
        return
    base_colors = [team_color.get(info[k]["team"], GRAY) for k in keys]
    dimmed = [desaturate(c, INACTIVE_DESAT) for c in base_colors]
    colors, flipped = (base_colors, dimmed) if active else (dimmed, base_colors)
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
                "opacity": 1.0 if active else INACTIVE_ALPHA,
                "line": bet_marker_line(info, keys),
            },
            text=[info[k]["label"] if k in captions else "" for k in keys],
            textposition=[captions.get(k, "top center") for k in keys],
            textfont={
                "color": _ACTIVE_LABEL_COLOR if active else GRAY,
                "size": label_size,
            },
            customdata=[[k, *info[k]["card"], 1 if active else 0] for k in keys],
            meta=flipped,
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
