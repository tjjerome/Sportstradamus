"""Receipt-ticket builder + thin render for Receipts' "Your slips" section.

``build_tickets`` is the pure display-precompute step (status/color/stamp,
per-leg hit/miss/push outcomes, EV beside Model EV, stake, return-vs-to-win) — unit-tested
directly against hand-built ``user_slips``-shaped fixture rows.
``render_tickets`` is the thin Streamlit render: a 2-column card grid, no
business logic.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.components.slip_readout import ev_metrics
from sportstradamus.leg_schema import leg_label

# Won/lost/partial share the graded look; pending is its own state marker
# (gold), not a value — same "state, not value" gold usage DESIGN.md already
# sanctions for the reliability diagram's alt-line series (§4a). Push (every leg
# voided — nightly._slip_status returns it when effective == 0) gets the same
# neutral gray as a pushed leg's own outcome tag below.
_STATUS_STYLE = {
    "won": (theme.GREEN, "WON"),
    "lost": (theme.RED, "LOST"),
    "partial": (theme.ORANGE, "PARTIAL"),
    "pending": (theme.GOLD, "PENDING"),
    "push": (theme.GRAY, "PUSH"),
}

# Text words, not glyphs: DESIGN.md §5 bans emoji/dingbats as status iconography and
# requires color *and* text together for status, never color alone.
_LEG_OUTCOMES = {0: "hit", 1: "miss"}
_LEG_OUTCOME_PUSH = "push"


def _leg_outcomes(row: pd.Series) -> list[tuple[str, str]]:
    """Per-leg ``(label, outcome)`` pairs: hit/miss/push when graded, blank while pending."""
    result_json = row.get("result")
    if not isinstance(result_json, str):
        return [(leg_label(leg), "") for leg in row["legs"]]
    outcomes = json.loads(result_json)
    return [
        (leg_label(leg), _LEG_OUTCOMES.get(outcome, _LEG_OUTCOME_PUSH))
        for leg, outcome in zip(row["legs"], outcomes, strict=True)
    ]


def _value(row: pd.Series, status: str) -> tuple[str, float]:
    """``(label, dollar_amount)``: realized "Return" once graded, potential
    "To win" while still pending. A ``partial`` row's ``payout_realized`` is
    NaN today (``nightly._grade_slip``'s realized dict has no ``"partial"``
    key), so this treats "ungraded dollar figure" as 0 rather than crashing
    or fabricating a number from NaN.
    """
    stake = float(row.get("stake", 0.0) or 0.0)
    if status == "pending":
        multiplier = float(row.get("payout_multiplier", 0.0) or 0.0)
        return "To win", stake * multiplier
    realized = row.get("payout_realized")
    realized = float(realized) if pd.notna(realized) else 0.0
    return "Return", stake * realized


def build_tickets(user_slips: pd.DataFrame) -> list[dict]:
    """One display-ready dict per slip: status/color/stamp, headline, meta
    line, per-leg ``(label, outcome)`` pairs, EV, Model EV, stake, and the value/label pair.

    ``ev`` is the fitted value the slip was locked with, ``None`` on a slip the fit did not
    price or one locked before it priced slips; ``model_ev`` is the model's claim.
    """
    if user_slips.empty:
        return []
    tickets = []
    for _, row in user_slips.iterrows():
        status = row.get("status", "pending") or "pending"
        color, stamp = _STATUS_STYLE[status]
        value_label, value = _value(row, status)
        n_legs = len(row["legs"])
        short_date = str(row["saved_at"])[:10]
        tickets.append(
            {
                "status": status,
                "color": color,
                "stamp": stamp,
                "headline": row.get("headline") or "Custom slip",
                "meta": f"{row['platform']} · {row.get('play_type', '')} · "
                f"{n_legs} legs · {short_date}".replace("  ", " "),
                "legs": _leg_outcomes(row),
                "ev": float(row["ev"]) if pd.notna(row.get("ev")) else None,
                "model_ev": float(row["model_ev"]),
                "stake": float(row.get("stake", 0.0) or 0.0),
                "value_label": value_label,
                "value": value,
            }
        )
    return tickets


# Streamlit's ``:color[text]`` markdown directive only accepts these keywords;
# map each stamp's theme hex to the closest one (§5: color *and* text together).
_BADGE_COLOR_KEYWORD = {
    theme.GREEN: "green",
    theme.RED: "red",
    theme.ORANGE: "orange",
    theme.GOLD: "orange",
    theme.GRAY: "gray",
}
_OUTCOME_COLOR_KEYWORD = {"hit": "green", "miss": "red", "push": "gray", "": "gray"}

_TICKETS_PER_ROW = 2

# Status rail: a colored strip distinct from the stamp badge (mirrors the rail concept
# gate_matrix.py/grid.py use for AG-Grid rows, adapted here since st.container(border=True)
# has no per-side/per-color border of its own — a top strip is the plain-HTML equivalent
# that still nests inside the container ahead of the native st.* calls below it).
_RAIL_HEIGHT_PX = 4


def render_tickets(tickets: list[dict]) -> None:
    """Render the ticket grid: 2 cards per row, status-colored rail + stamp badge."""
    for i in range(0, len(tickets), _TICKETS_PER_ROW):
        cols = st.columns(_TICKETS_PER_ROW)
        for col, ticket in zip(cols, tickets[i : i + _TICKETS_PER_ROW], strict=False):
            with col, st.container(border=True):
                st.markdown(
                    f'<div style="height:{_RAIL_HEIGHT_PX}px;margin:-1px -1px 10px -1px;'
                    f"border-radius:3px 3px 0 0;"
                    f'background:{html.escape(str(ticket["color"]), quote=True)}"></div>',
                    unsafe_allow_html=True,
                )
                badge = _BADGE_COLOR_KEYWORD.get(ticket["color"], "gray")
                st.markdown(f":{badge}[**{ticket['stamp']}**]  {ticket['headline']}")
                st.caption(ticket["meta"])
                for label, outcome in ticket["legs"]:
                    tag = outcome or "pending"
                    st.caption(f"{label}  :{_OUTCOME_COLOR_KEYWORD[outcome]}[{tag}]")
                metrics = [
                    *ev_metrics(ticket["ev"], ticket["model_ev"]),
                    ("Stake", f"${ticket['stake']:,.0f}"),
                    (ticket["value_label"], f"${ticket['value']:,.0f}"),
                ]
                for col, (label, value) in zip(st.columns(len(metrics)), metrics, strict=True):
                    col.metric(label, value)
