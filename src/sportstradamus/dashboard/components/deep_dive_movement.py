"""The Movement tab of the offer-detail dialog: how the app has moved this offer's line.

``deep_dive.show_detail`` matches the offer's ``current_line_movement`` row on
``LINE_MOVEMENT_KEYS`` and hands it here. The tab leads with
``spark_svg.movement_summary``, the wording every surface uses for a move, then draws the
row's per-poll ``changes`` as two steps through time. The fair line takes the Board
trace's color. It is where the app's price would be even money, so a multiplier move
shows while the posted line holds. The posted main line is gray. The dashed rule is the
offer's own line, which on a Sleeper alt rung sits off the main line the steps follow.
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.deep_dive_charts import app_line_rule
from sportstradamus.dashboard.components.spark_svg import move_color, movement_summary
from sportstradamus.dashboard.theme import GRAY


def movement_chart(changes: list[dict], line: float, bet: str) -> alt.Chart:
    """The fair and posted lines as steps through time, against this offer's dashed line.

    Each step holds until the next logged poll (``step-after``), since ``changes`` lists
    only the polls where a line moved, plus the first and the last. Times go in as
    tz-aware UTC, so Vega parses each as an instant and shows it in the viewer's own zone.

    Args:
        changes: The movement row's decoded ``changes``, oldest first: ``t`` (ISO-8601
            with its UTC offset), the posted ``line``, its implied ``p_over`` and the
            ``fair`` line, both lines in stat units.
        line: The offer's own line, drawn as the dashed rule.
        bet: The offer's side, which colors the fair step through :func:`move_color`.

    Returns:
        The layered chart, with no legend: the tab's caption names the encodings.
    """
    polls = pd.DataFrame(changes)
    polls["t"] = pd.to_datetime(polls["t"], utc=True)
    fair_color = move_color(polls["fair"].iloc[-1] - polls["fair"].iloc[0], bet)
    x = alt.X("t:T", title="")
    # Off zero: a 250.5 passing-yards line would otherwise flatten against the baseline.
    y_scale = alt.Scale(zero=False)
    base = alt.Chart(polls)
    posted = base.mark_line(interpolate="step-after", color=GRAY, strokeWidth=2).encode(
        x=x, y=alt.Y("line:Q", title="Line", scale=y_scale)
    )
    fair = base.mark_line(interpolate="step-after", color=fair_color, strokeWidth=2).encode(
        x=x, y=alt.Y("fair:Q", title="Line", scale=y_scale)
    )
    points = base.mark_circle(color=fair_color, size=40).encode(
        x=x,
        y="fair:Q",
        tooltip=[
            alt.Tooltip("t:T", title="Time", format="%b %-d, %-I:%M %p"),
            alt.Tooltip("line:Q", title="Posted line"),
            alt.Tooltip("p_over:Q", title="Implied over", format=".1%"),
            alt.Tooltip("fair:Q", title="Fair line"),
        ],
    )
    return posted + fair + app_line_rule(line, "y") + points


def render_movement_tab(movement: pd.Series | None, row: pd.Series) -> None:
    """The offer's line history: its summary, the step chart, and a key to the chart.

    A missing row reads as no history. So does a row from a snapshot written before the
    fair-line columns existed, whose ``changes`` stays blank until the next
    ``prophecize``; the Board treats it the same way. An offer polled once gets its
    summary but no chart.
    """
    if movement is None or not isinstance(movement["changes"], str):
        st.caption("No line history is recorded for this offer yet.")
        return
    summary = movement_summary(
        json.loads(movement["fair_series"]),
        json.loads(movement["series"]),
        n_moves=int(movement["n_moves"]),
        n_price_moves=int(movement["n_price_moves"]),
    )
    # Plex Mono for the numerals (DESIGN §2); a backtick code span would paint the line in
    # codeTextColor's green, whichever way the move went.
    st.markdown(
        f"<div style=\"font-family:'IBM Plex Mono',monospace\">{summary}</div>",
        unsafe_allow_html=True,
    )
    changes = json.loads(movement["changes"])
    if len(changes) == 1:
        st.caption("Only one poll so far, so there is no movement to chart yet.")
        return
    st.altair_chart(movement_chart(changes, row["Line"], row["Bet"]), width="stretch")
    st.caption(
        "Fair line (step with dots): where the app's price would be even money · "
        ":gray[the app's posted line (plain gray step)] · this offer's line (dashed white)"
    )
