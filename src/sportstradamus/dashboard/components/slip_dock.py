"""Mobile slip dock — a fixed bottom bar that expands into the slip sheet.

The phone replacement for the sidebar rail's ambient visibility (Phase M spec
§3): collapsed, a one-line summary of the active slip rides the bottom of the
viewport on every surface; expanded, the sheet lists the legs with remove
controls, prices the slip through the same ``score_slip`` path the builders
use, and carries the bankroll + Lock it in! controls. Renders nothing without
legs, so desktop (which never mounts it) and an empty phone session are
unaffected. Money is ``Decimal``; scoring reuses slip_state/slip_builder
helpers — no duplicated math.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from decimal import Decimal

import streamlit as st

from sportstradamus.dashboard.components.slip_builder import slip_shrinkage
from sportstradamus.dashboard.components.slip_state import (
    _BANKROLL,
    _LEGS,
    _PLATFORM,
    bankroll_input,
    clear_slip,
    lock_in,
    remove_leg,
)
from sportstradamus.dashboard.data import load_current_game_corr
from sportstradamus.dashboard.slip_engine import SlipScore, score_slip
from sportstradamus.leg_schema import leg_label
from sportstradamus.prediction.stories.legs import validate_parlay_legs

_OPEN = "slip_dock_open"

# Fixed-bottom chrome for the .st-key-slip_dock container. Token literals restated
# because an injected <style> can't read config.toml (same convention as the
# constellation component's index.html): surface #1A1D24, text #E6E9EF, gold #C9A227.
# The main-container bottom padding keeps the last page row tappable above the bar.
_DOCK_CSS = """
<style>
.st-key-slip_dock{position:fixed;left:0;right:0;bottom:0;z-index:400;
  background:#1A1D24;border-top:1px solid rgba(201,162,39,.42);border-radius:4px 4px 0 0;
  box-shadow:0 -4px 18px rgba(0,0,0,.45);padding:8px 12px 10px !important;
  max-height:70vh;overflow-y:auto;overflow-x:hidden}
.st-key-slip_dock [data-testid="stVerticalBlock"]{gap:.35rem}
/* Streamlit stacks st.columns under ~640px (each column goes flex-basis:100%); the
   summary+chevron bar must stay one row, so pin the row nowrap and re-flex the columns:
   text column grows, control columns shrink to content. */
.st-key-slip_dock [data-testid="stHorizontalBlock"]{flex-wrap:nowrap;align-items:center}
.st-key-slip_dock [data-testid="stColumn"]{flex:0 0 auto;width:auto !important;min-width:0}
.st-key-slip_dock [data-testid="stColumn"]:first-child{flex:1 1 auto}
.slip-dock-line{font-family:'IBM Plex Mono',monospace;font-size:13px;color:#E6E9EF;margin:2px 0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
[data-testid="stMainBlockContainer"]{padding-bottom:120px}
</style>
"""


def render_slip_dock() -> None:
    """Render the bar (and, when toggled open, the sheet) for a non-empty slip."""
    legs = st.session_state[_LEGS]
    if not legs:
        return
    st.markdown(_DOCK_CSS, unsafe_allow_html=True)
    score = _price(legs)
    with st.container(key="slip_dock"):
        bar_col, toggle_col = st.columns([5, 1])
        bar_col.markdown(
            f'<div class="slip-dock-line">{_summary(legs, score)}</div>',
            unsafe_allow_html=True,
        )
        opened = st.session_state.get(_OPEN, False)
        if toggle_col.button(
            ":material/expand_less:" if not opened else ":material/expand_more:",
            key="slip_dock_toggle",
            help="Open the slip",
        ):
            st.session_state[_OPEN] = not opened
            st.rerun()
        if opened:
            _render_sheet(legs, score)


def _price(legs: Sequence[dict]) -> SlipScore | None:
    """SlipScore for ≥2 legs, else None — the same gate the builders apply."""
    if len(legs) < 2:
        return None
    return score_slip(
        legs,
        load_current_game_corr(),
        platform=st.session_state[_PLATFORM],
        bankroll=Decimal(str(st.session_state[_BANKROLL])),
        shrinkage=slip_shrinkage(legs),
    )


def _summary(legs: Sequence[dict], score: SlipScore | None) -> str:
    if score is None:
        return f"{len(legs)} leg · add another to price"
    return (
        f"{len(legs)} legs · {float(score.payout):.2f}x · "
        f"EV {float(score.model_ev) - 1:+.0%} · ${score.stake}"
    )


def _render_sheet(legs: Sequence[dict], score: SlipScore | None) -> None:
    for i, leg in enumerate(legs):
        text_col, rm_col = st.columns([8, 1])
        text_col.markdown(
            f'<div class="slip-dock-line">{html.escape(leg_label(leg))} · '
            f"{html.escape(str(leg['league']))}</div>",
            unsafe_allow_html=True,
        )
        if rm_col.button(":material/close:", key=f"slip_dock_rm_{i}", help="Remove leg"):
            remove_leg(i)
            st.rerun()
    bankroll_input(key="_bankroll_dock")
    if score is None:
        st.caption("Select at least two legs to price the slip.")
        return
    valid, reason = validate_parlay_legs(legs)
    if not valid:
        st.warning(reason)
    st.markdown(
        f'<div class="slip-dock-line">Kelly stake ${score.stake} · '
        f"joint {float(score.joint_p):.0%}</div>",
        unsafe_allow_html=True,
    )
    lock_col, clear_col = st.columns(2)
    if lock_col.button("Lock it in!", key="slip_dock_lock", type="primary", disabled=not valid):
        lock_in(score, "", slip_shrinkage(legs))
    if clear_col.button("Clear", key="slip_dock_clear"):
        clear_slip()
        st.rerun()
