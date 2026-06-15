"""The sidebar locked-slip shelf — mounted once on every surface.

Reads ``user_slips.parquet`` (the slips "locked in" from either builder) and
lists them compactly under the global bankroll control. Each entry carries the
payout / EV / stake / status at a glance, a **View** dialog that lays out every
leg (the reminder you read while placing the bet in the Underdog/Sleeper app),
an **Edit** that reopens it in its builder, and a **Delete**. Grading status is
filled in by nightly ``reflect``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_state import bankroll_input, load_slip
from sportstradamus.dashboard.data import load_current_offers, load_user_slips
from sportstradamus.helpers.io import delete_user_slip

# Headline characters shown before truncation in the narrow sidebar.
_SHELF_HEADLINE_CHARS = 48


def render_locked_shelf() -> None:
    """Render the bankroll control + the user's locked slips (newest first)."""
    st.markdown("### Your slips")
    bankroll_input()
    slips = load_user_slips()
    if slips.empty:
        st.caption("Lock in a slip to track it here.")
        return
    offers = load_current_offers()
    for row in slips.sort_values("saved_at", ascending=False).to_dict("records"):
        _render_shelf_entry(row, offers)


def _render_shelf_entry(row: Mapping, offers: pd.DataFrame) -> None:
    with st.container(border=True):
        st.caption((row.get("headline") or "Custom slip")[:_SHELF_HEADLINE_CHARS])
        st.write(
            f"{int(row['bet_size'])} legs · {row['platform']} {row.get('play_type', '')}".rstrip()
        )
        st.caption(
            f"{_fnum(row.get('payout_multiplier')):.2f}x · "
            f"EV {_fnum(row.get('model_ev'), 1.0) - 1:+.0%} · "
            f"${_fnum(row.get('stake')):,.2f} · {row.get('status') or 'pending'}"
        )
        view_col, edit_col, del_col = st.columns(3)
        if view_col.button("View", key=f"shelf_view_{row['slip_id']}"):
            _slip_dialog(row)
        target = (
            "surfaces/games.py" if row["builder_type"] == "constellation" else "surfaces/board.py"
        )
        if edit_col.button("Edit", key=f"shelf_edit_{row['slip_id']}"):
            load_slip(row, offers)
            st.switch_page(target)
        if del_col.button("Delete", key=f"shelf_del_{row['slip_id']}"):
            delete_user_slip(row["slip_id"])
            st.rerun()


@st.dialog("Slip detail", width="large")
def _slip_dialog(row: Mapping) -> None:
    """The bet-placement reminder: every leg + the slip's numbers in one popup."""
    if row.get("headline"):
        st.markdown(f"#### {row['headline']}")
    st.write(
        f"**{row['platform']} {row.get('play_type', '')}** · "
        f"{int(row['bet_size'])} legs · {row.get('status') or 'pending'}".replace("  ", " ")
    )
    for leg in json.loads(row["legs"]):
        st.write(f"- {leg['desc']}  ·  {leg.get('league', '')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Payout", f"{_fnum(row.get('payout_multiplier')):.2f}x")
    c2.metric("EV", f"{_fnum(row.get('model_ev'), 1.0) - 1:+.1%}")
    c3.metric("Stake", f"${_fnum(row.get('stake')):,.2f}")
    if row.get("payout_approximate"):
        st.caption("Sleeper payout approximate — correlation factor pending.")


def _fnum(value, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) else default
