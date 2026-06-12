"""The sidebar locked-slip shelf — mounted once on every surface.

Reads ``user_slips.parquet`` (the slips "locked in" from either builder) and
lists them compactly under the global bankroll control. "Edit" reopens a slip in
its builder (constellation → Slips, simple → Board) for re-locking. Grading
status (filled by nightly ``reflect``) shows per entry.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_builder import bankroll_input, load_slip
from sportstradamus.dashboard.data import load_current_offers, load_user_slips

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
        status = row.get("status") or "pending"
        st.write(f"{int(row['bet_size'])} legs · {row['platform']} · {status}")
        target = (
            "surfaces/slips.py" if row["builder_type"] == "constellation" else "surfaces/board.py"
        )
        if st.button("Edit", key=f"shelf_edit_{row['slip_id']}"):
            load_slip(row, offers)
            st.switch_page(target)
