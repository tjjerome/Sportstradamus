"""The Board's mobile card list — one bordered card per offer (Phase M spec §4.3).

The phone replacement for the 10-column AG-Grid, which can't fit a 390px
viewport. Each card carries the row's read (player, market, line/side, Win %,
Model Edge, platform) with numerals in the mono code face, plus the grid's two
actions: Detail (the deep-dive dialog, via the shared ``detail_stack``) and Add
to slip (the same ``add_to_simple_slip`` path the grid selection uses). Pages
``_PAGE_SIZE`` at a time — a slate can run hundreds of offers and phone DOM is
the constraint.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_state import add_to_simple_slip

# Cards rendered before the "Show more" button extends the list; sized so a full
# slate stays scrollable without flooding the phone DOM.
_PAGE_SIZE = 30
_SHOWN = "offer_cards_shown"


def render_offer_cards(offers: pd.DataFrame) -> None:
    """Render the paged card list over a filtered, scored offers frame.

    ``Win Prob`` arrives as the raw 0–1 probability; ``Model Edge`` arrives
    ×100 (the grid path's own percent convention).
    """
    shown = st.session_state.setdefault(_SHOWN, _PAGE_SIZE)
    for idx, row in offers.head(shown).iterrows():
        _render_card(idx, row)
    if len(offers) > shown and st.button(
        f"Show more ({len(offers) - shown} left)", key="offer_cards_more"
    ):
        st.session_state[_SHOWN] = shown + _PAGE_SIZE
        st.rerun()


def _render_card(idx, row: pd.Series) -> None:
    arrow = "▲" if str(row.get("Bet", "")).lower().startswith("o") else "▼"
    win = float(row.get("Win Prob") or 0.0)
    edge = float(row.get("Model Edge") or 0.0)
    market = row.get("Market Display") or row.get("Market", "")
    with st.container(border=True):
        st.markdown(
            f"**{row['Player']}** · {market} {arrow} `{row['Line']:.10g}`  \n"
            f"Win `{win:.0%}` · Edge `{edge:+.1f}%` · {row.get('Platform', '')} · "
            f"{row.get('League', '')}"
        )
        detail_col, add_col = st.columns(2)
        if detail_col.button("Detail", key=f"offer_card_detail_{idx}"):
            st.session_state.detail_stack = [idx]
        if add_col.button("Add to slip", key=f"offer_card_add_{idx}"):
            add_to_simple_slip(row.to_dict())
            st.rerun()
