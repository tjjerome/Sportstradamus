# ARCHIVED 2026-07-05 from src/sportstradamus/dashboard/components/deep_dive.py
# Reason: P8 Phase C Task C6 replaced satellite_picker.py's expander-based pickers
#         (render_satellites / render_disliked_legs) with two star-map lenses; their
#         shared _render_pick_popover helper was this card's only caller, leaving
#         render_offer_card with zero callers.
# Last live SHA: 95bb350
# Original imports (now unresolved here):
#   from collections.abc import Mapping
#   import streamlit as st


def render_offer_card(row: Mapping) -> None:
    """Compact offer card — the Streamlit twin of the constellation's JS hover card.

    Player + bet/line/market + win/boost/Kelly. The headshot (a person icon) and
    the last-5 line are scarred placeholders, matching the constellation card: the
    offer snapshot carries no player-id or gamelog columns to fill them yet.
    """
    win = float(row.get("Win Prob", 0.0) or 0.0)
    boost = float(row.get("Boost", 1.0) or 1.0)
    kelly = float(row.get("Kelly", 0.0) or 0.0)
    st.markdown(f":material/account_circle: **{row['Player']}**")
    st.caption(f"{row['Bet']} {float(row['Line']):.10g} {row['Market']}")
    st.markdown(f"Win **{win:.0%}** · **{boost:.2f}×** · Kelly **{kelly:.0%}**")
    st.caption(":gray[Last 5 — coming soon]")
