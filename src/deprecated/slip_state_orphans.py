# ARCHIVED 2026-10-11 from src/sportstradamus/dashboard/components/slip_state.py
# Reason: The Games surface replaced the Slips page's "Seed builder" leg picker with
#         clickable candidate stars (06bea6e0), removing this seeder's only call site.
# Last live SHA: a63b9107
# Original imports (now unresolved here):
#   from collections.abc import Mapping, Sequence
#   import streamlit as st
#   from sportstradamus.dashboard.components.slip_state import (
#       _BUILDER, _EDIT_ID, _LEGS, _PLATFORM,
#   )
#   from sportstradamus.leg_schema import build_leg

# From src/sportstradamus/dashboard/components/slip_state.py.
def seed_from_legs(rows: Sequence[Mapping], platform: str, builder: str) -> None:
    """Seed a builder from selected offer rows (Game → constellation, Board → simple).

    A slip is single-platform; rows off ``platform`` are dropped.
    """
    st.session_state[_LEGS] = [build_leg(r) for r in rows if r["Platform"] == platform]
    st.session_state[_PLATFORM] = platform
    st.session_state[_BUILDER] = builder
    st.session_state[_EDIT_ID] = None
