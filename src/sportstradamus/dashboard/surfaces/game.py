"""Game — single-game offer view with context strip."""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.deep_dive import to_american
from sportstradamus.dashboard.components.slip_builder import seed_from_legs
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_meta,
    load_current_offers,
    sport_filtered,
)

st.title("Game")

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
st.caption(f"Last updated: {generated}")

offers = sport_filtered(load_current_offers())

if offers.empty:
    st.info("No current predictions. Run `poetry run prophecize` to generate offers.")
    st.stop()

if "Team" in offers.columns and "Opponent" in offers.columns:
    # Label format must match tonight.py's View-game query param; the date
    # suffix keeps a doubleheader's two games distinct.
    offers = offers.assign(_game=offers["Team"] + " vs " + offers["Opponent"])
    if "Date" in offers.columns:
        offers["_game"] += " · " + offers["Date"].astype(str)
    game_labels = offers["_game"].dropna().unique().tolist()
else:
    game_labels = []

# Read game from query params; fall back to selectbox
param_game = st.query_params.get("game", "")
default_idx = game_labels.index(param_game) if param_game and param_game in game_labels else 0

if not game_labels:
    st.info("No game grouping columns in offer data.")
    st.stop()

selected_game = st.selectbox("Select game", game_labels, index=default_idx)
game_offers = offers.loc[offers["_game"] == selected_game]

if game_offers.empty:
    st.info("No offers found for this game.")
    st.stop()

ctx = game_offers.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Moneyline", to_american(ctx.get("Moneyline")) if "Moneyline" in ctx.index else "N/A")
ou = ctx.get("O/U")
c2.metric("O/U Total", f"{ou:.1f}" if pd.notna(ou) and isinstance(ou, int | float) else "N/A")
c3.metric("League", ctx.get("League", "N/A"))

display_cols = [
    c
    for c in ["Player", "Market", "Bet", "Line", "Model EV", "Model", "Books", "Platform"]
    if c in game_offers.columns
]
st.dataframe(game_offers[display_cols], use_container_width=True, hide_index=True)

st.subheader("Build a slip from this game")
build_platforms = sorted(game_offers["Platform"].dropna().unique())
if build_platforms:
    build_platform = st.selectbox("Platform", build_platforms, key="game_build_platform")
    pool = game_offers.loc[game_offers["Platform"] == build_platform]
    leg_rows = {
        f"{r['Player']} {r['Bet']} {float(r['Line']):.10g} {r['Market']}": r
        for r in pool.to_dict("records")
        if pd.notna(r.get("Model P"))
    }
    picks = st.multiselect(
        "Legs (same game → constellation)", list(leg_rows), key="game_build_legs"
    )
    if len(picks) >= 2 and st.button(
        "Build in constellation", type="primary", key="game_build_btn"
    ):
        seed_from_legs([leg_rows[p] for p in picks], build_platform, "constellation")
        st.switch_page("surfaces/slips.py")

st.caption("Prophecies arrive with the next data wave.")
