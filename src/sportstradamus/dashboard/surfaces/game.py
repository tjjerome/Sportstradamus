"""Game — single-game offer view with context strip."""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_state import seed_from_legs
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_context,
    load_current_meta,
    load_current_offers,
    sport_filtered,
)
from sportstradamus.dashboard.narrative import context_strip

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

if not game_labels:
    st.info("No game grouping columns in offer data.")
    st.stop()

# Pre-select the clicked game: the in-app handoff (session, one-shot) wins, then
# a ?game= deep link. Seeding the widget key lets a later selectbox change stick
# across reruns instead of being re-forced; drop a stale value if the option set
# changed (e.g. the sport switch narrowed the slate).
preselect = st.session_state.pop("nav_game", "") or st.query_params.get("game", "")
if preselect in game_labels:
    st.session_state["game_select"] = preselect
if st.session_state.get("game_select") not in game_labels:
    st.session_state.pop("game_select", None)

selected_game = st.selectbox("Select game", game_labels, key="game_select")
game_offers = offers.loc[offers["_game"] == selected_game]

if game_offers.empty:
    st.info("No offers found for this game.")
    st.stop()

ctx_row = game_offers.iloc[0]
strip = context_strip(load_current_game_context(), game=ctx_row["Game"], date=ctx_row["Date"])
c1, c2, c3 = st.columns(3)
if strip:
    fav, spread = strip["fav_team"], strip["spread"]
    c1.metric("Total", f"{strip['game_total']:.1f}")
    c2.metric("Spread", f"{fav} -{spread:.1f}" if fav and spread > 0 else "Even")
    c3.metric("Shape", str(strip["shape"]).title())
else:
    c1.metric("Total", "N/A")
    c2.metric("Spread", "N/A")
    c3.metric("League", ctx_row.get("League", "N/A"))

display_cols = [
    c
    for c in ["Player", "Market", "Bet", "Line", "Projection", "Model EV", "Market EV", "Platform"]
    if c in game_offers.columns
]
st.dataframe(game_offers[display_cols], width="stretch", hide_index=True)

st.subheader("Build a slip from this game")
build_platforms = sorted(game_offers["Platform"].dropna().unique())
if build_platforms:
    build_platform = st.selectbox("Platform", build_platforms, key="game_build_platform")
    pool = game_offers.loc[game_offers["Platform"] == build_platform]
    leg_rows = {
        f"{r['Player']} {r['Bet']} {float(r['Line']):.10g} {r['Market']}": r
        for r in pool.to_dict("records")
        if pd.notna(r.get("Win Prob"))
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
