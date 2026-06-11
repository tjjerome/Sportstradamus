"""Game — single-game offer view with context strip."""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.data import format_ts, load_current_meta, load_current_offers


def _to_american(p: float) -> str:
    if not isinstance(p, float | int) or pd.isna(p) or p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    return f"+{round((1 - p) / p * 100)}"


st.title("Game")

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
st.caption(f"Last updated: {generated}")

offers = load_current_offers()

# Apply global sport switch pre-filter
_sport = st.session_state.get("sport", "All")
if _sport != "All" and not offers.empty and "League" in offers.columns:
    offers = offers.loc[offers["League"] == _sport]

if offers.empty:
    st.info("No current predictions. Run `poetry run prophecize` to generate offers.")
    st.stop()

if "Team" in offers.columns and "Opponent" in offers.columns:
    game_labels = (offers["Team"] + " vs " + offers["Opponent"]).dropna().unique().tolist()
else:
    game_labels = []

# Read game from query params; fall back to selectbox
param_game = st.query_params.get("game", "")
default_idx = game_labels.index(param_game) if param_game and param_game in game_labels else 0

if not game_labels:
    st.info("No game grouping columns in offer data.")
    st.stop()

selected_game = st.selectbox("Select game", game_labels, index=default_idx)

if "Team" in offers.columns and "Opponent" in offers.columns:
    parts = selected_game.split(" vs ", 1)
    if len(parts) == 2:
        team, opponent = parts
        game_offers = offers.loc[(offers["Team"] == team) & (offers["Opponent"] == opponent)]
    else:
        game_offers = pd.DataFrame()
else:
    game_offers = pd.DataFrame()

if game_offers.empty:
    st.info("No offers found for this game.")
    st.stop()

ctx = game_offers.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Moneyline", _to_american(ctx.get("Moneyline")) if "Moneyline" in ctx.index else "N/A")
ou = ctx.get("O/U")
c2.metric("O/U Total", f"{ou:.1f}" if pd.notna(ou) and isinstance(ou, int | float) else "N/A")
c3.metric("League", ctx.get("League", "N/A"))

display_cols = [
    c
    for c in ["Player", "Market", "Bet", "Line", "Model EV", "Model", "Books", "Platform"]
    if c in game_offers.columns
]
st.dataframe(game_offers[display_cols], use_container_width=True, hide_index=True)

st.caption("Prophecy constellation coming in a future phase.")
st.caption("Prophecies arrive with the next data wave.")
