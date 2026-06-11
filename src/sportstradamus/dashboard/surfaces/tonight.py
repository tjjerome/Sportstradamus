"""Tonight — game-card slate showing today's offers grouped by game."""
# pylint: disable=duplicate-code  # same offers preamble as game.py; different presentation logic

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.data import format_ts, load_current_meta, load_current_offers

st.title("Tonight")

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
    st.caption("Prophecies arrive with the next data wave.")
    st.stop()

game_key_cols = [c for c in ["League", "Date", "Team", "Opponent"] if c in offers.columns]
if not game_key_cols:
    st.info("Offer data missing game grouping columns.")
    st.stop()

for game_key, group in offers.groupby(game_key_cols, sort=False):
    if isinstance(game_key, str):
        game_key = (game_key,)
    key_dict = dict(zip(game_key_cols, game_key, strict=False))

    league = key_dict.get("League", "")
    team = key_dict.get("Team", "")
    opponent = key_dict.get("Opponent", "")
    date_raw = key_dict.get("Date", "")
    lock_time = format_ts(str(date_raw)) if date_raw else ""

    offer_count = len(group)
    top_ev = group["Model EV"].max() if "Model EV" in group.columns else None

    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{team} vs {opponent}**  ·  {league}")
            if lock_time:
                st.caption(lock_time)
            st.caption(f"{offer_count} offer{'s' if offer_count != 1 else ''}")
            if top_ev is not None and pd.notna(top_ev):
                ev_sign = f"{top_ev:+.2f}" if top_ev >= 0 else f"{top_ev:.2f}"
                st.caption(f"Top edge: {ev_sign}")
        with right:
            game_label = f"{team} vs {opponent}"
            if st.button("View game", key=f"tonight_{game_label}_{league}"):
                st.query_params["game"] = game_label
                st.switch_page("surfaces/game.py")

st.caption("Prophecies arrive with the next data wave.")
