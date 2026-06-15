"""Tonight — game-card slate showing today's offers grouped by game."""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.data import (
    format_ts,
    load_current_meta,
    load_current_offers,
    load_current_parlays,
    sport_filtered,
)
from sportstradamus.dashboard.narrative import top_thesis

st.title("Tonight")

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
st.caption(f"Last updated: {generated}")

offers = sport_filtered(load_current_offers())
parlays = load_current_parlays()

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
    top_ev = group["Projection"].max() if "Projection" in group.columns else None
    headline = top_thesis(parlays, game=group["Game"].iloc[0], date=date_raw)

    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{team} vs {opponent}**  ·  {league}")
            if headline:
                st.markdown(headline)
            if lock_time:
                st.caption(lock_time)
            st.caption(f"{offer_count} offer{'s' if offer_count != 1 else ''}")
            if top_ev is not None and pd.notna(top_ev):
                st.caption(f"Top edge: {top_ev:+.2f}")
        with right:
            game_label = f"{team} vs {opponent}"
            # Date keeps doubleheaders distinct; format must match the Slips
            # game-seed labels.
            game_param = f"{game_label} · {date_raw}" if date_raw else game_label
            if st.button("View game", key=f"tonight_{league}_{date_raw}_{game_label}"):
                # switch_page drops query params set the same run; hand off via
                # session state and let the Slips game-seed read it first (?game=
                # is the deep link).
                st.session_state["nav_game"] = game_param
                st.switch_page("surfaces/slips.py")

st.caption("Prophecies arrive with the next data wave.")
