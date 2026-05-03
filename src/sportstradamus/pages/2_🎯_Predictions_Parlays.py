"""Today's parlay candidates from the latest `prophecize` run."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from sportstradamus.dashboard_data import (
    load_current_meta,
    load_current_parlays,
    render_banner,
)

LEG_COLS = [f"Leg {i}" for i in range(1, 7)]
TOP_PARLAY_DEFAULT = 50

st.set_page_config(page_title="Predictions — Parlays", layout="wide")
st.title("Today's Parlay Candidates")

meta = load_current_meta()
generated = meta.get("generated_at", "no run on record")
render_banner("predictions", f"generated {generated}")

parlays = load_current_parlays()
if parlays.empty:
    st.info(
        "No current parlays found. Run `poetry run prophecize` to "
        "generate `current_parlays.parquet`."
    )
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    leagues = sorted(parlays["League"].dropna().unique())
    selected_leagues = st.multiselect("League", leagues, default=leagues)
with col2:
    platforms = sorted(parlays["Platform"].dropna().unique()) if "Platform" in parlays else []
    selected_platforms = st.multiselect("Platform", platforms, default=platforms)
with col3:
    top_n = st.number_input("Show top N parlays", min_value=10, max_value=500, value=TOP_PARLAY_DEFAULT, step=10)

view = parlays
if selected_leagues:
    view = view.loc[view["League"].isin(selected_leagues)]
if selected_platforms and "Platform" in view:
    view = view.loc[view["Platform"].isin(selected_platforms)]

view = view.sort_values("Model EV", ascending=False).head(int(top_n))
st.caption(f"Showing **{len(view):,}** of {len(parlays):,} parlays")

for game, group in view.groupby("Game", sort=False):
    top_ev = group["Model EV"].max()
    with st.expander(f"{game} — {len(group)} parlays, top EV {top_ev:.2f}"):
        for _, row in group.iterrows():
            with st.container(border=True):
                meta_cols = st.columns(5)
                meta_cols[0].metric("Model EV", f"{row['Model EV']:.2f}")
                meta_cols[1].metric("Books EV", f"{row.get('Books EV', float('nan')):.2f}")
                meta_cols[2].metric("Boost", f"{row.get('Boost', float('nan')):.2f}x")
                meta_cols[3].metric("Bet Size", row.get("Bet Size", "—"))
                meta_cols[4].metric("Fun", f"{row.get('Fun', float('nan')):.2f}")
                legs = [row.get(c) for c in LEG_COLS if isinstance(row.get(c), str) and row.get(c)]
                for leg in legs:
                    st.text(f"  • {leg}")
                if pd.notna(row.get("Rec Bet")):
                    st.caption(f"Recommended bet: {row['Rec Bet']:.2f} units")

with st.expander("Snapshot info"):
    st.json(meta)
