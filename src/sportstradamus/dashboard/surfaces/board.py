"""Today's scored offers from the latest `prophecize` run."""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard import columns
from sportstradamus.dashboard.components.deep_dive import init_detail_state, show_detail
from sportstradamus.dashboard.components.grid import render_themed_grid
from sportstradamus.dashboard.components.slip_builder import render_simple_builder
from sportstradamus.dashboard.components.slip_state import add_to_simple_slip
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_corr,
    load_current_meta,
    load_current_offers,
    render_banner,
    sport_filtered,
)
from sportstradamus.dashboard.lenses import LENSES, apply_lens

st.title("Today's Predictions")

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
render_banner("predictions", f"generated {generated}")

offers = sport_filtered(load_current_offers())

if offers.empty:
    st.info(
        "No current predictions found. Run `poetry run prophecize` to "
        "generate `current_offers.parquet`."
    )
    st.stop()

MAIN_COLS = [
    "League",
    "Date",
    "Team",
    "Opponent",
    "Player",
    "Market",
    "Bet",
    "Line",
    "Boost",
    "Win Prob",
    "Edge",
    "Kelly",
    "Model EV",
    "Market EV",
    "Platform",
]

RANGE_COLS = ["Win Prob", "Model EV", "Market EV"]

signal_cols = [c for c in ["Boost", "Model EV", "Market EV"] if c in offers.columns]
if signal_cols:
    signal = offers[signal_cols].fillna(0)
    offers = offers.loc[(signal != 0).any(axis=1)]

lens = st.segmented_control("Prophecy lens", list(LENSES), default="All", key="board_lens") or "All"

col1, col2, col3 = st.columns(3)
with col1:
    leagues = sorted(offers["League"].dropna().unique())
    selected_leagues = st.multiselect("League", leagues, default=leagues)
with col2:
    platforms = sorted(offers["Platform"].dropna().unique()) if "Platform" in offers else []
    selected_platforms = st.multiselect("Platform", platforms, default=platforms)
with col3:
    markets = sorted(offers["Market"].dropna().unique())
    selected_markets = st.multiselect("Market", markets, default=markets)

player_query = st.text_input("Player search", placeholder="e.g. Jokic")

filtered = apply_lens(offers, lens)
if selected_leagues:
    filtered = filtered.loc[filtered["League"].isin(selected_leagues)]
if selected_platforms and "Platform" in filtered:
    filtered = filtered.loc[filtered["Platform"].isin(selected_platforms)]
if selected_markets:
    filtered = filtered.loc[filtered["Market"].isin(selected_markets)]
if player_query:
    filtered = filtered.loc[filtered["Player"].str.contains(player_query, case=False, na=False)]

# Numeric range filters
range_cols = [c for c in RANGE_COLS if c in filtered.columns]
if range_cols:
    st.caption("Numeric range filters")
    rcols = st.columns(len(range_cols))
    for slot, col in zip(rcols, range_cols, strict=False):
        series = pd.to_numeric(filtered[col], errors="coerce").dropna()
        if series.empty:
            continue
        lo = float(series.min())
        hi = float(series.max())
        if lo == hi:
            continue
        sel = slot.slider(
            columns.LABELS.get(col, col), lo, hi, (lo, hi), step=(hi - lo) / 100 or 0.01
        )
        vals = pd.to_numeric(filtered[col], errors="coerce")
        filtered = filtered.loc[vals.between(sel[0], sel[1]) | vals.isna()]

if "Model EV" in filtered.columns:
    filtered = filtered.sort_values("Model EV", ascending=False)

filtered = filtered.reset_index(drop=True)
st.caption(f"Showing **{len(filtered):,}** of {len(offers):,} offers")

init_detail_state()

filtered = columns.add_edge(filtered)
display_cols = [c for c in MAIN_COLS if c in filtered.columns]
grid_df = filtered[display_cols].copy()

if "Win Prob" in grid_df.columns:
    grid_df["Win Prob"] = (grid_df["Win Prob"] * 100).apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else ""
    )
for col in ("Boost", "Edge", "Kelly", "Model EV", "Market EV"):
    if col in grid_df.columns:
        grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce").round(2)
grid_df = grid_df.rename(columns=columns.LABELS)

numeric_cols = [
    c
    for c in ("Line", "Boost", "Win %", "Edge", "Kelly", "Model EV", "Market EV")
    if c in grid_df.columns
]
selected_rows = render_themed_grid(
    grid_df,
    numeric_cols=numeric_cols,
    heatmap_col="Edge",
    heatmap_center=0.0,
    header_help=columns.HELP,
)
st.caption("Trend sparklines arrive with the L1 line-movement export.")

if st.session_state.corr_nav:
    # Rerun from "View →" button — keep the stack as-is, just clear the flag.
    st.session_state.corr_nav = False
elif selected_rows:
    r = selected_rows[0]
    current_key = (r.get("Player"), r.get("Market"))
    if current_key != st.session_state.last_grid_key:
        st.session_state.last_grid_key = current_key
        player, market = r["Player"], r.get("Market")
        mask = filtered["Player"] == player
        if market:
            mask &= filtered["Market"] == market
        matches = filtered.loc[mask]
        if not matches.empty:
            st.session_state.detail_stack = [matches.index[0]]
else:
    # Grid reports no selection (tab switch, filter cleared the table, etc.)
    # — close popup and reset tracking.
    st.session_state.detail_stack = []
    st.session_state.last_grid_key = None

if st.session_state.detail_stack:
    row_idx = st.session_state.detail_stack[-1]
    show_detail(filtered.loc[row_idx], filtered)
else:
    st.caption("Click a row to see charts and correlated bets.")

st.subheader("Build a cross-game slip")
if selected_rows:
    sel = selected_rows[0]
    mask = (
        (filtered["Player"] == sel.get("Player"))
        & (filtered["Market"] == sel.get("Market"))
        & (filtered["Bet"] == sel.get("Bet"))
        & (filtered["Platform"] == sel.get("Platform"))
    )
    match = filtered.loc[mask]
    if not match.empty and match.iloc[0]["Platform"] in ("Underdog", "Sleeper"):
        full = match.iloc[0]
        if st.button(
            f"Add {full['Player']} {full['Bet']} {full['Line']:.10g} to slip",
            key="board_add_slip",
        ):
            add_to_simple_slip(full.to_dict())
            st.rerun()
    else:
        st.caption("Select an Underdog or Sleeper row to add it to a slip.")
else:
    st.caption("Select a row above, then add it to your slip.")
render_simple_builder(filtered, load_current_game_corr())

with st.expander("Snapshot info"):
    st.json(meta)
