"""Today's scored offers from the latest `prophecize` run."""

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from sportstradamus.dashboard.components.deep_dive import init_detail_state, show_detail
from sportstradamus.dashboard.components.slip_builder import (
    add_to_simple_slip,
    render_simple_builder,
)
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_corr,
    load_current_meta,
    load_current_offers,
    render_banner,
    sport_filtered,
)

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

# Hot-row highlight: model EV > this and books P > this and boost below ceiling
# qualifies a row for the blue background. Tuned empirically vs bankroll ROI.
EV_HOT_THRESHOLD = 1.02
BOOKS_HOT_THRESHOLD = 0.95
# Boosted rows above this ceiling signal a line move, not genuine edge.
BOOST_HOT_CEILING = 2.5
HIGHLIGHT_BG = "#1f4e79"

# Main table columns
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
    "Model P",
    "Model",
    "Books",
    "Platform",
]

# Numeric columns for range filtering
RANGE_COLS = ["Model P", "Model", "Books"]

signal_cols = [c for c in ["Boost", "Model", "Books"] if c in offers.columns]
if signal_cols:
    signal = offers[signal_cols].fillna(0)
    offers = offers.loc[(signal != 0).any(axis=1)]

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

filtered = offers
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
        sel = slot.slider(col, lo, hi, (lo, hi), step=(hi - lo) / 100 or 0.01)
        vals = pd.to_numeric(filtered[col], errors="coerce")
        filtered = filtered.loc[vals.between(sel[0], sel[1]) | vals.isna()]

if "Model" in filtered.columns:
    filtered = filtered.sort_values("Model", ascending=False)

filtered = filtered.reset_index(drop=True)
st.caption(f"Showing **{len(filtered):,}** of {len(offers):,} offers")

init_detail_state()

display_cols = [c for c in MAIN_COLS if c in filtered.columns]
grid_df = filtered[display_cols].copy()
format_cols = {"Boost": 2, "Model": 2, "Books": 2}

if "Model P" in grid_df.columns:
    grid_df["Model P"] = (grid_df["Model P"] * 100).apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else ""
    )

for col, decimals in format_cols.items():
    if col in grid_df.columns:
        grid_df[col] = grid_df[col].round(decimals)

gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_grid_options(rowStyle={"cursor": "pointer"})

gb.configure_column(
    "Model",
    cellStyle={
        "function": (
            f"params.data.Model > {EV_HOT_THRESHOLD} && "
            f"params.data.Books > {BOOKS_HOT_THRESHOLD} && "
            f"params.data.Boost <= {BOOST_HOT_CEILING} "
            f"? {{'backgroundColor': '{HIGHLIGHT_BG}', 'color': 'white'}} : {{}}"
        )
    },
)

go = gb.build()
ag = AgGrid(
    grid_df,
    gridOptions=go,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=True,
    height=720,
    use_container_width=True,
)

selected = ag.selected_rows
# Newer streamlit-aggrid returns a DataFrame; older versions return a list.
if isinstance(selected, pd.DataFrame):
    selected_rows = selected.to_dict("records")
else:
    selected_rows = selected or []

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
