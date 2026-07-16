"""Today's scored offers from the latest `prophecize` run."""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard import columns
from sportstradamus.dashboard.components.deep_dive import init_detail_state, show_detail
from sportstradamus.dashboard.components.grid import render_themed_grid
from sportstradamus.dashboard.components.hero import page_hero
from sportstradamus.dashboard.components.offer_cards import render_offer_cards
from sportstradamus.dashboard.components.slip_builder import render_simple_builder
from sportstradamus.dashboard.components.slip_state import add_to_simple_slip
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_game_corr,
    load_current_meta,
    load_current_offers,
    sport_filtered,
)
from sportstradamus.dashboard.lenses import LENSES, apply_lens
from sportstradamus.dashboard.viewport import is_mobile
from sportstradamus.helpers import market_display_name

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
page_hero("THE BOARD", "Today's Predictions", generated)

mobile = is_mobile()
offers = sport_filtered(load_current_offers())

if offers.empty:
    st.info(
        "No current predictions found. Run `poetry run prophecize` to "
        "generate `current_offers.parquet`."
    )
    st.stop()

MAIN_COLS = [
    "League",
    "Match",
    "Player",
    "Market Display",
    "Line",
    "Boost",
    "Win Prob",
    "Model Edge",
    "Consensus Edge",
    "Platform",
]

RANGE_COLS = ["Win Prob", columns.MODEL_EDGE, columns.CONSENSUS_EDGE]

signal_cols = [c for c in ["Boost", "Model EV", "Market EV"] if c in offers.columns]
if signal_cols:
    signal = offers[signal_cols].fillna(0)
    offers = offers.loc[(signal != 0).any(axis=1)]

lens_col, side_col = st.columns([3, 1])
with lens_col:
    lens = (
        st.segmented_control("Prophecy lens", list(LENSES), default="Tonight", key="board_lens")
        or "Tonight"
    )
with side_col:
    side = st.segmented_control("Side", ["All", "Over", "Under"], default="All", key="board_side")

# Phone: the widget-heavy filters collapse into one expander; lens/side stay above.
filter_host = st.expander("Filters") if mobile else st.container()
with filter_host:
    col1, col2, col3 = st.columns(3)
    with col1:
        leagues = sorted(offers["League"].dropna().unique())
        selected_leagues = st.multiselect("League", leagues, default=leagues)
    with col2:
        platforms = sorted(offers["Platform"].dropna().unique()) if "Platform" in offers else []
        selected_platforms = st.multiselect("Platform", platforms, default=platforms)
    with col3:
        markets = sorted(offers["Market"].dropna().unique())
        # Chips show the prose market name; the stored value stays the slug the filter keys off.
        market_labels = {
            m: market_display_name(lg, m)
            for lg, m in zip(offers["League"], offers["Market"], strict=True)
            if pd.notna(m)
        }
        selected_markets = st.multiselect(
            "Market", markets, default=markets, format_func=lambda s: market_labels.get(s, s)
        )

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
if side and side != "All" and "Bet" in filtered.columns:
    filtered = filtered.loc[filtered["Bet"] == side]

# Derive Model/Consensus Edge before the sliders so they filter on edge, not raw EV.
filtered = columns.add_edges(filtered)

range_cols = [c for c in RANGE_COLS if c in filtered.columns]
if range_cols:
    with filter_host:
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

filtered = columns.add_match_column(filtered)
filtered = columns.add_market_display(filtered)

# Bet rides along hidden (grid.py's arrow_col auto-hides it) for the Line arrow
# cellRenderer's params.data.Bet; the Market slug rides along, renamed out of the way,
# for the row-selection/slip-building code below (which matches offers by slug) — it
# would otherwise collide with Market Display's LABELS rename to the same "Market"
# header. Neither is one of the displayed MAIN_COLS.
display_cols = [c for c in [*MAIN_COLS, "Bet", "Market"] if c in filtered.columns]
grid_df = filtered[display_cols].copy()
if "Market" in grid_df.columns:
    grid_df = grid_df.rename(columns={"Market": "Market Slug"})

if "Win Prob" in grid_df.columns:
    # Kept numeric (0–100) so the column click-sorts; the "%" is a display-only
    # valueFormatter (percent_cols below), same as the edge columns.
    grid_df["Win Prob"] = (pd.to_numeric(grid_df["Win Prob"], errors="coerce") * 100).round(1)
# Model/Consensus Edge are edge-vs-DFS percentages (EV - 1, x100); kept numeric so the
# heatmap can bucket Model Edge and the grid sorts them — the "%" is a display suffix
# (percent_cols below), not baked into the value.
for col in (columns.MODEL_EDGE, columns.CONSENSUS_EDGE):
    if col in grid_df.columns:
        grid_df[col] = (pd.to_numeric(grid_df[col], errors="coerce") * 100).round(1)
if "Boost" in grid_df.columns:
    grid_df["Boost"] = pd.to_numeric(grid_df["Boost"], errors="coerce").round(2)
grid_df = grid_df.rename(columns=columns.LABELS)

# columns.LABELS renames Consensus Edge -> Cons Edge before the grid ever sees the
# frame, so every lookup below keys off the post-rename name, same as "Win %" already
# does for Win Prob.
numeric_cols = [
    c for c in ("Line", "Boost", "Win %", columns.MODEL_EDGE, "Cons Edge") if c in grid_df.columns
]
if mobile:
    # Cards read the pre-rename filtered frame (raw 0–1 Win Prob; Model Edge ×100
    # below, matching the grid path's percent convention).
    selected_rows = []
    cards_df = filtered.copy()
    cards_df[columns.MODEL_EDGE] = (
        pd.to_numeric(cards_df[columns.MODEL_EDGE], errors="coerce") * 100
    ).round(1)
    render_offer_cards(cards_df)
else:
    selected_rows = render_themed_grid(
        grid_df,
        numeric_cols=numeric_cols,
        heatmap_col=columns.MODEL_EDGE,
        heatmap_center=0.0,
        header_help=columns.HELP,
        percent_cols=["Win %"],
        signed_percent_cols=[columns.MODEL_EDGE, "Cons Edge"],
        arrow_col="Line",
        hidden_cols=["Bet", "Market Slug"],
    )
    st.caption("Trend sparklines arrive with the L1 line-movement export.")

if st.session_state.corr_nav:
    # Rerun from "View →" button — keep the stack as-is, just clear the flag.
    st.session_state.corr_nav = False
elif selected_rows:
    r = selected_rows[0]
    current_key = (r.get("Player"), r.get("Market Slug"))
    if current_key != st.session_state.last_grid_key:
        st.session_state.last_grid_key = current_key
        player, market = r["Player"], r.get("Market Slug")
        mask = filtered["Player"] == player
        if market:
            mask &= filtered["Market"] == market
        matches = filtered.loc[mask]
        if not matches.empty:
            st.session_state.detail_stack = [matches.index[0]]
elif not mobile:
    # Grid reports no selection (tab switch, filter cleared the table, etc.)
    # — close popup and reset tracking. Mobile has no grid selection; its cards
    # set detail_stack directly, so the reset would race the card click.
    st.session_state.detail_stack = []
    st.session_state.last_grid_key = None

if st.session_state.detail_stack:
    row_idx = st.session_state.detail_stack[-1]
    show_detail(filtered.loc[row_idx], filtered)
else:
    st.caption("Click a row to see charts and correlated bets.")

if not mobile:
    # Mobile adds legs straight from each card; this selected-row path is grid-only.
    st.subheader("Build a cross-game slip")
    if selected_rows:
        sel = selected_rows[0]
        mask = (
            (filtered["Player"] == sel.get("Player"))
            & (filtered["Market"] == sel.get("Market Slug"))
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
