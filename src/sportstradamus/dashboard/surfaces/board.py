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
    load_current_line_movement,
    load_current_meta,
    load_current_offers,
    load_game_ctxs,
    load_pair_modifiers,
    load_play_record,
    sport_filtered,
)
from sportstradamus.dashboard.legs import record_line
from sportstradamus.dashboard.lenses import LENSES, apply_lens
from sportstradamus.dashboard.viewport import is_mobile
from sportstradamus.helpers import leg_kind, market_display_name

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
page_hero("THE BOARD", "Today's Predictions", generated)

mobile = is_mobile()
offers = sport_filtered(load_current_offers())

if offers.empty:
    st.info(
        "No current predictions found. Run `sportstradamus prophecize` to "
        "generate `current_offers.parquet`."
    )
    st.stop()

MAIN_COLS = [
    "League",
    "Match",
    "Player",
    "Market Display",
    "Line",
    columns.MOVE,
    "Boost",
    "Fit Prob",
    "EV",
    "Win Prob",
    "Model EV",
    columns.CONSENSUS_EDGE,
    "Platform",
]

RANGE_COLS = ["EV", "Model EV", columns.CONSENSUS_EDGE]

signal_cols = [c for c in ["Boost", "Model EV", "Market EV"] if c in offers.columns]
if signal_cols:
    signal = offers[signal_cols].fillna(0)
    offers = offers.loc[(signal != 0).any(axis=1)]

# wrap=True on the widgets placed straight in a column below: Streamlit otherwise keeps
# them on one scrolling row, which hides lenses on a phone and filter chips on a desktop.
lens_col, side_col = st.columns([3, 1])
with lens_col:
    lens = (
        st.segmented_control(
            "Prophecy lens", list(LENSES), default="Picks", key="board_lens", wrap=True
        )
        or "Picks"
    )
with side_col:
    side = st.segmented_control(
        "Side", ["All", "Over", "Under"], default="All", key="board_side", wrap=True
    )

# Phone: the widget-heavy filters collapse into one expander; lens/side stay above.
filter_host = st.expander("Filters") if mobile else st.container()
with filter_host:
    col1, col2, col3 = st.columns(3)
    with col1:
        leagues = sorted(offers["League"].dropna().unique())
        selected_leagues = st.multiselect("League", leagues, default=leagues, wrap=True)
    with col2:
        platforms = sorted(offers["Platform"].dropna().unique()) if "Platform" in offers else []
        selected_platforms = st.multiselect("Platform", platforms, default=platforms, wrap=True)
    with col3:
        markets = sorted(offers["Market"].dropna().unique())
        # Chips show the prose market name; the stored value stays the slug the filter keys off.
        market_labels = {
            m: market_display_name(lg, m)
            for lg, m in zip(offers["League"], offers["Market"], strict=True)
            if pd.notna(m)
        }
        selected_markets = st.multiselect(
            "Market",
            markets,
            default=markets,
            format_func=lambda s: market_labels.get(s, s),
            wrap=True,
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

# Derive Consensus Edge before the sliders so its slider filters on the edge, not Market EV.
filtered = columns.add_consensus_edge(filtered)

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

# EV ranks the board; a leg the fit did not price follows, by the model's own EV.
filtered = filtered.sort_values(["EV", "Model EV"], ascending=False, na_position="last")

filtered = filtered.reset_index(drop=True)
st.caption(f"Showing **{len(filtered):,}** of {len(offers):,} offers")

picks = filtered.loc[filtered["Pick"]]
if not picks.empty:
    record = load_play_record()
    kinds = leg_kind(picks["Boost"], picks["Platform"])
    with st.expander("Pick record", expanded=True):
        for league, kind in sorted(set(zip(picks["League"], kinds, strict=True))):
            line, hover = record_line(record, league, kind)
            st.caption(line, help=hover)

init_detail_state()

filtered = columns.add_match_column(filtered)
filtered = columns.add_market_display(filtered)
filtered = columns.add_line_movement(filtered, load_current_line_movement())

# Bet rides along hidden (grid.py's arrow_col auto-hides it) for the Line arrow
# cellRenderer's params.data.Bet; the Market slug rides along, renamed out of the way,
# for the row-selection/slip-building code below (which matches offers by slug) — it
# would otherwise collide with Market Display's LABELS rename to the same "Market"
# header. Move Spark carries the finished sparkline SVG the Move cellRenderer draws.
# None of the three is a displayed MAIN_COLS entry.
display_cols = [
    c for c in [*MAIN_COLS, "Bet", "Market", columns.MOVE_SPARK] if c in filtered.columns
]
grid_df = filtered[display_cols].copy()
if "Market" in grid_df.columns:
    grid_df = grid_df.rename(columns={"Market": "Market Slug"})

# The win chances and Consensus Edge (Market EV − 1) are kept numeric (0–100) so the
# columns click-sort; the "%" is a display-only valueFormatter (percent_cols below).
for col in ("Fit Prob", "Win Prob", columns.CONSENSUS_EDGE):
    if col in grid_df.columns:
        grid_df[col] = (pd.to_numeric(grid_df[col], errors="coerce") * 100).round(1)
for col in ("Boost", "EV", "Model EV"):
    if col in grid_df.columns:
        grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce").round(2)
grid_df = grid_df.rename(columns=columns.LABELS)

# columns.LABELS renames the columns before the grid ever sees the frame, so every lookup
# below keys off the post-rename name (Fit Prob -> Win %, Consensus Edge -> Cons Edge).
numeric_cols = [
    c
    for c in ("Line", columns.MOVE, "Boost", "Win %", "EV", "Model win %", "Model EV", "Cons Edge")
    if c in grid_df.columns
]
if mobile:
    # Cards read the pre-rename filtered frame: raw 0–1 win chances and EVs.
    selected_rows = []
    render_offer_cards(filtered)
else:
    selected_rows = render_themed_grid(
        grid_df,
        numeric_cols=numeric_cols,
        header_help=columns.HELP,
        percent_cols=["Win %", "Model win %"],
        signed_percent_cols=["Cons Edge"],
        arrow_col="Line",
        spark_col=columns.MOVE,
        spark_svg_col=columns.MOVE_SPARK,
        hidden_cols=["Bet", "Market Slug", columns.MOVE_SPARK],
    )

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
render_simple_builder(filtered, load_game_ctxs(), load_pair_modifiers())

with st.expander("Snapshot info"):
    st.json(meta)
