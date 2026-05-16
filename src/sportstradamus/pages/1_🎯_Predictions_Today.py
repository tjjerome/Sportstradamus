"""Today's scored offers from the latest `prophecize` run."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from sportstradamus.dashboard_data import (
    load_current_meta,
    load_current_offers,
    render_banner,
)

# Hot-row highlight: model edge above the boost cost (Model > 1.02), strong
# bookmaker agreement (Books > 0.95), and a moderate boost (≤ 2.5).
EV_HOT_THRESHOLD = 1.02
BOOKS_HOT_THRESHOLD = 0.95
BOOST_HOT_CEILING = 2.5

HIGHLIGHT_BG = "#1f4e79"

# Columns shown in the main grid. The remaining scored columns are clutter on
# the main screen and live in the per-row detail popup instead.
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

# Clutter columns surfaced only in the per-row detail popup.
DETAIL_COLS = [
    "Model EV",
    "Model STD",
    "Push P",
    "Avg 5",
    "Avg H2H",
    "Moneyline",
    "O/U",
    "DVPOA",
]

# Numeric columns the user can range-filter on the main screen.
RANGE_COLS = ["Model P", "Model", "Books"]

st.set_page_config(page_title="Predictions — Today", layout="wide")
st.title("Today's Predictions")

meta = load_current_meta()
generated = meta.get("generated_at", "no run on record")
render_banner("predictions", f"generated {generated}")

offers = load_current_offers()
if offers.empty:
    st.info(
        "No current predictions found. Run `poetry run prophecize` to "
        "generate `current_offers.parquet`."
    )
    st.stop()

# Defensive: a row with no boost, no model edge, and no book edge is a scoring
# artifact (the snapshot writer drops these, but older parquets may predate it).
signal_cols = [c for c in ["Boost", "Model", "Books"] if c in offers.columns]
if signal_cols:
    signal = offers[signal_cols].fillna(0)
    offers = offers.loc[(signal != 0).any(axis=1)]

# --- In-page filters (predictions section keeps its own controls) ---
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

# --- Numeric range filters for the hit-probability and EV columns ---
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

# Default ordering: descending by the Model column.
if "Model" in filtered.columns:
    filtered = filtered.sort_values("Model", ascending=False)

filtered = filtered.reset_index(drop=True)
st.caption(f"Showing **{len(filtered):,}** of {len(offers):,} offers")


def _highlight_hot(row: pd.Series) -> list[str]:
    is_hot = (
        pd.notna(row.get("Model"))
        and pd.notna(row.get("Books"))
        and pd.notna(row.get("Boost"))
        and row["Model"] > EV_HOT_THRESHOLD
        and row["Books"] > BOOKS_HOT_THRESHOLD
        and row["Boost"] <= BOOST_HOT_CEILING
    )
    style = f"background-color:{HIGHLIGHT_BG};color:white" if is_hot else ""
    return [style] * len(row)


@st.dialog("Offer detail", width="large")
def _show_detail(row: pd.Series) -> None:
    st.subheader(f"{row.get('Player', '?')} — {row.get('Market', '?')}")
    st.write(
        f"**{row.get('Bet', '?')} {row.get('Line', '?')}** · "
        f"{row.get('Team', '?')} vs {row.get('Opponent', '?')} · "
        f"{row.get('League', '?')} · {row.get('Platform', '?')}"
    )

    detail = [c for c in DETAIL_COLS if c in row.index]
    if detail:
        cols = st.columns(4)
        for i, c in enumerate(detail):
            val = row[c]
            cols[i % 4].metric(c, f"{val:.3f}" if isinstance(val, int | float) else str(val))

    team_corr = row.get("Team Correlation")
    opp_corr = row.get("Opp Correlation")
    st.markdown("**Recommended correlated bets**")
    if isinstance(team_corr, str) and team_corr.strip():
        st.markdown(f"_Same team:_ {team_corr}")
    if isinstance(opp_corr, str) and opp_corr.strip():
        st.markdown(f"_Opponent:_ {opp_corr}")
    if not (isinstance(team_corr, str) and team_corr.strip()) and not (
        isinstance(opp_corr, str) and opp_corr.strip()
    ):
        st.caption("No correlated legs cleared the display thresholds for this offer.")


display_cols = [c for c in MAIN_COLS if c in filtered.columns]
styled = (
    filtered[display_cols]
    .style.apply(_highlight_hot, axis=1)
    .format(
        {
            "Model P": "{:.3f}",
            "Model": "{:.3f}",
            "Books": "{:.3f}",
            "Line": "{:.2f}",
            "Boost": "{:.2f}",
        }
    )
)

event = st.dataframe(
    styled,
    use_container_width=True,
    height=720,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="offers_table",
)

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    _show_detail(filtered.iloc[selected_rows[0]])
else:
    st.caption("Select a row to see Model EV, Model std, push/odds context, and correlated bets.")

with st.expander("Snapshot info"):
    st.json(meta)
