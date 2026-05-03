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


styled = filtered.style.apply(_highlight_hot, axis=1).format(
    {
        "Model EV": "{:.3f}",
        "Model": "{:.3f}",
        "Books": "{:.3f}",
        "Line": "{:.2f}",
        "Boost": "{:.2f}",
        "Avg 5": "{:.2f}",
        "Avg H2H": "{:.2f}",
        "Moneyline": "{:.0f}",
        "O/U": "{:.1f}",
        "DVPOA": "{:.3f}",
    }
)

st.dataframe(styled, use_container_width=True, height=720)

with st.expander("Snapshot info"):
    st.json(meta)
