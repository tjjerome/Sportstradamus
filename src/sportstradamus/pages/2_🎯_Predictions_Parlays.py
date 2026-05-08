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
from sportstradamus.helpers import underdog_payouts

LEG_COLS = [f"Leg {i}" for i in range(1, 7)]
TOP_PARLAY_DEFAULT = 50
CONTEST_VARIANTS = ["power", "flex", "insurance", "rivals"]

st.set_page_config(page_title="Predictions — Parlays", layout="wide")
st.title("Today's Parlay Candidates")

meta = load_current_meta()
generated = meta.get("generated_at", "no run on record")
scored_variant = meta.get("contest_variant", "power")
render_banner("predictions", f"generated {generated} · scored as `{scored_variant}`")

parlays = load_current_parlays()
if parlays.empty:
    st.info(
        "No current parlays found. Run `poetry run prophecize` to "
        "generate `current_parlays.parquet`."
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    leagues = sorted(parlays["League"].dropna().unique())
    selected_leagues = st.multiselect("League", leagues, default=leagues)
with col2:
    platforms = sorted(parlays["Platform"].dropna().unique()) if "Platform" in parlays else []
    selected_platforms = st.multiselect("Platform", platforms, default=platforms)
with col3:
    # Filters parlays to bet sizes the chosen variant actually pays out.
    # EV columns reflect the run's scored variant (see banner).
    default_idx = (
        CONTEST_VARIANTS.index(scored_variant) if scored_variant in CONTEST_VARIANTS else 0
    )
    contest_variant = st.selectbox("Contest variant", CONTEST_VARIANTS, index=default_idx)
with col4:
    top_n = st.number_input(
        "Show top N parlays", min_value=10, max_value=500, value=TOP_PARLAY_DEFAULT, step=10
    )

if contest_variant != scored_variant:
    st.warning(
        f"Showing only bet sizes payable under **{contest_variant}**. EV columns "
        f"still reflect the **{scored_variant}** payout schedule the run was scored "
        "under — re-run `prophecize --contest-variant " + contest_variant + "` to re-score."
    )

view = parlays
if selected_leagues:
    view = view.loc[view["League"].isin(selected_leagues)]
if selected_platforms and "Platform" in view:
    view = view.loc[view["Platform"].isin(selected_platforms)]
if "Bet Size" in view.columns:
    allowed_sizes = {int(k) for k in underdog_payouts.get(contest_variant, {})}
    if allowed_sizes:
        view = view.loc[view["Bet Size"].isin(allowed_sizes)]

view = view.sort_values("Model EV", ascending=False).head(int(top_n))
st.caption(f"Showing **{len(view):,}** of {len(parlays):,} parlays")

for game, group in view.groupby("Game", sort=False):
    top_ev = group["Model EV"].max()
    with st.expander(f"{game} — {len(group)} parlays, top EV {top_ev:.2f}"):
        for _, row in group.iterrows():
            with st.container(border=True):
                # Six tiles when Indep P is available, five otherwise — keeps
                # the layout balanced for older runs that pre-date correlation.
                indep_p = row.get("Indep P")
                show_indep = pd.notna(indep_p)
                meta_cols = st.columns(6 if show_indep else 5)
                meta_cols[0].metric("Model EV", f"{row['Model EV']:.2f}")
                meta_cols[1].metric("Books EV", f"{row.get('Books EV', float('nan')):.2f}")
                meta_cols[2].metric("Boost", f"{row.get('Boost', float('nan')):.2f}x")
                meta_cols[3].metric("Bet Size", row.get("Bet Size", "—"))
                meta_cols[4].metric("Fun", f"{row.get('Fun', float('nan')):.2f}")
                if show_indep:
                    # Indep P is the no-correlation joint; comparing it to the
                    # joint P reveals how much correlation lifted (or sank) the row.
                    joint_p = row.get("P", float("nan"))
                    delta = (joint_p - indep_p) if pd.notna(joint_p) else None
                    meta_cols[5].metric(
                        "Joint vs Indep",
                        f"{joint_p:.3f}" if pd.notna(joint_p) else "—",
                        f"{delta:+.3f}" if delta is not None else None,
                        help="Correlation-aware joint P vs independence-assumption Indep P.",
                    )
                legs = [row.get(c) for c in LEG_COLS if isinstance(row.get(c), str) and row.get(c)]
                for leg in legs:
                    st.text(f"  • {leg}")
                if pd.notna(row.get("Rec Bet")):
                    st.caption(f"Recommended bet: {row['Rec Bet']:.2f} units")

with st.expander("Snapshot info"):
    st.json(meta)
