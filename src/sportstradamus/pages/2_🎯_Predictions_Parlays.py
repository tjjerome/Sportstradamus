"""Today's parlay candidates from the latest `prophecize` run."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from sportstradamus.dashboard_data import (
    format_ts,
    load_current_meta,
    load_current_offers,
    load_current_parlays,
    render_banner,
)
from sportstradamus.dashboard_detail import (
    _show_detail,
    family_labels_for_game,
    find_offer_idx,
    init_detail_state,
    parse_leg,
)

LEG_COLS = [f"Leg {i}" for i in range(1, 7)]
TOP_PARLAY_DEFAULT = 50

# User-facing parlay sort options → (column, ascending).
SORT_OPTIONS = {
    "Model EV": ("Model EV", False),
    "Recommended bet size": ("Rec Bet", False),
    "Fun metric": ("Fun", False),
}

st.set_page_config(page_title="Predictions — Parlays", layout="wide")
st.title("Today's Parlay Candidates")

meta = load_current_meta()
generated = format_ts(meta.get("generated_at", "no run on record"))
render_banner("predictions", f"generated {generated}")

parlays = load_current_parlays()
if parlays.empty:
    st.info(
        "No current parlays found. Run `poetry run prophecize` to "
        "generate `current_parlays.parquet`."
    )
    st.stop()

if "Family" not in parlays.columns:
    parlays["Family"] = 1

# Full scored offers back the per-leg detail dialog; stable integer index
# because the detail-navigation stack stores these positions.
offers = load_current_offers().reset_index(drop=True)

init_detail_state()
if st.session_state.corr_nav:
    # Rerun came from a "View →" button inside the dialog — keep the stack.
    st.session_state.corr_nav = False

col1, col2, col3, col4 = st.columns(4)
with col1:
    leagues = sorted(parlays["League"].dropna().unique())
    selected_leagues = st.multiselect("League", leagues, default=leagues)
with col2:
    platforms = sorted(parlays["Platform"].dropna().unique()) if "Platform" in parlays else []
    selected_platforms = st.multiselect("Platform", platforms, default=platforms)
with col3:
    sort_label = st.selectbox("Sort parlays by", list(SORT_OPTIONS), index=0)
with col4:
    top_n = st.number_input(
        "Show top N parlays", min_value=10, max_value=500, value=TOP_PARLAY_DEFAULT, step=10
    )

view = parlays
if selected_leagues:
    view = view.loc[view["League"].isin(selected_leagues)]
if selected_platforms and "Platform" in view:
    view = view.loc[view["Platform"].isin(selected_platforms)]

sort_col, ascending = SORT_OPTIONS[sort_label]
if sort_col in view.columns:
    view = view.sort_values(sort_col, ascending=ascending)
view = view.head(int(top_n))
st.caption(f"Showing **{len(view):,}** of {len(parlays):,} parlays")


def _render_parlay(row: pd.Series, offers: pd.DataFrame) -> None:
    with st.container(border=True):
        indep_p = row.get("Indep P")
        show_indep = pd.notna(indep_p)
        meta_cols = st.columns(6 if show_indep else 5)
        meta_cols[0].metric("Model EV", f"{row['Model EV']:.2f}")
        meta_cols[1].metric("Books EV", f"{row.get('Books EV', float('nan')):.2f}")
        meta_cols[2].metric("Boost", f"{row.get('Boost', float('nan')):.2f}x")
        meta_cols[3].metric("Bet Size", row.get("Bet Size", "—"))
        meta_cols[4].metric("Fun", f"{row.get('Fun', float('nan')):.2f}")
        if show_indep:
            # Indep P is the no-correlation joint; comparing it to the joint P
            # reveals how much correlation lifted (or sank) the row.
            joint_p = row.get("P", float("nan"))
            delta = (joint_p - indep_p) if pd.notna(joint_p) else None
            meta_cols[5].metric(
                "Joint vs Indep",
                f"{joint_p:.3f}" if pd.notna(joint_p) else "—",
                f"{delta:+.3f}" if delta is not None else None,
                help="Correlation-aware joint P vs independence-assumption Indep P.",
            )
        legs = [row.get(c) for c in LEG_COLS if isinstance(row.get(c), str) and row.get(c)]
        for leg_i, leg in enumerate(legs):
            if st.button(
                f"  • {leg}",
                key=f"plyleg::{row.name}::{leg_i}",
                use_container_width=True,
            ):
                idx = find_offer_idx(parse_leg(leg), offers, row.get("Platform"))
                if idx is not None:
                    st.session_state.detail_stack = [idx]
                    st.rerun()
                else:
                    st.toast("Detail unavailable — line moved since this parlay was built.")
        if pd.notna(row.get("Rec Bet")):
            st.caption(f"Recommended bet: {row['Rec Bet']:.2f} units")


for game, group in view.groupby("Game", sort=False):
    top_ev = group["Model EV"].max()
    with st.expander(f"{game} — {len(group)} parlays, top EV {top_ev:.2f}"):
        families = sorted(group["Family"].dropna().unique())
        labels = family_labels_for_game(group)
        if len(families) > 1:
            fam_choice = st.selectbox(
                "Bet family",
                ["All"] + [labels[f] for f in families],
                key=f"family_{game}",
                help=(
                    "The backend clusters this game's parlays into up to three "
                    "families by how independent they are from each other. Each "
                    "name highlights the player who sets that family apart."
                ),
            )
        else:
            fam_choice = "All"

        for fam in families:
            if fam_choice != "All" and fam_choice != labels[fam]:
                continue
            fam_group = group.loc[group["Family"] == fam]
            if len(families) > 1:
                st.markdown(f"**{labels[fam]}** — {len(fam_group)} parlays")
            for _, row in fam_group.iterrows():
                _render_parlay(row, offers)

if st.session_state.detail_stack:
    row_idx = st.session_state.detail_stack[-1]
    _show_detail(offers.loc[row_idx], offers)

with st.expander("Snapshot info"):
    st.json(meta)
