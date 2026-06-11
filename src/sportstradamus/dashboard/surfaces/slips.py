"""Today's parlay candidates and Pick'em entries — two tabs."""

from decimal import Decimal

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.deep_dive import init_detail_state, show_detail
from sportstradamus.dashboard.components.stories import family_labels_for_game
from sportstradamus.dashboard.data import (
    format_ts,
    load_current_meta,
    load_current_offers,
    load_current_parlays,
    load_current_pickem,
    render_banner,
    sport_filtered,
)
from sportstradamus.dashboard.legs import find_offer_idx, parse_leg
from sportstradamus.strategies.kelly import fractional_kelly_stake

LEG_COLS = [f"Leg {i}" for i in range(1, 7)]
# Parlays shown per family cluster per game (no global cap, so every game
# stays visible regardless of sort order).
TOP_PER_FAMILY_DEFAULT = 15

# User-facing parlay sort options → (column, ascending).
SORT_OPTIONS = {
    "Model EV": ("Model EV", False),
    "Recommended bet size": ("Rec Bet", False),
    "Fun metric": ("Fun", False),
}

# Iteration order controls section order in the Pick'em tab.
VARIANT_LABELS = {"power": "Power", "flex": "Flex", "rivals": "Rivals"}


def _compute_stake(row: pd.Series, bankroll: float) -> float:
    """Quarter-Kelly stake for a Pick'em entry scaled to ``bankroll``."""
    return float(
        fractional_kelly_stake(
            bankroll=Decimal(str(bankroll)),
            win_prob=float(row["joint_prob"]),
            payout_multiplier=Decimal(str(row["payout_multiplier"])),
            model_shrinkage=float(row["shrinkage"]),
        )
    )


def _render_parlay(row: pd.Series, offers: pd.DataFrame) -> None:
    with st.container(border=True):
        _render_parlay_metrics(row)
        _render_parlay_legs(row, offers)
        if pd.notna(row.get("Rec Bet")):
            st.caption(f"Recommended bet: {row['Rec Bet']:.2f} units")


def _render_parlay_metrics(row: pd.Series) -> None:
    """Top metric row; a sixth column appears only when Indep P is present."""
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


def _render_parlay_legs(row: pd.Series, offers: pd.DataFrame) -> None:
    """Per-leg buttons; a click opens the leg's detail dialog if the line still exists."""
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


tab_parlays, tab_pickem = st.tabs(["Parlays", "Pick'em"])

with tab_parlays:
    st.title("Today's Parlay Candidates")

    meta = load_current_meta()
    generated = format_ts(meta.get("generated_at", "no run on record"))
    render_banner("predictions", f"generated {generated}")

    parlays = sport_filtered(load_current_parlays())

    # No st.stop() inside a tab body — it would halt the whole script and
    # blank the other tab; each tab guards with if/else instead.
    if parlays.empty:
        st.info(
            "No current parlays found. Run `poetry run prophecize` to "
            "generate `current_parlays.parquet`."
        )
    else:
        if "Family" not in parlays.columns:
            parlays["Family"] = 1

        # Stable integer index because the detail-navigation stack stores these positions.
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
            platforms = (
                sorted(parlays["Platform"].dropna().unique()) if "Platform" in parlays else []
            )
            selected_platforms = st.multiselect("Platform", platforms, default=platforms)
        with col3:
            sort_label = st.selectbox("Sort parlays by", list(SORT_OPTIONS), index=0)
        with col4:
            top_n = st.number_input(
                "Top N per family",
                min_value=5,
                max_value=100,
                value=TOP_PER_FAMILY_DEFAULT,
                step=5,
                help=(
                    "Caps parlays shown per family cluster within each game. There is "
                    "no global cap, so every game stays visible whatever the sort."
                ),
            )

        view = parlays
        if selected_leagues:
            view = view.loc[view["League"].isin(selected_leagues)]
        if selected_platforms and "Platform" in view:
            view = view.loc[view["Platform"].isin(selected_platforms)]

        sort_col, ascending = SORT_OPTIONS[sort_label]
        if sort_col in view.columns:
            view = view.sort_values(sort_col, ascending=ascending)
        st.caption(
            f"Showing top **{int(top_n)}** parlays per family across "
            f"{view['Game'].nunique():,} games ({len(parlays):,} total)."
        )

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
                            "The backend clusters this game's parlays into up to four "
                            "independence families (auto-selected; one when they aren't "
                            "separable). Each name highlights the player who most "
                            "drives that family."
                        ),
                    )
                else:
                    fam_choice = "All"

                for fam in families:
                    if fam_choice != "All" and fam_choice != labels[fam]:
                        continue
                    fam_group = group.loc[group["Family"] == fam]
                    shown = fam_group.head(int(top_n))
                    st.markdown(
                        f"**{labels[fam]}** — showing {len(shown)} of {len(fam_group)} parlays"
                    )
                    for _, row in shown.iterrows():
                        _render_parlay(row, offers)

        if st.session_state.detail_stack:
            row_idx = st.session_state.detail_stack[-1]
            show_detail(offers.loc[row_idx], offers)

        with st.expander("Snapshot info"):
            st.json(meta)

with tab_pickem:
    st.title("Today's Pick'em Entries")

    meta = load_current_meta()
    generated = format_ts(meta.get("generated_at", "no run on record"))
    render_banner("predictions", f"generated {generated}")

    entries = sport_filtered(load_current_pickem())

    if entries.empty:
        st.info(
            "No current Pick'em entries found. Run `poetry run prophecize` to "
            "generate `current_pickem.parquet`."
        )
    else:
        bankroll = st.number_input(
            "Bankroll ($)",
            min_value=0.0,
            value=1000.0,
            step=50.0,
            help=(
                "Stakes are quarter-Kelly × the per-cell shrinkage. Adjusting bankroll "
                "rescales every entry — this replaces the offline `kelly` CLI."
            ),
        )

        view = entries.copy()
        view["Stake"] = view.apply(lambda r: _compute_stake(r, bankroll), axis=1)
        view = view.sort_values("ev", ascending=False)

        st.caption(
            f"{len(view)} entries across {view['contest_variant'].nunique()} contest variants."
        )

        for variant, label in VARIANT_LABELS.items():
            group = view.loc[view["contest_variant"] == variant]
            if group.empty:
                continue
            st.subheader(f"{label} — {len(group)} entries, ${group['Stake'].sum():,.2f} staked")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Legs": group["legs"].apply("  +  ".join),
                        "Size": group["entry_size"],
                        "Joint P": group["joint_prob"].map("{:.1%}".format),
                        "Payout": group["payout_multiplier"].map("{:.2f}x".format),
                        "EV": group["ev"].map("{:+.1%}".format),
                        "Shrinkage": group["shrinkage_source"],
                        "Stake": group["Stake"].map("${:,.2f}".format),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Snapshot info"):
            st.json(meta)
