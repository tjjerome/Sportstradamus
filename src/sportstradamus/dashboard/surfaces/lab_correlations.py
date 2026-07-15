"""Lab Correlations — correlation effectiveness, hit rates, parlay calibration."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.components.hero import page_hero
from sportstradamus.dashboard.components.lab_filters import render_lab_filters
from sportstradamus.dashboard.data import (
    format_ts,
    load_corr_market_summary,
    load_history,
    load_parlays,
    load_resolve_meta,
    load_stat_meta,
    sidebar_filters,
    sport_filtered,
)
from sportstradamus.dashboard.surfaces.lab_correlations_charts import (
    corr_heatmap,
    worst_driver_pair,
)

# Lab Correlations only ever plots/aggregates these scalar parlay columns; projecting the
# read skips the multi-GB list<float> struct columns (legs, Corr/Boost Pairs) in the 1.7M-row
# parlay_hist.parquet, turning a multi-second load into a fast one.
_PARLAY_SCALAR_COLS = (
    "Date",
    "League",
    "Platform",
    "Legs Resolved",
    "Misses",
    "Indep P",
    "P",
    "Boost",
    "Bet Size",
)
# The value-add scatter draws one point per resolved parlay; cap the plotted sample so a full
# history (1.7M rows) can't choke the browser. The metrics below still use the full frame.
_SCATTER_MAX = 20_000
_SCATTER_SEED = 17

page_hero("MODEL LAB · CORRELATIONS", "Correlations & Parlays")

# Mounted for session consistency with the other two Lab pages (widget keys are
# shared across all three); this page's frames are one-row-per-parlay or a
# market-pair grid, neither of which fits apply_lab_filters' (league, market)
# join contract, so no frame here is re-filtered through the selection.
render_lab_filters(load_stat_meta(), collapsed=True)

history = load_history()
parlays = load_parlays(columns=_PARLAY_SCALAR_COLS)

if parlays.empty:
    st.warning("No parlay history found. Run `prophecize` first.")
    st.stop()

# Sport narrow runs before the sidebar builds its league multiselect, so the
# sidebar only offers that sport's leagues.
history = sport_filtered(history)
parlays = sport_filtered(parlays)
if parlays.empty:
    st.info("No parlays match the current sport filter.")
    st.stop()

meta = load_resolve_meta()
if meta.get("last_run"):
    st.caption(f"Data last resolved: {format_ts(meta['last_run'])}")

filters = sidebar_filters(
    history if not history.empty else parlays, key_prefix="corr_", time_window_key="corr_time"
)
cutoff = filters["cutoff"]

pf = parlays.copy()
pf["_date"] = pd.to_datetime(pf["Date"], errors="coerce").dt.date
pf = pf.loc[pf["_date"].notna()]

if cutoff is not None:
    pf = pf.loc[pf["_date"] >= cutoff]

if filters["date_range"]:
    pf = pf.loc[
        (pf["_date"] >= filters["date_range"][0]) & (pf["_date"] <= filters["date_range"][1])
    ]
if filters["leagues"]:
    pf = pf.loc[pf["League"].isin(filters["leagues"])]
if filters["platforms"]:
    pf = pf.loc[pf["Platform"].isin(filters["platforms"])]

resolved = pf.dropna(subset=["Legs Resolved"]).copy()
if not resolved.empty:
    resolved[["Legs Resolved", "Misses"]] = resolved[["Legs Resolved", "Misses"]].astype(int)
    resolved["Hit"] = (resolved["Misses"] == 0).astype(int)

if resolved.empty:
    st.info("No resolved parlays match the current filters.")
    st.stop()

has_indep = "Indep P" in resolved.columns and resolved["Indep P"].notna().any()
has_corr_p = "P" in resolved.columns and resolved["P"].notna().any()

# Hoisted so the diagnostic callout below and "Correlation Value-Add" further
# down share one boosted/not-boosted split instead of computing it twice.
scatter_df = pd.DataFrame()
above_line = pd.DataFrame()
below_line = pd.DataFrame()
if has_indep and has_corr_p:
    scatter_df = resolved.dropna(subset=["Indep P", "P"]).copy()
    scatter_df["Outcome"] = scatter_df["Hit"].map({1: "Hit", 0: "Miss"})
    above_line = scatter_df.loc[scatter_df["P"] > scatter_df["Indep P"]]
    below_line = scatter_df.loc[scatter_df["P"] <= scatter_df["Indep P"]]

if len(above_line) > 0 and len(below_line) > 0:
    boosted_rate = above_line["Hit"].mean()
    unboosted_rate = below_line["Hit"].mean()
    if boosted_rate < unboosted_rate:
        driver = worst_driver_pair(above_line)
        driver_note = (
            f" {driver[0]} ↔ {driver[1]} is the most frequent driver pair — check it first."
            if driver is not None
            else ""
        )
        st.warning(
            f"Correlation isn't paying here: boosted parlays hit {boosted_rate:.1%} vs "
            f"{unboosted_rate:.1%} unboosted — the copula may be over-adjusting.{driver_note}"
        )

st.header("Stat-Pair Correlation Matrix")
# One matrix is per-league; reuse the sidebar's league selection instead of a second league
# picker. With exactly one league in scope there's nothing to choose, so the selectbox only
# appears when the sidebar leaves more than one league selected.
matrix_leagues = filters["leagues"] or sorted(parlays["League"].unique())
if len(matrix_leagues) == 1:
    heatmap_league = matrix_leagues[0]
else:
    heatmap_league = st.selectbox("League", matrix_leagues, key="corr_heatmap_league")
corr_summary = load_corr_market_summary(heatmap_league)

if corr_summary.empty:
    st.caption(f"No correlation matrices generated yet for {heatmap_league} — run `correlate`.")
else:
    scope = st.radio("Scope", ["same_team", "opposing"], horizontal=True, key="corr_heatmap_scope")
    # Empirical-vs-model rho overlay is the §6.4 follow-up — not built here.
    st.plotly_chart(corr_heatmap(corr_summary, heatmap_league, scope), width="stretch")

st.download_button(
    "Export parlays (CSV)",
    resolved.drop(columns=["Corr Pairs", "Boost Pairs", "Leg Probs"], errors="ignore").to_csv(
        index=False
    ),
    "parlays_filtered.csv",
    "text/csv",
)

st.header("Correlation Value-Add")

if has_indep and has_corr_p:
    plot_df = scatter_df
    if len(scatter_df) > _SCATTER_MAX:
        plot_df = scatter_df.sample(n=_SCATTER_MAX, random_state=_SCATTER_SEED)
        st.caption(
            f"Scatter shows a random {_SCATTER_MAX:,} of {len(scatter_df):,} resolved parlays; "
            "the metrics below use all of them."
        )
    fig_scatter = px.scatter(
        plot_df,
        x="Indep P",
        y="P",
        color="Outcome",
        color_discrete_map={"Hit": theme.GREEN, "Miss": theme.RED},
        opacity=0.5,
        labels={
            "Indep P": "Independent Probability (no correlation)",
            "P": "Correlated Probability",
        },
        title="Correlation Adjustment: Independent vs Correlated Probability",
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, scatter_df[["Indep P", "P"]].max().max()],
            y=[0, scatter_df[["Indep P", "P"]].max().max()],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="No adjustment line",
        )
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation boosted parlays", f"{len(above_line)}")
    col2.metric(
        "Hit rate (boosted)", f"{above_line['Hit'].mean():.1%}" if len(above_line) > 0 else "N/A"
    )
    col3.metric(
        "Hit rate (not boosted)",
        f"{below_line['Hit'].mean():.1%}" if len(below_line) > 0 else "N/A",
    )

elif has_corr_p:
    st.info(
        "Indep P data not yet available (pre-update predictions). "
        "Falling back to Leg Probs for independent rate estimation."
    )

st.subheader("Correlation Boost vs Hit Rate")
if "Boost" in resolved.columns:
    boost_df = resolved.copy()
    boost_df["Boost_Bucket"] = pd.qcut(boost_df["Boost"], q=5, duplicates="drop")

    boost_stats = (
        boost_df.groupby("Boost_Bucket", observed=False)
        .agg(
            Hit_Rate=("Hit", "mean"),
            Count=("Hit", "count"),
        )
        .reset_index()
    )
    boost_stats["Boost_Bucket"] = boost_stats["Boost_Bucket"].astype(str)

    fig_boost = px.bar(
        boost_stats,
        x="Boost_Bucket",
        y="Hit_Rate",
        text="Count",
        labels={"Boost_Bucket": "Boost Range", "Hit_Rate": "Hit Rate"},
    )
    fig_boost.update_layout(height=400)
    st.plotly_chart(fig_boost, width="stretch")

st.header("Hit Rate by Parlay Size")

for platform in sorted(resolved["Platform"].unique()):
    plat_df = resolved.loc[resolved["Platform"] == platform]
    st.subheader(f"{platform}")

    size_data = []
    for size in sorted(plat_df["Bet Size"].unique()):
        sdf = plat_df.loc[plat_df["Bet Size"] == size]
        if len(sdf) == 0:
            continue
        row = {
            "Size": int(size),
            "Count": len(sdf),
            "Hit Rate": round(sdf["Hit"].mean(), 4),
            "Hit All": int((sdf["Misses"] == 0).sum()),
            "Missed 1": int((sdf["Misses"] == 1).sum()),
            "Missed 2+": int((sdf["Misses"] >= 2).sum()),
        }
        if "P" in sdf.columns and sdf["P"].notna().any():
            row["Predicted P"] = round(sdf["P"].mean(), 4)
        if "Indep P" in sdf.columns and sdf["Indep P"].notna().any():
            row["Independent P"] = round(sdf["Indep P"].mean(), 4)
        elif "legs" in sdf.columns and sdf["legs"].notna().any():
            indep = sdf["legs"].apply(
                lambda legs: (
                    np.prod([leg["win_prob"] for leg in legs])
                    if isinstance(legs, list | tuple | np.ndarray) and len(legs) > 0
                    else np.nan
                )
            )
            row["Independent P"] = round(indep.mean(), 4)
        size_data.append(row)

    if size_data:
        size_df = pd.DataFrame(size_data)
        st.dataframe(size_df, width="stretch", hide_index=True)

        miss_cols = ["Hit All", "Missed 1", "Missed 2+"]
        miss_df = size_df[["Size", *miss_cols]].melt(
            id_vars="Size", var_name="Outcome", value_name="Count"
        )
        fig_miss = px.bar(
            miss_df,
            x="Size",
            y="Count",
            color="Outcome",
            barmode="stack",
            color_discrete_map={
                "Hit All": theme.GREEN,
                "Missed 1": theme.ORANGE,
                "Missed 2+": theme.RED,
            },
            labels={"Size": "Parlay Size", "Count": "Parlays"},
        )
        fig_miss.update_layout(height=350)
        st.plotly_chart(fig_miss, width="stretch")

if "P" in resolved.columns and resolved["P"].notna().any():
    st.header("Parlay Calibration Curve")
    cal_df = resolved.dropna(subset=["P"]).copy()
    bins = np.linspace(0, cal_df["P"].quantile(0.95), 11)
    cal_df["p_bin"] = pd.cut(cal_df["P"], bins=bins)
    cal_stats = (
        cal_df.groupby("p_bin", observed=False)
        .agg(
            Predicted=("P", "mean"),
            Actual=("Hit", "mean"),
            Count=("Hit", "count"),
        )
        .reset_index()
        .dropna(subset=["Predicted"])
    )

    fig_pcal = go.Figure()
    fig_pcal.add_trace(
        go.Scatter(
            x=[0, cal_stats["Predicted"].max()],
            y=[0, cal_stats["Predicted"].max()],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Perfect",
        )
    )
    fig_pcal.add_trace(
        go.Scatter(
            x=cal_stats["Predicted"],
            y=cal_stats["Actual"],
            mode="lines+markers",
            name="Model",
            text=[f"n={c}" for c in cal_stats["Count"]],
        )
    )
    fig_pcal.update_layout(
        xaxis_title="Predicted Correlated Probability",
        yaxis_title="Actual Hit Rate",
        height=400,
    )
    st.plotly_chart(fig_pcal, width="stretch")
