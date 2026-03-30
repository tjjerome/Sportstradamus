"""Page 3: Correlations & Parlays — correlation effectiveness, hit rates, calibration."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from sportstradamus.dashboard_data import (
    load_history, load_parlays, load_stats, load_stat_map,
    resolve_and_save, resolve_parlays_and_save, sidebar_filters,
)
from sportstradamus.analysis import (
    compute_parlay_metrics, PAYOUT_TABLE, TIMEFRAMES,
)

st.title("Correlations & Parlays")

# --- Load data ---
history = load_history()
parlays = load_parlays()

if parlays.empty:
    st.warning("No parlay history found. Run `prophecize` first.")
    st.stop()

stats = load_stats()
stat_map = load_stat_map()

if not history.empty:
    history = resolve_and_save(history, stats)
parlays = resolve_parlays_and_save(parlays, stats, stat_map)

# --- Sidebar ---
filters = sidebar_filters(history if not history.empty else parlays, key_prefix="corr_")

# Filter parlays
pf = parlays.copy()
pf["_date"] = pd.to_datetime(pf["Date"], errors="coerce").dt.date
pf = pf.loc[pf["_date"].notna()]
if filters["date_range"]:
    pf = pf.loc[(pf["_date"] >= filters["date_range"][0]) & (pf["_date"] <= filters["date_range"][1])]
if filters["leagues"]:
    pf = pf.loc[pf["League"].isin(filters["leagues"])]
if filters["platforms"]:
    pf = pf.loc[pf["Platform"].isin(filters["platforms"])]

# Resolve for display
resolved = pf.dropna(subset=["Legs"]).copy()
resolved[["Legs", "Misses"]] = resolved[["Legs", "Misses"]].astype(int)
resolved["Hit"] = (resolved["Misses"] == 0).astype(int)

if resolved.empty:
    st.info("No resolved parlays match the current filters.")
    st.stop()

st.download_button(
    "Export parlays (CSV)",
    resolved.drop(columns=["Corr Pairs", "Boost Pairs", "Leg Probs"], errors="ignore").to_csv(index=False),
    "parlays_filtered.csv",
    "text/csv",
)

# =====================================================================
# CORRELATION VALUE-ADD
# =====================================================================
st.header("Correlation Value-Add")

has_indep = "Indep P" in resolved.columns and resolved["Indep P"].notna().any()
has_corr_p = "P" in resolved.columns and resolved["P"].notna().any()

if has_indep and has_corr_p:
    scatter_df = resolved.dropna(subset=["Indep P", "P"]).copy()
    scatter_df["Outcome"] = scatter_df["Hit"].map({1: "Hit", 0: "Miss"})

    fig_scatter = px.scatter(
        scatter_df, x="Indep P", y="P",
        color="Outcome",
        color_discrete_map={"Hit": "#2ecc71", "Miss": "#e74c3c"},
        opacity=0.5,
        labels={"Indep P": "Independent Probability (no correlation)",
                "P": "Correlated Probability"},
        title="Correlation Adjustment: Independent vs Correlated Probability",
    )
    fig_scatter.add_trace(go.Scatter(
        x=[0, scatter_df[["Indep P", "P"]].max().max()],
        y=[0, scatter_df[["Indep P", "P"]].max().max()],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="No adjustment line",
    ))
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Quantify value-add
    above_line = scatter_df.loc[scatter_df["P"] > scatter_df["Indep P"]]
    below_line = scatter_df.loc[scatter_df["P"] <= scatter_df["Indep P"]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation boosted parlays", f"{len(above_line)}")
    col2.metric("Hit rate (boosted)", f"{above_line['Hit'].mean():.1%}" if len(above_line) > 0 else "N/A")
    col3.metric("Hit rate (not boosted)", f"{below_line['Hit'].mean():.1%}" if len(below_line) > 0 else "N/A")

elif has_corr_p:
    st.info("Indep P data not yet available (pre-update predictions). "
            "Falling back to Leg Probs for independent rate estimation.")

# --- Correlation Boost vs Hit Rate ---
st.subheader("Correlation Boost vs Hit Rate")
if "Boost" in resolved.columns:
    boost_df = resolved.copy()
    boost_df["Boost_Bucket"] = pd.qcut(boost_df["Boost"], q=5, duplicates="drop")

    boost_stats = boost_df.groupby("Boost_Bucket", observed=False).agg(
        Hit_Rate=("Hit", "mean"),
        Count=("Hit", "count"),
    ).reset_index()
    boost_stats["Boost_Bucket"] = boost_stats["Boost_Bucket"].astype(str)

    fig_boost = px.bar(boost_stats, x="Boost_Bucket", y="Hit_Rate",
                       text="Count",
                       labels={"Boost_Bucket": "Boost Range", "Hit_Rate": "Hit Rate"})
    fig_boost.update_layout(height=400)
    st.plotly_chart(fig_boost, use_container_width=True)

# =====================================================================
# HIT RATE BY PARLAY SIZE
# =====================================================================
st.header("Hit Rate by Parlay Size")

time_split = st.selectbox("Time window", ["All"] + [t[0] for t in TIMEFRAMES],
                          index=0, key="parlay_time")
if time_split != "All":
    days = dict(TIMEFRAMES)[time_split]
    cutoff = datetime.today().date() - timedelta(days=days)
    size_resolved = resolved.loc[resolved["_date"] >= cutoff]
else:
    size_resolved = resolved

for platform in sorted(size_resolved["Platform"].unique()):
    plat_df = size_resolved.loc[size_resolved["Platform"] == platform]
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
        elif "Leg Probs" in sdf.columns and sdf["Leg Probs"].notna().any():
            indep = sdf["Leg Probs"].apply(
                lambda lp: np.prod(lp) if isinstance(lp, (list, tuple)) and len(lp) > 0 else np.nan
            )
            row["Independent P"] = round(indep.mean(), 4)
        size_data.append(row)

    if size_data:
        size_df = pd.DataFrame(size_data)
        st.dataframe(size_df, use_container_width=True, hide_index=True)

        # Miss distribution chart
        miss_cols = ["Hit All", "Missed 1", "Missed 2+"]
        miss_df = size_df[["Size"] + miss_cols].melt(id_vars="Size", var_name="Outcome", value_name="Count")
        fig_miss = px.bar(miss_df, x="Size", y="Count", color="Outcome",
                          barmode="stack",
                          color_discrete_map={"Hit All": "#2ecc71", "Missed 1": "#f39c12", "Missed 2+": "#e74c3c"},
                          labels={"Size": "Parlay Size", "Count": "Parlays"})
        fig_miss.update_layout(height=350)
        st.plotly_chart(fig_miss, use_container_width=True)

# =====================================================================
# PARLAY CALIBRATION CURVE
# =====================================================================
if "P" in resolved.columns and resolved["P"].notna().any():
    st.header("Parlay Calibration Curve")
    cal_df = resolved.dropna(subset=["P"]).copy()
    bins = np.linspace(0, cal_df["P"].quantile(0.95), 11)
    cal_df["p_bin"] = pd.cut(cal_df["P"], bins=bins)
    cal_stats = cal_df.groupby("p_bin", observed=False).agg(
        Predicted=("P", "mean"),
        Actual=("Hit", "mean"),
        Count=("Hit", "count"),
    ).reset_index().dropna(subset=["Predicted"])

    fig_pcal = go.Figure()
    fig_pcal.add_trace(go.Scatter(
        x=[0, cal_stats["Predicted"].max()],
        y=[0, cal_stats["Predicted"].max()],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="Perfect",
    ))
    fig_pcal.add_trace(go.Scatter(
        x=cal_stats["Predicted"], y=cal_stats["Actual"],
        mode="lines+markers", name="Model",
        text=[f"n={c}" for c in cal_stats["Count"]],
    ))
    fig_pcal.update_layout(
        xaxis_title="Predicted Correlated Probability",
        yaxis_title="Actual Hit Rate",
        height=400,
    )
    st.plotly_chart(fig_pcal, use_container_width=True)
