"""Page 1: Overview — KPIs, accuracy trends, profit trends, and volume."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import brier_score_loss

from sportstradamus.dashboard_data import (
    get_filtered_history,
    load_history,
    load_parlays,
    load_resolve_meta,
    render_banner,
    sidebar_filters,
)

st.title("Overview")
render_banner("stats", "historical accuracy, profit, and volume")

# --- Load data (pre-resolved by nightly script) ---
history = load_history()
parlays = load_parlays()

if history.empty:
    st.warning("No prediction history found. Run `prophecize` first.")
    st.stop()

meta = load_resolve_meta()
if meta.get("last_run"):
    st.caption(f"Data last resolved: {meta['last_run']}")
else:
    st.warning(
        "Nightly resolution has not run yet. Run `poetry run reflect` to resolve prediction outcomes."
    )

# --- Sidebar ---
filters = sidebar_filters(history, parlays, key_prefix="overview_")
df = get_filtered_history(
    history,
    leagues=filters["leagues"],
    platforms=filters["platforms"],
    date_range=filters["date_range"],
)

if df.empty:
    st.info("No resolved predictions match the current filters.")
    st.stop()

# --- Prep ---
prob_col = "Model P" if "Model P" in df.columns and df["Model P"].notna().any() else "Model"
df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
df["Hit"] = (df["Bet"] == df["Result"]).astype(int)
df["Profit Unit"] = df["Hit"] * (100 / 110) - (1 - df["Hit"])

# --- KPI Cards ---
col1, col2, col3, col4, col5 = st.columns(5)
accuracy = df["Hit"].mean()
roi = df["Profit Unit"].sum() / len(df)
brier = brier_score_loss(df["Hit"], df[prob_col].clip(0, 1))
total_profit = df["Profit Unit"].sum()

col1.metric("Total Predictions", f"{len(df):,}")
col2.metric("Accuracy", f"{accuracy:.1%}")
col3.metric("ROI", f"{roi:+.1%}")
col4.metric("Brier Score", f"{brier:.4f}")
col5.metric("Profit Units", f"{total_profit:+.1f}")

st.download_button(
    "Export filtered history (CSV)",
    df.to_csv(index=False),
    "history_filtered.csv",
    "text/csv",
)

# --- Cumulative Accuracy Time Series ---
st.subheader("Rolling 30-Day Accuracy by League")
daily_league = (
    df.groupby(["_date", "League"])
    .agg(
        Hits=("Hit", "sum"),
        Bets=("Hit", "count"),
    )
    .reset_index()
)
daily_league.sort_values("_date", inplace=True)

fig_acc = go.Figure()
for league in sorted(daily_league["League"].unique()):
    ld = daily_league.loc[daily_league["League"] == league].copy()
    ld["CumHits"] = ld["Hits"].cumsum()
    ld["CumBets"] = ld["Bets"].cumsum()
    # Rolling 30-day accuracy
    ld["Roll30_Hits"] = ld["Hits"].rolling(30, min_periods=1).sum()
    ld["Roll30_Bets"] = ld["Bets"].rolling(30, min_periods=1).sum()
    ld["Roll30_Acc"] = ld["Roll30_Hits"] / ld["Roll30_Bets"]
    fig_acc.add_trace(
        go.Scatter(
            x=ld["_date"],
            y=ld["Roll30_Acc"],
            mode="lines",
            name=league,
        )
    )

fig_acc.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
fig_acc.update_layout(yaxis_title="Accuracy", xaxis_title="Date", height=400)
st.plotly_chart(fig_acc, use_container_width=True)

# --- Cumulative Profit Time Series ---
st.subheader("Cumulative Profit (Units)")
daily_profit = (
    df.groupby("_date")
    .agg(
        Profit=("Profit Unit", "sum"),
        Bets=("Hit", "count"),
    )
    .reset_index()
    .sort_values("_date")
)
daily_profit["Cumulative Profit"] = daily_profit["Profit"].cumsum()

fig_profit = px.area(
    daily_profit,
    x="_date",
    y="Cumulative Profit",
    labels={"_date": "Date", "Cumulative Profit": "Units"},
)
fig_profit.update_layout(height=400)
st.plotly_chart(fig_profit, use_container_width=True)

# --- Volume Heatmap ---
st.subheader("Prediction Volume")
volume = df.groupby(["_date", "League"]).size().reset_index(name="Count")
volume_pivot = volume.pivot_table(index="League", columns="_date", values="Count", fill_value=0)

if not volume_pivot.empty:
    fig_heat = px.imshow(
        volume_pivot.values,
        x=[str(d) for d in volume_pivot.columns],
        y=volume_pivot.index.tolist(),
        labels=dict(x="Date", y="League", color="Predictions"),
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)
