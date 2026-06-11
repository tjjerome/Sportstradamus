"""Receipts — overview KPIs, accuracy trends, profit trends, and volume."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import brier_score_loss

from sportstradamus import clv
from sportstradamus.dashboard.data import (
    filtered_history_or_stop,
    format_ts,
    load_history,
    load_parlays,
    load_resolve_meta,
    render_banner,
    sidebar_filters,
    sport_filtered,
)

# Standard -110 juice: risk $110 to win $100. Used for flat-unit P&L accounting.
_JUICE_PAYOUT = 100 / 110

st.title("Overview")
render_banner("stats", "historical accuracy, profit, and volume")

history = load_history()
parlays = load_parlays()

if history.empty:
    st.warning("No prediction history found. Run `prophecize` first.")
    st.stop()

meta = load_resolve_meta()
if meta.get("last_run"):
    st.caption(f"Data last resolved: {format_ts(meta['last_run'])}")
else:
    st.warning(
        "Nightly resolution has not run yet. Run `poetry run reflect` to resolve prediction outcomes."
    )

# Sport narrow runs before the sidebar builds its league multiselect, so the
# sidebar only offers that sport's leagues.
history = sport_filtered(history)
if history.empty:
    st.info("No resolved predictions match the current sport filter.")
    st.stop()

filters = sidebar_filters(history, parlays, key_prefix="overview_")
df = filtered_history_or_stop(history, filters)

prob_col = "Model P" if "Model P" in df.columns and df["Model P"].notna().any() else "Model"
df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
df["Hit"] = (df["Bet"] == df["Result"]).astype(int)
df["Profit Unit"] = df["Hit"] * _JUICE_PAYOUT - (1 - df["Hit"])

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

st.subheader("Prediction Volume")
volume = df.groupby(["_date", "League"]).size().reset_index(name="Count")
volume_pivot = volume.pivot_table(index="League", columns="_date", values="Count", fill_value=0)

if not volume_pivot.empty:
    fig_heat = px.imshow(
        volume_pivot.values,
        x=[str(d) for d in volume_pivot.columns],
        y=volume_pivot.index.tolist(),
        labels={"x": "Date", "y": "League", "color": "Predictions"},
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# CLV ignores the sidebar filters — it is a structural model property, not a
# view slice — but the global sport switch still applies (history is narrowed above).
st.subheader("Closing Line Value")
clv_summary = clv.summarize(history)

if clv_summary["n"] == 0:
    st.info(
        "No legs with closing-line data yet. CLV populates after `reflect` "
        "runs against archives that contain post-lock odds."
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolved legs (with close)", f"{clv_summary['n']:,}")
    c2.metric("Mean Market CLV", f"{clv_summary['market_clv_mean']:+.3f}")
    c3.metric(
        "Mean Model CLV",
        f"{clv_summary['model_clv_mean']:+.3f}" if pd.notna(clv_summary["model_clv_mean"]) else "—",
    )
    c4.metric("Beat-close rate", f"{clv_summary['frac_beat_close']:.1%}")

    segments = clv_summary["segments"]
    if not segments.empty:
        st.caption(
            f"Segments with at least {clv.CLV_SEGMENT_MIN_N} legs, sorted by mean Market CLV."
        )
        st.dataframe(
            segments.style.format({"market_clv": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
