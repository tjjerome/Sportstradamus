"""Page 4: Profit Simulation — preset and custom strategies with Monte Carlo."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sportstradamus.dashboard_data import (
    format_ts,
    get_filtered_history,
    load_history,
    load_resolve_meta,
    render_banner,
)
from sportstradamus.strategies.profit_sim import (
    N_MONTE_CARLO_DEFAULT,
    RANKING_MAP,
    simulate_strategy,
    summarize_runs,
)

st.title("Profit Simulation")
render_banner("stats", "Monte Carlo strategy backtesting")

# --- Load data (pre-resolved by nightly script) ---
history = load_history()
if history.empty:
    st.warning("No prediction history found.")
    st.stop()

meta = load_resolve_meta()
if meta.get("last_run"):
    st.caption(f"Data last resolved: {format_ts(meta['last_run'])}")

# --- Explode offers and filter to resolved, non-push ---
df = get_filtered_history(history)
if df.empty:
    st.info("No resolved predictions found.")
    st.stop()

df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
df = df.loc[df["_date"].notna()]

prob_col = "Model P" if "Model P" in df.columns and df["Model P"].notna().any() else "Model"

# Ensure derived columns exist
if "K" not in df.columns:
    df["K"] = df[prob_col] * df.get("Boost", 1)
if "Model" not in df.columns:
    df["Model"] = df["Model P"] * df["Boost"]

# --- Timeframe selector ---
st.sidebar.header("Simulation Settings")
tf_options = {"All time": None, "Last 30 days": 30, "3 months": 91, "6 months": 183, "1 year": 365}
tf_choice = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
tf_days = tf_options[tf_choice]
if tf_days is not None:
    cutoff = datetime.today().date() - timedelta(days=tf_days)
    df = df.loc[df["_date"] >= cutoff]

custom_range = st.sidebar.date_input(
    "Or custom date range",
    value=(),
    key="profit_custom_range",
)
if len(custom_range) == 2:
    df = df.loc[(df["_date"] >= custom_range[0]) & (df["_date"] <= custom_range[1])]

if df.empty:
    st.info("No data for selected timeframe.")
    st.stop()

# League filter
leagues = sorted(df["League"].unique())
selected_leagues = st.sidebar.multiselect("Leagues", leagues, default=leagues, key="profit_leagues")
df = df.loc[df["League"].isin(selected_leagues)]

# Platform filter
platforms = sorted(df["Platform"].unique())
selected_platforms = st.sidebar.multiselect(
    "Platforms", platforms, default=platforms, key="profit_platforms"
)
df = df.loc[df["Platform"].isin(selected_platforms)]

# --- Preset Strategies ---
PRESETS = {
    "Conservative": {
        "min_model_p": 0.65,
        "min_books_p": 0.52,
        "max_bets_day": 5,
        "sizing_pct": 1.0,
        "use_kelly": False,
        "ranking": "Kelly",
    },
    "Moderate": {
        "min_model_p": 0.60,
        "min_books_p": 0.52,
        "max_bets_day": 10,
        "sizing_pct": 1.0,
        "use_kelly": False,
        "ranking": "Kelly",
    },
    "Aggressive": {
        "min_model_p": 0.55,
        "min_books_p": 0.50,
        "max_bets_day": 20,
        "sizing_pct": 2.0,
        "use_kelly": False,
        "ranking": "EV",
    },
    "Kelly": {
        "min_model_p": 0.58,
        "min_books_p": 0.52,
        "max_bets_day": 15,
        "sizing_pct": 1.0,
        "use_kelly": True,
        "ranking": "Kelly",
    },
}

initial_bankroll = st.sidebar.number_input(
    "Initial Bankroll ($)", value=1000, min_value=100, step=100
)

# --- Custom Strategy ---
with st.sidebar.expander("Custom Strategy"):
    custom_min_p = st.slider("Min Model P", 0.50, 0.80, 0.60, 0.01, key="custom_min_p")
    custom_min_books = st.slider("Min Books P", 0.45, 0.60, 0.52, 0.01, key="custom_min_books")
    custom_max_bets = st.slider("Max Bets/Day", 1, 50, 10, key="custom_max_bets")
    custom_sizing = st.slider("Bet Size (%)", 0.5, 5.0, 1.0, 0.5, key="custom_sizing")
    custom_kelly = st.toggle("Kelly Sizing", value=False, key="custom_kelly")
    custom_ranking = st.selectbox(
        "Selection Ranking", list(RANKING_MAP.keys()), key="custom_ranking"
    )

# --- Run Simulations ---
st.header("Strategy Comparison")

with st.spinner("Running Monte Carlo simulations..."):
    all_results = {}

    for name, params in PRESETS.items():
        result = simulate_strategy(
            df,
            prob_col=prob_col,
            ranking=params["ranking"],
            min_model_p=params["min_model_p"],
            min_books_p=params["min_books_p"],
            max_bets_day=params["max_bets_day"],
            sizing_pct=params["sizing_pct"],
            use_kelly=params["use_kelly"],
            initial_bankroll=initial_bankroll,
            n_mc=N_MONTE_CARLO_DEFAULT,
        )
        if not result.empty:
            all_results[name] = result

    # Custom
    custom_result = simulate_strategy(
        df,
        prob_col=prob_col,
        ranking=custom_ranking,
        min_model_p=custom_min_p,
        min_books_p=custom_min_books,
        max_bets_day=custom_max_bets,
        sizing_pct=custom_sizing,
        use_kelly=custom_kelly,
        initial_bankroll=initial_bankroll,
        n_mc=N_MONTE_CARLO_DEFAULT,
    )
    if not custom_result.empty:
        all_results["Custom"] = custom_result

if not all_results:
    st.info("No bets matched any strategy criteria.")
    st.stop()

# --- Cumulative Bankroll Chart ---
fig_bank = go.Figure()

for name, result in all_results.items():
    agg = (
        result.groupby("date")
        .agg(
            mean_bankroll=("bankroll", "mean"),
            p10=("bankroll", lambda x: np.percentile(x, 10)),
            p90=("bankroll", lambda x: np.percentile(x, 90)),
        )
        .reset_index()
    )

    color_map = {
        "Conservative": "#3498db",
        "Moderate": "#2ecc71",
        "Aggressive": "#e74c3c",
        "Kelly": "#9b59b6",
        "Custom": "#f39c12",
    }
    color = color_map.get(name, "#95a5a6")

    # Confidence band
    fig_bank.add_trace(
        go.Scatter(
            x=pd.concat([agg["date"], agg["date"][::-1]]),
            y=pd.concat([agg["p90"], agg["p10"][::-1]]),
            fill="toself",
            fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba")
            if "rgb" in color
            else color + "1A",
            line=dict(width=0),
            name=f"{name} (10th-90th %ile)",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Mean line
    fig_bank.add_trace(
        go.Scatter(
            x=agg["date"],
            y=agg["mean_bankroll"],
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
        )
    )

fig_bank.add_hline(
    y=initial_bankroll, line_dash="dash", line_color="gray", annotation_text="Starting bankroll"
)
fig_bank.update_layout(
    xaxis_title="Date",
    yaxis_title="Bankroll ($)",
    height=500,
    title="Cumulative Bankroll (Monte Carlo Mean with 10th-90th Percentile Band)",
)
st.plotly_chart(fig_bank, use_container_width=True)

# --- Summary Table ---
st.subheader("Strategy Summary")
summary_rows = []
for name, result in all_results.items():
    summary = summarize_runs(result, initial_bankroll)
    summary_rows.append(
        {
            "Strategy": name,
            "Final Bankroll": f"${summary['mean_final']:,.0f}",
            "ROI": f"{summary['roi']:+.1%}",
            "Max Drawdown": f"{summary['max_drawdown']:.1%}",
            "Sharpe Ratio": f"{summary['sharpe']:.3f}",
            "Win% (daily)": f"{summary['win_rate']:.1%}",
        }
    )

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# --- Daily P&L for selected strategy ---
st.subheader("Daily P&L")
selected_strategy = st.selectbox("Strategy", list(all_results.keys()), key="pnl_strategy")
if selected_strategy in all_results:
    pnl_data = all_results[selected_strategy]
    daily_pnl_agg = pnl_data.groupby("date")["daily_pnl"].mean().reset_index()

    fig_pnl = px.bar(
        daily_pnl_agg,
        x="date",
        y="daily_pnl",
        color=daily_pnl_agg["daily_pnl"].apply(lambda x: "Profit" if x >= 0 else "Loss"),
        color_discrete_map={"Profit": "#2ecc71", "Loss": "#e74c3c"},
        labels={"date": "Date", "daily_pnl": "Daily P&L ($)"},
    )
    fig_pnl.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_pnl, use_container_width=True)

# --- Drawdown Chart ---
st.subheader("Drawdown Over Time")
if selected_strategy in all_results:
    dd_data = all_results[selected_strategy]
    dd_agg = (
        dd_data.groupby("date")
        .agg(
            mean_bankroll=("bankroll", "mean"),
        )
        .reset_index()
    )
    dd_agg["Peak"] = dd_agg["mean_bankroll"].cummax()
    dd_agg["Drawdown"] = (dd_agg["Peak"] - dd_agg["mean_bankroll"]) / dd_agg["Peak"].clip(lower=1)

    fig_dd = px.area(
        dd_agg,
        x="date",
        y="Drawdown",
        labels={"date": "Date", "Drawdown": "Drawdown (%)"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig_dd.update_layout(height=350, yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)

# --- Export ---
st.download_button(
    "Export simulation results (CSV)",
    summary_df.to_csv(index=False),
    "profit_simulation.csv",
    "text/csv",
)
