"""Profit-simulation panel — preset and custom strategies with Monte Carlo.

Folded into the Receipts surface (``surfaces/receipts.py``) as an expander. Controls
live inline (not on ``st.sidebar``, which the host page owns) so the panel embeds without
colliding with the page's own filters. ``render_profit_sim(history)`` takes the raw
prediction history and explodes it itself, keeping the sim independent of the Receipts
sidebar slice exactly as the old standalone page was.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sportstradamus.dashboard.data import get_filtered_history, sport_filtered
from sportstradamus.strategies.profit_sim import (
    N_MONTE_CARLO_DEFAULT,
    RANKING_MAP,
    simulate_strategy,
    summarize_runs,
)

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

# Strategy colors for the Monte Carlo chart — paired with PRESETS keys + "Custom".
_STRATEGY_COLORS = {
    "Conservative": "#3498db",
    "Moderate": "#2ecc71",
    "Aggressive": "#e74c3c",
    "Kelly": "#9b59b6",
    "Custom": "#f39c12",
}

_TIMEFRAMES = {
    "All time": None,
    "Last 30 days": 30,
    "3 months": 91,
    "6 months": 183,
    "1 year": 365,
}


def _band_fillcolor(hex_color: str) -> str:
    """10%-alpha rgba() for the percentile band; plotly rejects 8-digit hex."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},0.1)"


def _prepare_sim_frame(history: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Explode + sport-filter + backfill Kelly/Model EV; return ``(df, prob_col)``.

    Returns an empty frame (with an ``st.info`` already shown) when nothing resolves.
    """
    df = get_filtered_history(history)
    if df.empty:
        st.info("No resolved predictions found.")
        return df, ""
    df = sport_filtered(df)
    if df.empty:
        st.info("No resolved predictions match the current sport filter.")
        return df, ""

    df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.loc[df["_date"].notna()]
    prob_col = (
        "Win Prob" if "Win Prob" in df.columns and df["Win Prob"].notna().any() else "Model EV"
    )
    if "Kelly" not in df.columns:
        df["Kelly"] = df[prob_col] * df.get("Boost", 1)
    if "Model EV" not in df.columns:
        df["Model EV"] = df["Win Prob"] * df["Boost"]
    return df, prob_col


def _apply_sim_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, int, dict]:
    """Render the inline controls, apply timeframe/league/platform filters.

    Returns ``(filtered_df, initial_bankroll, custom_params)``; the frame is empty (with
    an ``st.info`` shown) when the timeframe slice removes everything.
    """
    c1, c2, c3 = st.columns(3)
    tf_choice = c1.selectbox("Timeframe", list(_TIMEFRAMES), index=0, key="profit_timeframe")
    initial_bankroll = c2.number_input(
        "Initial Bankroll ($)", value=1000, min_value=100, step=100, key="profit_bankroll"
    )
    custom_range = c3.date_input("Or custom date range", value=(), key="profit_custom_range")

    tf_days = _TIMEFRAMES[tf_choice]
    if tf_days is not None:
        df = df.loc[df["_date"] >= datetime.today().date() - timedelta(days=tf_days)]
    if len(custom_range) == 2:
        df = df.loc[(df["_date"] >= custom_range[0]) & (df["_date"] <= custom_range[1])]
    if df.empty:
        st.info("No data for selected timeframe.")
        return df, initial_bankroll, {}

    leagues = sorted(df["League"].unique())
    df = df.loc[
        df["League"].isin(st.multiselect("Leagues", leagues, leagues, key="profit_leagues"))
    ]
    platforms = sorted(df["Platform"].unique())
    df = df.loc[
        df["Platform"].isin(
            st.multiselect("Platforms", platforms, platforms, key="profit_platforms")
        )
    ]

    with st.expander("Custom strategy"):
        custom = {
            "min_model_p": st.slider(
                "Min Win Prob", 0.50, 0.80, 0.60, 0.01, key="profit_custom_min_p"
            ),
            "min_books_p": st.slider(
                "Min Market Prob", 0.45, 0.60, 0.52, 0.01, key="profit_custom_min_books"
            ),
            "max_bets_day": st.slider("Max Bets/Day", 1, 50, 10, key="profit_custom_max_bets"),
            "sizing_pct": st.slider("Bet Size (%)", 0.5, 5.0, 1.0, 0.5, key="profit_custom_sizing"),
            "use_kelly": st.toggle("Kelly Sizing", value=False, key="profit_custom_kelly"),
            "ranking": st.selectbox(
                "Selection Ranking", list(RANKING_MAP), key="profit_custom_ranking"
            ),
        }
    return df, initial_bankroll, custom


def _run_strategies(df: pd.DataFrame, prob_col: str, initial_bankroll: int, custom: dict) -> dict:
    """Run the preset strategies plus the custom one; drop those that match no bets."""
    all_results = {}
    for name, params in {**PRESETS, "Custom": custom}.items():
        result = simulate_strategy(
            df,
            prob_col=prob_col,
            initial_bankroll=initial_bankroll,
            n_mc=N_MONTE_CARLO_DEFAULT,
            **params,
        )
        if not result.empty:
            all_results[name] = result
    return all_results


def _render_bankroll_chart(all_results: dict, initial_bankroll: int) -> None:
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
        color = _STRATEGY_COLORS.get(name, "#95a5a6")
        fig_bank.add_trace(
            go.Scatter(
                x=pd.concat([agg["date"], agg["date"][::-1]]),
                y=pd.concat([agg["p90"], agg["p10"][::-1]]),
                fill="toself",
                fillcolor=_band_fillcolor(color),
                line={"width": 0},
                name=f"{name} (10th-90th %ile)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_bank.add_trace(
            go.Scatter(
                x=agg["date"],
                y=agg["mean_bankroll"],
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
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
    st.plotly_chart(fig_bank, width="stretch")


def _render_summary_table(all_results: dict, initial_bankroll: int) -> pd.DataFrame:
    """Build + render the per-strategy summary table; return it for the CSV export."""
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
    st.dataframe(summary_df, width="stretch", hide_index=True)
    return summary_df


def _render_pnl_drawdown(all_results: dict) -> None:
    st.subheader("Daily P&L")
    selected = st.selectbox("Strategy", list(all_results), key="profit_pnl_strategy")
    data = all_results[selected]

    daily_pnl_agg = data.groupby("date")["daily_pnl"].mean().reset_index()
    fig_pnl = px.bar(
        daily_pnl_agg,
        x="date",
        y="daily_pnl",
        color=daily_pnl_agg["daily_pnl"].apply(lambda x: "Profit" if x >= 0 else "Loss"),
        color_discrete_map={"Profit": "#2ecc71", "Loss": "#e74c3c"},
        labels={"date": "Date", "daily_pnl": "Daily P&L ($)"},
    )
    fig_pnl.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_pnl, width="stretch")

    st.subheader("Drawdown Over Time")
    dd_agg = data.groupby("date").agg(mean_bankroll=("bankroll", "mean")).reset_index()
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
    st.plotly_chart(fig_dd, width="stretch")


def render_profit_sim(history: pd.DataFrame) -> None:
    """Render the Monte-Carlo strategy panel for ``history`` (raw prediction rows)."""
    df, prob_col = _prepare_sim_frame(history)
    if df.empty:
        return
    df, initial_bankroll, custom = _apply_sim_controls(df)
    if df.empty:
        return

    with st.spinner("Running Monte Carlo simulations..."):
        all_results = _run_strategies(df, prob_col, initial_bankroll, custom)
    if not all_results:
        st.info("No bets matched any strategy criteria.")
        return

    _render_bankroll_chart(all_results, initial_bankroll)
    summary_df = _render_summary_table(all_results, initial_bankroll)
    _render_pnl_drawdown(all_results)

    st.download_button(
        "Export simulation results (CSV)",
        summary_df.to_csv(index=False),
        "profit_simulation.csv",
        "text/csv",
    )
