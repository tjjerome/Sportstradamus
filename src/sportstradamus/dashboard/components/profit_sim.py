"""Profit-simulation panel — precomputed grid + an opt-in live Monte-Carlo backtest.

Folded into the Receipts surface (``surfaces/receipts.py``). ``render_profit_sim_summary``
shows the strategy x horizon grid ``reflect`` precomputed (instant, no Monte-Carlo at page
load — the owner's "do the MC in the nightly run, not at load time"). ``render_profit_sim``
is the opt-in live run behind an expander: it re-simulates the same strategies over a custom
bankroll / look-back / consensus-edge filter, with the bankroll fan chart and P&L. Controls
live inline (not on ``st.sidebar``, which the host page owns).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sportstradamus.analysis import PROFIT_SIM_STRATEGIES, filter_bet_universe
from sportstradamus.dashboard import theme
from sportstradamus.dashboard.data import get_filtered_history, sport_filtered
from sportstradamus.strategies.profit_sim import (
    N_MONTE_CARLO_DEFAULT,
    simulate_strategy,
    summarize_runs,
)

# Look-back windows (owner spec: 1w / 1m / 3m / 6m / 1y). Days match analysis.TIMEFRAMES,
# whose raw labels key the precomputed grid; _HORIZON_LABELS maps those to these.
_HORIZONS = {
    "1 week": 7,
    "1 month": 30,
    "3 months": 91,
    "6 months": 183,
    "1 year": 365,
}
_HORIZON_LABELS = {
    "7d": "1 week",
    "30d": "1 month",
    "3m": "3 months",
    "6m": "6 months",
    "1y": "1 year",
}

# One distinct line color per strategy for the Monte-Carlo fan chart (brand gold leads).
_STRATEGY_COLORS = {
    "Flat · Conservative": "#4F86C6",
    "Flat · Moderate": "#3FA796",
    "Kelly · Conservative": "#C9A227",
    "Kelly · Moderate": "#D98E04",
}


def _band_fillcolor(hex_color: str) -> str:
    """10%-alpha rgba() for the percentile band; plotly rejects 8-digit hex."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},0.1)"


def render_profit_sim_summary(summary: pd.DataFrame) -> None:
    """Show the precomputed strategy grid for a chosen look-back — instant, no MC."""
    if summary.empty:
        st.caption("The strategy backtest runs nightly; no results yet.")
        return
    available = [h for h in _HORIZON_LABELS if h in set(summary["Horizon"])]
    default = "3m" if "3m" in available else available[-1]
    choice = st.selectbox(
        "Look-back window",
        available,
        index=available.index(default),
        format_func=lambda h: _HORIZON_LABELS[h],
        key="receipts_sim_horizon",
    )
    grid = summary.loc[summary["Horizon"] == choice].sort_values("ROI", ascending=False)
    display = pd.DataFrame(
        {
            "Strategy": grid["Strategy"],
            "ROI": grid["ROI"].map("{:+.1%}".format),
            "Sharpe": grid["Sharpe"].map("{:.2f}".format),
            "Max Drawdown": grid["Max Drawdown"].map("{:.1%}".format),
            "Win%": grid["Win%"].map("{:.1%}".format),
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


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


def _apply_sim_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Render the inline controls, apply the look-back window + edge filter.

    Returns ``(filtered_df, initial_bankroll, compound)``; the frame is empty (with an
    ``st.info`` shown) when the window or the edge filter removes everything.
    """
    c1, c2, c3 = st.columns(3)
    horizon = c1.selectbox("Look-back window", list(_HORIZONS), index=1, key="profit_horizon")
    initial_bankroll = c2.number_input(
        "Initial bankroll ($)", value=1000, min_value=100, step=100, key="profit_bankroll"
    )
    min_edge = c3.slider("Min consensus edge", 0.0, 0.10, 0.0, 0.01, key="profit_min_edge")
    compound = st.toggle(
        "Compound bankroll (size off current, not initial)", value=False, key="profit_compound"
    )

    df = df.loc[df["_date"] >= datetime.today().date() - timedelta(days=_HORIZONS[horizon])]
    df = filter_bet_universe(df, min_consensus_edge=min_edge)
    if df.empty:
        st.info("No positive-edge bets in this window.")
    return df, int(initial_bankroll), compound


def _run_strategies(
    df: pd.DataFrame, prob_col: str, initial_bankroll: int, *, compound: bool
) -> dict:
    """Run every strategy over ``df``; flip flat staking to off-current when compounding."""
    all_results = {}
    for name, params in PROFIT_SIM_STRATEGIES.items():
        call = dict(params)
        if not call["use_kelly"]:
            call["flat_off_initial"] = not compound
        result = simulate_strategy(
            df,
            prob_col=prob_col,
            initial_bankroll=initial_bankroll,
            n_mc=N_MONTE_CARLO_DEFAULT,
            **call,
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
        color_discrete_map={"Profit": theme.GREEN, "Loss": theme.RED},
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
        color_discrete_sequence=[theme.RED],
    )
    fig_dd.update_layout(height=350, yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, width="stretch")


def render_profit_sim(history: pd.DataFrame) -> None:
    """Render the live Monte-Carlo strategy panel for ``history`` (raw prediction rows)."""
    df, prob_col = _prepare_sim_frame(history)
    if df.empty:
        return
    df, initial_bankroll, compound = _apply_sim_controls(df)
    if df.empty:
        return

    with st.spinner("Running Monte Carlo simulations..."):
        all_results = _run_strategies(df, prob_col, initial_bankroll, compound=compound)
    if not all_results:
        st.info("No bets matched any strategy.")
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
