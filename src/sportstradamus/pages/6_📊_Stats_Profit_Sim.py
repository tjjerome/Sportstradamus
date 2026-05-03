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
    get_filtered_history,
    load_history,
    load_resolve_meta,
    render_banner,
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
    st.caption(f"Data last resolved: {meta['last_run']}")

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

N_MONTE_CARLO = 100

# Ranking column mapping
RANKING_MAP = {
    "Kelly": "K",
    "Probability": "Model P",  # raw predicted probability
    "EV": "Model",  # Model P * Boost
}


def compute_payout(row):
    """Compute payout multiplier for a winning bet."""
    platform = row.get("Platform", "")
    boost = row.get("Boost", 1)
    if platform == "Underdog":
        return (100 / 110) * boost
    elif platform == "Sleeper":
        return boost
    else:
        return (100 / 110) * boost


def simulate_strategy(
    df,
    min_model_p,
    min_books_p,
    max_bets_day,
    sizing_pct,
    use_kelly,
    ranking,
    initial_bankroll,
    n_mc=N_MONTE_CARLO,
):
    """Run Monte Carlo profit simulation on exploded offer data.

    Returns DataFrame with columns: date, run, bankroll, daily_pnl
    """
    rank_col = RANKING_MAP.get(ranking, "K")

    # Filter by thresholds
    eligible = df.loc[(df[prob_col] >= min_model_p) & (df["Books"].fillna(0) >= min_books_p)].copy()

    if eligible.empty:
        return pd.DataFrame()

    # Best offer per player-market-date: pick the offer with the highest
    # ranking value (e.g., best Kelly, highest Model P * Boost, or highest Model P)
    eligible = eligible.sort_values(rank_col, ascending=False).drop_duplicates(
        subset=["Player", "Market", "_date"], keep="first"
    )

    eligible["_player_base"] = eligible["Player"].str.split(r" \+ | vs\. ").str[0]

    # Pre-compute payouts
    eligible["Payout"] = eligible.apply(compute_payout, axis=1)

    eligible = eligible.sort_values("_date")
    dates = sorted(eligible["_date"].unique())
    all_runs = []
    rng = np.random.default_rng(42)

    for run_i in range(n_mc):
        bankroll = initial_bankroll
        run_data = []

        for date in dates:
            day_bets = eligible.loc[eligible["_date"] == date].copy()
            if day_bets.empty:
                run_data.append({"date": date, "run": run_i, "bankroll": bankroll, "daily_pnl": 0})
                continue

            # Select bets via weighted sampling, ensuring each player
            # is only picked once (avoid correlated markets on same player)
            if len(day_bets) > max_bets_day:
                weights = day_bets[rank_col].clip(lower=0.01).values
                weights = weights / weights.sum()
                chosen = []
                used_players = set()
                pool_idx = list(range(len(day_bets)))
                pool_weights = weights.copy()
                while len(chosen) < max_bets_day and pool_idx:
                    pool_weights_norm = pool_weights / pool_weights.sum()
                    pick = rng.choice(len(pool_idx), p=pool_weights_norm)
                    actual_idx = pool_idx[pick]
                    player_base = day_bets.iloc[actual_idx]["_player_base"]
                    if player_base not in used_players:
                        chosen.append(actual_idx)
                        used_players.add(player_base)
                    pool_idx.pop(pick)
                    pool_weights = np.delete(pool_weights, pick)
                    if pool_weights.sum() == 0:
                        break
                day_bets = day_bets.iloc[chosen]
            else:
                # Even when taking all, deduplicate by base player (keep best)
                day_bets = day_bets.sort_values(rank_col, ascending=False).drop_duplicates(
                    subset=["_player_base"], keep="first"
                )

            daily_pnl = 0
            for _, bet in day_bets.iterrows():
                if use_kelly:
                    payout_mult = bet["Payout"]
                    if payout_mult <= 1:
                        continue
                    kelly_f = (bet[prob_col] * payout_mult - 1) / (payout_mult - 1)
                    kelly_f = max(0, min(kelly_f, 0.05))  # cap at 5%
                    bet_size = bankroll * kelly_f
                else:
                    bet_size = bankroll * sizing_pct / 100

                if bet["Hit"]:
                    daily_pnl += bet_size * bet["Payout"]
                else:
                    daily_pnl -= bet_size

            bankroll += daily_pnl
            bankroll = max(bankroll, 0)
            run_data.append(
                {"date": date, "run": run_i, "bankroll": bankroll, "daily_pnl": daily_pnl}
            )

        all_runs.extend(run_data)

    return pd.DataFrame(all_runs)


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
            params["min_model_p"],
            params["min_books_p"],
            params["max_bets_day"],
            params["sizing_pct"],
            params["use_kelly"],
            params["ranking"],
            initial_bankroll,
        )
        if not result.empty:
            all_results[name] = result

    # Custom
    custom_result = simulate_strategy(
        df,
        custom_min_p,
        custom_min_books,
        custom_max_bets,
        custom_sizing,
        custom_kelly,
        custom_ranking,
        initial_bankroll,
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
    final_runs = result.groupby("run").last()

    final_bankrolls = final_runs["bankroll"].values
    mean_final = final_bankrolls.mean()
    roi = (mean_final - initial_bankroll) / initial_bankroll

    # Drawdown: compute per-run max drawdown and average
    drawdowns = []
    for run_i in range(N_MONTE_CARLO):
        run_data = result.loc[result["run"] == run_i, "bankroll"].values
        if len(run_data) == 0:
            continue
        peak = np.maximum.accumulate(run_data)
        dd = (peak - run_data) / np.where(peak > 0, peak, 1)
        drawdowns.append(dd.max())

    mean_dd = np.mean(drawdowns) if drawdowns else 0

    # Sharpe-like ratio
    daily_returns = result.groupby("run").apply(
        lambda x: x["daily_pnl"].values
        / np.maximum(x["bankroll"].shift(1).fillna(initial_bankroll).values, 1)
    )
    all_returns = np.concatenate(daily_returns.values)
    sharpe = np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0

    # Win rate
    daily_outcomes = result.groupby(["run", "date"])["daily_pnl"].sum().reset_index()
    win_rate = (daily_outcomes["daily_pnl"] > 0).mean()

    summary_rows.append(
        {
            "Strategy": name,
            "Final Bankroll": f"${mean_final:,.0f}",
            "ROI": f"{roi:+.1%}",
            "Max Drawdown": f"{mean_dd:.1%}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Win% (daily)": f"{win_rate:.1%}",
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
