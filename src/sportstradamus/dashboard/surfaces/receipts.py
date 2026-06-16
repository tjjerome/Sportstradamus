"""Receipts — the "prove it" surface.

Hero ("if you'd tailed every rec" units + record), skeptic checks (EV>5% record, CLV beat
rate, calibration, worst month — losers shown, never hidden), a by-league/market/platform
grid, your tracked slips, accuracy/profit/volume trends, the full CLV breakdown, and the
folded-in strategy simulator. A skeptic should be able to verify profitability unaided.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import brier_score_loss

from sportstradamus import clv
from sportstradamus.analysis import (
    JUICE_PAYOUT,
    compute_book_brier_skill_score,
    dedup_bets,
    ev_threshold_record,
    record_grid,
    tailed_record,
    worst_month,
)
from sportstradamus.dashboard.components.grid import render_themed_grid
from sportstradamus.dashboard.components.profit_sim import (
    render_profit_sim,
    render_profit_sim_summary,
)
from sportstradamus.dashboard.data import (
    filtered_history_or_stop,
    format_ts,
    load_history,
    load_parlays,
    load_profit_sim_summary,
    load_resolve_meta,
    load_user_slips,
    render_banner,
    sidebar_filters,
    sport_filtered,
)

st.title("Receipts")
render_banner("stats", "the track record, with the losers shown")

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

filters = sidebar_filters(history, parlays, key_prefix="receipts_")
df = filtered_history_or_stop(history, filters)

df = dedup_bets(df)
prob_col = "Win Prob" if "Win Prob" in df.columns and df["Win Prob"].notna().any() else "Model EV"
df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
df["Hit"] = (df["Bet"] == df["Result"]).astype(int)
df["Profit Unit"] = df["Hit"] * JUICE_PAYOUT - (1 - df["Hit"])

st.subheader("If you'd tailed every rec")
record = tailed_record(df)
h1, h2, h3 = st.columns(3)
h1.metric("ROI", f"{record['roi']:+.1%}")
h2.metric("Win rate", f"{record['win_pct']:.1%}")
h3.metric("Record", f"{record['wins']:,}–{record['losses']:,}")
st.caption(
    f"+{record['units']:.0f} units across {record['n']:,} unique resolved recs — flat -110 "
    "(win +0.91u, loss -1u), deduped across books, pushes excluded."
)

daily_profit = (
    df.groupby("_date")
    .agg(Profit=("Profit Unit", "sum"), Bets=("Hit", "count"))
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
fig_profit.update_layout(height=360)
st.plotly_chart(fig_profit, width="stretch")

st.download_button(
    "Export filtered history (CSV)",
    df.to_csv(index=False),
    "history_filtered.csv",
    "text/csv",
)

# CLV is a structural model property, not a view slice, so it reads the full
# (sport-narrowed) history rather than the sidebar-filtered df.
st.subheader("Skeptic checks")
clv_summary = clv.summarize(history)
ev_rec = ev_threshold_record(df)
brier = brier_score_loss(df["Hit"], df[prob_col].clip(0, 1))
book_skill = compute_book_brier_skill_score(df)
wm = worst_month(df)

s1, s2, s3, s4 = st.columns(4)
s1.metric(
    "Record at EV>5%",
    f"{ev_rec['win_pct']:.1%}" if ev_rec["n"] else "—",
    f"{ev_rec['n']} bets · {ev_rec['units']:+.1f}u" if ev_rec["n"] else "no qualifying recs",
)
s2.metric(
    "CLV beat rate",
    f"{clv_summary['frac_beat_close']:.1%}" if clv_summary["n"] else "—",
    f"{clv_summary['n']:,} legs with close" if clv_summary["n"] else "no close data yet",
)
s3.metric(
    "Brier",
    f"{brier:.4f}",
    f"{book_skill:+.1%} vs book" if pd.notna(book_skill) else "book baseline n/a",
)
if wm:
    s4.metric(
        "Worst month",
        wm["month"],
        f"{wm['units']:+.1f}u · {wm['n']} bets",
        delta_color="normal",  # negative units render red — losers are shown, never hidden
    )
else:
    s4.metric("Worst month", "—")

st.subheader("By league / market / platform")
dim = (
    st.segmented_control(
        "Group by", ["League", "Market", "Platform"], default="League", key="receipts_grid_dim"
    )
    or "League"
)
grid_df = record_grid(df, dim)
if grid_df.empty:
    st.caption("No resolved recs in this slice.")
else:
    # The themed grid's percent formatter expects percentage points, so scale the two
    # rate columns (record_grid returns fractions) for display only.
    grid_df = grid_df.assign(**{"Win%": grid_df["Win%"] * 100, "ROI": grid_df["ROI"] * 100})
    render_themed_grid(
        grid_df,
        numeric_cols=["Bets", "Win%", "Units", "ROI"],
        heatmap_col="Units",
        heatmap_center=0.0,
        percent_cols=["Win%", "ROI"],
        height=320,
        key="receipts_grid",
    )

st.subheader("Your slips")
user_slips = load_user_slips()
if user_slips.empty:
    st.caption("Lock in a slip on the Build or Board surface to track it here.")
else:
    has_status = "status" in user_slips.columns
    pending_n = int((user_slips["status"] == "pending").sum()) if has_status else len(user_slips)
    graded = (
        user_slips.loc[user_slips["status"] != "pending"] if has_status else user_slips.iloc[:0]
    )
    if not graded.empty:
        wins = int((graded["status"] == "won").sum())
        st.caption(
            f"{len(graded)} graded · {wins} won · {pending_n} pending — your record vs the model's."
        )
    else:
        st.caption(f"{pending_n} pending — graded nightly by `reflect`.")
    slip_cols = [
        c
        for c in [
            "saved_at",
            "platform",
            "play_type",
            "headline",
            "bet_size",
            "model_ev",
            "stake",
            "status",
        ]
        if c in user_slips.columns
    ]
    st.dataframe(
        user_slips.sort_values("saved_at", ascending=False)[slip_cols],
        hide_index=True,
        width="stretch",
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
st.plotly_chart(fig_acc, width="stretch")

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
    st.plotly_chart(fig_heat, width="stretch")

st.subheader("Closing Line Value")
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
            width="stretch",
            hide_index=True,
        )

st.subheader("Strategy simulator")
st.caption(
    "If you'd staked the model's positive-edge bets under each strategy. Computed nightly — "
    "ROI / Sharpe / drawdown / win-rate over each look-back window."
)
render_profit_sim_summary(load_profit_sim_summary())
with st.expander("Customize & run a live backtest"):
    render_profit_sim(history)
