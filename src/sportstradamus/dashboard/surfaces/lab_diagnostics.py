"""Lab Diagnostics — market diagnostics and forecast quality.

Combines market-level diagnostics with professional forecasting metrics
following Gneiting & Raftery (2007) and Murphy (1973).
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss

from sportstradamus.analysis import (
    compute_brier_skill_score,
    compute_coverage,
    murphy_decomposition,
)
from sportstradamus.dashboard.components.grid import render_themed_grid
from sportstradamus.dashboard.components.hero import page_hero
from sportstradamus.dashboard.components.lab_filters import apply_lab_filters, render_lab_filters
from sportstradamus.dashboard.data import (
    filtered_history_or_stop,
    get_prediction_history,
    load_resolved_history_or_stop,
    load_stat_meta,
    sidebar_filters,
    sport_filtered,
)
from sportstradamus.dashboard.surfaces.lab_diagnostics_charts import (
    bias_bar,
    bias_color,
    bss_bar,
    coverage_bar,
    crps_line,
    ev_divergence_bar,
    murphy_decomp_bar,
    reliability_diagram,
    sharpness_histogram,
)

# Minimum predicted-probability std; markets below this cluster too tightly
# to discriminate, flagged as likely under-dispersed.
_LOW_SHARPNESS_STD = 0.04

# "Start here" strip width: the worst-BSS cells surfaced as metric tiles above the
# market table (spec: find-the-weak-spots framing, the opposite intent from Receipts).
_START_HERE_COUNT = 3

page_hero("MODEL LAB · DIAGNOSTICS", "Market Diagnostics & Forecast Quality")

stat_meta = load_stat_meta()
lab_sel = render_lab_filters(stat_meta, collapsed=False)

history = load_resolved_history_or_stop()

# Sport narrow runs before the sidebar builds its league multiselect, so the
# sidebar only offers that sport's leagues.
history = sport_filtered(history)
if history.empty:
    st.info("No data matches the current sport filter.")
    st.stop()

filters = sidebar_filters(history, key_prefix="mkt_", time_window_key="mkt_time")
cutoff = filters["cutoff"]

df = filtered_history_or_stop(history, filters)
df = apply_lab_filters(df, stat_meta, lab_sel)
if df.empty:
    st.info("No data matches the current Lab filter selection.")
    st.stop()

prob_col = "Win Prob" if "Win Prob" in df.columns and df["Win Prob"].notna().any() else "Model EV"
df["Hit"] = (df["Bet"] == df["Result"]).astype(int)
df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

if cutoff is not None:
    df = df.loc[df["_date"] >= cutoff]

if df.empty:
    st.info("No data for selected time window.")
    st.stop()

# Coverage is a diagnostics fact (share of predictions carrying a stored distribution),
# not a filter — it lives in the page body, not the shared sidebar filter panel.
if "Dist" in df.columns:
    st.metric(
        "Distribution Data Coverage",
        f"{df['Dist'].notna().mean():.0%}",
        help="Share of predictions in the current filter carrying a stored distribution family.",
    )

# CRPS arrives as a precomputed per-prediction column; here we only aggregate
# the stored values instead of re-integrating per row on every rerun.
pred_df = get_prediction_history(
    history,
    leagues=filters["leagues"],
    date_range=filters["date_range"],
)
pred_df = apply_lab_filters(pred_df, stat_meta, lab_sel)
pred_df["_date"] = pd.to_datetime(pred_df["Date"], errors="coerce").dt.date
if cutoff is not None:
    pred_df = pred_df.loc[pred_df["_date"] >= cutoff]

crps_by_market = pd.Series(dtype=float)
crps_counts = pd.Series(dtype=int)
if "CRPS" in pred_df.columns:
    _crps_grp = pred_df.dropna(subset=["CRPS"]).groupby(["League", "Market"])["CRPS"]
    crps_by_market = _crps_grp.mean()
    crps_counts = _crps_grp.size()

st.header("Market-Level Diagnostics")

st.subheader("Accuracy by League-Market")
market_rows = []
for (league, market), grp in df.groupby(["League", "Market"]):
    if len(grp) < 5:
        continue
    hits = grp["Hit"]
    pred_over_pct = (grp["Bet"] == "Over").mean()
    actual_over_pct = (grp["Result"] == "Over").mean()
    balance = pred_over_pct - actual_over_pct

    brier = brier_score_loss(hits, grp[prob_col].clip(0, 1))
    bss = compute_brier_skill_score(grp)

    row = {
        "League": league,
        "Market": market,
        "Accuracy": round(hits.mean(), 4),
        "Balance": round(balance, 4),
        "Brier": round(brier, 4),
        "BSS": round(bss, 4) if not np.isnan(bss) else None,
        "Samples": len(grp),
    }

    if (league, market) in crps_by_market.index and crps_counts.get((league, market), 0) >= 5:
        row["CRPS"] = round(float(crps_by_market.loc[(league, market)]), 4)

    market_rows.append(row)

market_df = pd.DataFrame(market_rows)
if not market_df.empty:
    market_df = market_df.merge(
        stat_meta[["league", "market", "dist", "target_normalization"]],
        left_on=["League", "Market"],
        right_on=["league", "market"],
        how="left",
    ).rename(columns={"dist": "Family", "target_normalization": "Norm"})
    market_df = market_df.drop(columns=["league", "market"])
    market_df = market_df.sort_values("BSS", ascending=True, na_position="last")

    worst_cells = market_df.dropna(subset=["BSS"]).head(_START_HERE_COUNT)
    if not worst_cells.empty:
        st.caption("Start here — lowest Brier Skill Score in the current filter:")
        tiles = st.columns(len(worst_cells))
        for tile, (_, cell) in zip(tiles, worst_cells.iterrows(), strict=True):
            tile.metric(f"{cell['League']} - {cell['Market']}", f"{cell['BSS']:+.3f} BSS")

    numeric_cols = [
        c
        for c in ("Accuracy", "Balance", "Brier", "BSS", "CRPS", "Samples")
        if c in market_df.columns
    ]
    render_themed_grid(
        market_df,
        numeric_cols=numeric_cols,
        flag_col="BSS",
        flag_below=0.0,
        height=min(720, max(200, 40 + 32 * len(market_df))),
    )
    st.download_button(
        "Export market table (CSV)",
        market_df.to_csv(index=False),
        "market_diagnostics.csv",
        "text/csv",
    )

st.subheader("Prediction Bias by Market")
if not market_df.empty:
    bias_df = market_df.copy()
    bias_df["abs_balance"] = bias_df["Balance"].abs()
    bias_df["Color"] = bias_df["Balance"].apply(bias_color)
    bias_df["Label"] = bias_df["League"] + " - " + bias_df["Market"]
    bias_df = bias_df.sort_values("Balance")
    st.plotly_chart(bias_bar(bias_df), width="stretch")

st.subheader("Prediction Sharpness (Win Prob Distribution)")
selected_league = st.selectbox(
    "League for sharpness view", ["All", *sorted(df["League"].unique())], key="sharp_league"
)
sharp_df = df if selected_league == "All" else df.loc[df["League"] == selected_league]

if not sharp_df.empty:
    markets_to_show = sorted(sharp_df["Market"].value_counts().head(12).index)
    sharp_subset = sharp_df.loc[sharp_df["Market"].isin(markets_to_show)]
    st.plotly_chart(sharpness_histogram(sharp_subset, prob_col), width="stretch")

    sharpness_df = sharp_df.groupby("Market")[prob_col].std().reset_index()
    sharpness_df.columns = ["Market", "Std(Win Prob)"]
    sharpness_df = sharpness_df.sort_values("Std(Win Prob)")
    low_sharp = sharpness_df.loc[sharpness_df["Std(Win Prob)"] < _LOW_SHARPNESS_STD]
    if not low_sharp.empty:
        st.warning(
            f"Low sharpness (std < {_LOW_SHARPNESS_STD}) — predictions cluster too tightly: "
            f"{', '.join(low_sharp['Market'].tolist())}"
        )

st.subheader("Accuracy by Model-Book Divergence")
if "Projection" in df.columns and "Line" in df.columns:
    ev_df = df.dropna(subset=["Projection", "Line"]).copy()
    ev_df["EV_Div"] = (ev_df["Projection"] - ev_df["Line"]).abs() / ev_df["Line"].clip(lower=0.1)
    ev_df["Div_Bucket"] = pd.qcut(ev_df["EV_Div"], q=5, duplicates="drop")
    div_stats = (
        ev_df.groupby("Div_Bucket", observed=False)
        .agg(Accuracy=("Hit", "mean"), Count=("Hit", "count"))
        .reset_index()
    )
    div_stats["Div_Bucket"] = div_stats["Div_Bucket"].astype(str)
    st.plotly_chart(ev_divergence_bar(div_stats), width="stretch")

st.header("Forecast Quality (Proper Scoring Rules)")

st.subheader("Reliability Diagram")
cal_df = df.copy()
bins = np.linspace(0.5, 1.0, 11)
cal_df["bin"] = pd.cut(cal_df[prob_col], bins=bins)
cal_stats = (
    cal_df.groupby("bin", observed=False)
    .agg(
        Predicted=(prob_col, "mean"),
        Actual=("Hit", "mean"),
        Count=("Hit", "count"),
    )
    .reset_index()
)
st.plotly_chart(reliability_diagram(cal_stats), width="stretch")

st.subheader("Brier Skill Score by League-Market")
if not market_df.empty and "BSS" in market_df.columns:
    bss_df = market_df.dropna(subset=["BSS"]).copy()
    bss_df["Label"] = bss_df["League"] + " - " + bss_df["Market"]
    bss_df = bss_df.sort_values("BSS", ascending=True)
    st.plotly_chart(bss_bar(bss_df), width="stretch")

st.subheader("Brier Score Decomposition (Murphy 1973)")
decomp_rows = []
for (league, market), grp in df.groupby(["League", "Market"]):
    if len(grp) < 20:
        continue
    d = murphy_decomposition(grp)
    d["League"] = league
    d["Market"] = market
    decomp_rows.append(d)

if decomp_rows:
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df["Label"] = decomp_df["League"] + " - " + decomp_df["Market"]
    decomp_df = decomp_df.sort_values("Brier")
    st.plotly_chart(murphy_decomp_bar(decomp_df), width="stretch")
    st.caption(
        "BS = Reliability - Resolution + Uncertainty. "
        "Good models have low reliability (well-calibrated) and high resolution (discriminative)."
    )

has_crps = "CRPS" in pred_df.columns and pred_df["CRPS"].notna().any()
has_dist = "Dist" in pred_df.columns and "Actual" in pred_df.columns
if has_crps:
    crps_df = pred_df.dropna(subset=["CRPS"]).copy()
    if len(crps_df) >= 10:
        st.subheader("CRPS Over Time")
        crps_daily = (
            crps_df.groupby(["_date", "League"])
            .agg(CRPS=("CRPS", "mean"), Count=("CRPS", "count"))
            .reset_index()
        )
        st.plotly_chart(crps_line(crps_daily), width="stretch")

if has_dist:
    cov_df = pred_df.dropna(subset=["Dist", "Actual"])
    if len(cov_df) >= 20:
        st.subheader("Prediction Interval Coverage")
        coverage = compute_coverage(cov_df, levels=(0.5, 0.8, 0.9))

        cov_display = pd.DataFrame(
            [
                {
                    "Nominal Level": f"{int(level * 100)}%",
                    "Actual Coverage": f"{cov:.1%}",
                    "Status": "Good"
                    if abs(cov - level) < 0.05
                    else ("Overconfident" if cov < level else "Underconfident"),
                }
                for level, cov in coverage.items()
                if not np.isnan(cov)
            ]
        )
        if not cov_display.empty:
            st.dataframe(cov_display, width="stretch", hide_index=True)

            cov_rows = []
            for league, lgrp in cov_df.groupby("League"):
                if len(lgrp) < 10:
                    continue
                lcov = compute_coverage(lgrp, levels=(0.5, 0.8, 0.9))
                for level, cov_val in lcov.items():
                    if not np.isnan(cov_val):
                        cov_rows.append({"League": league, "Nominal": level, "Actual": cov_val})
            if cov_rows:
                st.plotly_chart(coverage_bar(pd.DataFrame(cov_rows)), width="stretch")
