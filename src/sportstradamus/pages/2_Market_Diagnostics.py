"""Page 2: Market Diagnostics & Forecast Quality.

Combines market-level diagnostics with professional forecasting metrics
following Gneiting & Raftery (2007) and Murphy (1973).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import brier_score_loss, log_loss

from sportstradamus.dashboard_data import (
    load_history, load_stats, sidebar_filters, get_filtered_history,
    get_prediction_history, resolve_and_save,
)
from sportstradamus.analysis import (
    compute_brier_skill_score, murphy_decomposition,
    compute_crps_row, compute_coverage, reconstruct_prob,
    reconstruct_quantile, TIMEFRAMES,
)

st.title("Market Diagnostics & Forecast Quality")

# --- Load data ---
history = load_history()
if history.empty:
    st.warning("No prediction history found.")
    st.stop()

stats = load_stats()
history = resolve_and_save(history, stats)
filters = sidebar_filters(history, key_prefix="mkt_")

df = get_filtered_history(
    history,
    leagues=filters["leagues"],
    platforms=filters["platforms"],
    date_range=filters["date_range"],
)
if df.empty:
    st.info("No resolved predictions match the current filters.")
    st.stop()

prob_col = "Model P" if "Model P" in df.columns and df["Model P"].notna().any() else "Model"
df["Hit"] = (df["Bet"] == df["Result"]).astype(int)
df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

# --- Time-split selector ---
time_split = st.selectbox("Time window", ["All"] + [t[0] for t in TIMEFRAMES], index=0)
if time_split != "All":
    days = dict(TIMEFRAMES)[time_split]
    cutoff = datetime.today().date() - timedelta(days=days)
    df = df.loc[df["_date"] >= cutoff]

if df.empty:
    st.info("No data for selected time window.")
    st.stop()

# =====================================================================
# MARKET DIAGNOSTICS
# =====================================================================
st.header("Market-Level Diagnostics")

# --- Accuracy Table ---
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

    # CRPS if distribution data available
    if "Dist" in grp.columns:
        dist_valid = grp.dropna(subset=["Dist", "Actual"])
        if len(dist_valid) >= 5:
            crps_vals = dist_valid.apply(compute_crps_row, axis=1)
            row["CRPS"] = round(crps_vals.mean(), 4)

    market_rows.append(row)

market_df = pd.DataFrame(market_rows)
if not market_df.empty:
    st.dataframe(
        market_df.sort_values("Accuracy", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button("Export market table (CSV)", market_df.to_csv(index=False),
                       "market_diagnostics.csv", "text/csv")

# --- Bias Detector ---
st.subheader("Prediction Bias by Market")
if not market_df.empty:
    bias_df = market_df.copy()
    bias_df["abs_balance"] = bias_df["Balance"].abs()

    def bias_color(b):
        ab = abs(b)
        if ab < 0.03:
            return "green"
        elif ab < 0.07:
            return "orange"
        return "red"

    bias_df["Color"] = bias_df["Balance"].apply(bias_color)
    bias_df["Label"] = bias_df["League"] + " - " + bias_df["Market"]
    bias_df = bias_df.sort_values("Balance")

    fig_bias = px.bar(
        bias_df, x="Balance", y="Label", orientation="h",
        color="Color",
        color_discrete_map={"green": "#2ecc71", "orange": "#f39c12", "red": "#e74c3c"},
        labels={"Balance": "Over Bias (Predicted − Actual)", "Label": ""},
    )
    fig_bias.update_layout(showlegend=False, height=max(300, len(bias_df) * 22))
    fig_bias.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_bias, use_container_width=True)

# --- Prediction Clustering / Sharpness ---
st.subheader("Prediction Sharpness (Model P Distribution)")
selected_league = st.selectbox("League for sharpness view",
                               ["All"] + sorted(df["League"].unique()),
                               key="sharp_league")
sharp_df = df if selected_league == "All" else df.loc[df["League"] == selected_league]

if not sharp_df.empty:
    markets_to_show = sorted(sharp_df["Market"].value_counts().head(12).index)
    sharp_subset = sharp_df.loc[sharp_df["Market"].isin(markets_to_show)]

    fig_sharp = px.histogram(
        sharp_subset, x=prob_col, facet_col="Market", facet_col_wrap=4,
        nbins=20, labels={prob_col: "Model P"},
        title="Distribution of predicted probabilities per market",
    )
    fig_sharp.update_layout(height=600)
    st.plotly_chart(fig_sharp, use_container_width=True)

    # Flag low-sharpness markets
    sharpness_df = sharp_df.groupby("Market")[prob_col].std().reset_index()
    sharpness_df.columns = ["Market", "Std(Model P)"]
    sharpness_df = sharpness_df.sort_values("Std(Model P)")
    low_sharp = sharpness_df.loc[sharpness_df["Std(Model P)"] < 0.04]
    if not low_sharp.empty:
        st.warning(f"Low sharpness (std < 0.04) — predictions cluster too tightly: "
                   f"{', '.join(low_sharp['Market'].tolist())}")

# --- Accuracy by EV Divergence ---
st.subheader("Accuracy by Model-Book Divergence")
if "Model EV" in df.columns and "Line" in df.columns:
    ev_df = df.dropna(subset=["Model EV", "Line"]).copy()
    ev_df["EV_Div"] = (ev_df["Model EV"] - ev_df["Line"]).abs() / ev_df["Line"].clip(lower=0.1)
    ev_df["Div_Bucket"] = pd.qcut(ev_df["EV_Div"], q=5, duplicates="drop")

    div_stats = ev_df.groupby("Div_Bucket", observed=False).agg(
        Accuracy=("Hit", "mean"),
        Count=("Hit", "count"),
    ).reset_index()
    div_stats["Div_Bucket"] = div_stats["Div_Bucket"].astype(str)

    fig_div = px.bar(div_stats, x="Div_Bucket", y="Accuracy",
                     text="Count", labels={"Div_Bucket": "|Model EV - Line| / Line"})
    fig_div.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_div.update_layout(height=400)
    st.plotly_chart(fig_div, use_container_width=True)

# =====================================================================
# PROPER SCORING METRICS
# =====================================================================
st.header("Forecast Quality (Proper Scoring Rules)")

# --- Calibration / Reliability Diagram ---
st.subheader("Reliability Diagram")
cal_df = df.copy()
bins = np.linspace(0.5, 1.0, 11)
cal_df["bin"] = pd.cut(cal_df[prob_col], bins=bins)
cal_stats = cal_df.groupby("bin", observed=False).agg(
    Predicted=(prob_col, "mean"),
    Actual=("Hit", "mean"),
    Count=("Hit", "count"),
).reset_index()

fig_rel = go.Figure()
# Perfect calibration line
fig_rel.add_trace(go.Scatter(
    x=[0.5, 1.0], y=[0.5, 1.0],
    mode="lines", line=dict(dash="dash", color="gray"),
    name="Perfect calibration", showlegend=True,
))
# Calibration points
fig_rel.add_trace(go.Scatter(
    x=cal_stats["Predicted"], y=cal_stats["Actual"],
    mode="lines+markers", name="Model",
    marker=dict(size=cal_stats["Count"].clip(upper=500) / 20 + 5),
    text=[f"n={c}" for c in cal_stats["Count"]],
    hovertemplate="Predicted: %{x:.3f}<br>Actual: %{y:.3f}<br>%{text}",
))
# Count histogram on secondary y-axis
fig_rel.add_trace(go.Bar(
    x=cal_stats["Predicted"], y=cal_stats["Count"],
    name="Sample count", yaxis="y2", opacity=0.3,
))
fig_rel.update_layout(
    yaxis=dict(title="Actual Hit Rate", range=[0.4, 1.0]),
    yaxis2=dict(title="Count", overlaying="y", side="right"),
    xaxis_title="Predicted Probability",
    height=450,
)
st.plotly_chart(fig_rel, use_container_width=True)

# --- Brier Skill Score by League ---
st.subheader("Brier Skill Score by League-Market")
if not market_df.empty and "BSS" in market_df.columns:
    bss_df = market_df.dropna(subset=["BSS"]).copy()
    bss_df["Label"] = bss_df["League"] + " - " + bss_df["Market"]
    bss_df = bss_df.sort_values("BSS", ascending=True)

    fig_bss = px.bar(bss_df, x="BSS", y="Label", orientation="h",
                     color="BSS", color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0,
                     labels={"BSS": "Brier Skill Score", "Label": ""})
    fig_bss.add_vline(x=0, line_dash="dash", line_color="gray",
                      annotation_text="No skill")
    fig_bss.update_layout(height=max(300, len(bss_df) * 22))
    st.plotly_chart(fig_bss, use_container_width=True)

# --- Murphy Decomposition ---
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

    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Bar(
        y=decomp_df["Label"], x=decomp_df["Reliability"],
        name="Reliability (lower=better)", orientation="h",
        marker_color="#e74c3c",
    ))
    fig_decomp.add_trace(go.Bar(
        y=decomp_df["Label"], x=-decomp_df["Resolution"],
        name="Resolution (higher=better)", orientation="h",
        marker_color="#2ecc71",
    ))
    fig_decomp.update_layout(
        barmode="relative",
        xaxis_title="Contribution to Brier Score",
        height=max(300, len(decomp_df) * 22),
    )
    st.plotly_chart(fig_decomp, use_container_width=True)
    st.caption("BS = Reliability - Resolution + Uncertainty. "
               "Good models have low reliability (well-calibrated) and high resolution (discriminative).")

# --- CRPS and Coverage use prediction-level data (no explosion needed) ---
pred_df = get_prediction_history(
    history,
    leagues=filters["leagues"],
    date_range=filters["date_range"],
)
if time_split != "All":
    pred_df["_date"] = pd.to_datetime(pred_df["Date"], errors="coerce").dt.date
    pred_df = pred_df.loc[pred_df["_date"] >= cutoff]
else:
    pred_df["_date"] = pd.to_datetime(pred_df["Date"], errors="coerce").dt.date

# --- CRPS Over Time ---
has_crps = "Dist" in pred_df.columns and "Actual" in pred_df.columns
if has_crps:
    crps_df = pred_df.dropna(subset=["Dist", "Actual"]).copy()
    if len(crps_df) >= 10:
        st.subheader("CRPS Over Time")
        crps_df["CRPS"] = crps_df.apply(compute_crps_row, axis=1)
        crps_daily = crps_df.groupby(["_date", "League"]).agg(
            CRPS=("CRPS", "mean"),
            Count=("CRPS", "count"),
        ).reset_index()

        fig_crps = px.line(crps_daily, x="_date", y="CRPS", color="League",
                           labels={"_date": "Date", "CRPS": "Mean CRPS (lower=better)"})
        fig_crps.update_layout(height=400)
        st.plotly_chart(fig_crps, use_container_width=True)

# --- Prediction Interval Coverage ---
if has_crps:
    cov_df = pred_df.dropna(subset=["Dist", "Actual"])
    if len(cov_df) >= 20:
        st.subheader("Prediction Interval Coverage")
        coverage = compute_coverage(cov_df, levels=(0.5, 0.8, 0.9))

        cov_display = pd.DataFrame([
            {"Nominal Level": f"{int(level*100)}%",
             "Actual Coverage": f"{cov:.1%}",
             "Status": "Good" if abs(cov - level) < 0.05 else
                       ("Overconfident" if cov < level else "Underconfident")}
            for level, cov in coverage.items()
            if not np.isnan(cov)
        ])
        if not cov_display.empty:
            st.dataframe(cov_display, use_container_width=True, hide_index=True)

            # Coverage by league
            cov_rows = []
            for league, lgrp in cov_df.groupby("League"):
                if len(lgrp) < 10:
                    continue
                lcov = compute_coverage(lgrp, levels=(0.5, 0.8, 0.9))
                for level, cov_val in lcov.items():
                    if not np.isnan(cov_val):
                        cov_rows.append({
                            "League": league,
                            "Nominal": level,
                            "Actual": cov_val,
                        })
            if cov_rows:
                cov_league_df = pd.DataFrame(cov_rows)
                fig_cov = px.bar(cov_league_df, x="League", y="Actual",
                                 color="Nominal", barmode="group",
                                 labels={"Actual": "Actual Coverage"})
                for nom in [0.5, 0.8, 0.9]:
                    fig_cov.add_hline(y=nom, line_dash="dot", line_color="gray",
                                      annotation_text=f"{int(nom*100)}% target")
                fig_cov.update_layout(height=400)
                st.plotly_chart(fig_cov, use_container_width=True)
