"""Chart builders for the Market Diagnostics surface."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Bias thresholds for Over/Under balance: < 3% is well-calibrated, 3–7% is
# marginal, > 7% is a notable prediction bias worth investigating.
_BIAS_OK = 0.03
_BIAS_WARN = 0.07


def bias_bar(bias_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of Over/Under bias by league-market."""
    fig = px.bar(
        bias_df,
        x="Balance",
        y="Label",
        orientation="h",
        color="Color",
        color_discrete_map={"green": "#2ecc71", "orange": "#f39c12", "red": "#e74c3c"},
        labels={"Balance": "Over Bias (Predicted − Actual)", "Label": ""},
    )
    fig.update_layout(showlegend=False, height=max(300, len(bias_df) * 22))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig


def sharpness_histogram(sharp_subset: pd.DataFrame, prob_col: str) -> go.Figure:
    """Faceted histogram of predicted probabilities per market."""
    fig = px.histogram(
        sharp_subset,
        x=prob_col,
        facet_col="Market",
        facet_col_wrap=4,
        nbins=20,
        labels={prob_col: "Win Prob"},
        title="Distribution of predicted probabilities per market",
    )
    fig.update_layout(height=600)
    return fig


def ev_divergence_bar(div_stats: pd.DataFrame) -> go.Figure:
    """Bar chart of accuracy bucketed by model-book EV divergence."""
    fig = px.bar(
        div_stats,
        x="Div_Bucket",
        y="Accuracy",
        text="Count",
        labels={"Div_Bucket": "|Model EV - Line| / Line"},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    return fig


def reliability_diagram(cal_stats: pd.DataFrame) -> go.Figure:
    """Calibration reliability diagram with sample-count histogram on y2."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.5, 1.0],
            y=[0.5, 1.0],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Perfect calibration",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cal_stats["Predicted"],
            y=cal_stats["Actual"],
            mode="lines+markers",
            name="Model",
            marker={"size": cal_stats["Count"].clip(upper=500) / 20 + 5},
            text=[f"n={c}" for c in cal_stats["Count"]],
            hovertemplate="Predicted: %{x:.3f}<br>Actual: %{y:.3f}<br>%{text}",
        )
    )
    fig.add_trace(
        go.Bar(
            x=cal_stats["Predicted"],
            y=cal_stats["Count"],
            name="Sample count",
            yaxis="y2",
            opacity=0.3,
        )
    )
    fig.update_layout(
        yaxis={"title": "Actual Hit Rate", "range": [0.4, 1.0]},
        yaxis2={"title": "Count", "overlaying": "y", "side": "right"},
        xaxis_title="Predicted Probability",
        height=450,
    )
    return fig


def bss_bar(bss_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of Brier Skill Scores by league-market."""
    fig = px.bar(
        bss_df,
        x="BSS",
        y="Label",
        orientation="h",
        color="BSS",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={"BSS": "Brier Skill Score", "Label": ""},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="No skill")
    fig.update_layout(height=max(300, len(bss_df) * 22))
    return fig


def murphy_decomp_bar(decomp_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of Murphy Brier decomposition."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=decomp_df["Label"],
            x=decomp_df["Reliability"],
            name="Reliability (lower=better)",
            orientation="h",
            marker_color="#e74c3c",
        )
    )
    fig.add_trace(
        go.Bar(
            y=decomp_df["Label"],
            x=-decomp_df["Resolution"],
            name="Resolution (higher=better)",
            orientation="h",
            marker_color="#2ecc71",
        )
    )
    fig.update_layout(
        barmode="relative",
        xaxis_title="Contribution to Brier Score",
        height=max(300, len(decomp_df) * 22),
    )
    return fig


def crps_line(crps_daily: pd.DataFrame) -> go.Figure:
    """Line chart of mean CRPS over time by league."""
    fig = px.line(
        crps_daily,
        x="_date",
        y="CRPS",
        color="League",
        labels={"_date": "Date", "CRPS": "Mean CRPS (lower=better)"},
    )
    fig.update_layout(height=400)
    return fig


def coverage_bar(cov_league_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of prediction interval coverage by league and nominal level."""
    fig = px.bar(
        cov_league_df,
        x="League",
        y="Actual",
        color="Nominal",
        barmode="group",
        labels={"Actual": "Actual Coverage"},
    )
    for nom in [0.5, 0.8, 0.9]:
        fig.add_hline(
            y=nom,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"{int(nom * 100)}% target",
        )
    fig.update_layout(height=400)
    return fig


def bias_color(b: float) -> str:
    ab = abs(b)
    if ab < _BIAS_OK:
        return "green"
    if ab < _BIAS_WARN:
        return "orange"
    return "red"
