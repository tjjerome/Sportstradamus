"""Chart builders for the Receipts calibration, cumulative-units and rolling-accuracy panels."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sportstradamus.dashboard import theme

# Dot-size formula for bin volume: floor of 5px plus 1px per ~4 offers, capped at a
# volume of 200 so one giant bin can't swamp the plot. Mirrors the idiom in
# lab_diagnostics_charts.reliability_diagram, tuned for calibration_summary's typical
# per-bin counts (tens to low hundreds, not the thousands that module's Count sees).
_MARKER_SIZE_CAP = 200
_MARKER_SIZE_DIVISOR = 4
_MARKER_SIZE_FLOOR = 5

_SERIES = {
    False: {"name": "Standard line", "color": theme.SEQUENTIAL_COLORS[4]},
    True: {"name": "Alt line / ladder", "color": theme.GOLD},
}


def _marker_size(n: pd.Series) -> pd.Series:
    return n.clip(upper=_MARKER_SIZE_CAP) / _MARKER_SIZE_DIVISOR + _MARKER_SIZE_FLOOR


def reliability_diagram(cal_summary: pd.DataFrame) -> go.Figure:
    """Two-series reliability diagram of posted legs: standard-line bins vs. alt-line/ladder bins.

    ``cal_summary`` is the nightly ``realized.calibration_summary`` frame; only its
    ``posted`` cohort is drawn. The recommended cohort is a subset of those legs, and its
    read against its hit rate is the hero's to show.

    Gold marks the alt/ladder series identity, not a data value — the point's own
    position (predicted vs. actual) carries the calibration signal; gold only says
    which population a dot belongs to, the same role as a legend swatch (DESIGN.md
    §2's second sanctioned gold-on-chart exception).
    """
    posted = cal_summary.loc[cal_summary["Cohort"] == "posted"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.40, 1.0],
            y=[0.40, 1.0],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Perfect calibration",
            showlegend=True,
        )
    )
    for alt_line, spec in _SERIES.items():
        split = posted.loc[posted["Alt Line"] == alt_line]
        if split.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=split["Predicted"],
                y=split["Actual"],
                mode="markers",
                name=spec["name"],
                marker={"size": _marker_size(split["N"]), "color": spec["color"]},
                text=[f"n={n}" for n in split["N"]],
                hovertemplate="Predicted: %{x:.3f}<br>Actual: %{y:.3f}<br>%{text}",
            )
        )
    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis={"title": "Actual Hit Rate", "range": [0.40, 1.0]},
        height=380,
    )
    return fig


# Cumulative-units chart height (px) — unchanged from receipts.py's prior px.area default.
_PROFIT_CHART_HEIGHT = 360


def cumulative_profit_chart(daily_profit: pd.DataFrame, wm: dict) -> go.Figure:
    """Cumulative-units area chart with month/signed-unit axes + a worst-drawdown marker.

    ``daily_profit`` is the surface's own per-day ``(_date, Profit, Cumulative Profit)``
    frame. ``wm`` is a ``worst_month()`` result (``{}`` when nothing resolved yet, in
    which case no marker is drawn). The marker lands on the *last* daily row that falls
    in ``wm["month"]`` — the running total once that month finished, the real-data
    analogue of the mockup's monthly-vertex illustration.
    """
    fig = px.area(
        daily_profit,
        x="_date",
        y="Cumulative Profit",
        labels={"_date": "Date", "Cumulative Profit": "Units"},
    )
    fig.update_xaxes(dtick="M1", tickformat="%b '%y")
    fig.update_yaxes(tickformat="+d")
    fig.update_layout(height=_PROFIT_CHART_HEIGHT)

    if wm:
        month_key = pd.to_datetime(daily_profit["_date"]).dt.strftime("%Y-%m")
        month_end = daily_profit.loc[month_key == wm["month"]].iloc[-1]
        month_abbrev = pd.to_datetime(wm["month"]).strftime("%b")
        fig.add_trace(
            go.Scatter(
                x=[month_end["_date"]],
                y=[month_end["Cumulative Profit"]],
                mode="markers+text",
                marker={"size": 8, "color": theme.RED},
                text=[f"{wm['units']:+.0f}u {month_abbrev}"],
                textposition="bottom center",
                textfont={"color": theme.RED},
                showlegend=False,
                hovertemplate=f"Worst month: {wm['month']}<br>{wm['units']:+.1f}u",
            )
        )
    return fig


# Trailing rows (days with settled legs) per league behind the "Rolling 30-Day" hit rate.
_ROLLING_DAYS = 30


def rolling_accuracy_chart(offers: pd.DataFrame, breakeven: float) -> go.Figure:
    """Rolling hit rate per league, with a dashed reference line at ``breakeven``.

    ``offers`` are ``realized.settled_offers`` rows carrying the surface's ``_date``; each
    league's line is its hits over its legs across its trailing ``_ROLLING_DAYS`` dated rows.
    ``breakeven`` is the hit rate the posted payouts need (``cohort_summary``'s
    ``breakeven_rate``), the line a hit rate has to clear to make money.
    """
    daily = (
        offers.groupby(["League", "_date"])
        .agg(Hits=("Hit", "sum"), Bets=("Hit", "size"))
        .reset_index()
    )
    fig = go.Figure()
    for league, rows in daily.groupby("League"):
        hits = rows["Hits"].rolling(_ROLLING_DAYS, min_periods=1).sum()
        bets = rows["Bets"].rolling(_ROLLING_DAYS, min_periods=1).sum()
        fig.add_trace(go.Scatter(x=rows["_date"], y=hits / bets, mode="lines", name=league))
    fig.add_hline(y=breakeven, line_dash="dash", line_color=theme.GRAY, annotation_text="breakeven")
    fig.update_layout(yaxis_title="Accuracy", xaxis_title="Date", height=400)
    return fig
