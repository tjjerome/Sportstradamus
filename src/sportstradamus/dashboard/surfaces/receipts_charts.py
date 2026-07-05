"""Chart builders for the Receipts calibration panel."""

import pandas as pd
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
    """Two-series reliability diagram: standard-line bins vs. alt-line/ladder bins.

    Gold marks the alt/ladder series identity, not a data value — the point's own
    position (predicted vs. actual) carries the calibration signal; gold only says
    which population a dot belongs to, the same role as a legend swatch (DESIGN.md
    §2's second sanctioned gold-on-chart exception).
    """
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
        split = cal_summary.loc[cal_summary["Alt Line"] == alt_line]
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
