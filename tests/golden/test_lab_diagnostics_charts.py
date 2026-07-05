"""Golden pins for Lab Diagnostics' pure chart builders (P8 Task B4).

Phase A2 already retoned every figure onto design tokens (no RdYlGn, no off-token
hexes); this task's gap is real titles/legends per spec §3.6 — ``bias_bar`` colored
its bars by bucket but suppressed the legend outright, so a reader had no way to know
what green/orange/red meant. The other seven builders already carry axis titles via
``labels=``/explicit ``xaxis_title``/``yaxis`` and legends via ``color=``/named traces;
those pins here are a compliance spot-check, not new behavior.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.surfaces.lab_diagnostics_charts import (
    bias_bar,
    bss_bar,
    coverage_bar,
    crps_line,
    ev_divergence_bar,
    murphy_decomp_bar,
    reliability_diagram,
    sharpness_histogram,
)

_BIAS_DF = pd.DataFrame(
    {
        "League": ["NBA", "NBA", "NBA"],
        "Market": ["PTS", "AST", "REB"],
        "Label": ["NBA - PTS", "NBA - AST", "NBA - REB"],
        "Balance": [0.01, 0.05, 0.09],
        "Color": ["green", "orange", "red"],
    }
)


def test_bias_bar_shows_a_legend_naming_the_three_buckets():
    """Bars are colored by calibration bucket; a legend is the only way to read that."""
    fig = bias_bar(_BIAS_DF)
    assert fig.layout.showlegend is not False
    trace_names = {t.name for t in fig.data}
    assert len(trace_names) == 3
    # Raw hex-name strings ("green"/"orange"/"red") are not a legend a reader can use —
    # the fix maps them to descriptive category labels.
    assert trace_names.isdisjoint({"green", "orange", "red"})


def test_bias_bar_legend_colors_stay_on_token():
    fig = bias_bar(_BIAS_DF)
    colors = {t.marker.color for t in fig.data}
    assert colors <= {theme.GREEN, theme.ORANGE, theme.RED}


def test_bias_bar_has_axis_title():
    fig = bias_bar(_BIAS_DF)
    assert fig.layout.xaxis.title.text


_CAL_STATS = pd.DataFrame(
    {
        "Predicted": [0.55, 0.65, 0.75],
        "Actual": [0.52, 0.68, 0.71],
        "Count": [40, 55, 30],
    }
)


def test_reliability_diagram_model_curve_never_gold():
    """DESIGN §3.1: gold is never a plotted data mark. The Model curve sets no explicit
    color, so it inherits the registered token template's colorway (primary blue first)
    rather than the spec background text's (self-contradicting) "gold curve" suggestion.
    """
    fig = reliability_diagram(_CAL_STATS)
    model_trace = next(t for t in fig.data if t.name == "Model")
    line_color = (model_trace.line.color or "") if model_trace.line else ""
    marker_color = model_trace.marker.color if model_trace.marker else None
    assert line_color != theme.GOLD
    assert marker_color != theme.GOLD


def test_reliability_diagram_has_axis_titles():
    fig = reliability_diagram(_CAL_STATS)
    assert fig.layout.xaxis.title.text
    assert fig.layout.yaxis.title.text


_CRPS_DAILY = pd.DataFrame(
    {
        "_date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-01", "2026-06-02"]),
        "League": ["NBA", "NBA", "WNBA", "WNBA"],
        "CRPS": [3.2, 4.8, 5.9, 3.7],
        "Count": [10, 12, 8, 9],
    }
)


def test_crps_line_plots_real_mean_crps_values_not_a_placeholder():
    fig = crps_line(_CRPS_DAILY)
    plotted = sorted(float(v) for trace in fig.data for v in trace.y)
    assert plotted == sorted(_CRPS_DAILY["CRPS"].tolist())


def test_crps_line_has_axis_titles_and_legend():
    fig = crps_line(_CRPS_DAILY)
    assert fig.layout.xaxis.title.text
    assert fig.layout.yaxis.title.text
    trace_names = {t.name for t in fig.data}
    assert trace_names == {"NBA", "WNBA"}


_SHARP_DF = pd.DataFrame(
    {
        "Market": ["PTS", "PTS", "AST", "AST"],
        "Win Prob": [0.55, 0.60, 0.52, 0.58],
    }
)


def test_sharpness_histogram_has_axis_title():
    fig = sharpness_histogram(_SHARP_DF, "Win Prob")
    assert fig.layout.xaxis.title.text


_DIV_STATS = pd.DataFrame(
    {
        "Div_Bucket": ["(0, 0.1]", "(0.1, 0.2]"],
        "Accuracy": [0.55, 0.62],
        "Count": [30, 25],
    }
)


def test_ev_divergence_bar_has_axis_title():
    fig = ev_divergence_bar(_DIV_STATS)
    assert fig.layout.xaxis.title.text


_BSS_DF = pd.DataFrame(
    {
        "League": ["NBA", "NBA"],
        "Market": ["PTS", "AST"],
        "Label": ["NBA - PTS", "NBA - AST"],
        "BSS": [-0.02, 0.03],
    }
)


def test_bss_bar_has_axis_title_and_diverging_colors_never_gold():
    fig = bss_bar(_BSS_DF)
    assert fig.layout.xaxis.title.text
    hexes = {stop[1] for stop in fig.layout.coloraxis.colorscale}
    assert hexes <= set(theme.DIVERGING_COLORS)
    assert theme.GOLD not in hexes


_DECOMP_DF = pd.DataFrame(
    {
        "League": ["NBA", "NBA"],
        "Market": ["PTS", "AST"],
        "Label": ["NBA - PTS", "NBA - AST"],
        "Reliability": [0.01, 0.02],
        "Resolution": [0.05, 0.03],
        "Uncertainty": [0.25, 0.24],
        "Brier": [0.21, 0.23],
    }
)


def test_murphy_decomp_bar_has_axis_title_and_legend():
    fig = murphy_decomp_bar(_DECOMP_DF)
    assert fig.layout.xaxis.title.text
    trace_names = {t.name for t in fig.data}
    assert len(trace_names) == 2


_COV_DF = pd.DataFrame(
    {
        "League": ["NBA", "NBA", "WNBA", "WNBA"],
        "Nominal": [0.8, 0.9, 0.8, 0.9],
        "Actual": [0.78, 0.87, 0.82, 0.91],
    }
)


def test_coverage_bar_groups_every_nominal_level_per_league():
    """``color="Nominal"`` on a raw float column makes ``px.bar`` treat it as continuous
    -> a single trace, one bar per League, and the ``barmode="group"`` has nothing to
    group (the second nominal level per league silently overplots the first). Casting
    to a discrete label fixes both the grouping and gives the legend real names.
    """
    fig = coverage_bar(_COV_DF)
    total_bars = sum(len(trace.x) for trace in fig.data)
    assert total_bars == len(_COV_DF)


def test_coverage_bar_has_axis_title_and_legend():
    fig = coverage_bar(_COV_DF)
    assert fig.layout.yaxis.title.text
    trace_names = {t.name for t in fig.data}
    assert trace_names == {"0.8", "0.9"}
