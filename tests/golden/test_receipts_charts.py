"""Golden pins for the Receipts surface's pure chart builders (P8 Task C7).

``cumulative_profit_chart`` is the drawdown-annotated cumulative-units area chart extracted
out of receipts.py so the plan's "drawdown annotation present" exit criterion is a real CI
assertion, not a manual-verify note — the same extraction precedent as ``reliability_diagram``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.surfaces.receipts_charts import cumulative_profit_chart


def _daily_profit() -> pd.DataFrame:
    # Three months of daily rows; Feb nets the worst month (-41 units) after a strong Jan.
    dates = pd.to_datetime(
        ["2026-01-10", "2026-01-20", "2026-02-05", "2026-02-15", "2026-03-01", "2026-03-10"]
    ).date
    profit = [20.0, 15.0, -25.0, -16.0, 10.0, 12.0]
    df = pd.DataFrame({"_date": dates, "Profit": profit})
    df["Cumulative Profit"] = df["Profit"].cumsum()
    return df


_WM = {"month": "2026-02", "units": -41.0, "n": 612, "win_pct": 0.45}


def test_marker_lands_on_the_worst_months_final_row():
    """The month-end (last-matching-row) point is where the running total sat once
    that bad month finished — the daily-data analogue of the mockup's monthly vertex.
    """
    daily_profit = _daily_profit()
    fig = cumulative_profit_chart(daily_profit, _WM)
    marker = next(t for t in fig.data if t.mode and "markers" in t.mode)
    feb_rows = daily_profit.loc[daily_profit["_date"].astype(str).str.startswith("2026-02")]
    expected_point = feb_rows.iloc[-1]
    assert marker.x[0] == expected_point["_date"]
    assert marker.y[0] == pytest.approx(expected_point["Cumulative Profit"])


def test_marker_is_red_not_gold():
    """DESIGN §3.1: gold is a chart data mark only for the two sanctioned exceptions
    (neither is this). The drawdown marker is a verdict flag, not a series identity —
    the plan spells out red explicitly.
    """
    fig = cumulative_profit_chart(_daily_profit(), _WM)
    marker = next(t for t in fig.data if t.mode and "markers" in t.mode)
    assert marker.marker.color == theme.RED
    assert marker.marker.color != theme.GOLD


def test_marker_label_shows_units_and_abbreviated_month():
    fig = cumulative_profit_chart(_daily_profit(), _WM)
    marker = next(t for t in fig.data if t.mode and "markers" in t.mode)
    label = marker.text[0] if marker.text else str(marker.hovertext)
    assert "-41" in label
    assert "Feb" in label


def test_empty_worst_month_produces_no_marker_trace():
    fig = cumulative_profit_chart(_daily_profit(), {})
    assert not any(t.mode and "markers" in t.mode for t in fig.data)


def test_axes_carry_signed_units_and_month_tick_format():
    fig = cumulative_profit_chart(_daily_profit(), _WM)
    assert fig.layout.yaxis.tickformat == "+d"
    assert fig.layout.xaxis.dtick == "M1"
    assert "%b" in fig.layout.xaxis.tickformat
    assert "%y" in fig.layout.xaxis.tickformat
