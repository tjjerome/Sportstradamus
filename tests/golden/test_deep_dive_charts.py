"""Deep-dive Model-tab distribution chart: the three line/point labels.

``distribution_chart`` overlays three references on the predictive curve — the app
betting line (white dashed rule, unchanged), the consensus market line (a distinct
solid rule), and the model projection (a dot + numeric label, not a rule). These pins
assert the altair layer spec carries each without re-shading the over/under split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.dashboard.components.deep_dive_charts import (
    distribution_chart,
    distribution_frame,
    resolve_std,
)
from sportstradamus.dashboard.theme import GOLD

_APP_LINE = "#FFFFFF"


def test_resolve_std_falls_back_when_projection_std_missing():
    # Model std present and positive → used as-is.
    assert resolve_std(2.0, 9.5, 0.55) == 2.0
    # NaN Projection STD (book-fallback offers carry none) → CV-implied spread, NOT NaN.
    # Regression: the old `Projection STD or ev*0.3` returned NaN because NaN is truthy,
    # so the x-range was all-NaN and the curve rendered invisible.
    assert resolve_std(float("nan"), 9.5, 0.55) == pytest.approx(9.5 * 0.55)
    # Non-positive std also falls back to the CV spread.
    assert resolve_std(0.0, 9.5, 0.55) == pytest.approx(9.5 * 0.55)
    # CV also missing → last-resort 0.3 fraction of the mean.
    assert resolve_std(float("nan"), 10.0, float("nan")) == pytest.approx(3.0)


def test_distribution_frame_renders_with_cv_fallback_std():
    # A book-fallback SkewNormal row (no sigma/skew/Projection STD) still produces a
    # finite, non-flat curve once std falls back to the CV-implied spread.
    std = resolve_std(float("nan"), 9.5, 0.55)
    df_pdf, _, is_cont = distribution_frame("SkewNormal", 9.5, std, {}, 8.5)
    assert is_cont
    assert df_pdf["x"].notna().all()
    assert df_pdf["P"].notna().all()
    assert df_pdf["P"].max() > 0


def _layers(chart) -> list[dict]:
    return chart.to_dict()["layer"]


def _rules(layers: list[dict]) -> list[dict]:
    return [lyr for lyr in layers if isinstance(lyr.get("mark"), dict) and lyr["mark"].get("type") == "rule"]


def _marks_of_type(layers: list[dict], *types: str) -> list[dict]:
    return [
        lyr for lyr in layers if isinstance(lyr.get("mark"), dict) and lyr["mark"].get("type") in types
    ]


def test_three_labels_present_continuous():
    df_pdf, y_title, is_cont = distribution_frame("SkewNormal", 20.0, 5.0, {}, 18.5)
    chart = distribution_chart(
        df_pdf, is_cont, 18.5, "PTS", y_title, projection=20.0, consensus_line=19.0
    )
    layers = _layers(chart)

    rules = _rules(layers)
    app = [r for r in rules if r["mark"].get("strokeDash")]
    consensus = [r for r in rules if not r["mark"].get("strokeDash")]

    # App betting line: still the white dashed rule, untouched.
    assert app and app[0]["mark"]["color"] == _APP_LINE
    # Consensus market line: a distinct solid rule, not the app-line white.
    assert consensus and consensus[0]["mark"]["color"] != _APP_LINE
    # Model projection: a dot in the brand accent + a numeric text label, no rule.
    points = _marks_of_type(layers, "point", "circle")
    texts = _marks_of_type(layers, "text")
    assert points and points[0]["mark"].get("color") == GOLD
    assert texts


def test_app_line_unchanged_without_overlays():
    # No projection / consensus → exactly the legacy base curve + the one app-line rule.
    df_pdf, y_title, is_cont = distribution_frame("SkewNormal", 20.0, 5.0, {}, 18.5)
    chart = distribution_chart(df_pdf, is_cont, 18.5, "PTS", y_title)
    rules = _rules(_layers(chart))
    assert len(rules) == 1
    assert rules[0]["mark"]["color"] == _APP_LINE and rules[0]["mark"].get("strokeDash")


def test_over_under_shading_not_resplit_by_consensus():
    # The consensus rule must not introduce a second Side split: the area/line color
    # encoding still keys on the original Over/Under field only.
    df_pdf, y_title, is_cont = distribution_frame("SkewNormal", 20.0, 5.0, {}, 18.5)
    chart = distribution_chart(
        df_pdf, is_cont, 18.5, "PTS", y_title, projection=20.0, consensus_line=22.0
    )
    consensus = [r for r in _rules(_layers(chart)) if not r["mark"].get("strokeDash")]
    assert consensus and "color" not in consensus[0].get("encoding", {})


def test_discrete_carries_overlays():
    df_pdf, y_title, is_cont = distribution_frame("NegBin", 6.0, 3.0, {"r": 4.0}, 5.5)
    chart = distribution_chart(
        df_pdf, is_cont, 5.5, "AST", y_title, projection=6.0, consensus_line=5.0
    )
    layers = _layers(chart)
    assert len(_rules(layers)) == 2  # app line + consensus
    assert _marks_of_type(layers, "point", "circle")


def test_nan_overlays_skipped():
    df_pdf, y_title, is_cont = distribution_frame("SkewNormal", 20.0, 5.0, {}, 18.5)
    chart = distribution_chart(
        df_pdf, is_cont, 18.5, "PTS", y_title, projection=float("nan"), consensus_line=None
    )
    # NaN projection + missing consensus collapse back to just the app line.
    assert len(_rules(_layers(chart))) == 1
    assert not _marks_of_type(_layers(chart), "point", "circle")
