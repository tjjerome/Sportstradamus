"""Deep-dive History/Model chart pins: colors, line overlays, gold tag.

``distribution_chart`` overlays three references on the predictive curve — the app
betting line (white dashed rule, unchanged), the consensus market line (a distinct
solid rule), and the model projection (a dot + numeric label, not a rule). These pins
assert the altair layer spec carries each without re-shading the over/under split, and
that both charts encode over/under with the theme green/red (P8 Phase C rev-3 recolor).
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.dashboard.components.deep_dive_charts import (
    distribution_chart,
    distribution_frame,
    history_chart,
    resolve_std,
)
from sportstradamus.dashboard.theme import GOLD, GREEN, RED

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
    return [
        lyr
        for lyr in layers
        if isinstance(lyr.get("mark"), dict) and lyr["mark"].get("type") == "rule"
    ]


def _marks_of_type(layers: list[dict], *types: str) -> list[dict]:
    return [
        lyr
        for lyr in layers
        if isinstance(lyr.get("mark"), dict) and lyr["mark"].get("type") in types
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


def _side_color_range(chart) -> list[str]:
    """The green/red range of the Side color scale, wherever it sits in the layer tree."""
    spec = chart.to_dict()
    stack = list(spec.get("layer", [spec]))
    while stack:
        node = stack.pop()
        if "layer" in node:
            stack.extend(node["layer"])
        scale = node.get("encoding", {}).get("color", {}).get("scale")
        if isinstance(scale, dict) and "range" in scale:
            return scale["range"]
    raise AssertionError("no Side color scale found")


def _color_legend_shown(chart) -> bool:
    spec = chart.to_dict()
    stack = list(spec.get("layer", [spec]))
    while stack:
        node = stack.pop()
        if "layer" in node:
            stack.extend(node["layer"])
        color = node.get("encoding", {}).get("color", {})
        if "scale" in color and color.get("legend") is not None:
            return True
    return False


def _history_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Label": ["@BOS", "vs NYK", "@PHI"],
            "StatValue": [22.0, 18.0, 26.0],
            "Hit": [True, False, True],
        }
    )


def test_history_bars_green_over_red_under():
    chart = history_chart(_history_df(), 20.0)
    # Over reads green, under red — keyed on Hit (StatValue >= line), theme tokens.
    assert _side_color_range(chart) == [GREEN, RED]


def test_history_gold_line_tag_present():
    chart = history_chart(_history_df(), 23.5)
    texts = _marks_of_type(_layers(chart), "text")
    assert texts and texts[0]["mark"].get("color") == GOLD
    # The tag reads "line {x}" with the app line value, %.10g-formatted (no trailing zero).
    tag_row = texts[0]["encoding"]["text"]
    assert tag_row["field"] == "tag"


def test_history_app_line_still_white_dashed():
    rules = _rules(_layers(history_chart(_history_df(), 20.0)))
    assert len(rules) == 1
    assert rules[0]["mark"]["color"] == _APP_LINE and rules[0]["mark"].get("strokeDash")


def test_distribution_side_recolored_green_red_continuous():
    df_pdf, y_title, is_cont = distribution_frame("SkewNormal", 20.0, 5.0, {}, 18.5)
    chart = distribution_chart(df_pdf, is_cont, 18.5, "PTS", y_title)
    assert _side_color_range(chart) == [GREEN, RED]


def test_distribution_side_recolored_green_red_discrete():
    df_pdf, y_title, is_cont = distribution_frame("NegBin", 6.0, 3.0, {"r": 4.0}, 5.5)
    chart = distribution_chart(df_pdf, is_cont, 5.5, "AST", y_title)
    assert _side_color_range(chart) == [GREEN, RED]


def test_distribution_drops_redundant_over_under_legend():
    # The fixed green-above/red-below split makes the color legend chart junk (rev 3).
    df_pdf, y_title, is_cont = distribution_frame("SkewNormal", 20.0, 5.0, {}, 18.5)
    chart = distribution_chart(df_pdf, is_cont, 18.5, "PTS", y_title, projection=20.0)
    assert not _color_legend_shown(chart)


def _recal_blob():
    """An isotonic PIT-recal map fit on a deliberately non-uniform PIT (a real g≠identity)."""
    from scipy import stats

    from sportstradamus.training import posthoc

    pit = stats.beta.rvs(2.0, 2.0, size=4000, random_state=1)
    blob, _ = posthoc.select_pit_recal(pit)
    return blob


def test_cdf_recal_curve_over_mass_matches_served_prob():
    # For a cdf_recal_isotonic cell the deep-dive must draw g∘F — the distribution the
    # model actually serves — so the shaded over-mass equals the served P(over) =
    # 1 − g(F(line)). The raw parametric curve (what the old code drew) does not.
    from scipy import stats
    from scipy.integrate import trapezoid

    from sportstradamus.training.posthoc import apply_cdf_recal

    ev, sigma, skew, line = 25.0, 32.0, 4.0, 30.0
    params = {"sigma": sigma, "skew_alpha": skew}
    blob = _recal_blob()

    served_over = 1.0 - float(
        apply_cdf_recal(blob, [float(stats.skewnorm.cdf(line, skew, loc=ev, scale=sigma))])[0]
    )

    df_recal, _, _ = distribution_frame("SkewNormal", ev, sigma, params, line, blob)
    over = df_recal[df_recal["x"] >= line]
    recal_over_mass = trapezoid(over["P"], over["x"]) / trapezoid(df_recal["P"], df_recal["x"])
    assert recal_over_mass == pytest.approx(served_over, abs=0.01)

    # Sanity: the recal actually moved the curve off the raw parametric shape.
    df_raw, _, _ = distribution_frame("SkewNormal", ev, sigma, params, line)
    raw_over = df_raw[df_raw["x"] >= line]
    raw_over_mass = trapezoid(raw_over["P"], raw_over["x"]) / trapezoid(df_raw["P"], df_raw["x"])
    assert abs(raw_over_mass - recal_over_mass) > 0.02


def test_cdf_recal_absent_leaves_curve_identical():
    # No recal map (the 99% case) must render the exact legacy parametric curve.
    params = {"sigma": 5.0, "skew_alpha": 2.0}
    base, _, _ = distribution_frame("SkewNormal", 20.0, 5.0, params, 18.5)
    with_none, _, _ = distribution_frame("SkewNormal", 20.0, 5.0, params, 18.5, None)
    assert (base["P"].to_numpy() == with_none["P"].to_numpy()).all()
