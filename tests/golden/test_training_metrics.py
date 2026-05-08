"""Golden tests for training/report.py write_model_stats + get_market_calibration.

Covers Phase 3 §4.c/§4.f/§4.g: the migrated raw-metric schema, the pinned
``row_kind="book_baseline"`` row, and the thin getter Kelly consumes.
"""

from __future__ import annotations

import math
import sys
from unittest import mock

import pandas as pd
import pytest

from sportstradamus.training.report import (
    _RAW_METRIC_KEYS,
    get_market_calibration,
    write_model_stats,
)

report_module = sys.modules["sportstradamus.training.report"]


def _model_metrics(brier: float = 0.20) -> dict:
    return {
        "brier_score": brier,
        "log_loss": 0.65,
        "roc_auc": 0.62,
        "expected_calibration_error": 0.02,
        "accuracy": 0.58,
        "precision_over": 0.60,
        "precision_under": 0.55,
        "predicted_over_rate": 0.51,
        "empirical_over_rate": 0.49,
        "prediction_std": 0.08,
        "nll": 0.65,
    }


def _make_model(*, with_book: bool = True, brier: float = 0.197) -> dict:
    book = _model_metrics(brier=0.218) if with_book else None
    bss = 1 - (brier / max(book["brier_score"], 1e-9)) if book is not None else float("nan")
    ks = max(0.0, min(1.0, bss)) if math.isfinite(bss) else float("nan")
    return {
        "distribution": "Gamma",
        "cv": 0.5,
        "std": 1.0,
        "hist_gate": 0.01,
        "stats": {
            "Accuracy": [0.50, 0.55, 0.58],
            "Over Prec": [0.50, 0.55, 0.60],
            "Under Prec": [0.50, 0.55, 0.55],
            "Over%": [0.49, 0.50, 0.51],
            "Sharpness": [0.04, 0.06, 0.08],
            "NLL": [0.70, 0.67, 0.65],
        },
        "metrics": {
            "model": _model_metrics(brier=brier),
            "book_baseline": book,
            "brier_skill_score": float(bss),
            "kelly_shrinkage": float(ks),
        },
        "diagnostics": {
            "model_weight": 0.30,
            "shape_label": "alpha",
            "start_shape": 1.0,
            "model_shape": 1.1,
            "empirical_shape": 1.0,
            "ev_minus_line": 0.05,
            "median_ev_diff": 0.04,
            "frac_ev_gt_line": 0.55,
            "over_pct_ev_gt": 0.60,
            "over_pct_ev_lt": 0.40,
            "cf_over_pct": 0.50,
            "dispersion_cal": 1.0,
            "marginal_shape": 1.0,
            "shape_ceiling": 2.0,
            "model_ev": 220.0,
            "mean_line": 215.0,
            "result_mean": 218.0,
        },
        "params": {
            "opt_rounds": 100,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        },
    }


@pytest.fixture
def patched_path(tmp_path):
    target = tmp_path / "model_stats.parquet"
    with mock.patch.object(report_module, "MODEL_STATS_PATH", target):
        yield target


def test_write_emits_book_baseline_and_model_rows(patched_path):
    league_models = {"NFL": {"player_pass_yds": _make_model()}}
    write_model_stats(
        league_models, {"NFL": {"player_pass_yds": 0.5}}, {"NFL": {"player_pass_yds": 1.0}}
    )

    df = pd.read_parquet(patched_path)
    assert set(df["row_kind"]) == {"book_baseline", "model"}
    book = df[df["row_kind"] == "book_baseline"]
    assert len(book) == 1
    assert book.iloc[0]["brier_score"] == pytest.approx(0.218)
    # Skill score on the baseline row is pinned at 0 — the book matches itself.
    assert book.iloc[0]["brier_skill_score"] == 0.0
    assert book.iloc[0]["kelly_shrinkage"] == 0.0

    model_rows = df[df["row_kind"] == "model"]
    assert set(model_rows["metric_row"]) == {"raw", "corrected", "calibrated"}
    cal = model_rows[model_rows["metric_row"] == "calibrated"].iloc[0]
    # Calibrated row carries the validation raw metrics and the skill score.
    assert cal["brier_score"] == pytest.approx(0.197)
    assert cal["brier_skill_score"] == pytest.approx(1 - 0.197 / 0.218)
    assert cal["kelly_shrinkage"] == pytest.approx(1 - 0.197 / 0.218)
    # Raw/corrected rows do not carry the validation raw metrics.
    raw = model_rows[model_rows["metric_row"] == "raw"].iloc[0]
    assert math.isnan(raw["brier_score"])
    assert math.isnan(raw["brier_skill_score"])


def test_useless_model_skill_score_zero(patched_path):
    # Same Brier as book → BSS = 0, kelly_shrinkage = 0.
    league_models = {"NFL": {"m": _make_model(brier=0.218)}}
    write_model_stats(league_models, {}, {})
    df = pd.read_parquet(patched_path)
    cal = df[(df["row_kind"] == "model") & (df["metric_row"] == "calibrated")].iloc[0]
    assert cal["brier_skill_score"] == pytest.approx(0.0, abs=1e-9)
    assert cal["kelly_shrinkage"] == pytest.approx(0.0, abs=1e-9)


def test_missing_book_baseline_skill_is_nan(patched_path):
    league_models = {"NFL": {"m": _make_model(with_book=False)}}
    write_model_stats(league_models, {}, {})
    df = pd.read_parquet(patched_path)
    book = df[df["row_kind"] == "book_baseline"].iloc[0]
    # No book metrics → all raw metric columns NaN.
    for k in _RAW_METRIC_KEYS:
        assert math.isnan(book[k])
    assert math.isnan(book["brier_skill_score"])
    assert math.isnan(book["kelly_shrinkage"])
    cal = df[(df["row_kind"] == "model") & (df["metric_row"] == "calibrated")].iloc[0]
    assert math.isnan(cal["brier_skill_score"])
    assert math.isnan(cal["kelly_shrinkage"])


def test_get_market_calibration_returns_calibrated_row(patched_path):
    league_models = {"NFL": {"player_pass_yds": _make_model()}}
    write_model_stats(league_models, {}, {})
    out = get_market_calibration("NFL", "player_pass_yds")
    assert out["kelly_shrinkage"] == pytest.approx(1 - 0.197 / 0.218)
    assert out["brier_skill_score"] == pytest.approx(1 - 0.197 / 0.218)
    assert out["model_weight"] == pytest.approx(0.30)


def test_get_market_calibration_missing_returns_nans(patched_path):
    league_models = {"NFL": {"player_pass_yds": _make_model()}}
    write_model_stats(league_models, {}, {})
    out = get_market_calibration("NBA", "missing_market")
    assert math.isnan(out["kelly_shrinkage"])
    assert math.isnan(out["brier_skill_score"])
    assert math.isnan(out["model_weight"])


def test_get_market_calibration_no_parquet(tmp_path):
    target = tmp_path / "does_not_exist.parquet"
    with mock.patch.object(report_module, "MODEL_STATS_PATH", target):
        out = get_market_calibration("NFL", "x")
    assert math.isnan(out["kelly_shrinkage"])
    assert math.isnan(out["brier_skill_score"])
    assert math.isnan(out["model_weight"])
