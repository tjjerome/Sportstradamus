"""Golden tests for training/report.py write_model_stats + get_market_calibration.

The parquet is the wide one-row-per-cell schema: each ``(league, market)`` pair
emits a single row carrying the model's validation-set metrics, the book
baseline counterparts, the per-cell diagnostics and hyperparameters, and pd.NA
placeholders for the ship-gate columns ``training.scorecard.compute_gates``
populates in a second pass.
"""

from __future__ import annotations

import math
import sys
from unittest import mock

import pandas as pd
import pytest

from sportstradamus.training.report import (
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
def patched_paths(tmp_path):
    parquet = tmp_path / "model_stats.parquet"
    csv = tmp_path / "model_stats.csv"
    with (
        mock.patch.object(report_module, "MODEL_STATS_PATH", parquet),
        mock.patch.object(report_module, "MODEL_STATS_CSV_PATH", csv),
    ):
        yield parquet, csv


def test_writes_one_row_per_cell(patched_paths):
    parquet, csv = patched_paths
    league_models = {"NFL": {"player_pass_yds": _make_model()}}
    write_model_stats(
        league_models,
        {"NFL": {"player_pass_yds": 0.5}},
        {"NFL": {"player_pass_yds": 1.0}},
        {"NFL": {"player_pass_yds": "devel"}},
    )

    df = pd.read_parquet(parquet)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["league"] == "NFL"
    assert row["market"] == "player_pass_yds"
    assert row["distribution"] == "Gamma"
    assert row["shipped"] == "devel"
    assert row["brier_book"] == pytest.approx(0.218)
    assert row["brier_model"] == pytest.approx(0.197)
    assert row["brier_skill_score"] == pytest.approx(1 - 0.197 / 0.218)
    assert row["kelly_shrinkage"] == pytest.approx(1 - 0.197 / 0.218)
    assert row["log_loss_model"] == pytest.approx(0.65)
    assert row["log_loss_book"] == pytest.approx(0.65)
    assert row["nll"] == pytest.approx(0.65)
    assert row["roc_auc"] == pytest.approx(0.62)
    assert row["accuracy"] == pytest.approx(0.58)
    assert row["model_weight"] == pytest.approx(0.30)
    assert row["model_shape"] == pytest.approx(1.1)
    assert row["shape_ratio"] == pytest.approx(1.1)
    assert row["mean_ev_diff"] == pytest.approx(0.05)
    assert row["cv"] == pytest.approx(0.5)
    assert row["std"] == pytest.approx(1.0)
    assert row["historical_zero_rate"] == pytest.approx(0.01)
    assert row["hp_rounds"] == pytest.approx(100)
    assert row["hp_leaves"] == pytest.approx(31)
    # Ship-gate columns are populated by training.scorecard.compute_gates;
    # at write time they're NaN / pd.NA.
    assert math.isnan(row["ece_equal_mass"])
    assert math.isnan(row["g1_brier_diff_mean"])
    assert math.isnan(row["g5_ece_debiased"])
    assert pd.isna(row["g1_pass"])
    assert pd.isna(row["ship"])

    # CSV mirror is rewritten alongside the parquet.
    assert csv.is_file()
    csv_df = pd.read_csv(csv)
    assert len(csv_df) == 1
    assert csv_df.iloc[0]["brier_skill_score"] == pytest.approx(1 - 0.197 / 0.218)


def test_useless_model_skill_score_zero(patched_paths):
    league_models = {"NFL": {"m": _make_model(brier=0.218)}}
    write_model_stats(league_models, {}, {})
    df = pd.read_parquet(patched_paths[0])
    row = df.iloc[0]
    assert row["brier_skill_score"] == pytest.approx(0.0, abs=1e-9)
    assert row["kelly_shrinkage"] == pytest.approx(0.0, abs=1e-9)


def test_missing_book_baseline_nans(patched_paths):
    league_models = {"NFL": {"m": _make_model(with_book=False)}}
    write_model_stats(league_models, {}, {})
    df = pd.read_parquet(patched_paths[0])
    row = df.iloc[0]
    assert math.isnan(row["brier_book"])
    assert math.isnan(row["log_loss_book"])
    assert math.isnan(row["brier_skill_score"])
    assert math.isnan(row["kelly_shrinkage"])
    # Model-side metrics still populate.
    assert row["brier_model"] == pytest.approx(0.197)


def test_default_shipped_is_withheld(patched_paths):
    """Cells absent from the stat_shipped map default to ``"withheld"``."""
    league_models = {"NFL": {"m": _make_model()}}
    write_model_stats(league_models, {}, {})  # no stat_shipped argument
    df = pd.read_parquet(patched_paths[0])
    assert df.iloc[0]["shipped"] == "withheld"


def test_get_market_calibration_returns_cell_row(patched_paths):
    league_models = {"NFL": {"player_pass_yds": _make_model()}}
    write_model_stats(league_models, {}, {})
    out = get_market_calibration("NFL", "player_pass_yds")
    assert out["kelly_shrinkage"] == pytest.approx(1 - 0.197 / 0.218)
    assert out["brier_skill_score"] == pytest.approx(1 - 0.197 / 0.218)
    assert out["model_weight"] == pytest.approx(0.30)


def test_get_market_calibration_missing_returns_nans(patched_paths):
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
