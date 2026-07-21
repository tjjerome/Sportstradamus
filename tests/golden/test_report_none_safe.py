"""Regression test for `_wide_row` tolerating None-valued diagnostics.

`pipeline.py:_step_select_distribution` previously stored
`marginal_shape = None` on SkewNormal cells (the count-branch concept
doesn't apply) and the value flowed into `model["diagnostics"]`.
`report.py:_wide_row` called `float(diag.get("marginal_shape", nan))`,
which returns the literal `None` (not the default) when the key is
present-but-None — and `float(None)` raises `TypeError`. Manifested as
the NBA FG3A crash in `meditate --bypass-withholding` after Optuna
completed:

    TypeError: float() argument must be a string or a real number, not 'NoneType'

The fix coerces `None` to `NaN` at every helper inside `_wide_row`
(belt) and stores the SkewNormal `marginal_shape` as `float("nan")`
upstream (suspenders).
"""

import math
from unittest import mock

import pytest

from sportstradamus.training.report import _wide_row


def _model_with_diag(diag: dict) -> dict:
    """Minimum filedict shape `_wide_row` needs to produce a row."""
    return {
        "distribution": "SkewNormal",
        "metrics": {"model": {}, "book_baseline": {}},
        "diagnostics": diag,
        "params": {},
        "hist_gate": 0.0,
    }


def test_wide_row_handles_none_marginal_shape():
    """The canonical NBA FG3A failure mode — `marginal_shape=None` must coerce to NaN."""
    model = _model_with_diag(
        {
            "marginal_shape": None,
            "model_shape": 1.0,
            "empirical_shape": 1.0,
        }
    )

    row, _identity = _wide_row(model, "NBA", "FG3A", "devel", {}, {})

    assert math.isnan(row["marginal_shape"])
    assert row["model_shape"] == 1.0
    assert row["empirical_shape"] == 1.0


def test_wide_row_handles_none_in_every_diag_key():
    """Defensive: every diag lookup must coerce None to NaN."""
    diag_keys = (
        "marginal_shape",
        "model_ev",
        "mean_line",
        "result_mean",
        "ev_minus_line",
        "median_ev_diff",
        "frac_ev_gt_line",
        "over_pct_ev_gt",
        "over_pct_ev_lt",
        "model_weight",
        "dispersion_cal",
        "model_shape",
        "empirical_shape",
    )
    model = _model_with_diag(dict.fromkeys(diag_keys))

    row, _identity = _wide_row(model, "NBA", "FG3A", "devel", {}, {})

    for col in ("marginal_shape", "model_ev", "dispersion_cal", "model_weight"):
        assert math.isnan(row[col]), f"{col} must be NaN when diag value is None"


def test_wide_row_handles_none_in_model_and_metrics_blocks():
    """`hist_gate`, `brier_skill_score`, `kelly_shrinkage` lookups are also None-safe."""
    model = {
        "distribution": "SkewNormal",
        "metrics": {
            "model": {"brier_score": None, "log_loss": None},
            "book_baseline": {"brier_score": None},
            "brier_skill_score": None,
            "kelly_shrinkage": None,
        },
        "diagnostics": {},
        "params": {},
        "hist_gate": None,
    }

    row, _identity = _wide_row(model, "NBA", "FG3A", "devel", {}, {})

    for col in (
        "historical_zero_rate",
        "brier_skill_score",
        "kelly_shrinkage",
        "brier_book",
        "brier_model",
    ):
        assert math.isnan(row[col]), f"{col} must be NaN when source value is None"


def test_wide_row_still_passes_real_floats_through():
    """Sanity: real numbers in the source dict survive the None-safe wrappers unchanged."""
    model = {
        "distribution": "ZINB",
        "metrics": {
            "model": {"brier_score": 0.21, "log_loss": 0.55},
            "book_baseline": {"brier_score": 0.24},
            "brier_skill_score": 0.13,
            "kelly_shrinkage": 0.13,
        },
        "diagnostics": {"marginal_shape": 3.7, "model_shape": 4.1, "empirical_shape": 3.9},
        "params": {},
        "hist_gate": 0.18,
    }

    row, _identity = _wide_row(model, "NBA", "BLK", "devel", {}, {})

    assert row["brier_book"] == 0.24
    assert row["brier_model"] == 0.21
    assert row["brier_skill_score"] == 0.13
    assert row["marginal_shape"] == 3.7
    assert row["historical_zero_rate"] == 0.18


def test_wide_row_surfaces_generic_strategy_identity():
    model = _model_with_diag({"model_shape": 1.0, "empirical_shape": 1.0})
    identity = mock.Mock(
        strategy_slug="rushing-qb-rb-affine-groupcdf-bookpool-v1",
        structural_strategy="rushing-qb-rb-affine-groupcdf-bookpool-v1",
        signature="canonical-signature",
        implementation_version=2,
        artifact_schema_version=3,
        status="active",
        controls_json='{"dist":"SkewNormal"}',
        corner_fingerprint="corner-fingerprint",
        matrix_hash="matrix-sha256",
        split_fingerprint="split-sha256",
    )
    with mock.patch(
        "sportstradamus.training.report.resolve_report_identity", return_value=identity
    ) as resolve_identity:
        row, returned_identity = _wide_row(model, "NFL", "rushing yards", "devel", {}, {})

    resolve_identity.assert_called_once_with(model, league="NFL", market="rushing yards")
    assert returned_identity is identity
    assert {
        "strategy_slug": row["strategy_slug"],
        "structural_strategy": row["structural_strategy"],
        "strategy_signature": row["strategy_signature"],
        "strategy_implementation_version": row["strategy_implementation_version"],
        "artifact_schema_version": row["artifact_schema_version"],
        "strategy_status": row["strategy_status"],
        "strategy_controls_json": row["strategy_controls_json"],
        "strategy_corner_fingerprint": row["strategy_corner_fingerprint"],
        "strategy_matrix_hash": row["strategy_matrix_hash"],
        "strategy_split_fingerprint": row["strategy_split_fingerprint"],
    } == {
        "strategy_slug": identity.strategy_slug,
        "structural_strategy": identity.structural_strategy,
        "strategy_signature": identity.signature,
        "strategy_implementation_version": identity.implementation_version,
        "artifact_schema_version": identity.artifact_schema_version,
        "strategy_status": identity.status,
        "strategy_controls_json": identity.controls_json,
        "strategy_corner_fingerprint": identity.corner_fingerprint,
        "strategy_matrix_hash": identity.matrix_hash,
        "strategy_split_fingerprint": identity.split_fingerprint,
    }


def test_wide_row_propagates_invalid_structural_identity():
    model = _model_with_diag({"model_shape": 1.0, "empirical_shape": 1.0})
    with (
        mock.patch(
            "sportstradamus.training.report.resolve_report_identity",
            side_effect=ValueError("stale structural strategy identity"),
        ),
        pytest.raises(ValueError, match="stale structural strategy identity"),
    ):
        _wide_row(model, "NFL", "rushing yards", "devel", {}, {})
