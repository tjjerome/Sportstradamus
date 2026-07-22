"""Bind report gate results to the exact model-strategy artifact."""

import sys
from unittest import mock

import pandas as pd
import pytest

from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.model_strategy import (
    artifact_identity_columns,
    build_artifact_identity,
)
from sportstradamus.training.model_strategy import (
    get_strategy,
    registered_strategies,
    strategy_controls,
)
from sportstradamus.training.model_strategy import resolve_report_identity
from sportstradamus.training.report import _layer_gates_from_test_set

report_module = sys.modules["sportstradamus.training.report"]


def _test_set_frame(identity=None) -> pd.DataFrame:
    frame = pd.DataFrame({"MeanYr": [1.0, 2.0], "Result": [1.0, 2.0], "Blended_EV": [1.0, 2.0]})
    if identity is None:
        return frame
    for column, value in artifact_identity_columns(identity.as_model_blob()).items():
        frame[column] = value
    spec = get_strategy(identity.strategy_slug)
    for column in spec.required_columns:
        frame[column] = identity.strategy_slug if column == spec.legacy_csv_identity_column else "x"
    return frame


def _test_set_path(tmp_path, league: str, market: str):
    return tmp_path / f"{market_file_slug(league, market)}.csv"


def _base_identity(controls_index: int = 0, matrix_hash: str = "matrix-a"):
    spec = get_strategy("SkewNormal")
    return build_artifact_identity(
        spec.slug,
        "NBA",
        "FGA",
        strategy_controls(spec)[controls_index],
        matrix_hash=matrix_hash,
    )


@pytest.mark.parametrize(
    "test_identity",
    [_base_identity(1), _base_identity(matrix_hash="matrix-b")],
    ids=["different-corner", "different-matrix"],
)
def test_gate_layer_rejects_same_cell_different_generic_identity(tmp_path, test_identity):
    _test_set_frame(test_identity).to_csv(_test_set_path(tmp_path, "NBA", "FGA"), index=False)
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
        pytest.raises(ValueError, match="model/test-set strategy identity mismatch"),
    ):
        _layer_gates_from_test_set({}, _base_identity(), "NBA", "FGA")
    compute.assert_not_called()


def test_gate_layer_rejects_different_split_fingerprint(tmp_path):
    spec = next(spec for spec in registered_strategies() if spec.split_fingerprint_path)
    league, market = "NFL", "receiving yards"
    controls = strategy_controls(spec)[0]

    def legacy(split):
        return {
            "status": "active",
            "validation_audit": {"split_fingerprint_sha256": split},
        }

    model_identity = build_artifact_identity(
        spec.slug, league, market, controls, legacy("split-a"), matrix_hash="matrix-a"
    )
    test_identity = build_artifact_identity(
        spec.slug, league, market, controls, legacy("split-b"), matrix_hash="matrix-a"
    )
    _test_set_frame(test_identity).to_csv(_test_set_path(tmp_path, league, market), index=False)
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
        pytest.raises(ValueError, match="model/test-set strategy identity mismatch"),
    ):
        _layer_gates_from_test_set({}, model_identity, league, market)
    compute.assert_not_called()


@pytest.mark.parametrize("generic_model", [True, False], ids=["generic-model", "generic-csv"])
def test_gate_layer_rejects_mixed_legacy_and_generic(tmp_path, generic_model):
    generic_identity = _base_identity()
    legacy_identity = resolve_report_identity({"distribution": "SkewNormal"}, "NBA", "FGA")
    test_identity = None if generic_model else generic_identity
    _test_set_frame(test_identity).to_csv(_test_set_path(tmp_path, "NBA", "FGA"), index=False)
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
        pytest.raises(ValueError, match="mixed legacy/generic strategy artifacts"),
    ):
        _layer_gates_from_test_set(
            {}, generic_identity if generic_model else legacy_identity, "NBA", "FGA"
        )
    compute.assert_not_called()


@pytest.mark.parametrize("generic", [False, True], ids=["legacy", "generic"])
def test_gate_layer_allows_matching_model_and_test_set(tmp_path, generic):
    identity = (
        _base_identity()
        if generic
        else resolve_report_identity({"distribution": "SkewNormal"}, "NBA", "FGA")
    )
    _test_set_frame(identity if generic else None).to_csv(
        _test_set_path(tmp_path, "NBA", "FGA"), index=False
    )
    row = {}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates", return_value={"ship": True}) as compute,
    ):
        _layer_gates_from_test_set(row, identity, "NBA", "FGA")
    compute.assert_called_once()
    assert row["ship"] is True
