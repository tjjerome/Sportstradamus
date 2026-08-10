"""Bind report gate results to the exact model-strategy artifact."""

import sys
from unittest import mock

import pandas as pd
import pytest

from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.model_strategy import (
    artifact_identity_columns,
    build_artifact_identity,
    get_strategy,
    registered_strategies,
    resolve_report_identity,
    strategy_controls,
)
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


def _split_fingerprint_cell():
    spec = next(spec for spec in registered_strategies() if spec.split_fingerprint_path)
    return spec, "NFL", "receiving yards"


def _split_identity(spec, league: str, market: str, split: str, status: str = "active"):
    return build_artifact_identity(
        spec.slug,
        league,
        market,
        strategy_controls(spec)[0],
        {"status": status, "validation_audit": {"split_fingerprint_sha256": split}},
        matrix_hash="matrix-a",
    )


@pytest.fixture
def report_caplog(caplog):
    # report's logger sets propagate=False (helpers.get_logger), so caplog's root
    # handler never sees its records — attach the handler to that logger directly.
    report_module.logger.addHandler(caplog.handler)
    yield caplog
    report_module.logger.removeHandler(caplog.handler)


def _only_warning(caplog) -> str:
    (record,) = [r for r in caplog.records if r.levelname == "WARNING"]
    return record.getMessage()


@pytest.mark.parametrize(
    "test_identity",
    [_base_identity(1), _base_identity(matrix_hash="matrix-b")],
    ids=["different-corner", "different-matrix"],
)
def test_gate_layer_skips_same_cell_different_generic_identity(
    tmp_path, report_caplog, test_identity
):
    _test_set_frame(test_identity).to_csv(_test_set_path(tmp_path, "NBA", "FGA"), index=False)
    row = {"ship": pd.NA}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
    ):
        _layer_gates_from_test_set(row, _base_identity(), "NBA", "FGA")
    compute.assert_not_called()
    assert row == {"ship": pd.NA}
    warning = _only_warning(report_caplog)
    assert "NBA/FGA" in warning
    assert "model/test-set strategy identity mismatch" in warning


def test_gate_layer_skips_different_split_fingerprint(tmp_path, report_caplog):
    spec, league, market = _split_fingerprint_cell()
    model_identity = _split_identity(spec, league, market, "split-a")
    test_identity = _split_identity(spec, league, market, "split-b")
    _test_set_frame(test_identity).to_csv(_test_set_path(tmp_path, league, market), index=False)
    row = {"ship": pd.NA}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
    ):
        _layer_gates_from_test_set(row, model_identity, league, market)
    compute.assert_not_called()
    assert row == {"ship": pd.NA}
    warning = _only_warning(report_caplog)
    assert f"{league}/{market}" in warning
    assert "model/test-set strategy identity mismatch" in warning


@pytest.mark.parametrize("generic_model", [True, False], ids=["generic-model", "generic-csv"])
def test_gate_layer_skips_mixed_legacy_and_generic(tmp_path, report_caplog, generic_model):
    generic_identity = _base_identity()
    legacy_identity = resolve_report_identity({"distribution": "SkewNormal"}, "NBA", "FGA")
    test_identity = None if generic_model else generic_identity
    _test_set_frame(test_identity).to_csv(_test_set_path(tmp_path, "NBA", "FGA"), index=False)
    row = {"ship": pd.NA}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
    ):
        _layer_gates_from_test_set(
            row, generic_identity if generic_model else legacy_identity, "NBA", "FGA"
        )
    compute.assert_not_called()
    assert row == {"ship": pd.NA}
    warning = _only_warning(report_caplog)
    assert "NBA/FGA" in warning
    assert "mixed legacy/generic strategy artifacts" in warning


def test_gate_layer_skips_test_set_missing_required_columns(tmp_path, report_caplog):
    spec, league, market = _split_fingerprint_cell()
    identity = _split_identity(spec, league, market, "split-a")
    dropped = next(c for c in spec.required_columns if c != spec.legacy_csv_identity_column)
    _test_set_frame(identity).drop(columns=[dropped]).to_csv(
        _test_set_path(tmp_path, league, market), index=False
    )
    row = {"ship": pd.NA}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
    ):
        _layer_gates_from_test_set(row, identity, league, market)
    compute.assert_not_called()
    assert row == {"ship": pd.NA}
    warning = _only_warning(report_caplog)
    assert f"{league}/{market}" in warning
    assert f"artifact missing required columns ['{dropped}']" in warning


def test_gate_layer_skips_self_killed_structural_artifact(tmp_path, report_caplog):
    """A structural experiment that self-kills to its pooled fallback stamps a non-active
    status; ``validate_strategy_frame`` rejects that with InactiveStrategyArtifactError. It is a
    legitimate per-cell training outcome, so the report skips the cell (gates stay pd.NA, so the
    fallback is never credited a ship) instead of aborting the run. Serving still refuses it.
    """
    spec, league, market = _split_fingerprint_cell()
    identity = _split_identity(spec, league, market, "split-a", status="killed_fallback")
    _test_set_frame(identity).to_csv(_test_set_path(tmp_path, league, market), index=False)
    row = {"ship": pd.NA}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
    ):
        _layer_gates_from_test_set(row, identity, league, market)
    compute.assert_not_called()
    assert row == {"ship": pd.NA}
    warning = _only_warning(report_caplog)
    assert f"{league}/{market}" in warning
    assert "inactive strategy artifact" in warning


@pytest.mark.parametrize(
    "csv_text", ["MeanYr,Result,Blended_EV\n1,2,3\n4,5,6,7,8\n", ""], ids=["ragged", "truncated"]
)
def test_gate_layer_skips_unreadable_test_set(tmp_path, report_caplog, csv_text):
    """A copy interrupted mid-write leaves a CSV pandas cannot parse (ParserError) or an empty
    file (EmptyDataError). Both must warn and skip, not abort the run training every other cell.
    """
    _test_set_path(tmp_path, "NBA", "FGA").write_text(csv_text)
    row = {"ship": pd.NA}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates") as compute,
    ):
        _layer_gates_from_test_set(row, _base_identity(), "NBA", "FGA")
    compute.assert_not_called()
    assert row == {"ship": pd.NA}
    warning = _only_warning(report_caplog)
    assert "NBA/FGA" in warning
    assert "retrain this cell" in warning
    assert "\n" not in warning  # pandas' ParserError text is multi-line; the log line is not


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
