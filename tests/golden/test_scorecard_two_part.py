"""Official-scorecard parity guards for the NFL receiving-yards v3 head."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy.stats import skewnorm

from sportstradamus.training import group_conditional_cdf as receiving
from sportstradamus.training.group_conditional_cdf._pool import (
    fixed_pool_blob,
)
from sportstradamus.training.model_strategy_artifacts import (
    STRATEGY_MARKET_CSV_COLUMN,
    STRATEGY_SIGNATURE_CSV_COLUMN,
    STRATEGY_STATUS_CSV_COLUMN,
    STRUCTURAL_STRATEGY_CSV_COLUMN,
    artifact_identity_columns,
    build_artifact_identity,
)
from sportstradamus.training.model_strategy_registry import get_strategy, strategy_controls
from sportstradamus.training.scorecard import (
    _pred_cdf_pmf,
    _randomized_pit_draws,
    gate_row,
    load_test_set,
    main,
)

_MATRIX_HASH = "a" * 64
_SPLIT_FINGERPRINT = "b" * 64


def _strategy_identity_columns() -> dict[str, object]:
    spec = get_strategy(receiving.CANDIDATE_NAME)
    legacy_payload = {
        "schema_version": receiving.SCHEMA_VERSION,
        "status": "active",
        "validation_audit": {"split_fingerprint_sha256": _SPLIT_FINGERPRINT},
    }
    identity = build_artifact_identity(
        spec.slug,
        "NFL",
        "receiving yards",
        strategy_controls(spec)[0],
        legacy_payload,
        matrix_hash=_MATRIX_HASH,
    )
    return artifact_identity_columns(identity.as_model_blob())


def _identity_map(lam: float) -> dict[str, object]:
    return {
        "kind": "isotonic_pit",
        "lam": lam,
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }


def _calibration_blob() -> dict[str, object]:
    curved = {
        "kind": "isotonic_pit",
        "lam": 1.0,
        "x": [0.0, 0.4, 1.0],
        "y": [0.0, 0.2, 1.0],
    }
    return {
        "kind": receiving.CANDIDATE_NAME,
        "schema_version": receiving.SCHEMA_VERSION,
        "line_probability_only": True,
        "temperature_fit_scope": "pre_map_raw_endpoint_settlement",
        "temperature": 1.4,
        "cdf": {
            "kind": "role_position_two_part_cdf",
            "role_boundary": {
                role: {
                    "kind": "two_part_role_boundary",
                    "intercept": -0.3 if role == "low" else 0.2,
                    "nonpositive": _identity_map(0.75),
                }
                for role in receiving.ROLE_VALUES
            },
            "positive": {
                f"{role}_pos{position}": curved.copy()
                for role in receiving.ROLE_VALUES
                for position in receiving.POSITION_CODES
            },
            "rb_boundary_residual": 0.25,
        },
        "probability_pool": fixed_pool_blob(),
    }


def _candidate_frame() -> pd.DataFrame:
    result = np.array([0.0, 4.0, 8.0, 12.0, 20.0, 30.0])
    loc = np.array([6.0, 8.0, 10.0, 12.0, 18.0, 24.0])
    scale = np.array([5.0, 5.5, 6.0, 6.5, 8.0, 9.0])
    alpha = np.array([1.0, 1.3, 1.6, 1.9, 1.2, 0.8])
    gate = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.05])
    base_f0 = skewnorm.cdf(0.0, alpha, loc=loc, scale=scale)
    f0 = gate + (1.0 - gate) * base_f0
    frame = pd.DataFrame(
        {
            "MeanYr": np.array([8.0, 10.0, 12.0, 15.0, 22.0, 28.0]),
            "Result": result,
            "Blended_EV": np.array([7.0, 9.0, 11.0, 14.0, 20.0, 26.0]),
            "Line": np.array([5.5, 7.5, 9.5, 11.5, 19.5, 27.5]),
            "P": np.array([0.53, 0.57, 0.51, 0.49, 0.55, 0.52]),
            "P_PrePool": np.array([0.58, 0.61, 0.54, 0.46, 0.60, 0.56]),
            "SN_Loc": loc,
            "SN_Scale": scale,
            "SN_Alpha": alpha,
            "Gate": gate,
            "StructuralAdapterStrategy": receiving.CANDIDATE_NAME,
            "StructuralRoute": np.array(["low", "low", "high", "high", "low", "high"]),
            "StructuralFallback": False,
            "StructuralCalibration": receiving.serialize_two_part_calibration(_calibration_blob()),
            "StructuralRole": np.array(["low", "low", "high", "high", "low", "high"]),
            "StructuralPosition": np.array([2, 3, 4, 2, 3, 4]),
            "StructuralF0": f0,
        }
    )
    for column, value in _strategy_identity_columns().items():
        frame[column] = value
    return frame


def test_receiving_v3_gate4_transforms_outcome_endpoints_before_randomization():
    frame = _candidate_frame()
    actual = frame["Result"].to_numpy(dtype=float)
    raw_upper, raw_mass = _pred_cdf_pmf(frame, "SkewNormal", actual, strategy="none")
    expected = receiving.two_part_randomized_pit(
        _calibration_blob(),
        raw_upper - raw_mass,
        raw_upper,
        frame["StructuralF0"].to_numpy(dtype=float),
        frame["StructuralRole"].to_numpy(),
        frame["StructuralPosition"].to_numpy(),
    )

    scored = np.asarray(_randomized_pit_draws(frame, "SkewNormal", actual, strategy="none"))
    np.testing.assert_array_equal(scored, expected)

    different_lines = frame.assign(Line=np.linspace(100.5, 600.5, len(frame)))
    rescored = np.asarray(
        _randomized_pit_draws(different_lines, "SkewNormal", actual, strategy="none")
    )
    np.testing.assert_array_equal(rescored, expected)


def test_load_test_set_retains_and_validates_receiving_v3_contract(tmp_path):
    path = tmp_path / "NFL_receiving-yards.csv"
    _candidate_frame().to_csv(path, index=False)

    loaded = load_test_set(path, "Blended_EV")

    assert {
        "StructuralAdapterStrategy",
        "StructuralCalibration",
        "StructuralRole",
        "StructuralPosition",
        "StructuralF0",
        "P_PrePool",
    }.issubset(loaded.columns)
    assert set(_strategy_identity_columns()).issubset(loaded.columns)
    assert loaded["StructuralCalibration"].nunique() == 1


def test_load_test_set_rejects_partial_or_identity_absent_receiving_contract(tmp_path):
    partial = _candidate_frame().drop(columns=STRATEGY_SIGNATURE_CSV_COLUMN)
    partial_path = tmp_path / "partial-identity.csv"
    partial.to_csv(partial_path, index=False)
    with pytest.raises(ValueError, match="StrategySignature"):
        load_test_set(partial_path, "Blended_EV")

    adapter_only = _candidate_frame().drop(columns=list(_strategy_identity_columns()))
    adapter_path = tmp_path / "adapter-only.csv"
    adapter_only.to_csv(adapter_path, index=False)
    with pytest.raises(ValueError, match="adapter strategy columns require generic"):
        load_test_set(adapter_path, "Blended_EV")


def test_load_test_set_rejects_inactive_or_mismatched_structural_identity(tmp_path):
    inactive = _candidate_frame()
    inactive[STRATEGY_STATUS_CSV_COLUMN] = "killed_fallback"
    inactive_path = tmp_path / "inactive.csv"
    inactive.to_csv(inactive_path, index=False)
    with pytest.raises(ValueError, match="inactive strategy artifact"):
        load_test_set(inactive_path, "Blended_EV")

    mismatched = _candidate_frame()
    mismatched[STRUCTURAL_STRATEGY_CSV_COLUMN] = "none"
    mismatched_path = tmp_path / "mismatched.csv"
    mismatched.to_csv(mismatched_path, index=False)
    with pytest.raises(ValueError, match="stale, mismatched, or wrong-cell"):
        load_test_set(mismatched_path, "Blended_EV")



def test_load_test_set_rejects_nonconstant_or_wrong_schema_receiving_blob(tmp_path):
    frame = _candidate_frame()
    frame.loc[0, "StructuralAdapterStrategy"] = "none"
    path = tmp_path / "nonconstant-experiment.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one constant nonmissing StructuralAdapterStrategy"):
        load_test_set(path, "Blended_EV")

    frame = _candidate_frame()
    frame.loc[0, "StructuralCalibration"] += " "
    path = tmp_path / "nonconstant.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one constant nonmissing JSON"):
        load_test_set(path, "Blended_EV")

    frame = _candidate_frame()
    blob = json.loads(frame["StructuralCalibration"].iloc[0])
    blob["schema_version"] = 99
    frame["StructuralCalibration"] = json.dumps(blob)
    path = tmp_path / "wrong-schema.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown receiving calibration blob kind or schema"):
        load_test_set(path, "Blended_EV")


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("StructuralRole", "slot", "only nonmissing low/high"),
        ("StructuralPosition", 1, "only WR/RB/TE codes"),
        ("StructuralF0", 1.2, "finite probabilities"),
        ("SN_Scale", 0.0, "strictly positive"),
        ("P_PrePool", np.nan, "P_PrePool must be finite"),
    ],
)
def test_load_test_set_rejects_invalid_receiving_v3_row_fields(
    tmp_path, column, bad_value, message
):
    frame = _candidate_frame()
    frame.loc[0, column] = bad_value
    path = tmp_path / f"invalid-{column}.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_test_set(path, "Blended_EV")


def test_load_test_set_rejects_partial_receiving_v3_contract(tmp_path):
    frame = _candidate_frame().drop(columns="StructuralF0")
    path = tmp_path / "partial.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"artifact missing required columns.*StructuralF0"):
        load_test_set(path, "Blended_EV")


def test_receiving_v3_gate4_rejects_f0_shape_drift():
    frame = _candidate_frame()
    frame["StructuralF0"] += 1e-4

    with pytest.raises(ValueError, match="does not match the persisted served SkewNormal shape"):
        _randomized_pit_draws(
            frame,
            "SkewNormal",
            frame["Result"].to_numpy(dtype=float),
            strategy="none",
        )


def test_receiving_v3_scorecard_contract_is_cell_scoped():
    with pytest.raises(ValueError, match="does not match the generic model-strategy identity"):
        gate_row(
            _candidate_frame(),
            "Blended_EV",
            league="NBA",
            market="PTS",
            strategy="none",
        )


def test_diff_cli_routes_explicit_cell_identity_to_receiving_v3_gate(tmp_path):
    path = tmp_path / "receiving-v3.csv"
    _candidate_frame().to_csv(path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "--baseline",
            str(path),
            "--candidate",
            str(path),
            "--league",
            "NFL",
            "--market",
            "receiving-yards",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "supersede:" in result.output

    inferred = CliRunner().invoke(
        main,
        ["--baseline", str(path), "--candidate", str(path)],
    )
    assert inferred.exit_code == 0, inferred.output


def test_diff_cli_rejects_mixed_legacy_and_generic_identity(tmp_path):
    candidate = _candidate_frame()
    candidate_path = tmp_path / "candidate.csv"
    candidate.to_csv(candidate_path, index=False)

    legacy = candidate.drop(
        columns=[
            *list(_strategy_identity_columns()),
            "StructuralAdapterStrategy",
            "StructuralRoute",
            "StructuralFallback",
            "StructuralCalibration",
            "StructuralF0",
            "StructuralRole",
            "StructuralPosition",
        ]
    )
    legacy_path = tmp_path / "legacy.csv"
    legacy.to_csv(legacy_path, index=False)

    result = CliRunner().invoke(
        main,
        ["--baseline", str(legacy_path), "--candidate", str(candidate_path)],
    )

    assert result.exit_code != 0
    assert "cannot mix legacy and generic strategy identities" in result.output
