"""Input and artifact validation for the group-conditional CDF engine.

The two corners share the finite-array primitive verbatim but diverge on book
handling (receiving tolerates a nonfinite book on non-authentic rows; rushing
requires every book probability finite in ``[0, 1]``), on the missing-identifier
predicate, and on their blob schemas. Divergent helpers carry a corner suffix so
both survive behind config; identical helpers are shared.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from sportstradamus.training.group_conditional_cdf._contracts import (
    AFFINE_INTERCEPT_BOUNDS,
    AFFINE_SLOPE_BOUNDS,
    BOUNDARY_LOGIT_CLIP,
    CDF_BRANCH_TOLERANCE,
    INTERCEPT_BRACKET,
    MARGINAL_MEAN_FLOOR,
    PIT_LAMBDAS,
    PLAYER_CV_FOLDS,
    POSITIVE_GROUPS,
    RECEIVING_CANDIDATE_NAME,
    RECEIVING_POSITION_CODES,
    RECEIVING_SCHEMA_VERSION,
    ROLE_VALUES,
    RUSHING_CANDIDATE_NAME,
    RUSHING_POSITION_CODES,
    RUSHING_SCHEMA_VERSION,
    TEMPERATURE_BOUNDS,
    CalibrationRows,
    GroupCdfBlob,
    ReceivingCalibrationBlob,
    RushingCalibrationBlob,
    RushingPredictive,
)
from sportstradamus.training.group_conditional_cdf.cdf_map_validation import cdf_map_blob_is_valid


def subset_rows(rows: CalibrationRows, indices: np.ndarray) -> CalibrationRows:
    return CalibrationRows(*(getattr(rows, field)[indices] for field in rows.__dataclass_fields__))


def finite_vector(values, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must contain {length} rows")
    return array


def probability_vector(values, name: str, length: int | None = None) -> np.ndarray:
    array = finite_vector(values, name, length)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def optional_book_vector(values, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"book_over must contain {length} row-aligned values")
    finite = np.isfinite(array)
    if np.any((array[finite] < 0.0) | (array[finite] > 1.0)):
        raise ValueError("finite book_over values must lie in [0, 1]")
    return array


def authentic_vector(values, length: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != length:
        raise ValueError(f"authentic must contain {length} row-aligned booleans")
    if not all(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError("authentic must contain explicit booleans")
    return raw.astype(bool)


def group_vector(values, name: str, length: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != length:
        raise ValueError(f"{name} must be a row-aligned one-dimensional array")
    if any(missing_identifier_receiving(value) for value in raw):
        raise ValueError(f"{name} identifiers must be nonempty")
    return raw.astype(str)


def missing_identifier_receiving(value) -> bool:
    if value is None:
        return True
    try:
        if bool(np.isnan(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in {"", "nan", "None", "<NA>"}


def roles(values, length: int) -> np.ndarray:
    role_array = group_vector(values, "role", length)
    if not np.isin(role_array, ROLE_VALUES).all():
        raise ValueError("role must contain only low and high")
    return role_array


def receiving_positions(values, length: int) -> np.ndarray:
    raw = finite_vector(values, "position", length)
    position_array = raw.astype(int)
    if (
        not np.array_equal(raw, position_array)
        or not np.isin(position_array, RECEIVING_POSITION_CODES).all()
    ):
        raise ValueError("position must contain only WR=2, RB=3, and TE=4 codes")
    return position_array


def temperature(value) -> float:
    if (
        not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not TEMPERATURE_BOUNDS[0] <= value <= TEMPERATURE_BOUNDS[1]
    ):
        raise ValueError("temperature must be finite and inside [1, 10]")
    return float(value)


def cdf_apply_inputs(cdf, f0, role, position):
    values = probability_vector(cdf, "cdf")
    n_rows = len(values)
    boundary = probability_vector(f0, "f0", n_rows)
    return values, boundary, roles(role, n_rows), receiving_positions(position, n_rows)


def receiving_fit_inputs(
    result_upper,
    result_lower,
    zero_cdf,
    line_low,
    line_high,
    result,
    over_result,
    book_over,
    authentic,
    player,
    role,
    position,
    position_codes,
) -> CalibrationRows:
    # style: allow-complexity — moved verbatim from the receiving corner; the
    # flat guard chain is the byte-for-byte input contract and must not split.
    upper = probability_vector(result_upper, "result_cdf_upper")
    n_rows = len(upper)
    lower = probability_vector(result_lower, "result_cdf_lower", n_rows)
    boundary = probability_vector(zero_cdf, "zero_cdf", n_rows)
    low = probability_vector(line_low, "line_cdf_low", n_rows)
    high = probability_vector(line_high, "line_cdf_high", n_rows)
    actual = finite_vector(result, "result", n_rows)
    outcome = probability_vector(over_result, "over_result", n_rows)
    books = optional_book_vector(book_over, n_rows)
    authentic_rows = authentic_vector(authentic, n_rows)
    if not np.isin(outcome, (0.0, 1.0)).all():
        raise ValueError("over_result must contain only binary 0/1 values")
    if np.any(lower > upper + CDF_BRANCH_TOLERANCE):
        raise ValueError("result_cdf_lower cannot exceed result_cdf_upper")
    if np.any(low > high + CDF_BRANCH_TOLERANCE):
        raise ValueError("line_cdf_low cannot exceed line_cdf_high")
    if not np.isfinite(books[authentic_rows]).all() or np.any(
        (books[authentic_rows] <= 0.0) | (books[authentic_rows] >= 1.0)
    ):
        raise ValueError("authentic book_over must be finite and strictly inside (0, 1)")
    if np.any((actual > 0.0) & (upper < boundary - CDF_BRANCH_TOLERANCE)) or np.any(
        (actual <= 0.0) & (upper > boundary + CDF_BRANCH_TOLERANCE)
    ):
        raise ValueError("result CDF values are inconsistent with the F(0) boundary")

    players = group_vector(player, "player", n_rows)
    roles_array = roles(role, n_rows)
    positions_array = receiving_positions(position, n_rows)
    if len(np.unique(players)) < PLAYER_CV_FOLDS:
        raise ValueError("nested calibration requires at least five validation players")
    for role_name in ROLE_VALUES:
        role_mask = roles_array == role_name
        if len(np.unique(players[role_mask])) < PLAYER_CV_FOLDS:
            raise ValueError(f"role {role_name} requires at least five validation players")
        if not np.any(role_mask & (actual > 0.0)) or not np.any(role_mask & (actual <= 0.0)):
            raise ValueError(f"role {role_name} requires positive and nonpositive support")
    for code in position_codes:
        position_mask = positions_array == code
        if not np.any(position_mask & (actual > 0.0)):
            raise ValueError(f"position {code} requires positive support")
    return CalibrationRows(
        upper,
        lower,
        boundary,
        low,
        high,
        actual,
        outcome,
        books,
        authentic_rows,
        players,
        roles_array,
        positions_array,
    )


def validate_endpoint_map(blob: Mapping[str, object]) -> None:
    valid = cdf_map_blob_is_valid(
        blob,
        kind="isotonic_pit",
        allowed_lambdas=PIT_LAMBDAS,
    )
    if not valid:
        raise ValueError("invalid receiving conditional CDF map")


def validated_receiving_blob(blob: Mapping[str, object]) -> ReceivingCalibrationBlob:
    # style: allow-complexity — moved verbatim from the receiving corner; the
    # flat guard chain is the byte-for-byte schema contract and must not split.
    from sportstradamus.training.group_conditional_cdf._pool import fixed_pool_blob

    if (
        blob.get("kind") != RECEIVING_CANDIDATE_NAME
        or blob.get("schema_version") != RECEIVING_SCHEMA_VERSION
    ):
        raise ValueError("unknown receiving calibration blob kind or schema")
    if blob.get("line_probability_only") is not True:
        raise ValueError("receiving calibration must declare its line-only probability layer")
    fitted = cast(ReceivingCalibrationBlob, blob)
    if fitted.get("temperature_fit_scope") != "pre_map_raw_endpoint_settlement":
        raise ValueError("receiving calibration has an unknown temperature fit scope")
    temperature(fitted.get("temperature"))
    models = fitted.get("cdf")
    if not isinstance(models, dict) or models.get("kind") != (
        "receiving_role_position_two_part_cdf"
    ):
        raise ValueError("receiving calibration requires the v3 CDF model")
    role_models = models.get("role_boundary")
    positive_maps = models.get("positive")
    if not isinstance(role_models, dict) or set(role_models) != set(ROLE_VALUES):
        raise ValueError("receiving calibration requires low and high role boundaries")
    if not isinstance(positive_maps, dict) or set(positive_maps) != set(POSITIVE_GROUPS):
        raise ValueError("receiving calibration requires all six positive CDF maps")
    residual = models.get("rb_boundary_residual")
    if (
        not isinstance(residual, (int, float))
        or not np.isfinite(residual)
        or not INTERCEPT_BRACKET[0] <= float(residual) <= INTERCEPT_BRACKET[1]
    ):
        raise ValueError("invalid receiving RB boundary residual")
    nonpositive_lambdas: set[float] = set()
    for role_name in ROLE_VALUES:
        model = role_models[role_name]
        if not isinstance(model, dict) or model.get("kind") != ("receiving_two_part_role_boundary"):
            raise ValueError(f"invalid boundary model for role {role_name}")
        intercept = model.get("intercept")
        if (
            not isinstance(intercept, (int, float))
            or not np.isfinite(intercept)
            or not INTERCEPT_BRACKET[0] <= float(intercept) <= INTERCEPT_BRACKET[1]
        ):
            raise ValueError(f"invalid boundary intercept for role {role_name}")
        nonpositive = model.get("nonpositive")
        if not isinstance(nonpositive, dict):
            raise ValueError(f"missing nonpositive CDF map for role {role_name}")
        validate_endpoint_map(nonpositive)
        nonpositive_lambdas.add(float(nonpositive["lam"]))
    if len(nonpositive_lambdas) != 1:
        raise ValueError("receiving nonpositive maps must share one lambda")
    positive_lambdas: set[float] = set()
    for group in POSITIVE_GROUPS:
        positive = positive_maps[group]
        if not isinstance(positive, dict):
            raise ValueError(f"missing positive CDF map for {group}")
        validate_endpoint_map(positive)
        positive_lambdas.add(float(positive["lam"]))
    if len(positive_lambdas) != 1:
        raise ValueError("receiving positive maps must share one lambda")
    if fitted.get("probability_pool") != fixed_pool_blob():
        raise ValueError("receiving probability pool must match the fixed policy prior")
    return fitted
