"""Input and artifact validation for the affine strategy.

Split out of ``_validation`` so each variant's validators stay a focused module.
The only shared primitive is ``finite_vector``; everything here — the finite
book-probability requirement, the casefold missing-identifier predicate, and the
affine/pool blob schema — is affine-specific and not byte-identical to the
two-part strategy's checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from sportstradamus.training.group_conditional_cdf._contracts import (
    AFFINE_INTERCEPT_BOUNDS,
    AFFINE_POSITION_CODES,
    AFFINE_SCHEMA_VERSION,
    AFFINE_SLOPE_BOUNDS,
    AFFINE_STRATEGY_NAME,
    LEGACY_AFFINE_SCHEMA_VERSION,
    MARGINAL_MEAN_FLOOR,
    PIT_LAMBDAS,
    PLAYER_CV_FOLDS,
    TEMPERATURE_BOUNDS,
    AffineCalibrationBlob,
    AffinePredictive,
    GroupCdfBlob,
)
from sportstradamus.training.group_conditional_cdf._validation import (
    authentic_vector,
    finite_vector,
    optional_book_vector,
)
from sportstradamus.training.group_conditional_cdf.cdf_map_validation import cdf_map_blob_is_valid


def affine_positions(
    values,
    length: int,
    *,
    expected_codes: tuple[int, ...] | None = None,
    legacy: bool = False,
) -> np.ndarray:
    raw = finite_vector(values, "position", length)
    positions = raw.astype(int)
    if not np.array_equal(raw, positions):
        raise ValueError("position must contain integer league roster codes")
    if legacy and not np.isin(positions, AFFINE_POSITION_CODES).all():
        raise ValueError("position must contain only QB=1 and RB=3 codes")
    if expected_codes is not None:
        expected = _validated_expected_codes(expected_codes)
        missing = set(expected).difference(positions)
        if missing:
            raise ValueError(f"position is missing expected calibration codes {sorted(missing)}")
    return positions


def validated_predictive(predictive: AffinePredictive) -> AffinePredictive:
    mean = finite_vector(predictive.marginal_mean, "marginal_mean")
    n_rows = len(mean)
    sigma = finite_vector(predictive.sigma, "sigma", n_rows)
    alpha = finite_vector(predictive.alpha, "alpha", n_rows)
    gate = finite_vector(predictive.gate, "gate", n_rows)
    if np.any(mean < 0.0) or np.any(sigma <= 0.0):
        raise ValueError("marginal means must be nonnegative and sigma must be positive")
    if np.any((gate < 0.0) | (gate >= 1.0)):
        raise ValueError("gate must lie in [0, 1)")
    return AffinePredictive(mean, sigma, alpha, gate)


def affine_apply_inputs(
    predictive,
    line,
    book_over,
    position,
    authentic=None,
    *,
    expected_codes: tuple[int, ...] | None = None,
    legacy: bool = False,
):
    pred = validated_predictive(predictive)
    n_rows = len(pred.marginal_mean)
    lines = finite_vector(line, "line", n_rows)
    positions = affine_positions(position, n_rows, expected_codes=expected_codes, legacy=legacy)
    if legacy:
        books = finite_vector(book_over, "book_over", n_rows)
        if np.any((books < 0.0) | (books > 1.0)):
            raise ValueError("book_over must lie in [0, 1]")
        authentic_rows = np.ones(n_rows, dtype=bool)
    else:
        if authentic is None:
            raise ValueError("schema-2 affine application requires explicit authenticity")
        books = optional_book_vector(book_over, n_rows)
        authentic_rows = authentic_vector(authentic, n_rows)
        if not np.isfinite(books[authentic_rows]).all() or np.any(
            (books[authentic_rows] <= 0.0) | (books[authentic_rows] >= 1.0)
        ):
            raise ValueError("authentic book_over must be finite and strictly inside (0, 1)")
    return pred, lines, books, positions, authentic_rows


def distribution_inputs(
    predictive,
    point,
    position,
    *,
    expected_codes: tuple[int, ...] | None = None,
    legacy: bool = False,
):
    pred = validated_predictive(predictive)
    n_rows = len(pred.marginal_mean)
    return (
        pred,
        finite_vector(point, "point", n_rows),
        affine_positions(position, n_rows, expected_codes=expected_codes, legacy=legacy),
    )


def affine_fit_inputs(
    predictive,
    result,
    line,
    book_over,
    authentic,
    position,
    player,
    expected_codes,
):
    codes = _validated_expected_codes(expected_codes)
    pred, lines, books, positions, authentic_rows = affine_apply_inputs(
        predictive,
        line,
        book_over,
        position,
        authentic,
        expected_codes=codes,
    )
    y = finite_vector(result, "result", len(lines))
    raw_players = np.asarray(player)
    if raw_players.ndim != 1 or len(raw_players) != len(lines):
        raise ValueError("player must be a row-aligned one-dimensional array")
    if any(missing_identifier_affine(value) for value in raw_players):
        raise ValueError("player identifiers must be nonempty")
    players = raw_players.astype(str)
    if len(np.unique(players)) < PLAYER_CV_FOLDS:
        raise ValueError("nested calibration requires at least five validation players")
    for code in codes:
        if len(np.unique(players[positions == code])) < PLAYER_CV_FOLDS:
            raise ValueError(f"position {code} requires at least five validation players")
    return pred, y, lines, books, authentic_rows, positions, players, codes


def validated_affine_blob(blob: Mapping[str, object]) -> AffineCalibrationBlob:
    # style: allow-complexity — moved verbatim from the affine strategy; the flat
    # guard chain is the byte-for-byte schema contract and must not split.
    from sportstradamus.training.group_conditional_cdf.probability_pool import (
        apply_probability_pool,
    )

    schema = blob.get("schema_version")
    if blob.get("kind") != AFFINE_STRATEGY_NAME or schema not in (
        LEGACY_AFFINE_SCHEMA_VERSION,
        AFFINE_SCHEMA_VERSION,
    ):
        raise ValueError("unknown affine calibration blob kind or schema")
    if blob.get("line_probability_only") is not True:
        raise ValueError("affine calibration must declare its line-only probability layer")
    fitted = cast(AffineCalibrationBlob, blob)
    affine = fitted.get("affine")
    maps = fitted.get("position_cdf")
    if not isinstance(affine, dict) or affine.get("kind") != "affine_marginal_mean":
        raise ValueError("invalid affine calibration blob")
    intercept = finite_scalar(affine.get("intercept"), "affine intercept")
    slope = finite_scalar(affine.get("slope"), "affine slope")
    floor = finite_scalar(affine.get("floor"), "affine floor")
    legacy = schema == LEGACY_AFFINE_SCHEMA_VERSION
    bounds = _validated_affine_bounds(fitted, legacy=legacy)
    intercept_bounds = bounds["intercept"]
    slope_bounds = bounds["slope"]
    if not intercept_bounds[0] <= intercept <= intercept_bounds[1]:
        raise ValueError("affine intercept is outside the calibrated bounds")
    if not slope_bounds[0] <= slope <= slope_bounds[1]:
        raise ValueError("affine slope is outside the calibrated bounds")
    if floor != MARGINAL_MEAN_FLOOR:
        raise ValueError("affine floor does not match the calibration schema")
    if legacy:
        codes = AFFINE_POSITION_CODES
        if not isinstance(maps, dict) or set(maps) != {str(code) for code in codes}:
            raise ValueError("affine calibration requires QB and RB CDF maps")
    else:
        codes = _validated_position_codes_field(fitted)
        if not isinstance(maps, dict) or set(maps) != {str(code) for code in codes}:
            raise ValueError("affine calibration maps must match its persisted position codes")
        _validate_generalized_affine_contract(fitted, codes)
    validate_map_blobs(maps)
    temperature_value = finite_scalar(fitted.get("temperature"), "temperature")
    if not TEMPERATURE_BOUNDS[0] <= temperature_value <= TEMPERATURE_BOUNDS[1]:
        raise ValueError("invalid temperature in affine calibration blob")
    pool = fitted.get("probability_pool")
    if not isinstance(pool, dict):
        raise ValueError("affine calibration is missing its probability pool")
    apply_probability_pool(pool, np.array([0.5]), np.array([0.5]))
    rho = finite_scalar(pool.get("rho"), "probability-pool rho")
    raw_rho = finite_scalar(pool.get("raw_rho"), "probability-pool raw_rho")
    if not np.isclose(rho, np.clip(raw_rho, 0.0, 1.0), rtol=0.0, atol=1e-12):
        raise ValueError("probability-pool rho is inconsistent with raw_rho")
    return fitted


def _validated_expected_codes(values) -> tuple[int, ...]:
    raw = tuple(values)
    if (
        not raw
        or any(isinstance(code, bool) or not isinstance(code, (int, np.integer)) for code in raw)
        or tuple(sorted({int(code) for code in raw})) != tuple(int(code) for code in raw)
    ):
        raise ValueError("expected position codes must be a sorted unique integer sequence")
    return tuple(int(code) for code in raw)


def _validated_position_codes_field(fitted: Mapping[str, object]) -> tuple[int, ...]:
    codes = fitted.get("position_codes")
    if not isinstance(codes, list):
        raise ValueError("schema-2 affine calibration is missing persisted position codes")
    return _validated_expected_codes(codes)


def affine_bounds_for_training(mean: np.ndarray, result: np.ndarray) -> dict[str, object]:
    """Return schema-2 bounds whose dimensional intercept follows the train target scale."""
    values = np.concatenate((np.abs(np.asarray(mean, dtype=float)), np.abs(np.asarray(result, dtype=float))))
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("affine bound scale requires finite training values")
    scale = max(1.0, float(np.quantile(values, 0.95)))
    limit = 2.0 * scale
    return {
        "kind": "train_scale_affine_bounds",
        "train_abs_p95": scale,
        "intercept": [-limit, limit],
        "slope": list(AFFINE_SLOPE_BOUNDS),
    }


def _validated_affine_bounds(
    fitted: Mapping[str, object], *, legacy: bool
) -> dict[str, tuple[float, float]]:
    if legacy:
        return {"intercept": AFFINE_INTERCEPT_BOUNDS, "slope": AFFINE_SLOPE_BOUNDS}
    raw = fitted.get("affine_bounds")
    if not isinstance(raw, dict) or raw.get("kind") != "train_scale_affine_bounds":
        raise ValueError("schema-2 affine calibration is missing train-scale bounds")
    scale = finite_scalar(raw.get("train_abs_p95"), "affine train scale")
    intercept = _finite_bound_pair(raw.get("intercept"), "affine intercept bounds")
    slope = _finite_bound_pair(raw.get("slope"), "affine slope bounds")
    if scale < 1.0 or not np.allclose(intercept, (-2.0 * scale, 2.0 * scale), rtol=0, atol=1e-12):
        raise ValueError("affine intercept bounds are inconsistent with the train scale")
    if slope != AFFINE_SLOPE_BOUNDS:
        raise ValueError("affine slope bounds do not match the schema")
    return {"intercept": intercept, "slope": slope}


def _finite_bound_pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"invalid {name} in affine calibration blob")
    low = finite_scalar(value[0], name)
    high = finite_scalar(value[1], name)
    if low >= high:
        raise ValueError(f"invalid {name} in affine calibration blob")
    return low, high


def _validate_generalized_affine_contract(
    fitted: Mapping[str, object], codes: tuple[int, ...]
) -> None:
    fallback = fitted.get("fallback")
    if fallback != {
        "kind": "model_only",
        "unseen_position": "raw_cdf_unpooled",
        "nonauthentic_quote": "candidate_unpooled",
    }:
        raise ValueError("schema-2 affine calibration has an unsupported fallback contract")
    audit = fitted.get("fit_audit")
    if not isinstance(audit, dict) or audit.get("expected_position_codes") != list(codes):
        raise ValueError("schema-2 affine calibration is missing its code-set fit audit")
    folds = audit.get("outer_folds")
    if not isinstance(folds, list) or len(folds) != PLAYER_CV_FOLDS:
        raise ValueError("schema-2 affine calibration requires five outer-fold support audits")


def validate_map_blobs(maps: Mapping[str, GroupCdfBlob]) -> None:
    for code, blob in maps.items():
        valid = cdf_map_blob_is_valid(
            blob,
            kind="group_empirical_cdf",
            allowed_lambdas=PIT_LAMBDAS,
        )
        if not valid:
            raise ValueError(f"invalid grouped CDF map for position {code}")


def finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"invalid {name} in affine calibration blob")
    return float(value)


def missing_identifier_affine(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.casefold() in {"nan", "<na>", "nat"}
