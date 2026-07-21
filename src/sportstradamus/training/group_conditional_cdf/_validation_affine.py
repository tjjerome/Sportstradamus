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
    MARGINAL_MEAN_FLOOR,
    PIT_LAMBDAS,
    PLAYER_CV_FOLDS,
    TEMPERATURE_BOUNDS,
    AffineCalibrationBlob,
    AffinePredictive,
    GroupCdfBlob,
)
from sportstradamus.training.group_conditional_cdf._validation import finite_vector
from sportstradamus.training.group_conditional_cdf.cdf_map_validation import cdf_map_blob_is_valid


def affine_positions(values, length: int) -> np.ndarray:
    raw = finite_vector(values, "position", length)
    positions = raw.astype(int)
    if not np.array_equal(raw, positions) or not np.isin(positions, AFFINE_POSITION_CODES).all():
        raise ValueError("position must contain only QB=1 and RB=3 codes")
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


def affine_apply_inputs(predictive, line, book_over, position):
    pred = validated_predictive(predictive)
    n_rows = len(pred.marginal_mean)
    lines = finite_vector(line, "line", n_rows)
    books = finite_vector(book_over, "book_over", n_rows)
    if np.any((books < 0.0) | (books > 1.0)):
        raise ValueError("book_over must lie in [0, 1]")
    return pred, lines, books, affine_positions(position, n_rows)


def distribution_inputs(predictive, point, position):
    pred = validated_predictive(predictive)
    n_rows = len(pred.marginal_mean)
    return pred, finite_vector(point, "point", n_rows), affine_positions(position, n_rows)


def affine_fit_inputs(predictive, result, line, book_over, position, player):
    pred, lines, books, positions = affine_apply_inputs(predictive, line, book_over, position)
    y = finite_vector(result, "result", len(lines))
    raw_players = np.asarray(player)
    if raw_players.ndim != 1 or len(raw_players) != len(lines):
        raise ValueError("player must be a row-aligned one-dimensional array")
    if any(missing_identifier_affine(value) for value in raw_players):
        raise ValueError("player identifiers must be nonempty")
    players = raw_players.astype(str)
    if len(np.unique(players)) < PLAYER_CV_FOLDS:
        raise ValueError("nested calibration requires at least five validation players")
    for code in AFFINE_POSITION_CODES:
        if len(np.unique(players[positions == code])) < PLAYER_CV_FOLDS:
            raise ValueError(f"position {code} requires at least five validation players")
    return pred, y, lines, books, positions, players


def validated_affine_blob(blob: Mapping[str, object]) -> AffineCalibrationBlob:
    # style: allow-complexity — moved verbatim from the affine strategy; the flat
    # guard chain is the byte-for-byte schema contract and must not split.
    from sportstradamus.training.group_conditional_cdf.probability_pool import (
        apply_probability_pool,
    )

    if (
        blob.get("kind") != AFFINE_STRATEGY_NAME
        or blob.get("schema_version") != AFFINE_SCHEMA_VERSION
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
    if not AFFINE_INTERCEPT_BOUNDS[0] <= intercept <= AFFINE_INTERCEPT_BOUNDS[1]:
        raise ValueError("affine intercept is outside the calibrated bounds")
    if not AFFINE_SLOPE_BOUNDS[0] <= slope <= AFFINE_SLOPE_BOUNDS[1]:
        raise ValueError("affine slope is outside the calibrated bounds")
    if floor != MARGINAL_MEAN_FLOOR:
        raise ValueError("affine floor does not match the calibration schema")
    if not isinstance(maps, dict) or set(maps) != {str(code) for code in AFFINE_POSITION_CODES}:
        raise ValueError("affine calibration requires QB and RB CDF maps")
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
