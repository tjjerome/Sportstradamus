"""Application and serialization for the group-conditional CDF family.

Every public entry point recovers the discovered group set from the fitted blob
(the ``position_cdf`` / ``positive`` map keys) so application iterates exactly the
codes the fit persisted. The two-part and affine strategies keep their own apply
bodies — a two-part conditional transform and an affine-plus-grouped transform —
selected by the blob kind.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from sportstradamus.helpers.distributions import (
    apply_temperature,
    skewnormal_loc_from_mean,
)
from sportstradamus.training.group_conditional_cdf._affine import (
    apply_affine,
    mapped_cdf_endpoints,
    mapped_over,
)
from sportstradamus.training.group_conditional_cdf._boundary import apply_models
from sportstradamus.training.group_conditional_cdf._contracts import (
    CDF_BRANCH_TOLERANCE,
    RANDOMIZED_PIT_DRAWS,
    RANDOMIZED_PIT_SEED,
    AffineCalibrationBlob,
    AffineCalibrationOutput,
    AffinePredictive,
    TwoPartCalibrationBlob,
    TwoPartLineOutput,
)
from sportstradamus.training.group_conditional_cdf._pool import pooled_probability
from sportstradamus.training.group_conditional_cdf._validation import (
    authentic_vector,
    cdf_apply_inputs,
    optional_book_vector,
    validated_two_part_blob,
)
from sportstradamus.training.group_conditional_cdf._validation_affine import (
    affine_apply_inputs,
    distribution_inputs,
    validated_affine_blob,
)
from sportstradamus.training.group_conditional_cdf.probability_pool import apply_probability_pool


def _blob_positive_groups(blob: TwoPartCalibrationBlob):
    return tuple(blob["cdf"]["positive"].keys())


def _blob_residual_positions(blob: TwoPartCalibrationBlob) -> list[int]:
    return blob["cdf"]["boundary_residual_positions"]


def _blob_grouping(blob: TwoPartCalibrationBlob) -> str:
    return blob["cdf"].get("grouping", "role_by_position")


def _blob_position_codes(blob: AffineCalibrationBlob) -> tuple[int, ...]:
    return tuple(sorted(int(code) for code in blob["position_cdf"]))


def apply_two_part_cdf(
    blob: Mapping[str, object],
    cdf: np.ndarray,
    f0: np.ndarray,
    role: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """Apply the fitted two-part transform to arbitrary row-aligned CDF values.

    The function is the complete distribution-CDF layer. It intentionally does
    not apply temperature or the bookmaker pool, both of which are calibrated
    only for a quoted binary over probability.
    """
    fitted = validated_two_part_blob(blob)
    values, boundary, roles, positions = cdf_apply_inputs(cdf, f0, role, position)
    return apply_models(
        fitted["cdf"],
        values,
        boundary,
        roles,
        positions,
        _blob_positive_groups(fitted),
        _blob_residual_positions(fitted),
        _blob_grouping(fitted),
    )


def two_part_cdf_endpoints(
    blob: Mapping[str, object],
    lower_cdf: np.ndarray,
    upper_cdf: np.ndarray,
    f0: np.ndarray,
    role: np.ndarray,
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform lower and upper endpoints separately, preserving the zero atom."""
    lower = apply_two_part_cdf(blob, lower_cdf, f0, role, position)
    upper = apply_two_part_cdf(blob, upper_cdf, f0, role, position)
    if np.any(lower > upper + CDF_BRANCH_TOLERANCE):
        raise ValueError("transformed lower CDF endpoint exceeds upper endpoint")
    return lower, upper


def two_part_randomized_pit(
    blob: Mapping[str, object],
    lower_cdf: np.ndarray,
    upper_cdf: np.ndarray,
    f0: np.ndarray,
    role: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """Generate the exact seeded 25-draw PIT matrix scored by Gate 4."""
    lower, upper = two_part_cdf_endpoints(blob, lower_cdf, upper_cdf, f0, role, position)
    uniforms = np.random.default_rng(RANDOMIZED_PIT_SEED).random((RANDOMIZED_PIT_DRAWS, len(lower)))
    return lower[None, :] + uniforms * (upper - lower)[None, :]


def apply_two_part_line(
    blob: Mapping[str, object],
    line_cdf_low: np.ndarray,
    line_cdf_high: np.ndarray,
    f0: np.ndarray,
    role: np.ndarray,
    position: np.ndarray,
    book_over: np.ndarray,
    authentic: np.ndarray,
) -> TwoPartLineOutput:
    """Apply the CDF, fixed temperature, and quoted-line probability pool.

    ``pooled_over`` is a per-quote binary head. It is not the survival function
    of the two-part distribution and must not be used to compute PIT or means.
    """
    fitted = validated_two_part_blob(blob)
    mapped_low = apply_two_part_cdf(fitted, line_cdf_low, f0, role, position)
    mapped_high = apply_two_part_cdf(fitted, line_cdf_high, f0, role, position)
    if np.any(mapped_low > mapped_high + CDF_BRANCH_TOLERANCE):
        raise ValueError("mapped line lower endpoint exceeds upper endpoint")
    mapped_under = 0.5 * (mapped_low + mapped_high)
    books = optional_book_vector(book_over, len(mapped_under))
    authentic_rows = authentic_vector(authentic, len(mapped_under))
    if not np.isfinite(books[authentic_rows]).all() or np.any(
        (books[authentic_rows] <= 0.0) | (books[authentic_rows] >= 1.0)
    ):
        raise ValueError("authentic book_over must be finite and strictly inside (0, 1)")
    candidate = apply_temperature(1.0 - mapped_under, fitted["temperature"])
    pooled = pooled_probability(candidate, books, authentic_rows)
    return TwoPartLineOutput(mapped_low, mapped_high, mapped_under, candidate, pooled)


def serialize_two_part_calibration(blob: Mapping[str, object]) -> str:
    """Serialize validated calibration state to deterministic portable JSON."""
    fitted = validated_two_part_blob(blob)
    return json.dumps(fitted, sort_keys=True, separators=(",", ":"), allow_nan=False)


def deserialize_two_part_calibration(payload: str) -> TwoPartCalibrationBlob:
    """Load and validate portable JSON calibration state."""
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("two-part calibration payload must contain a JSON object")
    return validated_two_part_blob(decoded)


def apply_affine_groupcdf(
    blob: Mapping[str, object],
    predictive: AffinePredictive,
    line: np.ndarray,
    book_over: np.ndarray,
    position: np.ndarray,
) -> AffineCalibrationOutput:
    """Apply the frozen validation fit to validation, held-out test, or live arrays.

    The returned ``candidate_over`` includes the grouped CDF map and temperature;
    ``pooled_over`` additionally includes the posted-line book pool. The reported
    mean and distribution parameters stop before those probability-only layers.
    """
    fitted = validated_affine_blob(blob)
    codes = _blob_position_codes(fitted)
    pred, lines, books, positions = affine_apply_inputs(predictive, line, book_over, position)
    affine = fitted["affine"]
    corrected = apply_affine(pred, affine["intercept"], affine["slope"], affine["floor"])
    candidate = apply_temperature(
        mapped_over(corrected, lines, positions, fitted["position_cdf"], codes),
        fitted["temperature"],
    )
    pooled = apply_probability_pool(fitted["probability_pool"], candidate, books)
    conditional = corrected.marginal_mean / (1.0 - corrected.gate)
    loc = skewnormal_loc_from_mean(conditional, corrected.sigma, corrected.alpha)
    return AffineCalibrationOutput(
        marginal_mean=corrected.marginal_mean,
        conditional_mean=conditional,
        loc=loc,
        sigma=corrected.sigma,
        alpha=corrected.alpha,
        gate=corrected.gate,
        candidate_over=candidate,
        pooled_over=pooled,
    )


def affine_cdf_endpoints(
    blob: Mapping[str, object],
    predictive: AffinePredictive,
    point: np.ndarray,
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grouped-CDF values immediately below and at row-aligned points.

    Mapping the two endpoints separately preserves the transformed mass of the
    zero atom. Temperature and the line-only book pool are intentionally absent.
    """
    fitted = validated_affine_blob(blob)
    codes = _blob_position_codes(fitted)
    pred, points, positions = distribution_inputs(predictive, point, position)
    affine = fitted["affine"]
    corrected = apply_affine(pred, affine["intercept"], affine["slope"], affine["floor"])
    return mapped_cdf_endpoints(corrected, points, positions, fitted["position_cdf"], codes)


def affine_randomized_pit(
    blob: Mapping[str, object],
    predictive: AffinePredictive,
    result: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """Generate the exact seeded 25-draw PIT matrix scored by Gate 4."""
    lower, upper = affine_cdf_endpoints(blob, predictive, result, position)
    uniforms = np.random.default_rng(RANDOMIZED_PIT_SEED).random((RANDOMIZED_PIT_DRAWS, len(lower)))
    return lower[None, :] + uniforms * (upper - lower)[None, :]


def serialize_affine_calibration(blob: Mapping[str, object]) -> str:
    """Serialize validated calibration state to deterministic portable JSON."""
    fitted = validated_affine_blob(blob)
    return json.dumps(fitted, sort_keys=True, separators=(",", ":"), allow_nan=False)


def deserialize_affine_calibration(payload: str) -> AffineCalibrationBlob:
    """Load and validate portable JSON calibration state."""
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("affine calibration payload must contain a JSON object")
    return validated_affine_blob(decoded)
