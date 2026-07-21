"""Unified group-conditional CDF calibration family.

One staged, config-driven engine backs both structural calibrators. A
:class:`StrategyConfig` selects the stages: affine mean correction (affine), the
per-role logit boundary with RB residual (two-part), the always-on per-group
endpoint-preserving isotonic map, and the always-on line-only temperature plus
book/model pool. Group codes are discovered from data and persisted as the map
keys. The old ``two_part_groupcdf`` and ``affine_groupcdf``
public names are re-exported here as thin config-constructing wrappers so callers
can re-point at this package without behavior change; the two blob slugs and
schema versions stay byte-identical.
"""

from __future__ import annotations

import numpy as np

from sportstradamus.training.group_conditional_cdf._apply import (
    affine_cdf_endpoints,
    affine_randomized_pit,
    apply_affine_groupcdf,
    apply_two_part_cdf,
    apply_two_part_line,
    deserialize_affine_calibration,
    deserialize_two_part_calibration,
    serialize_affine_calibration,
    serialize_two_part_calibration,
    two_part_cdf_endpoints,
    two_part_randomized_pit,
)
from sportstradamus.training.group_conditional_cdf._config import (
    AFFINE_CONFIG,
    TWO_PART_CONFIG,
    StrategyConfig,
)
from sportstradamus.training.group_conditional_cdf._contracts import (
    AFFINE_POSITION_CODES,
    AFFINE_SCHEMA_VERSION,
    AFFINE_STRATEGY_NAME,
    MARGINAL_MEAN_FLOOR,
    PLAYER_CV_FOLDS,
    RANDOMIZED_PIT_DRAWS,
    ROLE_VALUES,
    TWO_PART_POSITION_CODES,
    TWO_PART_SCHEMA_VERSION,
    TWO_PART_STRATEGY_NAME,
    AffineCalibrationBlob,
    AffineCalibrationFit,
    AffineCalibrationOutput,
    AffineMeanBlob,
    AffinePredictive,
    EndpointMapBlob,
    GroupCdfBlob,
    TwoPartCalibrationBlob,
    TwoPartCalibrationFit,
    TwoPartLineOutput,
)
from sportstradamus.training.group_conditional_cdf._fit import fit_group_conditional_cdf
from sportstradamus.training.group_conditional_cdf._maps import ks_supremum

CANDIDATE_NAME = TWO_PART_STRATEGY_NAME
SCHEMA_VERSION = TWO_PART_SCHEMA_VERSION
POSITION_CODES = TWO_PART_POSITION_CODES


def fit_two_part_groupcdf(
    result_cdf_upper: np.ndarray,
    result_cdf_lower: np.ndarray,
    zero_cdf: np.ndarray,
    line_cdf_low: np.ndarray,
    line_cdf_high: np.ndarray,
    result: np.ndarray,
    over_result: np.ndarray,
    book_over: np.ndarray,
    authentic: np.ndarray,
    player: np.ndarray,
    role: np.ndarray,
    position: np.ndarray,
) -> TwoPartCalibrationFit:
    """Fit v3 by nested five-fold Player CV on the validation split.

    Line settlement is endpoint-first: callers provide the raw CDF at
    ``ceil(line - 1)`` and ``floor(line + 1)``. Temperature is fit in each outer
    training fold on their raw arithmetic settlement, then applied only after
    both endpoints have been mapped. Only explicitly authentic book quotes use
    the fixed 80/20 pool; every other row remains model-only.
    """
    return fit_group_conditional_cdf(
        TWO_PART_CONFIG,
        result_cdf_upper,
        result_cdf_lower,
        zero_cdf,
        line_cdf_low,
        line_cdf_high,
        result,
        over_result,
        book_over,
        authentic,
        player,
        role,
        position,
    )


def fit_affine_groupcdf(
    predictive: AffinePredictive,
    result: np.ndarray,
    line: np.ndarray,
    book_over: np.ndarray,
    position: np.ndarray,
    player: np.ndarray,
) -> AffineCalibrationFit:
    """Fit the affine candidate by nested five-fold Player CV, then all validation rows.

    Outer folds generate honest mean, line-probability, and randomized-PIT arrays
    for the six-gate scorecard. Within each outer training partition, a second
    Player-grouped CV selects QB and RB map shrinkage. The affine fit, map,
    temperature, and pool weight never inspect the outer held-out players.
    """
    return fit_group_conditional_cdf(
        AFFINE_CONFIG, predictive, result, line, book_over, position, player
    )


__all__ = (
    "AFFINE_CONFIG",
    "AFFINE_POSITION_CODES",
    "AFFINE_SCHEMA_VERSION",
    "AFFINE_STRATEGY_NAME",
    "CANDIDATE_NAME",
    "MARGINAL_MEAN_FLOOR",
    "PLAYER_CV_FOLDS",
    "POSITION_CODES",
    "RANDOMIZED_PIT_DRAWS",
    "ROLE_VALUES",
    "SCHEMA_VERSION",
    "TWO_PART_CONFIG",
    "TWO_PART_POSITION_CODES",
    "TWO_PART_SCHEMA_VERSION",
    "TWO_PART_STRATEGY_NAME",
    "AffineCalibrationBlob",
    "AffineCalibrationFit",
    "AffineCalibrationOutput",
    "AffineMeanBlob",
    "AffinePredictive",
    "EndpointMapBlob",
    "GroupCdfBlob",
    "StrategyConfig",
    "TwoPartCalibrationBlob",
    "TwoPartCalibrationFit",
    "TwoPartLineOutput",
    "affine_cdf_endpoints",
    "affine_randomized_pit",
    "apply_affine_groupcdf",
    "apply_two_part_cdf",
    "apply_two_part_line",
    "deserialize_affine_calibration",
    "deserialize_two_part_calibration",
    "fit_affine_groupcdf",
    "fit_group_conditional_cdf",
    "fit_two_part_groupcdf",
    "ks_supremum",
    "serialize_affine_calibration",
    "serialize_two_part_calibration",
    "two_part_cdf_endpoints",
    "two_part_randomized_pit",
)
