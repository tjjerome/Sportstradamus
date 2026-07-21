"""Unified group-conditional CDF calibration family.

One staged, config-driven engine backs both NFL yards calibrators. A
:class:`StrategyConfig` selects the stages: affine mean correction (rushing), the
per-role logit boundary with RB residual (receiving), the always-on per-group
endpoint-preserving isotonic map, and the always-on line-only temperature plus
book/model pool. Group codes are discovered from data and persisted as the map
keys. The old ``nfl_receiving_two_part_groupcdf`` and ``nfl_rushing_affine_groupcdf``
public names are re-exported here as thin config-constructing wrappers so callers
can re-point at this package without behavior change; the two blob slugs and
schema versions stay byte-identical.
"""

from __future__ import annotations

import numpy as np

from sportstradamus.training.group_conditional_cdf._apply import (
    apply_receiving_two_part_cdf,
    apply_receiving_two_part_line,
    apply_rushing_affine_groupcdf,
    deserialize_receiving_calibration,
    deserialize_rushing_calibration,
    receiving_two_part_cdf_endpoints,
    receiving_two_part_randomized_pit,
    rushing_cdf_endpoints,
    rushing_randomized_pit,
    serialize_receiving_calibration,
    serialize_rushing_calibration,
)
from sportstradamus.training.group_conditional_cdf._config import (
    RECEIVING_CONFIG,
    RUSHING_CONFIG,
    StrategyConfig,
)
from sportstradamus.training.group_conditional_cdf._contracts import (
    MARGINAL_MEAN_FLOOR,
    PLAYER_CV_FOLDS,
    RANDOMIZED_PIT_DRAWS,
    RECEIVING_CANDIDATE_NAME,
    RECEIVING_POSITION_CODES,
    RECEIVING_SCHEMA_VERSION,
    ROLE_VALUES,
    RUSHING_CANDIDATE_NAME,
    RUSHING_POSITION_CODES,
    RUSHING_SCHEMA_VERSION,
    AffineMeanBlob,
    EndpointMapBlob,
    GroupCdfBlob,
    ReceivingCalibrationBlob,
    ReceivingCalibrationFit,
    ReceivingLineOutput,
    RushingCalibrationBlob,
    RushingCalibrationFit,
    RushingCalibrationOutput,
    RushingPredictive,
)
from sportstradamus.training.group_conditional_cdf._fit import fit_group_conditional_cdf

CANDIDATE_NAME = RECEIVING_CANDIDATE_NAME
SCHEMA_VERSION = RECEIVING_SCHEMA_VERSION
POSITION_CODES = RECEIVING_POSITION_CODES


def fit_receiving_two_part_groupcdf(
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
) -> ReceivingCalibrationFit:
    """Fit v3 by nested five-fold Player CV on the validation split.

    Line settlement is endpoint-first: callers provide the raw CDF at
    ``ceil(line - 1)`` and ``floor(line + 1)``. Temperature is fit in each outer
    training fold on their raw arithmetic settlement, then applied only after
    both endpoints have been mapped. Only explicitly authentic book quotes use
    the fixed 80/20 pool; every other row remains model-only.
    """
    return fit_group_conditional_cdf(
        RECEIVING_CONFIG,
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


def fit_rushing_affine_groupcdf(
    predictive: RushingPredictive,
    result: np.ndarray,
    line: np.ndarray,
    book_over: np.ndarray,
    position: np.ndarray,
    player: np.ndarray,
) -> RushingCalibrationFit:
    """Fit the rushing candidate by nested five-fold Player CV, then all validation rows.

    Outer folds generate honest mean, line-probability, and randomized-PIT arrays
    for the six-gate scorecard. Within each outer training partition, a second
    Player-grouped CV selects QB and RB map shrinkage. The affine fit, map,
    temperature, and pool weight never inspect the outer held-out players.
    """
    return fit_group_conditional_cdf(
        RUSHING_CONFIG, predictive, result, line, book_over, position, player
    )


__all__ = (
    "CANDIDATE_NAME",
    "MARGINAL_MEAN_FLOOR",
    "PLAYER_CV_FOLDS",
    "POSITION_CODES",
    "RANDOMIZED_PIT_DRAWS",
    "RECEIVING_CANDIDATE_NAME",
    "RECEIVING_CONFIG",
    "RECEIVING_POSITION_CODES",
    "RECEIVING_SCHEMA_VERSION",
    "ROLE_VALUES",
    "RUSHING_CANDIDATE_NAME",
    "RUSHING_CONFIG",
    "RUSHING_POSITION_CODES",
    "RUSHING_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "AffineMeanBlob",
    "EndpointMapBlob",
    "GroupCdfBlob",
    "ReceivingCalibrationBlob",
    "ReceivingCalibrationFit",
    "ReceivingLineOutput",
    "RushingCalibrationBlob",
    "RushingCalibrationFit",
    "RushingCalibrationOutput",
    "RushingPredictive",
    "StrategyConfig",
    "apply_receiving_two_part_cdf",
    "apply_receiving_two_part_line",
    "apply_rushing_affine_groupcdf",
    "deserialize_receiving_calibration",
    "deserialize_rushing_calibration",
    "fit_group_conditional_cdf",
    "fit_receiving_two_part_groupcdf",
    "fit_rushing_affine_groupcdf",
    "receiving_two_part_cdf_endpoints",
    "receiving_two_part_randomized_pit",
    "rushing_cdf_endpoints",
    "rushing_randomized_pit",
    "serialize_receiving_calibration",
    "serialize_rushing_calibration",
)
