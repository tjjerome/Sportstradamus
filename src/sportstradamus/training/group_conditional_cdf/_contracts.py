"""Constants and portable state contracts for group-conditional CDF calibration.

This module unifies the frozen contracts of the two structural calibrators. The
scalar constants that genuinely differ between the two-part strategy and
the affine strategy (candidate slug, group codes, schema version) carry a
variant prefix; everything the two variants agree on is shared verbatim. The blob
TypedDicts and dataclasses of both variants live here unchanged so their JSON and
pickle layout is byte-identical to the packages this replaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np

from sportstradamus.helpers.integer_distribution import RANDOMIZED_PIT_DRAWS, RANDOMIZED_PIT_SEED
from sportstradamus.training.group_conditional_cdf.probability_pool import ProbabilityPoolBlob

TWO_PART_STRATEGY_NAME = "role-position-two-part-groupcdf-fixedlinear-v3"
TWO_PART_SCHEMA_VERSION = 3
AFFINE_STRATEGY_NAME = "affine-groupcdf-bookpool-v1"
AFFINE_SCHEMA_VERSION = 1

ROLE_VALUES: tuple[str, str] = ("low", "high")
TWO_PART_POSITION_CODES: tuple[int, int, int] = (2, 3, 4)
AFFINE_POSITION_CODES: tuple[int, int] = (1, 3)

PLAYER_CV_FOLDS = 5
PIT_BINS = 10
PIT_LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
PIT_LAMBDA_TIE_TOLERANCE = 1e-4

CDF_CLIP = 1e-12
BOUNDARY_LOGIT_CLIP = 1e-9
CDF_BRANCH_TOLERANCE = 1e-14
INTERCEPT_BRACKET: tuple[float, float] = (-30.0, 30.0)
TEMPERATURE_BOUNDS: tuple[float, float] = (1.0, 10.0)
TEMPERATURE_REGULARIZATION = 0.01
PROBABILITY_CLIP = 1e-6
FIXED_MODEL_WEIGHT = 0.2
FIXED_BOOK_WEIGHT = 0.8

AFFINE_INTERCEPT_BOUNDS: tuple[float, float] = (-20.0, 20.0)
AFFINE_SLOPE_BOUNDS: tuple[float, float] = (0.5, 1.5)
MARGINAL_MEAN_FLOOR = 0.1

GLOBAL_PIT_GUARD = 0.05
ROLE_FULL_PIT_GUARDS: dict[str, float] = {"low": 0.09, "high": 0.09}
ROLE_POSITIVE_PIT_GUARDS: dict[str, float] = {"low": 0.08, "high": 0.09}
POSITION_FULL_PIT_GUARD = 0.08
POSITION_POSITIVE_PIT_GUARD = 0.09


class EndpointMapBlob(TypedDict):
    """Portable endpoint-preserving empirical-CDF map for one conditional side."""

    kind: Literal["isotonic_pit"]
    lam: float
    x: list[float]
    y: list[float]


class GroupCdfBlob(TypedDict):
    """Portable endpoint-preserving empirical-CDF map for one position."""

    kind: Literal["group_empirical_cdf"]
    lam: float
    x: list[float]
    y: list[float]


class RoleBoundaryBlob(TypedDict):
    """Portable boundary intercept and nonpositive conditional map."""

    kind: Literal["two_part_role_boundary"]
    intercept: float
    nonpositive: EndpointMapBlob


class TwoPartCdfBlob(TypedDict):
    """The role-boundary and role-by-position positive CDF layer.

    ``rb_boundary_residual`` is one shared logit offset (heritage name: it was
    first fit for the NFL RB position) applied to the roster codes persisted in
    ``boundary_residual_positions``. An empty list ⇒ no residual, so cells that do
    not opt in serve the plain per-role boundary.
    """

    kind: Literal["role_position_two_part_cdf"]
    role_boundary: dict[str, RoleBoundaryBlob]
    positive: dict[str, EndpointMapBlob]
    rb_boundary_residual: float
    boundary_residual_positions: list[int]


class FixedProbabilityPoolBlob(TypedDict):
    """Auditable policy prior for authentic quoted-line probabilities."""

    kind: Literal["fixed_linear_probability_pool"]
    model_weight: float
    book_weight: float
    source: Literal["policy_prior"]
    estimated: Literal[False]
    authenticity: str
    missing_book: Literal["unpooled_model_fallback"]


class TwoPartCalibrationBlob(TypedDict):
    """JSON/pickle-safe state fitted only from the full validation split."""

    kind: Literal["role-position-two-part-groupcdf-fixedlinear-v3"]
    schema_version: int
    line_probability_only: bool
    temperature_fit_scope: Literal["pre_map_raw_endpoint_settlement"]
    temperature: float
    cdf: TwoPartCdfBlob
    probability_pool: FixedProbabilityPoolBlob


class AffineMeanBlob(TypedDict):
    """Portable affine marginal-mean correction."""

    kind: Literal["affine_marginal_mean"]
    intercept: float
    slope: float
    floor: float


class AffineCalibrationBlob(TypedDict):
    """JSON/pickle-safe state fitted only from the full validation split."""

    kind: Literal["affine-groupcdf-bookpool-v1"]
    schema_version: int
    line_probability_only: bool
    affine: AffineMeanBlob
    position_cdf: dict[str, GroupCdfBlob]
    temperature: float
    probability_pool: ProbabilityPoolBlob


@dataclass(frozen=True)
class TwoPartLineOutput:
    """Mapped quoted-line probabilities before and after bookmaker pooling."""

    mapped_low: np.ndarray
    mapped_high: np.ndarray
    mapped_under: np.ndarray
    candidate_over: np.ndarray
    pooled_over: np.ndarray


@dataclass(frozen=True)
class AffinePredictive:
    """Row-aligned pre-calibration zero-inflated SkewNormal parameters.

    ``marginal_mean`` includes the zero gate. ``sigma`` and ``alpha`` describe
    the conditional SkewNormal, while ``gate`` is its zero-atom probability.
    """

    marginal_mean: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    gate: np.ndarray


@dataclass(frozen=True)
class AffineCalibrationOutput:
    """Applied distribution parameters and quoted-line probabilities."""

    marginal_mean: np.ndarray
    conditional_mean: np.ndarray
    loc: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    gate: np.ndarray
    candidate_over: np.ndarray
    pooled_over: np.ndarray


@dataclass(frozen=True)
class TwoPartCalibrationFit:
    """Full-validation blob plus honest outer-Player-CV scorecard inputs."""

    blob: TwoPartCalibrationBlob
    oof_pit_draws: np.ndarray
    oof_positive_pit: np.ndarray
    oof_mapped_low: np.ndarray
    oof_mapped_high: np.ndarray
    oof_mapped_under: np.ndarray
    oof_candidate_over: np.ndarray
    oof_pooled_over: np.ndarray
    fold_lambdas: tuple[tuple[float, float], ...]
    fold_intercepts: tuple[dict[str, float], ...]
    fold_temperatures: tuple[float, ...]
    fold_selection_keys: tuple[tuple[float, float, float, float, float], ...]


@dataclass(frozen=True)
class AffineCalibrationFit:
    """Full-validation blob plus honest outer-Player-CV gate inputs."""

    blob: AffineCalibrationBlob
    oof_marginal_mean: np.ndarray
    oof_candidate_over: np.ndarray
    oof_pooled_over: np.ndarray
    oof_pit_draws: np.ndarray
    fold_affines: tuple[tuple[float, float], ...]
    fold_lambdas: tuple[tuple[float, float], ...]
    fold_temperatures: tuple[float, ...]
    fold_rhos: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationRows:
    result_upper: np.ndarray
    result_lower: np.ndarray
    zero_cdf: np.ndarray
    line_low: np.ndarray
    line_high: np.ndarray
    result: np.ndarray
    over_result: np.ndarray
    book_over: np.ndarray
    authentic: np.ndarray
    player: np.ndarray
    role: np.ndarray
    position: np.ndarray


@dataclass(frozen=True)
class CrossfitSelection:
    lambdas: tuple[float, float]
    key: tuple[float, float, float, float, float]
