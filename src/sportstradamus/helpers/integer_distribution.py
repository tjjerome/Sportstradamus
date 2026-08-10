"""Exact CDF endpoints and settlement for nonnegative integer outcomes.

This is the shared parity prerequisite specified by
``/tmp/researcher_structured_count_transfer_20260720.md`` (SHA-256
``413c7e2049ddd9a0918a7d315dd1340f6bc26f22ac9069a1434abae0c002edf2``).
It keeps observation randomization and whole/half-line settlement on the same
integer-lattice convention without embedding any league or market policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Gate 4 averages this fixed draw set so count-cell calibration stays reproducible.
RANDOMIZED_PIT_DRAWS = 25
RANDOMIZED_PIT_SEED = 4517

_LATTICE_TOLERANCE = 1e-12
_PROBABILITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SettlementProbabilities:
    """Mutually exclusive under, push, and over probabilities for each line."""

    under: np.ndarray
    push: np.ndarray
    over: np.ndarray


def observation_cdf_points(outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the integer points for the two sides of each observed CDF jump.

    For an observed count ``y``, the returned arrays are ``y - 1`` and ``y``.
    A nonnegative-integer CDF therefore evaluates them as ``F(y - 1)`` and
    ``F(y)``; in particular, the lower point for a zero outcome is ``-1``.

    Args:
        outcomes: Finite, nonnegative integer observations. The returned arrays
            preserve its shape and contain integer-valued CDF evaluation points.

    Returns:
        The point immediately below each observation and the observation itself.

    Raises:
        ValueError: If an outcome is nonfinite, negative, or nonintegral.
    """
    outcome_array = np.asarray(outcomes, dtype=float)
    if not np.all(np.isfinite(outcome_array)):
        raise ValueError("outcomes must contain only finite values")
    if np.any(outcome_array < 0.0):
        raise ValueError("outcomes must be nonnegative")
    if np.any(outcome_array != np.floor(outcome_array)):
        raise ValueError("outcomes must be integers")

    upper_points = outcome_array.astype(np.int64)
    return upper_points - 1, upper_points


def line_cdf_points(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact lower and upper CDF evaluation points for integer lines.

    The convention is ``ceil(line) - 1`` and ``floor(line)``. At a whole line
    ``k`` these are ``k - 1`` and ``k``; at a half line ``k + 0.5`` both are
    ``k``. Inputs within ``1e-12`` of the half-unit lattice are canonicalized
    before the integer points are formed.

    Args:
        lines: Finite whole- or half-unit market lines.

    Returns:
        Row-aligned lower and upper integer CDF evaluation points.

    Raises:
        ValueError: If a line is nonfinite or is not whole/half-unit valued.
    """
    line_array = np.asarray(lines, dtype=float)
    if not np.all(np.isfinite(line_array)):
        raise ValueError("lines must contain only finite values")

    doubled_lines = 2.0 * line_array
    lattice_index = np.rint(doubled_lines)
    if np.any(
        ~np.isclose(
            doubled_lines,
            lattice_index,
            rtol=0.0,
            atol=_LATTICE_TOLERANCE,
        )
    ):
        raise ValueError("lines must be whole or half-unit values")

    canonical_lines = lattice_index / 2.0
    lower_points = np.ceil(canonical_lines).astype(np.int64) - 1
    upper_points = np.floor(canonical_lines).astype(np.int64)
    return lower_points, upper_points


def randomized_pit(
    outcomes: np.ndarray,
    lower_cdf: np.ndarray,
    upper_cdf: np.ndarray,
    *,
    draws: int = RANDOMIZED_PIT_DRAWS,
    seed: int = RANDOMIZED_PIT_SEED,
) -> np.ndarray:
    """Spread each observed count uniformly across its exact CDF jump.

    The endpoint arrays must be row-aligned evaluations of ``F(y - 1)`` and
    ``F(y)``. The output shape is ``(draws, *outcomes.shape)`` and every draw is
    ``lower + V * (upper - lower)`` for a seeded ``V ~ Uniform(0, 1)``. CDF
    values are validated but never clipped.

    Args:
        outcomes: Finite, nonnegative integer observations.
        lower_cdf: CDF values immediately below the observations.
        upper_cdf: CDF values at the observations.
        draws: Number of randomized PIT replicates.
        seed: NumPy random-generator seed.

    Returns:
        Seeded randomized PIT draws in probability units.

    Raises:
        ValueError: If outcomes or CDF endpoints violate the integer-CDF contract.
    """
    lower_points, upper_points = observation_cdf_points(outcomes)
    lower, upper = _validated_cdf_endpoints(
        lower_cdf,
        upper_cdf,
        lower_points,
        upper_points,
    )
    uniforms = np.random.default_rng(seed).random((draws, *lower.shape))
    return lower[None, ...] + uniforms * (upper - lower)[None, ...]


def settlement_probabilities(
    lines: np.ndarray,
    lower_cdf: np.ndarray,
    upper_cdf: np.ndarray,
) -> SettlementProbabilities:
    """Partition each integer-market line into under, push, and over mass.

    ``lower_cdf`` and ``upper_cdf`` are the CDF evaluated at the points returned
    by :func:`line_cdf_points`. Whole lines produce ``F(k-1)``, the atom at
    ``k``, and ``1-F(k)``. Half lines require the two endpoint values to match,
    so their push probability is exactly zero. Inputs and outputs are
    probabilities in ``[0, 1]`` and are never clipped.

    Args:
        lines: Finite whole- or half-unit market lines.
        lower_cdf: CDF at ``ceil(line) - 1``.
        upper_cdf: CDF at ``floor(line)``.

    Returns:
        Named, mutually exclusive under/push/over probability arrays.

    Raises:
        ValueError: If lines, endpoints, or the resulting probability partition
            violate the integer-settlement contract.
    """
    lower_points, upper_points = line_cdf_points(lines)
    lower, upper = _validated_cdf_endpoints(
        lower_cdf,
        upper_cdf,
        lower_points,
        upper_points,
    )

    half_lines = lower_points == upper_points
    if np.any(half_lines & (lower != upper)):
        raise ValueError("half-line CDF endpoints must match")

    under = lower
    push = upper - lower
    over = 1.0 - upper
    total = under + push + over
    if not np.allclose(total, np.ones_like(total), rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError("under, push, and over probabilities must sum to one")
    return SettlementProbabilities(under=under, push=push, over=over)


def _validated_cdf_endpoints(
    lower_cdf: np.ndarray,
    upper_cdf: np.ndarray,
    lower_points: np.ndarray,
    upper_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower_cdf, dtype=float)
    upper = np.asarray(upper_cdf, dtype=float)
    if lower.shape != lower_points.shape or upper.shape != upper_points.shape:
        raise ValueError("CDF endpoints must match the observation or line shape")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("CDF endpoints must contain only finite values")
    if np.any((lower < 0.0) | (lower > 1.0) | (upper < 0.0) | (upper > 1.0)):
        raise ValueError("CDF endpoints must lie in [0, 1]")
    if np.any(lower > upper):
        raise ValueError("CDF endpoints must be monotone within each row")
    if np.any(lower[lower_points < 0] != 0.0) or np.any(upper[upper_points < 0] != 0.0):
        raise ValueError("CDF endpoints below nonnegative support must be zero")
    return lower, upper
