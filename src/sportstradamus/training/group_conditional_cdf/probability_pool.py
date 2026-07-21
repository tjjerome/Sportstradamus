"""Analytic bookmaker probability pooling shared by NFL yards calibrators.

The pool operates on one quoted over-probability at a time.  It deliberately
does not claim to define a full predictive CDF: callers that need arbitrary-line
coherence must supply and validate coherent book and model distributions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, cast

import numpy as np

PROBABILITY_POOL_KIND = "nfl_yards_probability_pool"
PROBABILITY_POOL_SCHEMA_VERSION = 1
RHO_FIT_BOUNDS: tuple[float, float] = (0.0, 1.0)
RHO_STABILITY_BOUNDS: tuple[float, float] = (0.2, 0.8)
RHO_MAX_FOLD_RANGE = 0.2


class ProbabilityPoolBlob(TypedDict):
    """JSON/pickle-safe state for a fitted line-probability pool."""

    kind: Literal["nfl_yards_probability_pool"]
    schema_version: int
    rho: float
    raw_rho: float


def fit_probability_pool(
    candidate_over: np.ndarray,
    book_over: np.ndarray,
    over_result: np.ndarray,
) -> ProbabilityPoolBlob:
    """Fit the Brier-optimal weight on a candidate over-probability.

    The served probability is ``book + rho * (candidate - book)``.  Ordinary
    least squares through the fixed book endpoint gives the analytic optimum;
    ``rho`` is clipped to ``[0, 1]`` so application remains a convex pool.

    Args:
        candidate_over: Candidate over-probabilities for validation rows.
        book_over: Bookmaker over-probabilities on the same rows and lines.
        over_result: Binary realized-over indicators on the same rows.

    Returns:
        A plain-typed, portable blob containing clipped and unconstrained weights.

    Raises:
        ValueError: If inputs are invalid or candidate and book are identical.
    """
    candidate = _probability_array(candidate_over, "candidate_over")
    book = _probability_array(book_over, "book_over")
    outcome = _probability_array(over_result, "over_result")
    if candidate.shape != book.shape or candidate.shape != outcome.shape:
        raise ValueError("candidate, book, and outcome arrays must have identical shapes")
    if not np.isin(outcome, (0.0, 1.0)).all():
        raise ValueError("over_result must contain only binary 0/1 values")

    direction = candidate - book
    denominator = float(np.dot(direction.ravel(), direction.ravel()))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("candidate and book probabilities must differ on at least one row")
    raw_rho = float(np.dot(direction.ravel(), (outcome - book).ravel()) / denominator)
    rho = float(np.clip(raw_rho, *RHO_FIT_BOUNDS))
    return {
        "kind": PROBABILITY_POOL_KIND,
        "schema_version": PROBABILITY_POOL_SCHEMA_VERSION,
        "rho": rho,
        "raw_rho": raw_rho,
    }


def apply_probability_pool(
    blob: Mapping[str, object], candidate_over: np.ndarray, book_over: np.ndarray
) -> np.ndarray:
    """Apply a fitted convex pool to validation, test, or live line probabilities.

    Args:
        blob: State returned by :func:`fit_probability_pool`, possibly JSON-loaded.
        candidate_over: Candidate over-probabilities at the requested lines.
        book_over: Bookmaker over-probabilities at those same lines.

    Returns:
        Convex-pooled probabilities with the inputs' shared shape.
    """
    rho = _rho_from_blob(blob)
    candidate = _probability_array(candidate_over, "candidate_over")
    book = _probability_array(book_over, "book_over")
    if candidate.shape != book.shape:
        raise ValueError("candidate and book arrays must have identical shapes")
    return np.clip(book + rho * (candidate - book), 0.0, 1.0)


def validate_probability_pool_weights(fold_rhos: Sequence[float], final_rho: float) -> None:
    """Enforce the preregistered NFL-yards weight and fold-stability guards.

    Every outer-Player-CV weight and the full-validation weight must lie in
    ``[0.2, 0.8]``.  The outer-fold range must not exceed ``0.2``.  These guards
    reject book-only/model-only boundary solutions and unstable calibration fits.

    Raises:
        ValueError: If a weight is nonfinite, outside bounds, or unstable by fold.
    """
    folds = np.asarray(tuple(fold_rhos), dtype=float)
    if folds.size == 0:
        raise ValueError("at least one outer-fold rho is required")
    all_weights = np.append(folds, float(final_rho))
    low, high = RHO_STABILITY_BOUNDS
    if not np.isfinite(all_weights).all() or np.any((all_weights < low) | (all_weights > high)):
        raise ValueError(f"probability-pool rho must remain within [{low}, {high}]")
    fold_range = float(np.ptp(folds))
    if fold_range > RHO_MAX_FOLD_RANGE:
        raise ValueError(
            f"probability-pool outer-fold rho range {fold_range:.6g} exceeds {RHO_MAX_FOLD_RANGE}"
        )


def _probability_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite array")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _rho_from_blob(blob: Mapping[str, object]) -> float:
    if blob.get("kind") != PROBABILITY_POOL_KIND:
        raise ValueError("unknown probability-pool blob kind")
    if blob.get("schema_version") != PROBABILITY_POOL_SCHEMA_VERSION:
        raise ValueError("unsupported probability-pool blob schema")
    rho = cast(float, blob.get("rho"))
    if not isinstance(rho, (int, float)) or not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("probability-pool blob contains an invalid rho")
    return float(rho)
