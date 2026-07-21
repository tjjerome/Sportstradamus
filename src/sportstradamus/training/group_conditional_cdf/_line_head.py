"""Line-only temperature fits for the group-conditional engine.

The two corners fit a line-only Brier temperature with the same 0.01 anchor
regularization but not by identical code: the receiving corner validates that the
outcome is binary and wraps the optimizer output through the temperature bounds
validator; the rushing corner does neither. Both are kept behind ``pool_weight``'s
sibling switch so each corner reproduces its exact result.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from sportstradamus.training.group_conditional_cdf._contracts import (
    PROBABILITY_CLIP,
    TEMPERATURE_BOUNDS,
    TEMPERATURE_REGULARIZATION,
)
from sportstradamus.training.group_conditional_cdf._validation import (
    probability_vector,
    temperature,
)


def _optimize_temperature(logits: np.ndarray, outcome: np.ndarray, *, error: str) -> float:
    """Minimize the Brier + anchor-regularized objective over the temperature bounds."""

    def objective(value: float) -> float:
        calibrated = expit(logits / value)
        return float(
            np.mean((calibrated - outcome) ** 2) + TEMPERATURE_REGULARIZATION * (value - 1.0) ** 2
        )

    result = minimize_scalar(objective, bounds=TEMPERATURE_BOUNDS, method="bounded")
    if not result.success:
        raise ValueError(error)
    return float(result.x)


def fit_temperature_receiving(probability: np.ndarray, outcome: np.ndarray) -> float:
    probabilities = probability_vector(probability, "temperature probability")
    outcomes = probability_vector(outcome, "temperature outcome", len(probabilities))
    if not np.isin(outcomes, (0.0, 1.0)).all():
        raise ValueError("temperature outcome must contain only binary 0/1 values")
    logits = logit(np.clip(probabilities, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP))
    return temperature(
        _optimize_temperature(logits, outcomes, error="receiving temperature fit did not converge")
    )


def fit_temperature_rushing(probability: np.ndarray, outcome: np.ndarray) -> float:
    logits = logit(np.clip(probability, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP))
    return _optimize_temperature(logits, outcome, error="temperature optimization failed")
