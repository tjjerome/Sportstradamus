"""Book/model probability pooling for the group-conditional engine.

Two pool families sit behind ``pool_weight``. ``fixed_0.8`` is the receiving
corner's auditable 80/20 policy prior, applied only to explicitly authentic
quotes and otherwise leaving the model probability untouched. ``fit`` is the
rushing corner's Brier-optimal convex pool, delegated verbatim to the shared
probability-pool helper.
"""

from __future__ import annotations

import numpy as np

from sportstradamus.training.group_conditional_cdf._contracts import (
    FIXED_BOOK_WEIGHT,
    FIXED_MODEL_WEIGHT,
    FixedProbabilityPoolBlob,
)
from sportstradamus.training.group_conditional_cdf._validation import (
    authentic_vector,
    optional_book_vector,
    probability_vector,
)


def fixed_pool_blob() -> FixedProbabilityPoolBlob:
    return {
        "kind": "fixed_linear_probability_pool",
        "model_weight": FIXED_MODEL_WEIGHT,
        "book_weight": FIXED_BOOK_WEIGHT,
        "source": "policy_prior",
        "estimated": False,
        "authenticity": "Archived==True and Odds_synthetic is explicitly False",
        "missing_book": "unpooled_model_fallback",
    }


def pooled_probability(
    model_probability: np.ndarray,
    book_probability: np.ndarray,
    authentic: np.ndarray,
) -> np.ndarray:
    model = probability_vector(model_probability, "model_probability")
    books = optional_book_vector(book_probability, len(model))
    authentic_rows = authentic_vector(authentic, len(model))
    pooled = model.copy()
    pooled[authentic_rows] = (
        FIXED_BOOK_WEIGHT * books[authentic_rows] + FIXED_MODEL_WEIGHT * model[authentic_rows]
    )
    return pooled
