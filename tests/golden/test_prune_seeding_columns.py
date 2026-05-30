"""Regression test for `_prune_uninformative_features` preserving seeding columns.

`set_model_start_values` (helpers/distributions.py:461) unconditionally reads
`X_data[["MeanYr", "STDYr", "ZeroYr"]]` to populate model start values. The
pruner introduced in the 2026-05-27 no-filter rewire drops constant /
all-NaN columns to shave Optuna wall-time, but it must NOT drop those three
even when they happen to be constant in a given cell's training slice. The
canonical failure mode: NBA MIN training data has `ZeroYr ≡ 0` for every
player (the NBA gamelog excludes DNPs, so minutes are never recorded as 0)
— the pruner would drop ZeroYr, then `set_model_start_values` KeyErrors at
the unconditional column select.
"""

import numpy as np
import pandas as pd

from sportstradamus.training.pipeline import (
    _SEEDING_REQUIRED_COLUMNS,
    _prune_uninformative_features,
)


def test_seeding_columns_kept_when_constant():
    """All three seeding columns survive pruning even with one distinct value."""
    n = 20
    X = pd.DataFrame(
        {
            "MeanYr": np.full(n, 30.0),
            "STDYr": np.full(n, 5.0),
            "ZeroYr": np.zeros(n),
            "informative_feature": np.arange(n, dtype=float),
            "unrelated_constant": np.zeros(n),
        }
    )

    keep = _prune_uninformative_features(X, categorical_cols=[])

    for col in _SEEDING_REQUIRED_COLUMNS:
        assert col in keep, f"{col} must survive pruning (consumed by set_model_start_values)"
    assert "informative_feature" in keep
    assert "unrelated_constant" not in keep, "non-seeding constant columns should still be pruned"


def test_seeding_columns_kept_when_all_nan():
    """Seeding columns survive even when entirely NaN (defensive; shouldn't happen in practice)."""
    n = 10
    X = pd.DataFrame(
        {
            "MeanYr": np.full(n, np.nan),
            "STDYr": np.full(n, np.nan),
            "ZeroYr": np.full(n, np.nan),
            "informative_feature": np.arange(n, dtype=float),
        }
    )

    keep = _prune_uninformative_features(X, categorical_cols=[])

    for col in _SEEDING_REQUIRED_COLUMNS:
        assert col in keep


def test_normal_pruning_still_works_alongside_protection():
    """Informative / constant / NaN classification still holds for non-seeding columns."""
    n = 30
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {
            "MeanYr": rng.standard_normal(n),
            "STDYr": rng.standard_normal(n),
            "ZeroYr": rng.uniform(0, 0.3, size=n),
            "varying_feature": rng.standard_normal(n),
            "constant_feature": np.full(n, 7.0),
            "all_nan_feature": np.full(n, np.nan),
        }
    )

    keep = _prune_uninformative_features(X, categorical_cols=[])

    assert set(keep) == {"MeanYr", "STDYr", "ZeroYr", "varying_feature"}


def test_categorical_columns_still_kept():
    """Categorical-cols allowlist behaviour is unchanged by the seeding protection."""
    n = 15
    X = pd.DataFrame(
        {
            "MeanYr": np.full(n, 10.0),
            "STDYr": np.full(n, 2.0),
            "ZeroYr": np.zeros(n),
            "Home": np.zeros(n),
            "Player position": np.ones(n),
        }
    )

    keep = _prune_uninformative_features(X, categorical_cols=["Home", "Player position"])

    assert "Home" in keep
    assert "Player position" in keep
