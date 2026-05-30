"""Unit tests for ``training.pipeline._prune_uninformative_features``.

The prune is invoked inside ``_step_build_splits`` to drop columns that
LightGBM cannot meaningfully split on. These tests pin the contract:
zero-variance and all-NaN columns are dropped, categoricals are kept,
and informative columns survive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.training.pipeline import _prune_uninformative_features


def test_drops_all_nan_columns() -> None:
    X_train = pd.DataFrame({
        "informative": [1.0, 2.0, 3.0, 4.0],
        "all_nan": [np.nan, np.nan, np.nan, np.nan],
    })
    kept = _prune_uninformative_features(X_train, categorical_cols=[])
    assert kept == ["informative"]


def test_drops_zero_variance_columns() -> None:
    X_train = pd.DataFrame({
        "informative": [1.0, 2.0, 3.0, 4.0],
        "constant": [7.0, 7.0, 7.0, 7.0],
    })
    kept = _prune_uninformative_features(X_train, categorical_cols=[])
    assert kept == ["informative"]


def test_keeps_categorical_columns_even_when_degenerate() -> None:
    X_train = pd.DataFrame({
        "informative": [1.0, 2.0, 3.0, 4.0],
        "Home": pd.Series([True, True, True, True], dtype="category"),
    })
    kept = _prune_uninformative_features(X_train, categorical_cols=["Home"])
    # Home is constant but categorical — it must survive.
    assert "Home" in kept
    assert "informative" in kept


def test_keeps_column_with_some_nans_and_two_distinct_values() -> None:
    X_train = pd.DataFrame({
        "sparse_but_varied": [1.0, np.nan, 2.0, np.nan, 1.0],
    })
    kept = _prune_uninformative_features(X_train, categorical_cols=[])
    # Has 2 distinct non-NaN values → informative.
    assert kept == ["sparse_but_varied"]


def test_drops_column_with_one_distinct_value_and_nans() -> None:
    X_train = pd.DataFrame({
        "near_constant": [1.0, np.nan, 1.0, np.nan, 1.0],
    })
    kept = _prune_uninformative_features(X_train, categorical_cols=[])
    # nunique(dropna=True) == 1 → cannot split → dropped.
    assert kept == []


def test_preserves_column_order() -> None:
    X_train = pd.DataFrame({
        "a": [1.0, 2.0],
        "b": [3.0, 3.0],  # constant, dropped
        "c": [4.0, 5.0],
        "d": [np.nan, np.nan],  # all-nan, dropped
        "e": [6.0, 7.0],
    })
    kept = _prune_uninformative_features(X_train, categorical_cols=[])
    assert kept == ["a", "c", "e"]
