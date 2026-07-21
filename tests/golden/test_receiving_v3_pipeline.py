"""Golden pins for receiving-v3 train-only gate state and CDF endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from sportstradamus.helpers import skewnormal_loc_from_mean
from sportstradamus.training.group_conditional_cdf._pipeline_steps_receiving import (
    _skewnormal_cdf_endpoints,
)
from sportstradamus.training.nfl_yards_context import build_receiving_role_context
from sportstradamus.training.nfl_yards_experiments import ROLE_COLUMNS


def _role_frame(index: pd.Index, scores: list[float]) -> pd.DataFrame:
    positions = np.resize(np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]), len(index))
    return pd.DataFrame(
        {
            ROLE_COLUMNS[0]: scores,
            ROLE_COLUMNS[1]: 1.0,
            ROLE_COLUMNS[2]: 1.0,
            ROLE_COLUMNS[3]: 1.0,
            "Player position": positions,
        },
        index=index,
    )


def test_receiving_context_fits_role_gate_rates_from_train_outcomes_only():
    train_index = pd.Index(range(12))
    validation_index = pd.Index(range(20, 32))
    test_index = pd.Index(range(40, 52))
    scores = [1.0, 3.0, 2.0, 4.0] * 3
    splits = {
        "X_train": _role_frame(train_index, scores),
        "X_validation": _role_frame(validation_index, scores),
        "X_test": _role_frame(test_index, scores),
        "y_train": pd.DataFrame(
            {"Result": [0.0, 0.0, 0.0, 5.0, 0.0, 8.0, 0.0, 9.0, 0.0, 7.0, 5.0, 6.0]},
            index=train_index,
        ),
        "players_train": pd.Series([f"train-{i}" for i in train_index], index=train_index),
        "players_validation": pd.Series(
            [f"validation-{i}" for i in validation_index], index=validation_index
        ),
    }
    tiny_floor = {
        "train_rows": 1,
        "validation_rows": 1,
        "test_rows": 1,
        "train_players": 1,
        "validation_players": 1,
    }

    context = build_receiving_role_context(
        splits, league="NFL", market="receiving yards", support_floor=tiny_floor
    )

    assert context["status"] == "active"
    assert context["thresholds"] == {"2": 2.5, "3": 2.5, "4": 2.5}
    assert context["gate_rates"] == {"low": 5 / 6, "high": 1 / 6}
    assert context["fallback_gate"] == 0.5
    assert context["routes"]["validation"].tolist() == ["low", "high", "low", "high"] * 3


def test_receiving_endpoint_helper_preserves_role_specific_zero_atom():
    mean = np.array([18.0, 30.0])
    sigma = np.array([8.0, 12.0])
    alpha = np.array([0.4, 1.1])
    gate = np.array([0.18, 0.06])
    points = np.array([0.0, 12.0])

    lower, upper = _skewnormal_cdf_endpoints(mean, sigma, alpha, gate, points)
    loc = skewnormal_loc_from_mean(mean, sigma, alpha)
    base = skewnorm.cdf(points, alpha, loc=loc, scale=sigma)

    assert upper[0] - lower[0] == gate[0]
    assert upper[1] == (gate[1] + (1.0 - gate[1]) * base[1])
    assert lower[1] == upper[1]
