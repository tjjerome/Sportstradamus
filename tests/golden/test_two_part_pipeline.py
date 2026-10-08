"""Golden pins for receiving-v3 train-only gate state and CDF endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers import skewnormal_loc_from_mean
from sportstradamus.training.group_conditional_cdf._pipeline_steps_two_part import (
    _skewnormal_cdf_endpoints,
    _step_apply_two_part_groupcdf_candidate,
    _two_part_support_rows,
    pin_two_part_grouping,
)
from sportstradamus.training.structural_context import build_two_part_context
from sportstradamus.training.structural_strategies import ROLE_COLUMNS


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

    context = build_two_part_context(
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


def _supported_splits(rows: int, thin_position_positives: int | None = None):
    """A validation frame dense enough to clear every two-part support floor.

    ``thin_position_positives`` starves roster code 4 of positive rows, which is what makes
    ``role_by_position`` unsupportable while ``role_only`` still holds.
    """
    rng = np.random.default_rng(11)
    index = pd.Index(range(rows))
    positions = np.resize(np.array([2, 3, 4]), rows)
    roles = np.where(np.arange(rows) % 2 == 0, "low", "high")
    result = rng.gamma(2.0, 6.0, rows)
    result[rng.random(rows) < 0.25] = 0.0
    if thin_position_positives is not None:
        starved = np.flatnonzero(positions == 4)
        result[starved[thin_position_positives:]] = 0.0
    # The line sits at the median of the positive outcomes, which keeps both over/under classes
    # populated for the temperature floors however many rows the starving above zeroed.
    line = float(np.median(result[result > 0.0]))
    splits = {
        "X_validation": pd.DataFrame({"Player position": positions}, index=index),
        "y_validation": pd.DataFrame({"Result": result}, index=index),
        "B_validation": pd.DataFrame({"Line": np.full(rows, line)}, index=index),
        "players_validation": pd.Series([f"player-{i % 400}" for i in range(rows)], index=index),
        "quote_authenticity_validation": pd.Series("authentic", index=index),
    }
    context = {
        "routes": {"validation": pd.Series(roles, index=index, dtype="object")},
        "boundary_residual_positions": [],
    }
    return splits, context, index


def test_pin_two_part_grouping_keeps_position_granularity_when_every_partition_supports_it():
    splits, context, index = _supported_splits(3000)
    partitions = [index, index[: int(len(index) * 0.8)], index[int(len(index) * 0.2) :]]

    assert pin_two_part_grouping(splits, context, partitions) == "role_by_position"


def test_pin_two_part_grouping_demotes_once_any_single_partition_cannot_hold_positions():
    """One thin partition decides for all of them — folds may not each pick their own grouping."""
    splits, context, index = _supported_splits(3000, thin_position_positives=40)
    partitions = [index, index[: int(len(index) * 0.8)]]

    assert pin_two_part_grouping(splits, context, partitions) == "role_only"


def test_pin_two_part_grouping_fails_the_corner_when_neither_grouping_is_supported():
    splits, context, index = _supported_splits(300)

    with pytest.raises(ValueError, match="no two-part grouping is supported"):
        pin_two_part_grouping(splits, context, [index])


def test_two_part_step_runs_on_tie_rows_and_counts_each_as_half_an_over():
    rows = 3000
    splits, context, index = _supported_splits(rows)
    # Outcomes drawn from the same gated SkewNormal the step is handed, so every guard passes and
    # the step returns; one row in sixty lands exactly on its line.
    rng = np.random.default_rng(11)
    gate_rates = {"low": 0.30, "high": 0.10}
    gate = context["routes"]["validation"].map(gate_rates).to_numpy(dtype=float)
    mean, sigma, alpha = rng.uniform(25.0, 70.0, rows), np.full(rows, 22.0), np.full(rows, 1.5)
    loc = skewnormal_loc_from_mean(mean, sigma, alpha)
    result = skewnorm.rvs(alpha, loc, sigma, random_state=rng)
    result[rng.random(rows) < gate] = 0.0
    line = np.round(mean * (1.0 - gate)) + 0.5
    tie = np.arange(rows) % 60 == 0
    result[tie] = line[tie]
    book = (1.0 - gate) * skewnorm.sf(line, alpha, loc, sigma) + rng.normal(0.0, 0.08, rows)
    priced = pd.DataFrame({"Line": line, "Odds": np.clip(book, 0.02, 0.98)}, index=index)
    frame = splits["X_validation"].assign(MeanYr=mean * (1.0 - gate), Mean10=mean * (1.0 - gate))
    splits["X_validation"] = splits["X_test"] = frame
    splits["B_validation"] = splits["B_test"] = priced
    splits["y_validation"] = pd.DataFrame({"Result": result}, index=index)
    splits["dates_validation"] = pd.Series(pd.date_range("2020-01-01", periods=rows), index=index)
    splits["quote_authenticity_test"] = splits["quote_authenticity_validation"]
    context["routes"]["test"] = context["routes"]["validation"]
    context |= {"status": "active", "gate_rates": gate_rates, "positions": [2, 3, 4]}
    context |= dict.fromkeys(["thresholds", "role_columns", "support", "fallback_gate"])
    fused = {"model_weight": 1.0, "weighted_mean_val": mean, "weighted_mean": mean}
    for split in ("val", "test"):
        fused |= {f"sn_sigma_blend_{split}": sigma, f"sn_alpha_blend_{split}": alpha}

    assert set(_two_part_support_rows(splits, context, index)[1][tie]) == {0.5}
    context["pinned_grouping"] = pin_two_part_grouping(splits, context, [index])
    calibrated, _ = _step_apply_two_part_groupcdf_candidate({}, fused, splits, context)

    # Both numbers move if a tie is scored as an Over: it would join class 1, and cost the book
    # (1 - p)^2 where half an Over costs it (0.5 - p)^2.
    blob = calibrated["structural_calibration_blob"]
    support = blob["support"]["nested_calibration"]["temperature_support"][0]
    assert support["class_0_rows"] + support["class_1_rows"] == rows - tie.sum()
    book_brier = np.mean((priced["Odds"] - np.where(tie, 0.5, result > line)) ** 2)
    assert blob["validation_audit"]["gate1"]["book_brier"] == pytest.approx(book_brier)
