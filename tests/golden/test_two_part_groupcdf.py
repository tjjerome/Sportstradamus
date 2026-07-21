"""Golden guards for the receiving-yards v3 two-part grouped-CDF calibrator."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import expit, logit

from sportstradamus.training import group_conditional_cdf as receiving
from sportstradamus.training.group_conditional_cdf._pool import (
    fixed_pool_blob,
)


def _identity_map(lam: float = 0.0):
    return {
        "kind": "isotonic_pit",
        "lam": lam,
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }


def _manual_blob():
    curved = {
        "kind": "isotonic_pit",
        "lam": 1.0,
        "x": [0.0, 0.5, 1.0],
        "y": [0.0, 0.25, 1.0],
    }
    positive = {
        f"{role}_pos{position}": (curved if role == "low" and position == 2 else _identity_map(1.0))
        for role in receiving.ROLE_VALUES
        for position in receiving.POSITION_CODES
    }
    return {
        "kind": receiving.CANDIDATE_NAME,
        "schema_version": receiving.SCHEMA_VERSION,
        "line_probability_only": True,
        "temperature_fit_scope": "pre_map_raw_endpoint_settlement",
        "temperature": 2.0,
        "cdf": {
            "kind": "role_position_two_part_cdf",
            "role_boundary": {
                "low": {
                    "kind": "two_part_role_boundary",
                    "intercept": -0.4,
                    "nonpositive": _identity_map(),
                },
                "high": {
                    "kind": "two_part_role_boundary",
                    "intercept": 0.2,
                    "nonpositive": _identity_map(),
                },
            },
            "positive": positive,
            "rb_boundary_residual": 0.3,
            "boundary_residual_positions": [3],
        },
        "probability_pool": fixed_pool_blob(),
    }


def _synthetic_validation():
    n_players, rows_per_player = 60, 30
    player = np.repeat(np.arange(n_players), rows_per_player)
    within = np.tile(np.arange(rows_per_player), n_players)
    n_rows = len(player)
    role = np.where(within < rows_per_player // 2, "low", "high")
    position = np.take(np.array([2, 3, 4]), (player + within) % 3)
    zero_cdf = np.full(n_rows, 0.2)

    role_rank = within % 15
    nonpositive = role_rank < 3
    result = np.where(nonpositive, 0.0, role_rank - 2.0)
    positive_quantile = np.clip((role_rank - 2.5) / 12.0, 0.01, 0.99)
    result_upper = np.where(
        nonpositive,
        zero_cdf,
        zero_cdf + (1.0 - zero_cdf) * positive_quantile,
    )
    result_lower = np.where(nonpositive, 0.0, result_upper)

    high_candidate = within % 2 == 0
    line_low = np.where(high_candidate, 0.15, 0.7)
    line_high = line_low + 0.1
    book_over = np.where(high_candidate, 0.65, 0.35)
    authentic = within % 5 != 0
    high_rank = within // 2
    over_result = np.where(high_candidate, high_rank < 10, high_rank < 5).astype(float)
    return (
        result_upper,
        result_lower,
        zero_cdf,
        line_low,
        line_high,
        result,
        over_result,
        book_over,
        authentic,
        player,
        role,
        position,
    )


def test_piecewise_transform_uses_role_position_boundary_and_positive_map():
    blob = _manual_blob()
    f0 = np.array([0.2, 0.2, 0.6])
    cdf = np.array([0.1, 0.6, 0.3])
    roles = np.array(["low", "low", "high"])
    positions = np.array([3, 2, 4])

    transformed = receiving.apply_two_part_cdf(blob, cdf, f0, roles, positions)

    q_low_rb = expit(logit(0.2) - 0.4 + 0.3)
    q_low_wr = expit(logit(0.2) - 0.4)
    q_high_te = expit(logit(0.6) + 0.2)
    raw_positive = (0.6 - 0.2) / 0.8
    mapped_positive = np.interp(raw_positive, [0.0, 0.5, 1.0], [0.0, 0.25, 1.0])
    np.testing.assert_allclose(
        transformed,
        [
            q_low_rb * 0.5,
            q_low_wr + (1.0 - q_low_wr) * mapped_positive,
            q_high_te * 0.5,
        ],
    )


def test_zero_atom_endpoints_are_transformed_before_randomization():
    blob = _manual_blob()
    args = (
        blob,
        np.array([0.08]),
        np.array([0.2]),
        np.array([0.2]),
        np.array(["low"]),
        np.array([3]),
    )
    lower, upper = receiving.two_part_cdf_endpoints(*args)
    draws = receiving.two_part_randomized_pit(*args)

    q = expit(logit(0.2) - 0.4 + 0.3)
    assert lower[0] == pytest.approx(q * 0.4)
    assert upper[0] == pytest.approx(q)
    assert np.all((draws[:, 0] >= lower[0]) & (draws[:, 0] <= upper[0]))
    assert np.ptp(draws[:, 0]) > 0.0


def test_line_apply_maps_endpoints_first_and_pools_only_authentic_quotes():
    blob = _manual_blob()
    low = np.array([0.35, 0.65, 0.3])
    high = np.array([0.45, 0.75, 0.4])
    f0 = np.array([0.2, 0.4, 0.2])
    role = np.array(["low", "high", "low"])
    position = np.array([2, 4, 3])
    book = np.array([0.48, np.nan, 0.54])
    authentic = np.array([True, False, True])

    output = receiving.apply_two_part_line(
        blob, low, high, f0, role, position, book, authentic
    )
    mapped_low = receiving.apply_two_part_cdf(blob, low, f0, role, position)
    mapped_high = receiving.apply_two_part_cdf(blob, high, f0, role, position)

    np.testing.assert_array_equal(output.mapped_low, mapped_low)
    np.testing.assert_array_equal(output.mapped_high, mapped_high)
    np.testing.assert_allclose(output.mapped_under, 0.5 * (mapped_low + mapped_high))
    np.testing.assert_allclose(
        output.pooled_over[authentic],
        0.8 * book[authentic] + 0.2 * output.candidate_over[authentic],
    )
    assert output.pooled_over[1] == output.candidate_over[1]


def test_line_apply_rejects_missing_authentic_book_but_allows_fallback():
    args = (
        _manual_blob(),
        np.array([0.2]),
        np.array([0.3]),
        np.array([0.1]),
        np.array(["low"]),
        np.array([2]),
        np.array([np.nan]),
    )
    with pytest.raises(ValueError, match="authentic book_over"):
        receiving.apply_two_part_line(*args, np.array([True]))
    output = receiving.apply_two_part_line(*args, np.array([False]))
    np.testing.assert_array_equal(output.pooled_over, output.candidate_over)


def test_portable_blob_round_trip_reproduces_cdf_and_line_application():
    blob = _manual_blob()
    payload = receiving.serialize_two_part_calibration(blob)
    restored = receiving.deserialize_two_part_calibration(payload)
    cdf = np.array([0.1, 0.7])
    f0 = np.array([0.2, 0.4])
    role = np.array(["low", "high"])
    position = np.array([2, 3])
    book = np.array([0.45, 0.55])
    authentic = np.array([True, False])

    np.testing.assert_array_equal(
        receiving.apply_two_part_cdf(blob, cdf, f0, role, position),
        receiving.apply_two_part_cdf(restored, cdf, f0, role, position),
    )
    original = receiving.apply_two_part_line(
        blob, cdf, cdf, f0, role, position, book, authentic
    )
    reloaded = receiving.apply_two_part_line(
        restored, cdf, cdf, f0, role, position, book, authentic
    )
    for field in original.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(original, field), getattr(reloaded, field))
    assert json.loads(payload)["kind"] == receiving.CANDIDATE_NAME


def test_nested_player_cv_fit_returns_finite_reusable_outputs():
    inputs = _synthetic_validation()
    fit = receiving.fit_two_part_groupcdf(*inputs, residual_positions=(3,))
    line = receiving.apply_two_part_line(
        fit.blob,
        inputs[3],
        inputs[4],
        inputs[2],
        inputs[10],
        inputs[11],
        inputs[7],
        inputs[8],
    )

    n_rows = len(inputs[5])
    assert fit.oof_pit_draws.shape == (receiving.RANDOMIZED_PIT_DRAWS, n_rows)
    assert fit.oof_positive_pit.shape == (n_rows,)
    assert len(fit.fold_lambdas) == receiving.PLAYER_CV_FOLDS
    assert len(fit.fold_intercepts) == receiving.PLAYER_CV_FOLDS
    assert len(fit.fold_temperatures) == receiving.PLAYER_CV_FOLDS
    assert len(fit.fold_selection_keys) == receiving.PLAYER_CV_FOLDS
    assert all(1.0 <= value <= 10.0 for value in fit.fold_temperatures)
    assert all(
        np.isfinite(values).all()
        for values in (
            fit.oof_pit_draws,
            fit.oof_candidate_over,
            fit.oof_pooled_over,
            line.pooled_over,
        )
    )
    assert np.isfinite(fit.oof_positive_pit[inputs[5] > 0.0]).all()
    assert set(fit.blob["cdf"]["role_boundary"]) == {"low", "high"}
    assert set(fit.blob["cdf"]["positive"]) == {
        f"{role}_pos{position}"
        for role in receiving.ROLE_VALUES
        for position in receiving.POSITION_CODES
    }


def test_blob_validation_rejects_malformed_map_and_policy_drift():
    blob = _manual_blob()
    blob["cdf"]["positive"]["high_pos4"]["x"] = [0.0, 1.0, 1.0]
    blob["cdf"]["positive"]["high_pos4"]["y"] = [0.0, 0.9, 1.0]
    with pytest.raises(ValueError, match="invalid two-part conditional CDF map"):
        receiving.serialize_two_part_calibration(blob)

    blob = _manual_blob()
    blob["probability_pool"]["model_weight"] = 0.3
    with pytest.raises(ValueError, match="fixed policy prior"):
        receiving.serialize_two_part_calibration(blob)


def test_fit_rejects_missing_player_before_string_cast():
    inputs = list(_synthetic_validation())
    players = inputs[9].astype(object)
    players[0] = None
    inputs[9] = players
    with pytest.raises(ValueError, match="identifiers must be nonempty"):
        receiving.fit_two_part_groupcdf(*inputs)
