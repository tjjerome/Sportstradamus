"""Golden live-parity pins for the surviving NFL yards candidates."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers import skewnormal_loc_from_mean
from sportstradamus.prediction.model_prob import (
    _affine_candidate_calibration,
    _apply_affine_candidate_distribution,
    _apply_two_part_candidate_distribution,
    _build_prob_params,
    _two_part_candidate_state,
)
from sportstradamus.training.group_conditional_cdf import (
    AffinePredictive,
    apply_affine_groupcdf,
    apply_two_part_line,
    deserialize_two_part_calibration,
    serialize_two_part_calibration,
)
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY,
    ROLE_COLUMNS,
    TWO_PART_POSITIONS,
    TWO_PART_STRATEGY,
)

model_prob_module = importlib.import_module("sportstradamus.prediction.model_prob")


def _endpoint_map(lam: float, midpoint: float) -> dict:
    return {
        "kind": "isotonic_pit",
        "lam": lam,
        "x": [0.0, 0.5, 1.0],
        "y": [0.0, midpoint, 1.0],
    }


def _two_part_calibration_blob() -> dict:
    return {
        "kind": TWO_PART_STRATEGY,
        "schema_version": 3,
        "line_probability_only": True,
        "temperature_fit_scope": "pre_map_raw_endpoint_settlement",
        "temperature": 1.4,
        "cdf": {
            "kind": "role_position_two_part_cdf",
            "role_boundary": {
                "low": {
                    "kind": "two_part_role_boundary",
                    "intercept": -0.15,
                    "nonpositive": _endpoint_map(0.75, 0.45),
                },
                "high": {
                    "kind": "two_part_role_boundary",
                    "intercept": 0.1,
                    "nonpositive": _endpoint_map(0.75, 0.45),
                },
            },
            "positive": {
                f"{role}_pos{position}": _endpoint_map(1.0, 0.55)
                for role in ("low", "high")
                for position in TWO_PART_POSITIONS
            },
            "rb_boundary_residual": 0.05,
        },
        "probability_pool": {
            "kind": "fixed_linear_probability_pool",
            "model_weight": 0.2,
            "book_weight": 0.8,
            "source": "policy_prior",
            "estimated": False,
            "authenticity": "Archived==True and Odds_synthetic is explicitly False",
            "missing_book": "unpooled_model_fallback",
        },
    }


def _two_part_candidate_blob(calibration: dict | None = None) -> dict:
    return {
        "schema_version": 3,
        "slug": TWO_PART_STRATEGY,
        "status": "active",
        "kill_reason": None,
        "line_probability_only": True,
        "routing": {
            "kind": "pregame_role_by_position",
            "thresholds": {"2": 0.5, "3": 0.5, "4": 0.5},
            "role_columns": list(ROLE_COLUMNS),
            "gate_rates": {"low": 0.30, "high": 0.60},
            "served_gate_rates": {"low": 0.06, "high": 0.12},
            "gate_model_weight": 0.2,
            "positions": {str(code): label for code, label in TWO_PART_POSITIONS.items()},
        },
        "endpoint_contract": {
            "gate_source": "train_result_eq_zero_by_role",
            "gate_rates_are_model_endpoint": True,
            "book_endpoint_gate": 0.0,
            "fallback_gate": 0.45,
            "dispersion": "raw_fused_shape",
        },
        "calibration": calibration or _two_part_calibration_blob(),
    }


def _two_part_frame(book_over=(0.58, 0.44)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Player position": [2, 3],
            "Line": [42.5, 24.0],
            "Market EV": book_over,
            "Model Sigma": [14.0, 9.0],
            "Model Skew": [0.6, 0.2],
            "Model Gate": [0.91, 0.91],
            ROLE_COLUMNS[0]: [10.0, 10.0],
            ROLE_COLUMNS[1]: [0.5, 0.5],
            ROLE_COLUMNS[2]: [0.05, 0.20],
            ROLE_COLUMNS[3]: [1.0, 1.0],
        }
    )


def test_receiving_candidate_live_endpoint_operator_matches_shared_application():
    calibration, routing = _two_part_candidate_state(_two_part_candidate_blob())
    frame = _two_part_frame()
    mean = np.array([38.0, 27.0])
    line = frame["Line"].to_numpy()
    sigma = frame["Model Sigma"].to_numpy()
    alpha = frame["Model Skew"].to_numpy()
    gate = np.array([routing["served_gate_rates"][role] for role in ("low", "high")])
    loc = skewnormal_loc_from_mean(mean, sigma, alpha)

    def gated_cdf(point):
        base = skewnorm.cdf(point, alpha, loc=loc, scale=sigma)
        return np.where(point < 0.0, (1.0 - gate) * base, gate + (1.0 - gate) * base)

    low = gated_cdf(np.ceil(line - 1.0))
    high = gated_cdf(np.floor(line + 1.0))
    f0 = gated_cdf(np.zeros(len(frame)))
    expected = apply_two_part_line(
        calibration,
        low,
        high,
        f0,
        np.array(["low", "high"]),
        frame["Player position"].to_numpy(),
        frame["Market EV"].to_numpy(),
        np.array([True, True]),
    )

    served = _apply_two_part_candidate_distribution(frame, mean, calibration, routing)

    for field in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(served, field), getattr(expected, field))
    np.testing.assert_array_equal(frame["Model Gate"].to_numpy(), gate)
    np.testing.assert_array_equal(frame["Projection"].to_numpy(), (1.0 - gate) * mean)


def test_receiving_candidate_calibration_roundtrip_is_serve_exact():
    calibration = _two_part_calibration_blob()
    restored = deserialize_two_part_calibration(serialize_two_part_calibration(calibration))
    original_state = _two_part_candidate_state(_two_part_candidate_blob(calibration))
    restored_state = _two_part_candidate_state(_two_part_candidate_blob(restored))
    frame = _two_part_frame()
    mean = np.array([38.0, 27.0])

    original = _apply_two_part_candidate_distribution(frame, mean, *original_state)
    reloaded = _apply_two_part_candidate_distribution(frame, mean, *restored_state)

    for field in original.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(original, field), getattr(reloaded, field))


def test_receiving_candidate_pools_only_finite_current_book_probability():
    calibration, routing = _two_part_candidate_state(_two_part_candidate_blob())
    frame = _two_part_frame(book_over=(0.62, np.nan))

    served = _apply_two_part_candidate_distribution(
        frame,
        np.array([38.0, 27.0]),
        calibration,
        routing,
    )

    assert served.pooled_over[0] == pytest.approx(
        0.8 * frame.loc[0, "Market EV"] + 0.2 * served.candidate_over[0]
    )
    assert served.pooled_over[1] == served.candidate_over[1]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda blob: blob.update(schema_version=2), "schema 3"),
        (
            lambda blob: blob["routing"].update(role_columns=list(ROLE_COLUMNS[:-1])),
            "feature routing",
        ),
        (
            lambda blob: blob["routing"]["served_gate_rates"].update(low=0.07),
            "served gates",
        ),
        (
            lambda blob: blob["endpoint_contract"].update(dispersion="calibrated_shape"),
            "endpoint contract",
        ),
    ],
)
def test_receiving_candidate_rejects_partial_artifact_state(mutate, message):
    experiment = _two_part_candidate_blob()
    mutate(experiment)

    with pytest.raises(ValueError, match=message):
        _two_part_candidate_state(experiment)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns=[ROLE_COLUMNS[0]]), "role column"),
        (lambda frame: frame.assign(**{"Player position": [2, 1]}), "position codes"),
    ],
)
def test_receiving_candidate_fails_loud_for_invalid_eligible_route(mutate, message):
    calibration, routing = _two_part_candidate_state(_two_part_candidate_blob())

    with pytest.raises(ValueError, match=message):
        _apply_two_part_candidate_distribution(
            mutate(_two_part_frame()),
            np.array([38.0, 27.0]),
            calibration,
            routing,
        )


def _affine_candidate_blob() -> dict:
    identity = {
        "kind": "group_empirical_cdf",
        "lam": 0.0,
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }
    calibration = {
        "kind": AFFINE_STRATEGY,
        "schema_version": 1,
        "line_probability_only": True,
        "affine": {
            "kind": "affine_marginal_mean",
            "intercept": 2.0,
            "slope": 1.1,
            "floor": 0.1,
        },
        "position_cdf": {"1": identity, "3": identity},
        "temperature": 1.3,
        "probability_pool": {
            "kind": "structural_probability_pool",
            "schema_version": 1,
            "rho": 0.4,
            "raw_rho": 0.4,
        },
    }
    return {
        "slug": AFFINE_STRATEGY,
        "status": "active",
        "line_probability_only": True,
        "calibration": calibration,
    }


def test_rushing_candidate_live_distribution_matches_shared_application():
    experiment = _affine_candidate_blob()
    calibration = _affine_candidate_calibration(experiment)
    frame = pd.DataFrame(
        {
            "Player position": [1, 3],
            "Line": [20.5, 52.5],
            "Market EV": [0.48, 0.53],
            "Model Sigma": [12.0, 20.0],
            "Model Skew": [0.5, 0.8],
            "Model Gate": [0.08, 0.03],
        }
    )
    conditional_mean = np.array([25.0, 58.0])
    predictive = AffinePredictive(
        conditional_mean * (1.0 - frame["Model Gate"].to_numpy()),
        frame["Model Sigma"].to_numpy(),
        frame["Model Skew"].to_numpy(),
        frame["Model Gate"].to_numpy(),
    )
    expected = apply_affine_groupcdf(
        calibration,
        predictive,
        frame["Line"].to_numpy(),
        frame["Market EV"].to_numpy(),
        frame["Player position"].to_numpy(),
    )

    served_mean, served_over = _apply_affine_candidate_distribution(
        frame,
        conditional_mean,
        calibration,
    )

    np.testing.assert_allclose(served_mean, expected.conditional_mean)
    np.testing.assert_allclose(served_over, expected.pooled_over)
    np.testing.assert_allclose(frame["Projection"], expected.marginal_mean)


def test_rushing_candidate_live_prediction_routes_qb_rb_experts(monkeypatch):
    class _Model:
        def __init__(self, marker):
            self.marker = marker

        def predict(self, rows, pred_type):
            assert pred_type == "parameters"
            return pd.DataFrame(
                {
                    "loc": np.full(len(rows), self.marker),
                    "scale": np.full(len(rows), 2.0),
                    "alpha": np.zeros(len(rows)),
                },
                index=rows.index,
            )

    class _Stats:
        volume_stats = set()

    monkeypatch.setattr(model_prob_module, "set_model_start_values", lambda *_a, **_k: None)
    player_stats = pd.DataFrame(
        {
            "Home": [0, 1, 0],
            "Player position": [1, 3, 2],
        },
        index=["qb", "rb", "wr"],
    )
    filedict = {
        "model": _Model(10.0),
        "expert_models": {1: _Model(101.0), 3: _Model(303.0)},
        "structural_calibration": _affine_candidate_blob(),
        "sn_param": "direct",
    }

    params = _build_prob_params(
        filedict,
        "rushing yards",
        _Stats(),
        player_stats,
        "SkewNormal",
        True,
        "ratio_meanyr",
        AFFINE_STRATEGY,
    )

    assert params.loc["qb", "loc"] == 101.0
    assert params.loc["rb", "loc"] == 303.0
    assert params.loc["wr", "loc"] == 10.0

    pooled = _build_prob_params(
        filedict,
        "rushing yards",
        _Stats(),
        player_stats,
        "SkewNormal",
        True,
        "ratio_meanyr",
        "none",
    )
    assert pooled["loc"].eq(10.0).all()
