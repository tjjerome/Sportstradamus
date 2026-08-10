import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers.distributions import skewnormal_loc_from_mean
from sportstradamus.training import group_conditional_cdf as rushing
from sportstradamus.training.group_conditional_cdf import _affine as rushing_cdf
from sportstradamus.training.group_conditional_cdf.probability_pool import (
    apply_probability_pool,
    fit_probability_pool,
    validate_probability_pool_weights,
)


def _identity_map(lam=0.0):
    return {
        "kind": "group_empirical_cdf",
        "lam": lam,
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }


def _manual_blob():
    return {
        "kind": rushing.AFFINE_STRATEGY_NAME,
        "schema_version": rushing.LEGACY_AFFINE_SCHEMA_VERSION,
        "line_probability_only": True,
        "affine": {
            "kind": "affine_marginal_mean",
            "intercept": 2.0,
            "slope": 1.1,
            "floor": rushing.MARGINAL_MEAN_FLOOR,
        },
        "position_cdf": {
            "1": {
                "kind": "group_empirical_cdf",
                "lam": 1.0,
                "x": [0.0, 0.5, 1.0],
                "y": [0.0, 0.25, 1.0],
            },
            "3": _identity_map(),
        },
        "temperature": 2.0,
        "probability_pool": {
            "kind": "structural_probability_pool",
            "schema_version": 1,
            "rho": 0.4,
            "raw_rho": 0.4,
        },
    }


def _generalized_blob(codes=(1, 2, 3, 4, 5)):
    return {
        "kind": rushing.AFFINE_STRATEGY_NAME,
        "schema_version": rushing.AFFINE_SCHEMA_VERSION,
        "line_probability_only": True,
        "affine": {
            "kind": "affine_marginal_mean",
            "intercept": 2.0,
            "slope": 1.1,
            "floor": rushing.MARGINAL_MEAN_FLOOR,
        },
        "position_cdf": {str(code): _identity_map() for code in codes},
        "position_codes": list(codes),
        "fallback": {
            "kind": "model_only",
            "unseen_position": "raw_cdf_unpooled",
            "nonauthentic_quote": "candidate_unpooled",
        },
        "affine_bounds": {
            "kind": "train_scale_affine_bounds",
            "train_abs_p95": 20.0,
            "intercept": [-40.0, 40.0],
            "slope": [0.5, 1.5],
        },
        "fit_audit": {
            "expected_position_codes": list(codes),
            "outer_folds": [{} for _ in range(rushing.PLAYER_CV_FOLDS)],
        },
        "temperature": 2.0,
        "probability_pool": {
            "kind": "structural_probability_pool",
            "schema_version": 1,
            "rho": 0.4,
            "raw_rho": 0.4,
        },
    }


def _synthetic_validation():
    rng = np.random.default_rng(812)
    n_players, rows_per_player = 100, 16
    player = np.repeat(np.arange(n_players), rows_per_player)
    position = np.where(player % 2 == 0, 1, 3)
    n_rows = len(player)
    phase = rng.normal(size=n_rows)
    gate = np.where(position == 1, 0.09, 0.025)
    conditional_true = np.where(position == 1, 26.0, 61.0) + 7.0 * phase
    sigma = np.where(position == 1, 15.0, 22.0)
    alpha = np.where(position == 1, 1.2, 0.7)
    true_marginal = (1.0 - gate) * conditional_true
    model_marginal = np.clip(true_marginal + rng.normal(0.0, 9.0, n_rows), 0.5, None)
    loc = skewnormal_loc_from_mean(conditional_true, sigma, alpha)
    result = skewnorm.rvs(alpha, loc=loc, scale=sigma, random_state=rng)
    result = np.where(rng.random(n_rows) < gate, 0.0, result)
    line = np.floor(true_marginal / 5.0) * 5.0 + 0.5
    true_under = gate + (1.0 - gate) * skewnorm.cdf(line, alpha, loc=loc, scale=sigma)
    book_over = np.clip(1.0 - true_under + rng.normal(0.0, 0.13, n_rows), 0.02, 0.98)
    predictive = rushing.AffinePredictive(model_marginal, sigma, alpha, gate)
    return predictive, result, line, book_over, position, player


def test_probability_pool_uses_analytic_clipped_brier_optimum():
    candidate = np.array([0.12, 0.75, 0.58, 0.31, 0.85])
    book = np.array([0.25, 0.62, 0.44, 0.49, 0.71])
    outcome = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    direction = candidate - book
    raw = float(np.dot(direction, outcome - book) / np.dot(direction, direction))

    blob = fit_probability_pool(candidate, book, outcome)

    assert blob["raw_rho"] == pytest.approx(raw)
    assert blob["rho"] == pytest.approx(np.clip(raw, 0.0, 1.0))
    np.testing.assert_allclose(
        apply_probability_pool(blob, candidate, book),
        book + blob["rho"] * (candidate - book),
    )


def test_probability_pool_rejects_degenerate_and_unstable_fits():
    with pytest.raises(ValueError, match="must differ"):
        fit_probability_pool(np.full(4, 0.5), np.full(4, 0.5), np.array([0, 1, 0, 1]))
    validate_probability_pool_weights([0.31, 0.44, 0.48, 0.39, 0.43], 0.42)
    with pytest.raises(ValueError, match="within"):
        validate_probability_pool_weights([0.19, 0.44, 0.48, 0.39, 0.43], 0.42)
    with pytest.raises(ValueError, match="range"):
        validate_probability_pool_weights([0.25, 0.46, 0.48, 0.39, 0.43], 0.42)


def test_zero_atom_maps_lower_and_upper_cdf_endpoints_separately():
    blob = _manual_blob()
    blob["affine"] = {
        "kind": "affine_marginal_mean",
        "intercept": 0.0,
        "slope": 1.0,
        "floor": rushing.MARGINAL_MEAN_FLOOR,
    }
    predictive = rushing.AffinePredictive(
        marginal_mean=np.array([9.0]),
        sigma=np.array([5.0]),
        alpha=np.array([0.0]),
        gate=np.array([0.2]),
    )
    conditional = 9.0 / 0.8
    raw_lower = 0.8 * skewnorm.cdf(0.0, 0.0, loc=conditional, scale=5.0)
    raw_upper = raw_lower + 0.2
    expected_lower = np.interp(raw_lower, [0.0, 0.5, 1.0], [0.0, 0.25, 1.0])
    expected_upper = np.interp(raw_upper, [0.0, 0.5, 1.0], [0.0, 0.25, 1.0])

    lower, upper = rushing.affine_cdf_endpoints(blob, predictive, np.array([0.0]), np.array([1]))
    pit = rushing.affine_randomized_pit(blob, predictive, np.array([0.0]), np.array([1]))

    assert lower[0] == pytest.approx(expected_lower)
    assert upper[0] == pytest.approx(expected_upper)
    assert np.all((pit[:, 0] >= lower[0]) & (pit[:, 0] <= upper[0]))
    assert np.ptp(pit[:, 0]) > 0.0


def test_apply_preserves_distribution_shape_and_marks_pool_as_line_only():
    blob = _manual_blob()
    predictive = rushing.AffinePredictive(
        marginal_mean=np.array([10.0, 30.0]),
        sigma=np.array([6.0, 12.0]),
        alpha=np.array([0.5, -0.25]),
        gate=np.array([0.1, 0.05]),
    )
    line = np.array([12.5, 31.5])
    book = np.array([0.46, 0.51])

    output = rushing.apply_affine_groupcdf(blob, predictive, line, book, np.array([1, 3]))

    expected_mean = 2.0 + 1.1 * predictive.marginal_mean
    np.testing.assert_allclose(output.marginal_mean, expected_mean)
    np.testing.assert_allclose(output.conditional_mean, expected_mean / (1.0 - predictive.gate))
    np.testing.assert_array_equal(output.sigma, predictive.sigma)
    np.testing.assert_array_equal(output.alpha, predictive.alpha)
    np.testing.assert_array_equal(output.gate, predictive.gate)
    np.testing.assert_allclose(output.pooled_over, book + 0.4 * (output.candidate_over - book))
    assert blob["line_probability_only"] is True


def test_portable_blob_round_trip_reproduces_application():
    blob = _manual_blob()
    payload = rushing.serialize_affine_calibration(blob)
    restored = rushing.deserialize_affine_calibration(payload)
    predictive = rushing.AffinePredictive(
        np.array([15.0, 45.0]),
        np.array([8.0, 18.0]),
        np.array([0.2, 0.8]),
        np.array([0.04, 0.02]),
    )
    args = (predictive, np.array([14.5, 47.5]), np.array([0.52, 0.43]), np.array([1, 3]))

    original = rushing.apply_affine_groupcdf(blob, *args)
    reloaded = rushing.apply_affine_groupcdf(restored, *args)

    assert json.loads(payload)["kind"] == rushing.AFFINE_STRATEGY_NAME
    for field in original.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(original, field), getattr(reloaded, field))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda blob: blob["affine"].update(intercept=np.nan), "affine intercept"),
        (lambda blob: blob["affine"].update(slope=2.0), "affine slope"),
        (lambda blob: blob["affine"].update(floor=0.0), "affine floor"),
        (lambda blob: blob.update(temperature=0.0), "temperature"),
        (
            lambda blob: blob["probability_pool"].update(raw_rho=0.9),
            "inconsistent with raw_rho",
        ),
        (lambda blob: blob["position_cdf"]["1"].update(x=[0.0, 1.1]), "CDF map"),
    ],
)
def test_blob_validation_rejects_malformed_numeric_state(mutation, message):
    blob = _manual_blob()
    mutation(blob)

    with pytest.raises(ValueError, match=message):
        rushing.serialize_affine_calibration(blob)


@pytest.mark.parametrize("missing", [None, np.nan])
def test_fit_rejects_missing_player_before_string_cast(missing):
    predictive, result, line, book, position, player = _synthetic_validation()
    player = player.astype(object)
    player[0] = missing

    with pytest.raises(ValueError, match="player identifiers"):
        rushing.fit_affine_groupcdf(
            predictive,
            result,
            line,
            book,
            np.ones(len(result), dtype=bool),
            position,
            player,
            expected_codes=(1, 3),
        )


def test_lambda_ties_prefer_lower_shrink(monkeypatch):
    monkeypatch.setattr(rushing_cdf, "ks_uniform", lambda _values: 0.25)
    draws = np.linspace(0.01, 0.99, 250).reshape(25, 10)
    players = np.array([f"p{i}" for i in range(10)])

    assert rushing_cdf.select_lambda(draws, players) == 0.0


def test_nested_player_cv_fit_is_finite_stable_and_reusable():
    predictive, result, line, book, position, player = _synthetic_validation()

    authentic = np.arange(len(result)) % 5 != 0
    book = book.copy()
    book[~authentic] = np.nan
    fit = rushing.fit_affine_groupcdf(
        predictive,
        result,
        line,
        book,
        authentic,
        position,
        player,
        expected_codes=(1, 3),
    )
    applied = rushing.apply_affine_groupcdf(
        fit.blob,
        predictive,
        line,
        book,
        position,
        authentic,
    )

    assert len(fit.fold_affines) == rushing.PLAYER_CV_FOLDS
    assert len(fit.fold_lambdas) == rushing.PLAYER_CV_FOLDS
    assert len(fit.fold_temperatures) == rushing.PLAYER_CV_FOLDS
    assert len(fit.fold_rhos) == rushing.PLAYER_CV_FOLDS
    assert all(0.2 <= rho <= 0.8 for rho in fit.fold_rhos)
    assert np.ptp(fit.fold_rhos) <= 0.2
    assert fit.oof_pit_draws.shape == (rushing.RANDOMIZED_PIT_DRAWS, len(result))
    assert all(
        np.isfinite(values).all()
        for values in (
            fit.oof_marginal_mean,
            fit.oof_candidate_over,
            fit.oof_pooled_over,
            fit.oof_pit_draws,
            applied.loc,
            applied.pooled_over,
        )
    )
    assert set(fit.blob["position_cdf"]) == {"1", "3"}
    assert fit.blob["position_codes"] == [1, 3]
    assert fit.blob["fit_audit"]["expected_position_codes"] == [1, 3]
    assert fit.blob["fit_audit"]["authentic_rows"] == int(authentic.sum())
    np.testing.assert_array_equal(
        applied.pooled_over[~authentic],
        applied.candidate_over[~authentic],
    )
    assert all(
        group_map["x"][0] == group_map["y"][0] == 0.0
        and group_map["x"][-1] == group_map["y"][-1] == 1.0
        for group_map in fit.blob["position_cdf"].values()
    )


@pytest.mark.parametrize("codes", [(1, 2, 3, 4, 5), (1, 2, 3, 4), (1, 2, 3)])
def test_schema2_generalizes_code_shapes_and_uses_model_only_fallback(codes):
    blob = _generalized_blob(codes)
    positions = np.array([codes[0], 99])
    predictive = rushing.AffinePredictive(
        np.array([12.0, 14.0]),
        np.array([4.0, 5.0]),
        np.array([0.2, 0.3]),
        np.array([0.05, 0.05]),
    )
    output = rushing.apply_affine_groupcdf(
        blob,
        predictive,
        np.array([11.5, 13.5]),
        np.array([0.55, np.nan]),
        positions,
        np.array([True, False]),
    )

    assert np.isfinite(output.candidate_over).all()
    assert output.pooled_over[0] != output.candidate_over[0]
    assert output.pooled_over[1] == output.candidate_over[1]


def test_schema2_nonauthentic_supported_quote_is_never_pooled():
    blob = _generalized_blob((1, 3))
    predictive = rushing.AffinePredictive(
        np.array([12.0, 14.0]),
        np.array([4.0, 5.0]),
        np.array([0.2, 0.3]),
        np.array([0.05, 0.05]),
    )
    output = rushing.apply_affine_groupcdf(
        blob,
        predictive,
        np.array([11.5, 13.5]),
        np.array([0.55, 0.45]),
        np.array([1, 3]),
        np.array([True, False]),
    )

    assert output.pooled_over[0] != output.candidate_over[0]
    assert output.pooled_over[1] == output.candidate_over[1]


def test_schema2_rejects_fractional_position_before_code_discovery():
    blob = _generalized_blob((1, 3))
    predictive = rushing.AffinePredictive(
        np.array([12.0]),
        np.array([4.0]),
        np.array([0.2]),
        np.array([0.05]),
    )

    with pytest.raises(ValueError, match="integer league roster codes"):
        rushing.apply_affine_groupcdf(
            blob,
            predictive,
            np.array([11.5]),
            np.array([0.55]),
            np.array([1.5]),
            np.array([True]),
        )


def test_schema2_roundtrip_persists_code_set_and_fallback_contract():
    blob = _generalized_blob((1, 2, 3, 4))
    restored = rushing.deserialize_affine_calibration(rushing.serialize_affine_calibration(blob))

    assert restored["position_codes"] == [1, 2, 3, 4]
    assert restored["fallback"]["unseen_position"] == "raw_cdf_unpooled"


def test_schema2_uses_train_scaled_intercept_bounds_without_changing_legacy_bounds():
    generalized = _generalized_blob((1, 3))
    generalized["affine"]["intercept"] = 30.0
    rushing.serialize_affine_calibration(generalized)

    legacy = _manual_blob()
    legacy["affine"]["intercept"] = 30.0
    with pytest.raises(ValueError, match="affine intercept"):
        rushing.serialize_affine_calibration(legacy)


def test_unsupported_strategy_config_fails_before_fit_dispatch():
    unsupported = replace(rushing.AFFINE_CONFIG, boundary=True)

    with pytest.raises(ValueError, match="unsupported group-conditional CDF strategy config"):
        rushing.fit_group_conditional_cdf(unsupported)
