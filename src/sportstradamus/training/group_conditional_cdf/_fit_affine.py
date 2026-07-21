"""Nested-CV fit body for the rushing affine corner.

Moved verbatim from the rushing calibrator: a five-fold Player-grouped outer CV
that fits the affine mean correction and per-position maps, then the line-only
temperature and Brier-optimal book pool, scoring the honest out-of-fold arrays.
Group codes are discovered from the position column and persisted as the
``position_cdf`` keys.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from sportstradamus.helpers.distributions import apply_temperature
from sportstradamus.training.group_conditional_cdf._affine import (
    apply_affine,
    fit_affine,
    fit_group_maps,
    mapped_cdf_endpoints,
    mapped_over,
    subset_predictive,
)
from sportstradamus.training.group_conditional_cdf._config import StrategyConfig, discover_codes
from sportstradamus.training.group_conditional_cdf._contracts import (
    AFFINE_INTERCEPT_BOUNDS,
    AFFINE_SLOPE_BOUNDS,
    MARGINAL_MEAN_FLOOR,
    PLAYER_CV_FOLDS,
    RANDOMIZED_PIT_DRAWS,
    RANDOMIZED_PIT_SEED,
    TEMPERATURE_BOUNDS,
    AffineCalibrationBlob,
    AffineCalibrationFit,
)
from sportstradamus.training.group_conditional_cdf._line_head import fit_temperature_affine
from sportstradamus.training.group_conditional_cdf._validation_affine import (
    affine_fit_inputs,
    validated_affine_blob,
)
from sportstradamus.training.group_conditional_cdf.probability_pool import (
    apply_probability_pool,
    fit_probability_pool,
    validate_probability_pool_weights,
)


def fit_rushing(config: StrategyConfig, predictive, result, line, book_over, position, player):
    """Fit the rushing affine corner by nested five-fold Player CV."""
    codes = discover_codes(position)
    pred, y, lines, books, positions, players = affine_fit_inputs(
        predictive, result, line, book_over, position, player
    )
    n_rows = len(y)
    uniforms = np.random.default_rng(RANDOMIZED_PIT_SEED).random((RANDOMIZED_PIT_DRAWS, n_rows))
    oof_mean = np.empty(n_rows, dtype=float)
    oof_candidate = np.empty(n_rows, dtype=float)
    oof_pooled = np.empty(n_rows, dtype=float)
    oof_pit = np.empty((RANDOMIZED_PIT_DRAWS, n_rows), dtype=float)
    affine_folds: list[tuple[float, float]] = []
    lambda_folds: list[tuple[float, float]] = []
    temperatures: list[float] = []
    rhos: list[float] = []
    over_result = (y >= lines).astype(float)

    splitter = GroupKFold(n_splits=PLAYER_CV_FOLDS)
    for train, hold in splitter.split(np.zeros(n_rows), groups=players):
        affine = fit_affine(pred.marginal_mean[train], y[train])
        corrected = apply_affine(pred, *affine)
        train_pred, hold_pred = (
            subset_predictive(corrected, train),
            subset_predictive(corrected, hold),
        )
        maps = fit_group_maps(
            train_pred, y[train], positions[train], players[train], uniforms[:, train], codes
        )
        lower, upper = mapped_cdf_endpoints(hold_pred, y[hold], positions[hold], maps, codes)
        oof_pit[:, hold] = lower[None, :] + uniforms[:, hold] * (upper - lower)[None, :]
        train_raw_over = mapped_over(train_pred, lines[train], positions[train], maps, codes)
        temperature = fit_temperature_affine(train_raw_over, over_result[train])
        train_candidate = apply_temperature(train_raw_over, temperature)
        hold_candidate = apply_temperature(
            mapped_over(hold_pred, lines[hold], positions[hold], maps, codes), temperature
        )
        pool = fit_probability_pool(train_candidate, books[train], over_result[train])
        oof_candidate[hold] = hold_candidate
        oof_pooled[hold] = apply_probability_pool(pool, hold_candidate, books[hold])
        oof_mean[hold] = corrected.marginal_mean[hold]
        affine_folds.append(affine)
        lambda_folds.append(tuple(maps[str(code)]["lam"] for code in codes))
        temperatures.append(temperature)
        rhos.append(pool["rho"])

    final_affine = fit_affine(pred.marginal_mean, y)
    corrected = apply_affine(pred, *final_affine)
    final_maps = fit_group_maps(corrected, y, positions, players, uniforms, codes)
    final_raw_over = mapped_over(corrected, lines, positions, final_maps, codes)
    final_temperature = fit_temperature_affine(final_raw_over, over_result)
    final_candidate = apply_temperature(final_raw_over, final_temperature)
    final_pool = fit_probability_pool(final_candidate, books, over_result)
    blob: AffineCalibrationBlob = {
        "kind": config.candidate_name,
        "schema_version": config.schema_version,
        "line_probability_only": True,
        "affine": {
            "kind": "affine_marginal_mean",
            "intercept": final_affine[0],
            "slope": final_affine[1],
            "floor": MARGINAL_MEAN_FLOOR,
        },
        "position_cdf": final_maps,
        "temperature": final_temperature,
        "probability_pool": final_pool,
    }
    _validate_affine_fit(
        blob, affine_folds, temperatures, rhos, (oof_mean, oof_candidate, oof_pooled, oof_pit)
    )
    return AffineCalibrationFit(
        blob=blob,
        oof_marginal_mean=oof_mean,
        oof_candidate_over=oof_candidate,
        oof_pooled_over=oof_pooled,
        oof_pit_draws=oof_pit,
        fold_affines=tuple(affine_folds),
        fold_lambdas=tuple(lambda_folds),
        fold_temperatures=tuple(temperatures),
        fold_rhos=tuple(rhos),
    )


def _validate_affine_fit(blob, affine_folds, temperatures, rhos, outputs):
    fitted = validated_affine_blob(blob)
    affines = [*affine_folds, (fitted["affine"]["intercept"], fitted["affine"]["slope"])]
    a_low, a_high = AFFINE_INTERCEPT_BOUNDS
    b_low, b_high = AFFINE_SLOPE_BOUNDS
    if any(not a_low <= a <= a_high or not b_low <= b <= b_high for a, b in affines):
        raise ValueError("affine marginal-mean fit breached preregistered bounds")
    all_temperatures = np.asarray([*temperatures, fitted["temperature"]], dtype=float)
    low, high = TEMPERATURE_BOUNDS
    if not np.isfinite(all_temperatures).all() or np.any(
        (all_temperatures < low) | (all_temperatures > high)
    ):
        raise ValueError("temperature fit breached preregistered bounds")
    validate_probability_pool_weights(rhos, fitted["probability_pool"]["rho"])
    if any(not np.isfinite(output).all() for output in outputs):
        raise ValueError("nested calibration produced nonfinite OOF output")
