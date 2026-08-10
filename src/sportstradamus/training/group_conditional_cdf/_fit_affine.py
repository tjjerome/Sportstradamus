"""Nested-CV fit body for the affine strategy.

Moved verbatim from the affine calibrator: a five-fold Player-grouped outer CV
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
    fit_affine_mean,
    fit_group_maps,
    mapped_cdf_endpoints,
    mapped_over,
    subset_predictive,
)
from sportstradamus.training.group_conditional_cdf._config import StrategyConfig
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
    affine_bounds_for_training,
    affine_fit_inputs,
    validated_affine_blob,
)
from sportstradamus.training.group_conditional_cdf.probability_pool import (
    apply_probability_pool,
    fit_probability_pool,
    validate_probability_pool_weights,
)


def fit_affine(
    config: StrategyConfig,
    predictive,
    result,
    line,
    book_over,
    authentic,
    position,
    player,
    expected_codes,
):
    """Fit the affine strategy by nested five-fold Player CV."""
    pred, y, lines, books, authentic_rows, positions, players, codes = affine_fit_inputs(
        predictive,
        result,
        line,
        book_over,
        authentic,
        position,
        player,
        expected_codes,
    )
    n_rows = len(y)
    eligible = np.isin(positions, codes)
    bounds = affine_bounds_for_training(pred.marginal_mean[eligible], y[eligible])
    uniforms = np.random.default_rng(RANDOMIZED_PIT_SEED).random((RANDOMIZED_PIT_DRAWS, n_rows))
    oof_mean = np.empty(n_rows, dtype=float)
    oof_candidate = np.empty(n_rows, dtype=float)
    oof_pooled = np.empty(n_rows, dtype=float)
    oof_pit = np.empty((RANDOMIZED_PIT_DRAWS, n_rows), dtype=float)
    affine_folds: list[tuple[float, float]] = []
    lambda_folds: list[tuple[float, float]] = []
    temperatures: list[float] = []
    rhos: list[float] = []
    fold_support: list[dict[str, object]] = []
    over_result = (y >= lines).astype(float)

    splitter = GroupKFold(n_splits=PLAYER_CV_FOLDS)
    folds = tuple(splitter.split(np.zeros(n_rows), groups=players))
    _preflight_affine_folds(folds, positions, players, authentic_rows, eligible, codes)
    for train, hold in folds:
        train_eligible = eligible[train]
        hold_eligible = eligible[hold]
        affine = fit_affine_mean(
            pred.marginal_mean[train][train_eligible],
            y[train][train_eligible],
        )
        corrected = apply_affine(pred, *affine)
        train_pred, hold_pred = (
            subset_predictive(corrected, train),
            subset_predictive(corrected, hold),
        )
        maps = fit_group_maps(
            train_pred, y[train], positions[train], players[train], uniforms[:, train], codes
        )
        lower, upper = mapped_cdf_endpoints(
            hold_pred,
            y[hold],
            positions[hold],
            maps,
            codes,
            model_only_fallback=True,
        )
        oof_pit[:, hold] = lower[None, :] + uniforms[:, hold] * (upper - lower)[None, :]
        train_raw_over = mapped_over(
            train_pred,
            lines[train],
            positions[train],
            maps,
            codes,
            model_only_fallback=True,
        )
        temperature = fit_temperature_affine(
            train_raw_over[train_eligible],
            over_result[train][train_eligible],
        )
        train_candidate = apply_temperature(train_raw_over, temperature)
        hold_candidate = apply_temperature(
            mapped_over(
                hold_pred,
                lines[hold],
                positions[hold],
                maps,
                codes,
                model_only_fallback=True,
            ),
            temperature,
        )
        train_pool_rows = train_eligible & authentic_rows[train]
        hold_pool_rows = hold_eligible & authentic_rows[hold]
        pool = fit_probability_pool(
            train_candidate[train_pool_rows],
            books[train][train_pool_rows],
            over_result[train][train_pool_rows],
        )
        oof_candidate[hold] = hold_candidate
        oof_pooled[hold] = _apply_affine_pool(
            pool,
            hold_candidate,
            books[hold],
            hold_pool_rows,
        )
        oof_mean[hold] = corrected.marginal_mean[hold]
        affine_folds.append(affine)
        lambda_folds.append(tuple(maps[str(code)]["lam"] for code in codes))
        temperatures.append(temperature)
        rhos.append(pool["rho"])
        fold_support.append(
            _fold_support_audit(
                train,
                hold,
                positions,
                authentic_rows,
                eligible,
                codes,
            )
        )

    final_affine = fit_affine_mean(pred.marginal_mean[eligible], y[eligible])
    corrected = apply_affine(pred, *final_affine)
    final_maps = fit_group_maps(corrected, y, positions, players, uniforms, codes)
    final_raw_over = mapped_over(
        corrected,
        lines,
        positions,
        final_maps,
        codes,
        model_only_fallback=True,
    )
    final_temperature = fit_temperature_affine(
        final_raw_over[eligible],
        over_result[eligible],
    )
    final_candidate = apply_temperature(final_raw_over, final_temperature)
    final_pool_rows = eligible & authentic_rows
    final_pool = fit_probability_pool(
        final_candidate[final_pool_rows],
        books[final_pool_rows],
        over_result[final_pool_rows],
    )
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
        "position_codes": list(codes),
        "fallback": {
            "kind": "model_only",
            "unseen_position": "raw_cdf_unpooled",
            "nonauthentic_quote": "candidate_unpooled",
        },
        "affine_bounds": bounds,
        "fit_audit": {
            "expected_position_codes": list(codes),
            "rows": n_rows,
            "eligible_rows": int(eligible.sum()),
            "authentic_rows": int(authentic_rows.sum()),
            "eligible_authentic_rows": int(final_pool_rows.sum()),
            "outer_folds": fold_support,
        },
        "temperature": final_temperature,
        "probability_pool": final_pool,
    }
    _validate_affine_fit(
        blob,
        affine_folds,
        temperatures,
        rhos,
        (oof_mean, oof_candidate, oof_pooled, oof_pit),
        bounds,
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


def _validate_affine_fit(blob, affine_folds, temperatures, rhos, outputs, bounds):
    fitted = validated_affine_blob(blob)
    affines = [*affine_folds, (fitted["affine"]["intercept"], fitted["affine"]["slope"])]
    a_low, a_high = bounds["intercept"]
    b_low, b_high = bounds["slope"]
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


def _apply_affine_pool(pool, candidate, books, use_rows):
    pooled = np.asarray(candidate, dtype=float).copy()
    if np.any(use_rows):
        pooled[use_rows] = apply_probability_pool(
            pool,
            pooled[use_rows],
            np.asarray(books, dtype=float)[use_rows],
        )
    return pooled


def _preflight_affine_folds(folds, positions, players, authentic, eligible, codes):
    for fold_number, (train, _hold) in enumerate(folds):
        for code in codes:
            if len(np.unique(players[train][positions[train] == code])) < PLAYER_CV_FOLDS:
                raise ValueError(
                    f"outer fold {fold_number} position {code} lacks five map-validation players"
                )
        if int((eligible[train] & authentic[train]).sum()) < 2:
            raise ValueError(
                f"outer fold {fold_number} lacks authentic eligible rows for probability pooling"
            )


def _fold_support_audit(train, hold, positions, authentic, eligible, codes):
    return {
        "train_rows": len(train),
        "hold_rows": len(hold),
        "train_eligible_rows": int(eligible[train].sum()),
        "hold_eligible_rows": int(eligible[hold].sum()),
        "train_authentic_rows": int(authentic[train].sum()),
        "hold_authentic_rows": int(authentic[hold].sum()),
        "train_eligible_authentic_rows": int((eligible[train] & authentic[train]).sum()),
        "hold_eligible_authentic_rows": int((eligible[hold] & authentic[hold]).sum()),
        "train_position_rows": {str(code): int((positions[train] == code).sum()) for code in codes},
        "hold_position_rows": {str(code): int((positions[hold] == code).sum()) for code in codes},
    }
