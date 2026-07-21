"""Nested-CV fit body for the two-part strategy.

Moved verbatim from the two-part calibrator: a five-fold Player-grouped outer CV
that fits the boundary-plus-positive maps, settles the line endpoints, and scores
the honest out-of-fold PIT and quoted-line arrays. Group codes are discovered from
the position column and persisted as the positive-map keys.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from sportstradamus.helpers.distributions import apply_temperature
from sportstradamus.training.group_conditional_cdf._boundary import (
    apply_models,
    fit_models,
    mapped_boundary,
    positive_conditional_pit,
)
from sportstradamus.training.group_conditional_cdf._config import (
    StrategyConfig,
    discover_codes,
    positive_groups,
)
from sportstradamus.training.group_conditional_cdf._contracts import (
    CDF_BRANCH_TOLERANCE,
    PLAYER_CV_FOLDS,
    RANDOMIZED_PIT_DRAWS,
    RANDOMIZED_PIT_SEED,
    ROLE_VALUES,
    TwoPartCalibrationBlob,
    TwoPartCalibrationFit,
)
from sportstradamus.training.group_conditional_cdf._line_head import fit_temperature_two_part
from sportstradamus.training.group_conditional_cdf._pool import fixed_pool_blob, pooled_probability
from sportstradamus.training.group_conditional_cdf._selection import select_lambdas_crossfit
from sportstradamus.training.group_conditional_cdf._validation import (
    subset_rows,
    two_part_fit_inputs,
    validated_two_part_blob,
)


def fit_two_part(config: StrategyConfig, *args, residual_positions=()) -> TwoPartCalibrationFit:
    """Fit the two-part strategy by nested five-fold Player CV."""
    position_codes = discover_codes(args[11])
    groups = positive_groups(position_codes)
    rows = two_part_fit_inputs(*args, position_codes)
    n_rows = len(rows.result)
    uniforms = np.random.default_rng(RANDOMIZED_PIT_SEED).random((RANDOMIZED_PIT_DRAWS, n_rows))
    oof_pit = np.empty((RANDOMIZED_PIT_DRAWS, n_rows), dtype=float)
    oof_positive = np.full(n_rows, np.nan, dtype=float)
    oof_line_low = np.empty(n_rows, dtype=float)
    oof_line_high = np.empty(n_rows, dtype=float)
    oof_mapped_under = np.empty(n_rows, dtype=float)
    oof_candidate = np.empty(n_rows, dtype=float)
    oof_pooled = np.empty(n_rows, dtype=float)
    fold_lambdas: list[tuple[float, float]] = []
    fold_intercepts: list[dict[str, float]] = []
    fold_temperatures: list[float] = []
    fold_selection_keys: list[tuple[float, float, float, float, float]] = []
    raw_over = 1.0 - 0.5 * (rows.line_low + rows.line_high)

    splitter = GroupKFold(n_splits=PLAYER_CV_FOLDS)
    for train, hold in splitter.split(np.zeros(n_rows), groups=rows.player):
        train_rows = subset_rows(rows, train)
        hold_rows = subset_rows(rows, hold)
        selection = select_lambdas_crossfit(
            train_rows, uniforms[:, train], position_codes, groups, residual_positions
        )
        fold_temperature = fit_temperature_two_part(raw_over[train], rows.over_result[train])
        models = fit_models(
            train_rows, uniforms[:, train], *selection.lambdas, groups, residual_positions
        )
        mapped_lower = apply_models(
            models,
            hold_rows.result_lower,
            hold_rows.zero_cdf,
            hold_rows.role,
            hold_rows.position,
            groups,
            residual_positions,
        )
        mapped_upper = apply_models(
            models,
            hold_rows.result_upper,
            hold_rows.zero_cdf,
            hold_rows.role,
            hold_rows.position,
            groups,
            residual_positions,
        )
        boundary = mapped_boundary(
            models, hold_rows.zero_cdf, hold_rows.role, hold_rows.position, residual_positions
        )
        mapped_line_low = apply_models(
            models,
            hold_rows.line_low,
            hold_rows.zero_cdf,
            hold_rows.role,
            hold_rows.position,
            groups,
            residual_positions,
        )
        mapped_line_high = apply_models(
            models,
            hold_rows.line_high,
            hold_rows.zero_cdf,
            hold_rows.role,
            hold_rows.position,
            groups,
            residual_positions,
        )
        mapped_under = 0.5 * (mapped_line_low + mapped_line_high)
        hold_candidate = apply_temperature(1.0 - mapped_under, fold_temperature)

        oof_pit[:, hold] = (
            mapped_lower[None, :] + uniforms[:, hold] * (mapped_upper - mapped_lower)[None, :]
        )
        positive = hold_rows.result > 0.0
        oof_positive[hold[positive]] = positive_conditional_pit(
            mapped_upper[positive], boundary[positive]
        )
        oof_line_low[hold] = mapped_line_low
        oof_line_high[hold] = mapped_line_high
        oof_mapped_under[hold] = mapped_under
        oof_candidate[hold] = hold_candidate
        oof_pooled[hold] = pooled_probability(
            hold_candidate, hold_rows.book_over, hold_rows.authentic
        )
        fold_lambdas.append(selection.lambdas)
        fold_intercepts.append(
            {
                **{
                    role_name: models["role_boundary"][role_name]["intercept"]
                    for role_name in ROLE_VALUES
                },
                "rb_residual": models["rb_boundary_residual"],
            }
        )
        fold_temperatures.append(fold_temperature)
        fold_selection_keys.append(selection.key)

    final_selection = select_lambdas_crossfit(
        rows, uniforms, position_codes, groups, residual_positions
    )
    final_temperature = fit_temperature_two_part(raw_over, rows.over_result)
    final_models = fit_models(rows, uniforms, *final_selection.lambdas, groups, residual_positions)
    blob: TwoPartCalibrationBlob = {
        "kind": config.candidate_name,
        "schema_version": config.schema_version,
        "line_probability_only": True,
        "temperature_fit_scope": "pre_map_raw_endpoint_settlement",
        "temperature": final_temperature,
        "cdf": final_models,
        "probability_pool": fixed_pool_blob(),
    }
    validated_two_part_blob(blob)
    if (
        not all(
            np.isfinite(values).all()
            for values in (
                oof_pit,
                oof_line_low,
                oof_line_high,
                oof_mapped_under,
                oof_candidate,
                oof_pooled,
            )
        )
        or not np.isfinite(oof_positive[rows.result > 0.0]).all()
    ):
        raise ValueError("nested two-part calibration produced nonfinite OOF output")
    if np.any(oof_line_low > oof_line_high + CDF_BRANCH_TOLERANCE):
        raise ValueError("mapped line lower endpoint exceeds upper endpoint")
    return TwoPartCalibrationFit(
        blob=blob,
        oof_pit_draws=oof_pit,
        oof_positive_pit=oof_positive,
        oof_mapped_low=oof_line_low,
        oof_mapped_high=oof_line_high,
        oof_mapped_under=oof_mapped_under,
        oof_candidate_over=oof_candidate,
        oof_pooled_over=oof_pooled,
        fold_lambdas=tuple(fold_lambdas),
        fold_intercepts=tuple(fold_intercepts),
        fold_temperatures=tuple(fold_temperatures),
        fold_selection_keys=tuple(fold_selection_keys),
    )
