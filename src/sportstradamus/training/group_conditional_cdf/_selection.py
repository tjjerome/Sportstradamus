"""Nested Player-CV lambda selection (two-part strategy).

Moved verbatim from the two-part calibrator: a grid search over the
positive/nonpositive shrink pair, scored by a guard-normalized KS key across the
global, per-role, and per-position PIT. Position codes and the positive-group
list are threaded in so the engine iterates the discovered/persisted set.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.model_selection import GroupKFold

from sportstradamus.training.group_conditional_cdf._boundary import (
    apply_models,
    fit_models,
    mapped_boundary,
    positive_conditional_pit,
)
from sportstradamus.training.group_conditional_cdf._contracts import (
    GLOBAL_PIT_GUARD,
    PIT_LAMBDAS,
    PLAYER_CV_FOLDS,
    POSITION_FULL_PIT_GUARD,
    POSITION_POSITIVE_PIT_GUARD,
    RANDOMIZED_PIT_DRAWS,
    ROLE_FULL_PIT_GUARDS,
    ROLE_POSITIVE_PIT_GUARDS,
    ROLE_VALUES,
    CalibrationRows,
    CrossfitSelection,
)
from sportstradamus.training.group_conditional_cdf._maps import ks_uniform_finite, mean_ks
from sportstradamus.training.group_conditional_cdf._validation import subset_rows


def select_lambdas_crossfit(
    rows: CalibrationRows,
    uniforms: np.ndarray,
    position_codes,
    positive_groups,
) -> CrossfitSelection:
    n_rows = len(rows.result)
    if uniforms.shape != (RANDOMIZED_PIT_DRAWS, n_rows):
        raise ValueError("uniform draws must be the seeded 25-by-row matrix")
    splitter = GroupKFold(n_splits=PLAYER_CV_FOLDS)
    best_key: tuple[float, float, float, float, float] | None = None
    best_lambdas: tuple[float, float] | None = None
    for positive_lam, nonpositive_lam in product(PIT_LAMBDAS, repeat=2):
        oof_lower = np.empty(n_rows, dtype=float)
        oof_upper = np.empty(n_rows, dtype=float)
        oof_boundary = np.empty(n_rows, dtype=float)
        for train, hold in splitter.split(np.zeros(n_rows), groups=rows.player):
            train_rows = subset_rows(rows, train)
            hold_rows = subset_rows(rows, hold)
            models = fit_models(
                train_rows,
                uniforms[:, train],
                positive_lam,
                nonpositive_lam,
                positive_groups,
            )
            oof_lower[hold] = apply_models(
                models,
                hold_rows.result_lower,
                hold_rows.zero_cdf,
                hold_rows.role,
                hold_rows.position,
                positive_groups,
            )
            oof_upper[hold] = apply_models(
                models,
                hold_rows.result_upper,
                hold_rows.zero_cdf,
                hold_rows.role,
                hold_rows.position,
                positive_groups,
            )
            oof_boundary[hold] = mapped_boundary(
                models,
                hold_rows.zero_cdf,
                hold_rows.role,
                hold_rows.position,
            )
        draws = oof_lower[None, :] + uniforms * (oof_upper - oof_lower)[None, :]
        positive_pit = np.full(n_rows, np.nan, dtype=float)
        positive = rows.result > 0.0
        positive_pit[positive] = positive_conditional_pit(
            oof_upper[positive],
            oof_boundary[positive],
        )
        key = selection_key(
            draws,
            positive_pit,
            rows.result,
            rows.role,
            rows.position,
            positive_lam,
            nonpositive_lam,
            position_codes,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_lambdas = (positive_lam, nonpositive_lam)
    if best_lambdas is None or best_key is None:
        raise ValueError("lambda selection did not produce a candidate")
    return CrossfitSelection(best_lambdas, best_key)


def selection_key(
    draws: np.ndarray,
    positive_pit: np.ndarray,
    result: np.ndarray,
    role: np.ndarray,
    position: np.ndarray,
    positive_lam: float,
    nonpositive_lam: float,
    position_codes,
) -> tuple[float, float, float, float, float]:
    positive = result > 0.0
    global_ks = mean_ks(draws)
    normalized = [global_ks / GLOBAL_PIT_GUARD]
    for role_name in ROLE_VALUES:
        role_mask = role == role_name
        normalized.append(mean_ks(draws[:, role_mask]) / ROLE_FULL_PIT_GUARDS[role_name])
        normalized.append(
            ks_uniform_finite(positive_pit[role_mask & positive])
            / ROLE_POSITIVE_PIT_GUARDS[role_name]
        )
    for code in position_codes:
        position_mask = position == code
        normalized.append(mean_ks(draws[:, position_mask]) / POSITION_FULL_PIT_GUARD)
        normalized.append(
            ks_uniform_finite(positive_pit[position_mask & positive]) / POSITION_POSITIVE_PIT_GUARD
        )
    return (
        max(normalized),
        global_ks,
        positive_lam + nonpositive_lam,
        positive_lam,
        nonpositive_lam,
    )
