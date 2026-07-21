"""Two-part logit-boundary and positive group-CDF kernels (two-part strategy).

Moved verbatim from the two-part calibrator. The per-role boundary
intercept, RB-position residual, role nonpositive maps, and role-by-position
positive maps together define the conditional split whose endpoints the positive
group-CDF consumes. Position codes and the positive-group list are threaded in as
arguments so the engine iterates the discovered/persisted set.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from sportstradamus.training.group_conditional_cdf._contracts import (
    BOUNDARY_LOGIT_CLIP,
    CDF_BRANCH_TOLERANCE,
    CDF_CLIP,
    INTERCEPT_BRACKET,
    RB_CODE,
    ROLE_VALUES,
    CalibrationRows,
    TwoPartCdfBlob,
)
from sportstradamus.training.group_conditional_cdf._maps import apply_endpoint_map, fit_endpoint_map


def positive_group(role: np.ndarray, position: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{role_name}_pos{code}" for role_name, code in zip(role, position, strict=True)],
        dtype=str,
    )


def fit_models(
    rows: CalibrationRows,
    uniforms: np.ndarray,
    positive_lam: float,
    nonpositive_lam: float,
    positive_groups,
) -> TwoPartCdfBlob:
    role_models = {}
    for role_name in ROLE_VALUES:
        mask = rows.role == role_name
        role_nonpositive = mask & (rows.result <= 0.0)
        intercept = fit_boundary_intercept(
            rows.zero_cdf[mask],
            rows.result[mask] <= 0.0,
        )
        nonpositive_lower = rows.result_lower[role_nonpositive] / np.maximum(
            rows.zero_cdf[role_nonpositive], CDF_CLIP
        )
        nonpositive_width = (
            rows.result_upper[role_nonpositive] - rows.result_lower[role_nonpositive]
        ) / np.maximum(rows.zero_cdf[role_nonpositive], CDF_CLIP)
        nonpositive_samples = np.clip(
            nonpositive_lower[None, :] + uniforms[:, role_nonpositive] * nonpositive_width[None, :],
            0.0,
            1.0,
        ).reshape(-1)
        role_models[role_name] = {
            "kind": "two_part_role_boundary",
            "intercept": intercept,
            "nonpositive": fit_endpoint_map(nonpositive_samples, nonpositive_lam),
        }

    rb = rows.position == RB_CODE
    rb_eta = logit(
        np.clip(
            rows.zero_cdf[rb],
            BOUNDARY_LOGIT_CLIP,
            1.0 - BOUNDARY_LOGIT_CLIP,
        )
    ) + np.asarray(
        [role_models[role_name]["intercept"] for role_name in rows.role[rb]],
        dtype=float,
    )
    rb_target = float(np.mean(rows.result[rb] <= 0.0))
    try:
        rb_residual = float(
            brentq(
                lambda value: float(np.mean(expit(rb_eta + value)) - rb_target),
                *INTERCEPT_BRACKET,
            )
        )
    except ValueError as exc:
        raise ValueError("two-part RB boundary residual is outside its fit bracket") from exc

    groups = positive_group(rows.role, rows.position)
    positive_maps = {}
    for group in positive_groups:
        mask = (rows.result > 0.0) & (groups == group)
        samples = np.clip(
            (rows.result_upper[mask] - rows.zero_cdf[mask])
            / np.maximum(1.0 - rows.zero_cdf[mask], CDF_CLIP),
            0.0,
            1.0,
        )
        positive_maps[group] = fit_endpoint_map(samples, positive_lam)
    return {
        "kind": "role_position_two_part_cdf",
        "role_boundary": role_models,
        "positive": positive_maps,
        "rb_boundary_residual": rb_residual,
    }


def fit_boundary_intercept(f0: np.ndarray, nonpositive: np.ndarray) -> float:
    target = float(np.mean(nonpositive))
    if not 0.0 < target < 1.0:
        raise ValueError("each role fit requires both positive and nonpositive outcomes")
    eta = logit(np.clip(f0, BOUNDARY_LOGIT_CLIP, 1.0 - BOUNDARY_LOGIT_CLIP))

    def score(intercept: float) -> float:
        return float(np.mean(expit(eta + intercept)) - target)

    try:
        return float(brentq(score, *INTERCEPT_BRACKET))
    except ValueError as exc:
        raise ValueError("two-part boundary intercept is outside its fit bracket") from exc


def apply_models(models, cdf, f0, roles, positions, positive_groups):
    output = np.empty_like(cdf, dtype=float)
    groups = positive_group(roles, positions)
    for role_name in ROLE_VALUES:
        mask = roles == role_name
        raw_boundary = np.clip(f0[mask], CDF_CLIP, 1.0 - CDF_CLIP)
        fitted_boundary = mapped_boundary(
            models,
            f0[mask],
            roles[mask],
            positions[mask],
        )
        lower_side = cdf[mask] <= f0[mask] + CDF_BRANCH_TOLERANCE
        conditional = np.empty(mask.sum(), dtype=float)
        conditional[lower_side] = np.clip(
            cdf[mask][lower_side] / raw_boundary[lower_side],
            0.0,
            1.0,
        )
        conditional[~lower_side] = np.clip(
            (cdf[mask][~lower_side] - f0[mask][~lower_side]) / (1.0 - raw_boundary[~lower_side]),
            0.0,
            1.0,
        )
        transformed = np.empty(mask.sum(), dtype=float)
        transformed[lower_side] = fitted_boundary[lower_side] * apply_endpoint_map(
            models["role_boundary"][role_name]["nonpositive"], conditional[lower_side]
        )
        local_groups = groups[mask]
        for group in positive_groups:
            group_mask = (~lower_side) & (local_groups == group)
            if not np.any(group_mask):
                continue
            transformed[group_mask] = fitted_boundary[group_mask] + (
                1.0 - fitted_boundary[group_mask]
            ) * apply_endpoint_map(models["positive"][group], conditional[group_mask])
        output[mask] = transformed
    return np.clip(output, 0.0, 1.0)


def mapped_boundary(models, f0, roles, positions):
    boundary = np.empty_like(f0, dtype=float)
    for role_name in ROLE_VALUES:
        mask = roles == role_name
        residual = models["rb_boundary_residual"] * (positions[mask] == RB_CODE)
        boundary[mask] = expit(
            logit(np.clip(f0[mask], BOUNDARY_LOGIT_CLIP, 1.0 - BOUNDARY_LOGIT_CLIP))
            + models["role_boundary"][role_name]["intercept"]
            + residual
        )
    return boundary


def positive_conditional_pit(mapped_upper, mapped_boundary_values):
    return np.clip(
        (mapped_upper - mapped_boundary_values)
        / np.maximum(1.0 - mapped_boundary_values, CDF_CLIP),
        0.0,
        1.0,
    )
