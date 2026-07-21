"""Train-market fit/gate/apply step for the rushing QB/RB affine group-CDF candidate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sportstradamus.training.group_conditional_cdf import fit_affine_groupcdf
from sportstradamus.training.group_conditional_cdf._apply import apply_affine_groupcdf
from sportstradamus.training.group_conditional_cdf._contracts import (
    AffineCalibrationFit,
    AffinePredictive,
)
from sportstradamus.training.group_conditional_cdf._pipeline_steps_shared import (
    _ks_uniform,
    _oof_brier_arrays,
    _validation_split_fingerprint,
)
from sportstradamus.training.scorecard import (
    _bootstrap_ratio_ci_clustered,
    _gate4_pit_ks_threshold,
    _gate5_ece_debiased,
    _gate23_segment_match,
    _segment_masks,
)
from sportstradamus.training.structural_strategies import (
    AFFINE_POSITIONS,
    AFFINE_STRATEGY,
)


def _affine_candidate_oof_audit(
    fit: AffineCalibrationFit,
    splits: dict,
    positions: np.ndarray,
) -> dict:
    """Validate the rushing candidate without consulting the held-out test outcomes."""
    # style: allow-complexity -- aggregates independent out-of-fold guards on the rushing candidate
    result, players, outcome, _book, brier_delta, row_ci, player_ci = _oof_brier_arrays(fit, splits)

    pit_global = float(np.mean([_ks_uniform(draw) for draw in fit.oof_pit_draws]))
    pit_by_position = {
        AFFINE_POSITIONS[code]: float(
            np.mean([_ks_uniform(draw[positions == code]) for draw in fit.oof_pit_draws])
        )
        for code in AFFINE_POSITIONS
    }
    gate_frame = splits["X_validation"].copy()
    gate_frame["Result"] = result
    gate_frame["Player"] = players
    star_mask, bench_mask = _segment_masks(gate_frame)
    gate2_z = _gate23_segment_match(fit.oof_marginal_mean, result, star_mask)[-1]
    gate3_z = _gate23_segment_match(fit.oof_marginal_mean, result, bench_mask)[-1]

    meanyr = gate_frame["MeanYr"].to_numpy(dtype=float)
    mean10 = gate_frame["Mean10"].to_numpy(dtype=float)
    stable = (
        (meanyr > 0) & (mean10 > 0) & (np.abs(mean10 / np.clip(meanyr, 1e-12, None) - 1.0) <= 0.12)
    )
    star = stable & (meanyr >= np.quantile(meanyr[meanyr > 0], 0.75))
    recent_ratio = _bootstrap_ratio_ci_clustered(
        fit.oof_marginal_mean[star],
        mean10[star],
        players[star],
        np.random.default_rng(1729),
    )
    citl_ratio = _bootstrap_ratio_ci_clustered(
        fit.oof_marginal_mean[star],
        result[star],
        players[star],
        np.random.default_rng(1729),
    )
    ece = _gate5_ece_debiased(fit.oof_pooled_over, outcome)
    rate_error = abs(float(fit.oof_pooled_over.mean() - outcome.mean()))
    pit_limit = _gate4_pit_ks_threshold(len(result))
    audit = {
        "validation_rows": len(result),
        "validation_players": int(pd.Series(players).nunique()),
        "split_fingerprint_sha256": _validation_split_fingerprint(splits),
        "brier_delta": float(brier_delta.mean()),
        "brier_delta_row_ci": tuple(float(value) for value in row_ci),
        "brier_delta_player_ci": tuple(float(value) for value in player_ci),
        "pit_ks": pit_global,
        "pit_ks_limit": pit_limit,
        "pit_ks_by_position": pit_by_position,
        "gate2_z": float(gate2_z),
        "gate3_z": float(gate3_z),
        "gate5_ece_debiased": float(ece),
        "over_rate_error": rate_error,
        "gate6_recent_ratio_ci": tuple(float(value) for value in recent_ratio),
        "gate6_citl_ratio_ci": tuple(float(value) for value in citl_ratio),
        "fold_affines": [list(values) for values in fit.fold_affines],
        "fold_lambdas": [list(values) for values in fit.fold_lambdas],
        "fold_temperatures": list(fit.fold_temperatures),
        "fold_probability_pool_rhos": list(fit.fold_rhos),
    }
    guards = {
        "gate1_row_ci": row_ci[2] < 0.005,
        "gate1_player_ci": player_ci[2] < 0.005,
        "gate2": np.isfinite(gate2_z) and gate2_z < 0.5,
        "gate3": np.isfinite(gate3_z) and gate3_z < 0.5,
        "gate4_global": np.isfinite(pit_global) and pit_global <= pit_limit,
        "gate4_positions": all(
            np.isfinite(value) and value <= 0.075 for value in pit_by_position.values()
        ),
        "gate5": np.isfinite(ece) and ece < 0.075 and rate_error < 0.035,
        "gate6_recent": np.isfinite(recent_ratio[2]) and recent_ratio[2] >= 0.91,
        "gate6_citl": np.isfinite(citl_ratio[2]) and citl_ratio[2] >= 0.97,
    }
    audit["guards"] = guards
    failed = [name for name, passed in guards.items() if not passed]
    if failed:
        raise ValueError("rushing candidate failed validation guard(s): " + ", ".join(failed))
    return audit


def _step_apply_affine_groupcdf_candidate(
    calibrated: dict,
    fused: dict,
    splits: dict,
    context: dict,
) -> tuple[dict, dict]:
    """Fit and apply the frozen QB/RB distribution and quoted-line probability heads."""
    if context.get("status") != "active":
        raise ValueError("rushing affine/group-CDF candidate requires active QB/RB routing")
    index_val = splits["X_validation"].index
    index_test = splits["X_test"].index
    positions_val = context["positions"]["validation"].reindex(index_val).to_numpy(dtype=float)
    positions_test = context["positions"]["test"].reindex(index_test).to_numpy(dtype=float)
    players_val = splits["players_validation"].reindex(index_val).astype(str).to_numpy()

    def _gate_vector(value, length: int) -> np.ndarray:
        if value is None:
            return np.zeros(length, dtype=float)
        gate = np.asarray(value, dtype=float)
        if gate.ndim == 0:
            return np.full(length, float(gate), dtype=float)
        if gate.shape != (length,):
            raise ValueError("rushing candidate gate is not row-aligned")
        return gate

    gate_val = _gate_vector(fused["gate_blend_val"], len(index_val))
    gate_test = _gate_vector(fused["gate_blend_test"], len(index_test))
    predictive_val = AffinePredictive(
        marginal_mean=np.asarray(fused["weighted_mean_val"], dtype=float) * (1.0 - gate_val),
        sigma=np.asarray(fused["sn_sigma_blend_val"], dtype=float),
        alpha=np.asarray(fused["sn_alpha_blend_val"], dtype=float),
        gate=gate_val,
    )
    predictive_test = AffinePredictive(
        marginal_mean=np.asarray(fused["weighted_mean"], dtype=float) * (1.0 - gate_test),
        sigma=np.asarray(fused["sn_sigma_blend_test"], dtype=float),
        alpha=np.asarray(fused["sn_alpha_blend_test"], dtype=float),
        gate=gate_test,
    )
    fit = fit_affine_groupcdf(
        predictive_val,
        splits["y_validation"]["Result"].reindex(index_val).to_numpy(dtype=float),
        splits["B_validation"]["Line"].reindex(index_val).to_numpy(dtype=float),
        splits["B_validation"]["Odds"].reindex(index_val).to_numpy(dtype=float),
        positions_val,
        players_val,
    )
    audit = _affine_candidate_oof_audit(fit, splits, positions_val.astype(int))
    validation_output = apply_affine_groupcdf(
        fit.blob,
        predictive_val,
        splits["B_validation"]["Line"].reindex(index_val).to_numpy(dtype=float),
        splits["B_validation"]["Odds"].reindex(index_val).to_numpy(dtype=float),
        positions_val,
    )
    test_output = apply_affine_groupcdf(
        fit.blob,
        predictive_test,
        splits["B_test"]["Line"].reindex(index_test).to_numpy(dtype=float),
        splits["B_test"]["Odds"].reindex(index_test).to_numpy(dtype=float),
        positions_test,
    )

    fused["weighted_mean"] = test_output.conditional_mean
    fused["weighted_mean_val"] = validation_output.conditional_mean
    calibrated.update(
        {
            "c_opt": 1.0,
            "skew_cal": 0.0,
            "sn_sigma_blend_test": test_output.sigma,
            "sn_sigma_blend_val": validation_output.sigma,
            "sn_alpha_blend_test": test_output.alpha,
            "sn_alpha_blend_val": validation_output.alpha,
            "val_weighted_mean_val": validation_output.conditional_mean,
            "pit_recal_blob": None,
            "rushing_candidate_fit": fit,
            "rushing_candidate_test": test_output,
            "rushing_candidate_test_base": predictive_test,
            "rushing_candidate_validation": validation_output,
            "nfl_yards_routes": context["routes"],
            "nfl_yards_pit_maps": pd.Series(positions_test.astype(int), index=index_test).map(
                {
                    code: json.dumps(
                        fit.blob["position_cdf"][str(code)],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for code in AFFINE_POSITIONS
                }
            ),
            "nfl_yards_experiment_blob": {
                "schema_version": 1,
                "slug": AFFINE_STRATEGY,
                "status": "active",
                "kill_reason": None,
                "line_probability_only": True,
                "routing": {
                    "kind": "player_position",
                    "experts": {str(code): label for code, label in AFFINE_POSITIONS.items()},
                },
                "support": context["support"],
                "calibration": fit.blob,
                "validation_audit": audit,
                "fallback_test_rows": int(context["routes"]["test"].eq("pooled_fallback").sum()),
            },
        }
    )
    return calibrated, context
