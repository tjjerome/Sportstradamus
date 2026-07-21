"""Train-market fit/gate/apply step for the receiving role×position candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from sportstradamus.helpers import skewnormal_loc_from_mean
from sportstradamus.training.group_conditional_cdf import fit_receiving_two_part_groupcdf
from sportstradamus.training.group_conditional_cdf._apply import (
    apply_receiving_two_part_line,
    serialize_receiving_calibration,
)
from sportstradamus.training.group_conditional_cdf._contracts import (
    RECEIVING_POSITION_CODES,
    ReceivingCalibrationFit,
)
from sportstradamus.training.group_conditional_cdf._contracts import (
    ROLE_VALUES as RECEIVING_ROLE_VALUES,
)
from sportstradamus.training.group_conditional_cdf._pipeline_steps_receiving_support import (
    _receiving_nested_support_audit,
)
from sportstradamus.training.group_conditional_cdf._pipeline_steps_shared import (
    _ks_uniform,
    _oof_brier_arrays,
    _validation_split_fingerprint,
)
from sportstradamus.training.scorecard import (
    _bootstrap_ratio_ci_clustered,
    _gate5_ece_debiased,
    _gate23_segment_match,
    _segment_masks,
)
from sportstradamus.training.structural_strategies import (
    RECEIVING_ROLE_POSITION_TWO_PART_GROUPCDF_FIXEDLINEAR,
)


def _explicit_boolean_mask(values, expected: bool) -> np.ndarray:
    """Recognize only real bool scalars; strings and integer lookalikes are not provenance."""
    return np.asarray(
        [
            isinstance(value, (bool, np.bool_)) and bool(value) is expected
            for value in np.asarray(values, dtype=object)
        ],
        dtype=bool,
    )


def _receiving_gate_vector(value, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=float)
    gate = np.asarray(value, dtype=float)
    if gate.ndim == 0:
        gate = np.full(length, float(gate), dtype=float)
    if gate.shape != (length,) or not np.isfinite(gate).all():
        raise ValueError("receiving candidate gate is not finite and row-aligned")
    if np.any((gate < 0.0) | (gate >= 1.0)):
        raise ValueError("receiving candidate gate must lie in [0, 1)")
    return gate


def _skewnormal_cdf_endpoints(mean, sigma, alpha, gate, point) -> tuple[np.ndarray, np.ndarray]:
    """Return the served zero-adjusted SkewNormal CDF immediately below/at each point."""
    means = np.asarray(mean, dtype=float)
    scales = np.asarray(sigma, dtype=float)
    shapes = np.asarray(alpha, dtype=float)
    points = np.asarray(point, dtype=float)
    n_rows = len(means)
    gates = _receiving_gate_vector(gate, n_rows)
    if any(array.shape != (n_rows,) for array in (scales, shapes, points)):
        raise ValueError("receiving predictive inputs must be row-aligned vectors")
    if not all(np.isfinite(array).all() for array in (means, scales, shapes, points)):
        raise ValueError("receiving predictive inputs must be finite")
    if np.any(scales <= 0.0):
        raise ValueError("receiving predictive scales must be positive")

    loc = skewnormal_loc_from_mean(means, scales, shapes)
    base = skewnorm.cdf(points, shapes, loc=loc, scale=scales)
    upper = np.where(
        points < 0.0,
        (1.0 - gates) * base,
        gates + (1.0 - gates) * base,
    )
    atom = np.where(np.isclose(points, 0.0), gates, 0.0)
    return np.clip(upper - atom, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def _receiving_pit_diagnostics(
    fit: ReceivingCalibrationFit,
    result: np.ndarray,
    roles: np.ndarray,
    positions: np.ndarray,
) -> dict:
    positive = result > 0.0

    def mean_ks(draws: np.ndarray) -> float:
        return float(np.mean([_ks_uniform(draw) for draw in draws]))

    return {
        "global": mean_ks(fit.oof_pit_draws),
        "role_full": {
            role: mean_ks(fit.oof_pit_draws[:, roles == role]) for role in RECEIVING_ROLE_VALUES
        },
        "role_positive": {
            role: _ks_uniform(fit.oof_positive_pit[(roles == role) & positive])
            for role in RECEIVING_ROLE_VALUES
        },
        "position_full": {
            str(position): mean_ks(fit.oof_pit_draws[:, positions == position])
            for position in RECEIVING_POSITION_CODES
        },
        "position_positive": {
            str(position): _ks_uniform(fit.oof_positive_pit[(positions == position) & positive])
            for position in RECEIVING_POSITION_CODES
        },
    }


def _receiving_candidate_oof_audit(
    fit: ReceivingCalibrationFit,
    fused: dict,
    splits: dict,
    roles: np.ndarray,
    positions: np.ndarray,
    gate: np.ndarray,
) -> dict:
    """Run all six guards on nested OOF validation predictions before test access."""
    # style: allow-complexity -- aggregates independent out-of-fold guards on the receiving candidate
    result, players, outcome, book, brier_delta, row_ci, player_ci = _oof_brier_arrays(fit, splits)

    marginal_mean = np.asarray(fused["weighted_mean_val"], dtype=float) * (1.0 - gate)
    gate_frame = splits["X_validation"].copy()
    gate_frame["Result"] = result
    gate_frame["Player"] = players
    star_mask, bench_mask = _segment_masks(gate_frame)
    gate2_z = _gate23_segment_match(marginal_mean, result, star_mask)[-1]
    gate3_z = _gate23_segment_match(marginal_mean, result, bench_mask)[-1]
    pits = _receiving_pit_diagnostics(fit, result, roles, positions)
    ece = _gate5_ece_debiased(fit.oof_pooled_over, outcome)
    rate_error = abs(float(fit.oof_pooled_over.mean() - outcome.mean()))

    meanyr = gate_frame["MeanYr"].to_numpy(dtype=float)
    mean10 = gate_frame["Mean10"].to_numpy(dtype=float)
    stable = (
        (meanyr > 0.0)
        & (mean10 > 0.0)
        & (np.abs(mean10 / np.clip(meanyr, 1e-12, None) - 1.0) <= 0.12)
    )
    stable_star = stable & (meanyr >= np.quantile(meanyr[meanyr > 0.0], 0.75))
    recent_ratio = _bootstrap_ratio_ci_clustered(
        marginal_mean[stable_star],
        mean10[stable_star],
        players[stable_star],
        np.random.default_rng(1729),
    )
    citl_ratio = _bootstrap_ratio_ci_clustered(
        marginal_mean[stable_star],
        result[stable_star],
        players[stable_star],
        np.random.default_rng(1729),
    )
    recent_corr = float(np.corrcoef(mean10, result)[0, 1])
    guards = {
        "gate1_row_ci": np.isfinite(row_ci[2]) and row_ci[2] < 0.005,
        "gate1_player_ci": np.isfinite(player_ci[2]) and player_ci[2] < 0.005,
        "gate2": np.isfinite(gate2_z) and gate2_z < 0.5,
        "gate3": np.isfinite(gate3_z) and gate3_z < 0.5,
        "gate4_global": pits["global"] <= 0.05,
        "gate4_roles": all(value <= 0.09 for value in pits["role_full"].values()),
        "gate4_role_positive": (
            pits["role_positive"]["low"] <= 0.08 and pits["role_positive"]["high"] <= 0.09
        ),
        "gate4_positions": all(value <= 0.08 for value in pits["position_full"].values()),
        "gate4_position_positive": all(
            value <= 0.09 for value in pits["position_positive"].values()
        ),
        "gate5": np.isfinite(ece) and ece < 0.075 and rate_error < 0.035,
        "gate6_recent": recent_corr < 0.58 or recent_ratio[2] >= 0.91,
        "gate6_citl": np.isfinite(citl_ratio[2]) and citl_ratio[2] >= 0.97,
    }
    audit = {
        "guards": guards,
        "validation_rows": len(result),
        "validation_players": int(pd.Series(players).nunique()),
        "split_fingerprint_sha256": _validation_split_fingerprint(splits),
        "gate1": {
            "model_brier": float(np.mean((fit.oof_pooled_over - outcome) ** 2)),
            "book_brier": float(np.mean((book - outcome) ** 2)),
            "delta": float(brier_delta.mean()),
            "row_ci": tuple(float(value) for value in row_ci),
            "player_clustered_ci": tuple(float(value) for value in player_ci),
        },
        "gate2_z": float(gate2_z),
        "gate3_z": float(gate3_z),
        "gate4": pits,
        "gate5": {"ece_debiased": float(ece), "over_rate_error": rate_error},
        "gate6": {
            "recent_corr": recent_corr,
            "recent_ratio_ci": tuple(float(value) for value in recent_ratio),
            "citl_ratio_ci": tuple(float(value) for value in citl_ratio),
            "stable_star_rows": int(stable_star.sum()),
            "stable_star_players": len(np.unique(players[stable_star])),
        },
        "fold_lambdas": [list(values) for values in fit.fold_lambdas],
        "fold_intercepts": list(fit.fold_intercepts),
        "fold_temperatures": list(fit.fold_temperatures),
        "fold_selection_keys": [list(values) for values in fit.fold_selection_keys],
    }
    failed = [name for name, passed in guards.items() if not passed]
    if failed:
        raise ValueError("receiving candidate failed validation guard(s): " + ", ".join(failed))
    return audit


def _step_apply_receiving_two_part_groupcdf_candidate(
    calibrated: dict,
    fused: dict,
    splits: dict,
    context: dict,
) -> tuple[dict, dict]:
    """Fit receiving v3 on validation OOF data, gate it, then price held-out lines."""
    # style: allow-complexity -- linear fit -> gate -> price pipeline for the receiving two-part candidate
    if context.get("status") != "active":
        raise ValueError(
            "receiving role/position two-part candidate requires active routing: "
            f"{context.get('kill_reason')}"
        )
    index_val = splits["X_validation"].index
    index_test = splits["X_test"].index

    def _required_series(key: str, index: pd.Index, *, allow_missing: bool = False) -> pd.Series:
        series = splits.get(key)
        if series is None:
            raise ValueError(f"receiving candidate requires {key} provenance")
        aligned = series.reindex(index)
        if not allow_missing and aligned.isna().any():
            raise ValueError(f"receiving candidate {key} does not align to its split")
        return aligned

    raw_players = _required_series("players_validation", index_val)
    dates = pd.to_datetime(_required_series("dates_validation", index_val), errors="coerce")
    player_missing = raw_players.astype(str).str.strip().isin(("", "nan", "None", "<NA>"))
    identity = pd.DataFrame({"Player": raw_players, "Date": dates}, index=index_val)
    if player_missing.any() or dates.isna().any() or identity.duplicated().any():
        raise ValueError("receiving validation Player/Date identity must be complete and unique")
    players = raw_players.astype(str).to_numpy()

    roles_val = context["routes"]["validation"].reindex(index_val).astype(str).to_numpy()
    roles_test = context["routes"]["test"].reindex(index_test).astype(str).to_numpy()
    positions_val = pd.to_numeric(
        splits["X_validation"]["Player position"], errors="coerce"
    ).to_numpy(dtype=float)
    positions_test = pd.to_numeric(splits["X_test"]["Player position"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(positions_val).all() or not np.isfinite(positions_test).all():
        raise ValueError("receiving candidate positions must be finite")
    positions_val = positions_val.astype(int)
    positions_test = positions_test.astype(int)

    raw_gate_rates = context.get("gate_rates")
    if not isinstance(raw_gate_rates, dict) or set(raw_gate_rates) != set(RECEIVING_ROLE_VALUES):
        raise ValueError("receiving candidate requires train-only low/high gate rates")
    model_weight = float(fused["model_weight"])
    if not np.isfinite(model_weight) or not 0.0 <= model_weight <= 1.0:
        raise ValueError("receiving candidate model weight must lie in [0, 1]")
    served_gate_rates = {
        role: model_weight * float(raw_gate_rates[role]) for role in RECEIVING_ROLE_VALUES
    }
    gate_val = _receiving_gate_vector(
        np.asarray([served_gate_rates[role] for role in roles_val]), len(index_val)
    )
    gate_test = _receiving_gate_vector(
        np.asarray([served_gate_rates[role] for role in roles_test]), len(index_test)
    )
    mean_val = np.asarray(fused["weighted_mean_val"], dtype=float)
    mean_test = np.asarray(fused["weighted_mean"], dtype=float)
    sigma_val = np.asarray(fused["sn_sigma_blend_val"], dtype=float)
    sigma_test = np.asarray(fused["sn_sigma_blend_test"], dtype=float)
    alpha_val = np.asarray(fused["sn_alpha_blend_val"], dtype=float)
    alpha_test = np.asarray(fused["sn_alpha_blend_test"], dtype=float)

    result = splits["y_validation"]["Result"].reindex(index_val).to_numpy(dtype=float)
    line_val = splits["B_validation"]["Line"].reindex(index_val).to_numpy(dtype=float)
    book_val = splits["B_validation"]["Odds"].reindex(index_val).to_numpy(dtype=float)
    line_test = splits["B_test"]["Line"].reindex(index_test).to_numpy(dtype=float)
    book_test = splits["B_test"]["Odds"].reindex(index_test).to_numpy(dtype=float)
    result_lower, result_upper = _skewnormal_cdf_endpoints(
        mean_val, sigma_val, alpha_val, gate_val, result
    )
    _, zero_val = _skewnormal_cdf_endpoints(
        mean_val, sigma_val, alpha_val, gate_val, np.zeros(len(index_val))
    )
    _, zero_test = _skewnormal_cdf_endpoints(
        mean_test, sigma_test, alpha_test, gate_test, np.zeros(len(index_test))
    )
    _, line_low_val = _skewnormal_cdf_endpoints(
        mean_val, sigma_val, alpha_val, gate_val, np.ceil(line_val - 1.0)
    )
    _, line_high_val = _skewnormal_cdf_endpoints(
        mean_val, sigma_val, alpha_val, gate_val, np.floor(line_val + 1.0)
    )
    _, line_low_test = _skewnormal_cdf_endpoints(
        mean_test, sigma_test, alpha_test, gate_test, np.ceil(line_test - 1.0)
    )
    _, line_high_test = _skewnormal_cdf_endpoints(
        mean_test, sigma_test, alpha_test, gate_test, np.floor(line_test + 1.0)
    )

    authentic_val = _explicit_boolean_mask(
        _required_series("archived_validation", index_val, allow_missing=True), True
    ) & _explicit_boolean_mask(
        _required_series("odds_synthetic_validation", index_val, allow_missing=True), False
    )
    authentic_test = _explicit_boolean_mask(
        _required_series("archived_test", index_test, allow_missing=True), True
    ) & _explicit_boolean_mask(
        _required_series("odds_synthetic_test", index_test, allow_missing=True), False
    )
    outcome = (result >= line_val).astype(float)
    support_audit = _receiving_nested_support_audit(
        result,
        outcome,
        authentic_val,
        players,
        roles_val,
        positions_val,
    )
    fit = fit_receiving_two_part_groupcdf(
        result_upper,
        result_lower,
        zero_val,
        line_low_val,
        line_high_val,
        result,
        outcome,
        book_val,
        authentic_val,
        players,
        roles_val,
        positions_val,
    )
    validation_audit = _receiving_candidate_oof_audit(
        fit, fused, splits, roles_val, positions_val, gate_val
    )
    test_output = apply_receiving_two_part_line(
        fit.blob,
        line_low_test,
        line_high_test,
        zero_test,
        roles_test,
        positions_test,
        book_test,
        authentic_test,
    )

    calibration_payload = serialize_receiving_calibration(fit.blob)
    fused["gate_blend_val"] = gate_val
    fused["gate_blend_test"] = gate_test
    calibrated.update(
        {
            "c_opt": 1.0,
            "skew_cal": 0.0,
            "sn_sigma_blend_val": sigma_val,
            "sn_sigma_blend_test": sigma_test,
            "sn_alpha_blend_val": alpha_val,
            "sn_alpha_blend_test": alpha_test,
            "val_weighted_mean_val": mean_val,
            "pit_recal_blob": None,
            "receiving_candidate_fit": fit,
            "receiving_candidate_test": test_output,
            "receiving_candidate_test_f0": zero_test,
            "receiving_candidate_test_role": pd.Series(roles_test, index=index_test),
            "receiving_candidate_test_position": pd.Series(positions_test, index=index_test),
            "receiving_candidate_calibration_payload": calibration_payload,
            "nfl_yards_routes": context["routes"],
            "nfl_yards_experiment_blob": {
                "schema_version": 3,
                "slug": RECEIVING_ROLE_POSITION_TWO_PART_GROUPCDF_FIXEDLINEAR,
                "status": "active",
                "kill_reason": None,
                "line_probability_only": True,
                "routing": {
                    "kind": "pregame_role_by_position",
                    "thresholds": context["thresholds"],
                    "role_columns": context["role_columns"],
                    "gate_rates": raw_gate_rates,
                    "served_gate_rates": served_gate_rates,
                    "gate_model_weight": model_weight,
                    "positions": context["positions"],
                },
                "support": {
                    "routing": context["support"],
                    "nested_calibration": support_audit,
                },
                "endpoint_contract": {
                    "gate_source": "train_result_eq_zero_by_role",
                    "gate_rates_are_model_endpoint": True,
                    "book_endpoint_gate": 0.0,
                    "fallback_gate": context["fallback_gate"],
                    "dispersion": "raw_fused_shape",
                },
                "calibration": fit.blob,
                "validation_audit": validation_audit,
                "fallback_test_rows": int((~authentic_test).sum()),
            },
        }
    )
    return calibrated, context
