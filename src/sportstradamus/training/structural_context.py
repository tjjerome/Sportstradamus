"""Train-only routing and support contexts for structural structural strategies."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from sportstradamus.training.role_specs import (
    RoleSpec,
    league_position_codes,
    position_label,
    role_spec_for,
)
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY,
    AFFINE_SUPPORT,
    TWO_PART_STRATEGY,
    TWO_PART_SUPPORT,
)


def _validated_positions(frame: pd.DataFrame, codes: tuple[int, ...]) -> pd.Series:
    if "Player position" not in frame.columns:
        raise ValueError("missing Player position")
    position = pd.to_numeric(frame["Player position"], errors="coerce")
    if position.isna().any() or not position.isin(codes).all():
        raise ValueError(f"position code outside league roster {list(codes)}")
    return position.astype(int)


def _support_for_group(
    routes: Mapping[str, pd.Series], players: Mapping[str, pd.Series | None], group: str
) -> dict[str, int]:
    counts = {
        "train_rows": int(routes["train"].eq(group).sum()),
        "validation_rows": int(routes["validation"].eq(group).sum()),
        "test_rows": int(routes["test"].eq(group).sum()),
    }
    for split in ("train", "validation"):
        aligned_players = players[split]
        if aligned_players is None:
            raise ValueError(f"missing {split} player metadata")
        aligned_players = aligned_players.reindex(routes[split].index)
        if aligned_players.isna().any():
            raise ValueError(f"missing {split} player identity")
        counts[f"{split}_players"] = int(
            aligned_players.loc[routes[split].eq(group)].nunique(dropna=True)
        )
    return counts


def _fallback_routes(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {
        split: pd.Series("pooled_fallback", index=frames[split].index, dtype="object")
        for split in ("train", "validation", "test")
    }


def build_two_part_context(
    splits: Mapping[str, object],
    *,
    league: str,
    market: str,
    support_floor: Mapping[str, int] = TWO_PART_SUPPORT,
    slug: str = TWO_PART_STRATEGY,
) -> dict:
    """Derive train-only role thresholds, routes, and support for the two-part method."""
    # style: allow-complexity — one try/except sequences threshold→route→support→gate
    # derivation so any single failure collapses to one killed_fallback payload.
    if slug != TWO_PART_STRATEGY:
        raise ValueError(f"unsupported two-part strategy {slug!r}")
    spec = role_spec_for(league, market)
    if spec is None:
        raise ValueError(f"no role spec registered for {league} {market}")
    codes = league_position_codes(league)
    frames = {split: splits[f"X_{split}"] for split in ("train", "validation", "test")}
    thresholds: dict[str, float] = {}
    positions_map: dict[str, str] = {}
    support: dict[str, dict[str, int]] = {}
    gate_rates: dict[str, float] = {}
    fallback_gate: float | None = None
    try:
        scores = {split: spec.role_score(frame) for split, frame in frames.items()}
        positions = {split: _validated_positions(frame, codes) for split, frame in frames.items()}
        for code in codes:
            position_scores = scores["train"].loc[positions["train"].eq(code)]
            if position_scores.empty:
                continue  # league position the market never fields — leave it untiered
            threshold = float(position_scores.median())
            if not np.isfinite(threshold):
                raise ValueError(f"nonfinite train role threshold for position code {code}")
            thresholds[str(code)] = threshold
            positions_map[str(code)] = position_label(league, code)
        if not thresholds:
            raise ValueError("no train rows for any league position code")

        routes: dict[str, pd.Series] = {}
        threshold_by_position = {int(code): value for code, value in thresholds.items()}
        for split in ("train", "validation", "test"):
            row_threshold = positions[split].map(threshold_by_position)
            if row_threshold.isna().any():
                raise ValueError("holdout carries a position code absent from train tiers")
            routes[split] = pd.Series(
                np.where(scores[split] < row_threshold, "low", "high"),
                index=frames[split].index,
                dtype="object",
            )

        players = {
            "train": splits.get("players_train"),
            "validation": splits.get("players_validation"),
        }
        support = {group: _support_for_group(routes, players, group) for group in ("low", "high")}
        failures = [
            f"{group}.{key}={counts[key]}<{minimum}"
            for group, counts in support.items()
            for key, minimum in support_floor.items()
            if counts[key] < minimum
        ]
        if failures:
            raise ValueError(f"two-part role support guard failed: {', '.join(failures)}")

        train_result = splits["y_train"]["Result"].reindex(frames["train"].index)
        if train_result.isna().any():
            raise ValueError("two-part train outcomes do not align to role routes")
        fallback_gate = float(train_result.eq(0.0).mean())
        gate_rates = {
            group: float(train_result.loc[routes["train"].eq(group)].eq(0.0).mean())
            for group in ("low", "high")
        }
        if not np.isfinite([fallback_gate, *gate_rates.values()]).all() or not all(
            0.0 <= value < 1.0 for value in (fallback_gate, *gate_rates.values())
        ):
            raise ValueError("two-part role gate rates must be finite in [0, 1)")
    except ValueError as exc:
        return {
            "slug": slug,
            "status": "killed_fallback",
            "kill_reason": str(exc),
            "thresholds": thresholds,
            "support": support,
            "gate_rates": gate_rates,
            "fallback_gate": fallback_gate,
            "routes": _fallback_routes(frames),
            "role_columns": list(spec.role_columns),
            "positions": positions_map,
            "boundary_residual_positions": list(spec.boundary_residual_positions),
        }

    return {
        "slug": slug,
        "status": "active",
        "kill_reason": None,
        "thresholds": thresholds,
        "support": support,
        "gate_rates": gate_rates,
        "fallback_gate": fallback_gate,
        "routes": routes,
        "role_columns": list(spec.role_columns),
        "positions": positions_map,
        "boundary_residual_positions": list(spec.boundary_residual_positions),
    }


def build_affine_expert_context(
    splits: Mapping[str, object],
    *,
    league: str,
    support_floor: Mapping[str, int] = AFFINE_SUPPORT,
    slug: str = AFFINE_STRATEGY,
) -> dict:
    """Discover the market's fielded positions and enforce support floors per expert.

    Mirrors :func:`build_two_part_context`: the expert set is the league roster codes
    (:func:`league_position_codes`) the market's train rows actually field, labelled via
    :func:`position_label`; codes the market never fields are pruned, and every other row
    routes to the pooled fallback. The fit runs fresh on each cell's own matrix, so the
    kept codes are emergent — the NFL rushing pilot reproduces its QB/RB experts because
    those are the only roster codes its matrix carries.
    """
    if slug != AFFINE_STRATEGY:
        raise ValueError(f"unsupported affine strategy {slug!r}")
    frames = {split: splits[f"X_{split}"] for split in ("train", "validation", "test")}
    codes = league_position_codes(league)
    experts: dict[int, str] = {}
    support: dict[str, dict[str, int]] = {}
    positions: dict[str, pd.Series] = {}
    try:
        for split, frame in frames.items():
            positions[split] = _validated_positions(frame, codes)
        expert_labels = {
            code: position_label(league, code)
            for code in codes
            if positions["train"].eq(code).any()  # skip roster codes this market never fields
        }
        if not expert_labels:
            raise ValueError("no train rows for any league position code")

        routes = {
            split: positions[split].map(expert_labels).fillna("pooled_fallback").astype("object")
            for split in ("train", "validation", "test")
        }
        players = {
            "train": splits.get("players_train"),
            "validation": splits.get("players_validation"),
        }
        support = {
            str(code): _support_for_group(routes, players, label)
            for code, label in expert_labels.items()
        }
        failures = [
            f"{expert_labels[int(code)]}.{key}={counts[key]}<{minimum}"
            for code, counts in support.items()
            for key, minimum in support_floor.items()
            if counts[key] < minimum
        ]
        if failures:
            raise ValueError(f"affine expert support guard failed: {', '.join(failures)}")
        experts = expert_labels
    except ValueError as exc:
        return {
            "slug": slug,
            "status": "killed_fallback",
            "kill_reason": str(exc),
            "experts": experts,
            "support": support,
            "routes": _fallback_routes(frames),
            "positions": positions,
        }

    return {
        "slug": slug,
        "status": "active",
        "kill_reason": None,
        "experts": experts,
        "support": support,
        "routes": routes,
        "positions": positions,
    }
