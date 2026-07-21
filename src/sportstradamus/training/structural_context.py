"""Train-only routing and support contexts for structural NFL-yards strategies."""

from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from operator import add, mul

import numpy as np
import pandas as pd

from sportstradamus.training.role_specs import (
    RoleSpec,
    league_position_codes,
    position_label,
    role_spec_for,
)
from sportstradamus.training.structural_strategies import (
    AFFINE_POSITIONS,
    AFFINE_STRATEGY,
    AFFINE_SUPPORT,
    TWO_PART_STRATEGY,
    TWO_PART_SUPPORT,
)


def _role_score(frame: pd.DataFrame, spec: RoleSpec) -> pd.Series:
    columns = spec.role_columns
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing role column(s): {', '.join(missing)}")
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite role input")
    volume = reduce(mul, (values[column] for column in spec.volume_cols))
    if len(spec.usage_share_cols) == 1:
        clipped = (
            values[column].clip(lower=0)
            for column in (*spec.usage_share_cols, *spec.efficiency_cols)
        )
        score = reduce(mul, clipped, volume)
    else:
        components = (
            values[usage].clip(lower=0) * values[efficiency].clip(lower=0)
            for usage, efficiency in zip(spec.usage_share_cols, spec.efficiency_cols, strict=True)
        )
        score = volume * reduce(add, components)
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite role score")
    return score


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
        raise ValueError(f"unsupported receiving experiment {slug!r}")
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
        scores = {split: _role_score(frame, spec) for split, frame in frames.items()}
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
            raise ValueError(f"receiving role support guard failed: {', '.join(failures)}")

        train_result = splits["y_train"]["Result"].reindex(frames["train"].index)
        if train_result.isna().any():
            raise ValueError("receiving train outcomes do not align to role routes")
        fallback_gate = float(train_result.eq(0.0).mean())
        gate_rates = {
            group: float(train_result.loc[routes["train"].eq(group)].eq(0.0).mean())
            for group in ("low", "high")
        }
        if not np.isfinite([fallback_gate, *gate_rates.values()]).all() or not all(
            0.0 <= value < 1.0 for value in (fallback_gate, *gate_rates.values())
        ):
            raise ValueError("receiving role gate rates must be finite in [0, 1)")
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
    }


def build_affine_expert_context(
    splits: Mapping[str, object],
    *,
    support_floor: Mapping[str, int] = AFFINE_SUPPORT,
    slug: str = AFFINE_STRATEGY,
) -> dict:
    """Route QB/RB rows and enforce support floors for the final rushing candidate."""
    if slug != AFFINE_STRATEGY:
        raise ValueError(f"unsupported rushing experiment {slug!r}")
    frames = {split: splits[f"X_{split}"] for split in ("train", "validation", "test")}
    support: dict[str, dict[str, int]] = {}
    positions: dict[str, pd.Series] = {}
    try:
        routes: dict[str, pd.Series] = {}
        for split, frame in frames.items():
            if "Player position" not in frame.columns:
                raise ValueError("missing Player position")
            position = pd.to_numeric(frame["Player position"], errors="coerce")
            positions[split] = position
            routes[split] = position.map(AFFINE_POSITIONS).fillna("pooled_fallback")

        players = {
            "train": splits.get("players_train"),
            "validation": splits.get("players_validation"),
        }
        support = {
            str(code): _support_for_group(routes, players, label)
            for code, label in AFFINE_POSITIONS.items()
        }
        failures = [
            f"{AFFINE_POSITIONS[int(code)]}.{key}={counts[key]}<{minimum}"
            for code, counts in support.items()
            for key, minimum in support_floor.items()
            if counts[key] < minimum
        ]
        if failures:
            raise ValueError(f"rushing expert support guard failed: {', '.join(failures)}")
    except ValueError as exc:
        return {
            "slug": slug,
            "status": "killed_fallback",
            "kill_reason": str(exc),
            "support": support,
            "routes": _fallback_routes(frames),
            "positions": positions,
        }

    return {
        "slug": slug,
        "status": "active",
        "kill_reason": None,
        "support": support,
        "routes": routes,
        "positions": positions,
    }
