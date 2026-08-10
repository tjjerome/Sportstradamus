"""Train-market helpers shared by the two-part and affine structural steps."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from sportstradamus.training.group_conditional_cdf._maps import ks_supremum
from sportstradamus.training.model_strategy import BASE_STRUCTURAL_STRATEGY, get_strategy
from sportstradamus.training.scorecard import (
    _bootstrap_mean_ci,
    _bootstrap_mean_ci_clustered,
)

# Every fit-dependent structural value the served CSV carries per row. Cross-fit training
# stitches these fold by fold, so each one is a row-indexed Series (``None`` where the active
# method does not produce it) rather than a constant the whole frame shares.
STRUCTURAL_ROW_FIELDS: tuple[str, ...] = (
    "test_route",
    "prepool_over",
    "pit_recal_by_row",
    "two_part_calibration_by_row",
    "two_part_f0",
    "two_part_role",
    "two_part_position",
)
_TWO_PART_ROW_FIELDS: tuple[str, ...] = (
    "two_part_calibration_by_row",
    "two_part_f0",
    "two_part_role",
    "two_part_position",
)


def _adapter_schema_version(slug: str) -> int:
    """The registered artifact-schema version a fitted adapter blob must echo.

    Read from the registry rather than repeated as a literal so the pickle wrapper and the
    spec ``model_strategy.identity._validate_adapter`` checks it against cannot drift apart.
    """
    return get_strategy(slug).artifact_schema_version


def _aligned_rows(rows: dict[str, pd.Series | None], field: str, index: pd.Index) -> pd.Series:
    """One structural row field reindexed onto the persisted rows, failing loud on a gap."""
    values = rows[field].reindex(index)
    if values.isna().any():
        raise ValueError(f"structural {field} does not align to persisted rows")
    return values


def _persist_structural_columns(
    X_test: pd.DataFrame, *, structural_strategy: str, rows: dict[str, pd.Series | None]
) -> None:
    """Write the structural routing + two-part audit columns onto ``X_test``.

    Absent for a base-strategy cell (no columns added => byte-identical to a
    pre-structural artifact). Under cross-fit training each field carries the value fitted
    without its own row, so the two-part calibration payload varies across rows.
    """
    if structural_strategy != BASE_STRUCTURAL_STRATEGY:
        if rows["test_route"] is None:
            raise ValueError("structural candidate persistence requires auditable routes")
        test_routes = _aligned_rows(rows, "test_route", X_test.index)
        X_test["StructuralAdapterStrategy"] = structural_strategy
        X_test["StructuralRoute"] = test_routes
        X_test["StructuralFallback"] = test_routes.eq("pooled_fallback")
    present = [rows[field] is not None for field in _TWO_PART_ROW_FIELDS]
    if not any(present):
        return
    if not all(present):
        raise ValueError("two-part persistence requires its complete row contract")
    f0 = _aligned_rows(rows, "two_part_f0", X_test.index).to_numpy(dtype=float)
    if not np.isfinite(f0).all():
        raise ValueError("two-part F(0) must be finite and row-aligned")
    X_test["StructuralCalibration"] = _aligned_rows(
        rows, "two_part_calibration_by_row", X_test.index
    ).to_numpy()
    X_test["StructuralF0"] = f0
    X_test["StructuralRole"] = _aligned_rows(rows, "two_part_role", X_test.index).to_numpy()
    X_test["StructuralPosition"] = _aligned_rows(rows, "two_part_position", X_test.index).to_numpy(
        dtype=int
    )


def _ks_uniform(values: np.ndarray) -> float:
    """One-sample KS distance to Uniform(0, 1), without scipy fit state."""
    ordered = np.sort(np.clip(np.asarray(values, dtype=float), 0.0, 1.0))
    if len(ordered) == 0:
        return float("nan")
    return ks_supremum(ordered)


def _validation_split_fingerprint(splits: dict) -> str:
    """Fingerprint the exact row identities and priced outcomes used by a calibrator."""
    index = splits["X_validation"].index
    players = splits["players_validation"].reindex(index).astype(str)
    fingerprint_frame = pd.DataFrame(
        {
            "Player": players,
            "Result": splits["y_validation"]["Result"].reindex(index),
            "Line": splits["B_validation"]["Line"].reindex(index),
            "BookOver": splits["B_validation"]["Odds"].reindex(index),
        },
        index=index,
    )
    if fingerprint_frame.isna().any().any():
        raise ValueError("structural validation fingerprint contains unaligned rows")
    row_hashes = pd.util.hash_pandas_object(fingerprint_frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _oof_brier_arrays(fit, splits: dict):
    """Priced validation outcomes and book-relative Brier CIs for a structural candidate.

    The two-part and affine OOF audits open with the identical setup — same
    validation index, the same priced ``(result >= line)`` outcome, and the same
    seeded row/player-clustered Brier-delta CIs against ``fit.oof_pooled_over``.
    Returns ``(result, players, outcome, book, brier_delta, row_ci, player_ci)``;
    each audit's candidate-specific guards diverge after this point.
    """
    index = splits["X_validation"].index
    result = splits["y_validation"]["Result"].reindex(index).to_numpy(dtype=float)
    line = splits["B_validation"]["Line"].reindex(index).to_numpy(dtype=float)
    book = splits["B_validation"]["Odds"].reindex(index).to_numpy(dtype=float)
    players = splits["players_validation"].reindex(index).astype(str).to_numpy()
    outcome = (result >= line).astype(float)
    brier_delta = (fit.oof_pooled_over - outcome) ** 2 - (book - outcome) ** 2
    row_ci = _bootstrap_mean_ci(brier_delta, np.random.default_rng(1729))
    player_ci = _bootstrap_mean_ci_clustered(brier_delta, players, np.random.default_rng(1729))
    return result, players, outcome, book, brier_delta, row_ci, player_ci
