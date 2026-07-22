"""Train-market helpers shared by the two-part and affine structural steps."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from sportstradamus.training.group_conditional_cdf._maps import ks_supremum
from sportstradamus.training.model_strategy import BASE_STRUCTURAL_STRATEGY
from sportstradamus.training.scorecard import (
    _bootstrap_mean_ci,
    _bootstrap_mean_ci_clustered,
)


def _persist_structural_columns(
    X_test: pd.DataFrame,
    *,
    structural_strategy: str,
    structural_routes: dict[str, pd.Series] | None,
    receiving_calibration_payload: str | None,
    receiving_f0: np.ndarray | None,
    receiving_role: pd.Series | None,
    receiving_position: pd.Series | None,
) -> None:
    """Write the structural structural + two-part audit columns onto ``X_test``.

    Absent for a base-strategy cell (no columns added => byte-identical to a
    pre-structural artifact). Raises the same alignment/contract ValueErrors as
    the inline block it replaces.
    """
    if structural_strategy != BASE_STRUCTURAL_STRATEGY:
        if structural_routes is None:
            raise ValueError("structural candidate persistence requires auditable routes")
        test_routes = structural_routes["test"].reindex(X_test.index)
        if test_routes.isna().any():
            raise ValueError("structural candidate routes do not align to test rows")
        X_test["StructuralAdapterStrategy"] = structural_strategy
        X_test["StructuralRoute"] = test_routes
        X_test["StructuralFallback"] = test_routes.eq("pooled_fallback")
    receiving_metadata = (
        receiving_calibration_payload,
        receiving_f0,
        receiving_role,
        receiving_position,
    )
    if any(value is not None for value in receiving_metadata):
        if any(value is None for value in receiving_metadata):
            raise ValueError("two-part persistence requires its complete row contract")
        f0 = np.asarray(receiving_f0, dtype=float)
        if f0.shape != (len(X_test),) or not np.isfinite(f0).all():
            raise ValueError("two-part F(0) must be finite and row-aligned")
        roles = receiving_role.reindex(X_test.index)
        positions = receiving_position.reindex(X_test.index)
        if roles.isna().any() or positions.isna().any():
            raise ValueError("two-part role/position metadata does not align to test rows")
        X_test["StructuralCalibration"] = receiving_calibration_payload
        X_test["StructuralF0"] = f0
        X_test["StructuralRole"] = roles.to_numpy()
        X_test["StructuralPosition"] = positions.to_numpy(dtype=int)


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
