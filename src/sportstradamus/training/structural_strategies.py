"""Shared constants for the group-conditional-CDF structural calibration methods.

Both methods are graduated into the single-valued ``posthoc`` calibration pool
(:data:`sportstradamus.training.posthoc.STRUCTURAL_STAGE`); selection and dispatch live
there and in ``training.pipeline`` / ``prediction.model_prob``. This module holds only the
method-slug constants, the per-cell support floors, and the affine position/role column
literals the two builders read.
"""

from __future__ import annotations

TWO_PART_STRATEGY = "role-position-two-part-groupcdf-fixedlinear-v3"
AFFINE_STRATEGY = "affine-groupcdf-bookpool-v1"

AFFINE_EXPERT_EXPERIMENTS: frozenset[str] = frozenset({AFFINE_STRATEGY})

ROLE_COLUMNS: tuple[str, ...] = (
    "Team plays_per_game",
    "Team pass_rate",
    "Player target share",
    "Player yards per target",
)
AFFINE_POSITIONS: dict[int, str] = {1: "QB", 3: "RB"}
TWO_PART_SUPPORT: dict[str, int] = {
    "train_rows": 4000,
    "validation_rows": 400,
    "test_rows": 400,
    "train_players": 100,
    "validation_players": 100,
}
AFFINE_SUPPORT: dict[str, int] = {
    "train_rows": 1500,
    "validation_rows": 300,
    "test_rows": 300,
    "train_players": 60,
    "validation_players": 50,
}
