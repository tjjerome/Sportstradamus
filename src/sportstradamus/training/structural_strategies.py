"""Shared constants for the group-conditional-CDF structural calibration methods.

Both methods are graduated into the single-valued ``posthoc`` calibration pool
(:data:`sportstradamus.training.posthoc.STRUCTURAL_STAGE`); selection and dispatch live
there and in ``training.pipeline`` / ``prediction.model_prob``. This module holds only the
method-slug constants, the per-cell support floors, and the shared role-column literals a
builder reads. Both methods discover their per-cell position codes from the training matrix
(:func:`sportstradamus.training.role_specs.league_position_codes`), so no position literals
live here.
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
