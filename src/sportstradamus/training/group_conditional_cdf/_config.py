"""Stage-selection config for the group-conditional CDF engine.

One :class:`StrategyConfig` value fully determines which stages run and how each
group is keyed. The two-part strategy and the affine strategy are
the two configs the engine ships; every other field combination is reserved for
future variants. Group codes are discovered from the training data at fit time,
not frozen here — the config only names the column that forms the key and the
kind of routing it drives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sportstradamus.training.group_conditional_cdf._contracts import (
    AFFINE_SCHEMA_VERSION,
    AFFINE_STRATEGY_NAME,
    ROLE_VALUES,
    TWO_PART_SCHEMA_VERSION,
    TWO_PART_STRATEGY_NAME,
)

GroupingKind = Literal["categorical_code", "role_by_position"]
LambdaMode = Literal["shared", "per_group"]
PoolWeight = Literal["fixed_0.8", "fit"]
MapImpl = Literal["isotonic_pit", "group_empirical_cdf"]


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen switchboard for one group-conditional CDF variant.

    Attributes:
        candidate_name: The persisted blob slug (byte-stable per variant).
        schema_version: The persisted blob schema version.
        affine_marginal: Run the affine mean-correction stage.
        boundary: Run the per-role logit-boundary stage. When on, the positive
            group-CDF consumes the boundary's conditional-split endpoints; when
            off, the group-CDF reads raw endpoints.
        grouping: Column that forms the positive group key and its routing kind.
            ``categorical_code`` keys directly on an integer position code;
            ``role_by_position`` keys on ``f"{role}_pos{code}"``.
        boundary_grouping: Role key column plus residual sub-code, only used
            when ``boundary`` is on.
        lambda_mode: ``shared`` selects one ``(positive, nonpositive)`` lambda
            pair for every group; ``per_group`` selects a lambda per group.
        pool_weight: ``fixed_0.8`` uses the 80/20 policy pool on authentic
            quotes; ``fit`` fits the Brier-optimal book/model rho.
        map_impl: Which endpoint-map kernel the group-CDF stage uses.
    """

    candidate_name: str
    schema_version: int
    affine_marginal: bool
    boundary: bool
    grouping: GroupingKind
    boundary_grouping: str | None
    lambda_mode: LambdaMode
    pool_weight: PoolWeight
    map_impl: MapImpl


TWO_PART_CONFIG = StrategyConfig(
    candidate_name=TWO_PART_STRATEGY_NAME,
    schema_version=TWO_PART_SCHEMA_VERSION,
    affine_marginal=False,
    boundary=True,
    grouping="role_by_position",
    boundary_grouping="role_rb_residual",
    lambda_mode="shared",
    pool_weight="fixed_0.8",
    map_impl="isotonic_pit",
)

AFFINE_CONFIG = StrategyConfig(
    candidate_name=AFFINE_STRATEGY_NAME,
    schema_version=AFFINE_SCHEMA_VERSION,
    affine_marginal=True,
    boundary=False,
    grouping="categorical_code",
    boundary_grouping=None,
    lambda_mode="per_group",
    pool_weight="fit",
    map_impl="group_empirical_cdf",
)

TWO_PART_ROLES = ROLE_VALUES


def discover_codes(position: np.ndarray) -> tuple[int, ...]:
    """Recover the sorted integer group codes present in the training column."""
    return tuple(int(code) for code in np.unique(np.asarray(position, dtype=float).astype(int)))


def positive_groups(position_codes: tuple[int, ...]) -> tuple[str, ...]:
    """Cross the fixed role axis with the discovered codes into ``role_posN`` keys."""
    return tuple(f"{role_name}_pos{code}" for role_name in ROLE_VALUES for code in position_codes)
