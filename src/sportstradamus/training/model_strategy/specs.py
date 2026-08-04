"""Plain-data declarations for built-in model strategies."""

from __future__ import annotations

from sportstradamus.training.calibration import DEFAULT_BLENDING as _DEFAULT_BLENDING
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY,
    TWO_PART_STRATEGY,
)

_CONTROL_FLAGS = {
    "dist": "--dist",
    "normalization": "--target-normalization",
    "dist_training_loss": "--dist-training-loss",
    "sn_param": "--sn-param",
    "blending_loss_fn": "--blending-loss-fn",
    "zinb_mode": "--zinb-mode",
    "count_dispersion_objective": "--count-dispersion-objective",
    "hpo_selection": "--hpo-selection",
    "stabilization": "--stabilization",
    "posthoc": "--posthoc",
}
_SHIP = ("deterministic_train", "full_hpo", "score", "serve", "confirm")
_RESEARCH = ("deterministic_train", "full_hpo", "score")
_SERVE_ONLY = ("serve",)

# Above this target mean the count branch's dispersion cap (pipeline._COUNT_BRANCH_R_CAP = 50)
# pins r below the empirical spread, so a count corner can only fail Gate 4 — pure budget
# waste. Kills count families on NFL passing yards (mean 227) and admits NBA PTS (13.1).
COUNT_ADMISSION_MEAN_CEILING: float = 50.0

# Every base family is offered on both classes: which family fits is decided from the cell's own
# target facts (integer lattice, mean), not from the class its stat_meta ``dist`` happens to name.
_BOTH_CLASSES: tuple[str, ...] = ("continuous", "count")
# "count" stays in the tuple even though the class no longer gates admission —
# identity.validate_model_recipe branches on it to require target_normalization="none" for a
# corner that carries no normalization control.
_COUNT_APPLICABILITY = {
    "distribution_classes": _BOTH_CLASSES,
    "requires_integer_target": True,
    "max_global_mean": COUNT_ADMISSION_MEAN_CEILING,
}


def _base(
    slug: str,
    axes: dict[str, tuple[str, ...]],
    persist: dict[str, str],
    *,
    capabilities: tuple[str, ...] = _SHIP,
    fixed_controls: dict[str, str] | None = None,
    fixed_persist: dict[str, str] | None = None,
    applicability: dict | None = None,
) -> dict:
    return {
        "slug": slug,
        "implementation_version": 1,
        "artifact_schema_version": 1,
        "family": slug,
        "applicability": applicability or {"distribution_classes": _BOTH_CLASSES},
        "enrollments": (),
        "capabilities": capabilities,
        "axes": axes,
        "fixed_controls": fixed_controls or {},
        "cli_flags": _CONTROL_FLAGS,
        "persist": persist,
        "fixed_persist": fixed_persist or {},
        "required_columns": (),
        "shared_csv_columns": (),
        "structural": False,
        "artifact_namespace_suffix": False,
        "legacy_model_key": None,
        "legacy_csv_identity_column": None,
        "legacy_schema_field": "schema_version",
        "legacy_status_field": "status",
        "split_fingerprint_path": (),
        "matrix_hash_path": (),
    }


# ``none`` is deliberately absent: ship_config._validate_cell refuses to ship a continuous cell
# carrying target_normalization="none", so such a corner could never confirm.
_NORMS = (
    "ratio_meanyr",
    "centered_additive_mean10",
    "centered_additive_eb_meanyr_k10",
    "ratio_projvol",
)
_BLENDING = ("crps", "nll")
_COUNT_DISPERSION = ("crps", "pit_ks")
# The ``posthoc`` calibration pool as a search axis. POSTHOC_SLUGS also carries two structural
# slugs, but those are registered specs of their own and never axis values here — selecting one
# means selecting that spec, which pins ``posthoc`` to its own slug.
_POSTHOC = (
    "none",
    "prob_recal_isotonic",
    "prob_recal_platt",
    "roe_mean",
    "isotonic_mean",
)
# pipeline gates the whole-CDF stage on ``posthoc_slug in CDF_STAGE and dist == "SkewNormal"``; on
# any other family it falls through to the scalar path, so the corner would train identically to
# ``none`` while carrying a different fingerprint.
_POSTHOC_SKEWNORMAL = (*_POSTHOC, "cdf_recal_isotonic")
# A base spec deliberately leaves ``hpo_selection`` and ``stabilization`` out of its corner. Neither
# value is decided by the corner: ``cli._retry_calibrated_if_g4_only`` flips a cell to ``calibrated``
# and persists it after a Gate-4-only failure (8 cells carry it today, 6 of them shipped), and the
# pipeline silently upgrades ``stabilization`` to L2 for centered SkewNormal and Mixture. Pinning
# either would sign a value the run did not train with and would make every such cell fail
# ``validate_strategy_selection``. A structural spec is different — it pins the entire recipe.

# With the class unlocked a count corner can win on a cell whose stat_meta carries a continuous
# normalization; persisting ``none`` alongside keeps ship_config._validate_cell satisfied.
_COUNT_FIXED_PERSIST = {"target_normalization": "none"}

_SKEWNORMAL_PERSIST = {
    "dist": "dist",
    "normalization": "target_normalization",
    "dist_training_loss": "dist_training_loss",
    "sn_param": "sn_param",
    "blending_loss_fn": "blending",
    "posthoc": "posthoc",
}
_YARDS_PERSIST = {**_SKEWNORMAL_PERSIST, "hpo_selection": "hpo_selection"}
_COUNT_PERSIST = {
    "dist": "dist",
    "dist_training_loss": "dist_training_loss",
    "count_dispersion_objective": "count_dispersion_objective",
    "blending_loss_fn": "blending",
    "posthoc": "posthoc",
}
# The count constructors accept `nll` alone, and a count spec declares no dist_training_loss axis,
# so without this pin meditate inherits the *cell's* value — on a cell configured continuous with
# `crps` that reached the constructor and killed every count corner it had.
_COUNT_FIXED_CONTROLS = {"dist_training_loss": "nll"}

_BASE_SPECS = (
    _base(
        "SkewNormal",
        {
            "normalization": _NORMS,
            "dist_training_loss": ("crps", "nll"),
            "sn_param": ("direct", "centered"),
            "blending_loss_fn": _BLENDING,
            "posthoc": _POSTHOC_SKEWNORMAL,
        },
        _SKEWNORMAL_PERSIST,
        fixed_controls={"dist": "SkewNormal"},
    ),
    _base(
        "Mixture",
        {
            "dist": ("Mixture",),
            "normalization": _NORMS,
            "blending_loss_fn": _BLENDING,
            "posthoc": _POSTHOC,
        },
        {
            "dist": "dist",
            "normalization": "target_normalization",
            "dist_training_loss": "dist_training_loss",
            "blending_loss_fn": "blending",
            "posthoc": "posthoc",
        },
        capabilities=_RESEARCH,
        # lightgbmlss rejects crps mixture components and fails loud in the constructor, so a
        # swept dist_training_loss axis would guarantee crashed trials.
        fixed_controls={"dist_training_loss": "nll"},
    ),
    _base(
        "ZINB",
        {
            "dist": ("ZINB",),
            "zinb_mode": ("joint", "hurdle"),
            "count_dispersion_objective": _COUNT_DISPERSION,
            "blending_loss_fn": _BLENDING,
            "posthoc": _POSTHOC,
        },
        {**_COUNT_PERSIST, "zinb_mode": "zinb_mode"},
        fixed_controls=_COUNT_FIXED_CONTROLS,
        fixed_persist=_COUNT_FIXED_PERSIST,
        applicability=_COUNT_APPLICABILITY,
    ),
    _base(
        "NegBin",
        {
            "dist": ("NegBin",),
            "count_dispersion_objective": _COUNT_DISPERSION,
            "blending_loss_fn": _BLENDING,
            "posthoc": _POSTHOC,
        },
        _COUNT_PERSIST,
        fixed_controls=_COUNT_FIXED_CONTROLS,
        fixed_persist=_COUNT_FIXED_PERSIST,
        applicability=_COUNT_APPLICABILITY,
    ),
    _base(
        "DPO",
        {
            "dist": ("DPO",),
            "count_dispersion_objective": _COUNT_DISPERSION,
            "blending_loss_fn": _BLENDING,
            "posthoc": _POSTHOC,
        },
        _COUNT_PERSIST,
        fixed_controls=_COUNT_FIXED_CONTROLS,
        fixed_persist=_COUNT_FIXED_PERSIST,
        applicability=_COUNT_APPLICABILITY,
    ),
)


def _compatibility(slug: str) -> dict:
    return _base(
        slug,
        {},
        {},
        capabilities=_SERVE_ONLY,
        fixed_controls={"dist": slug},
        applicability={"distribution_classes": ("continuous",), "distributions": (slug,)},
    )


_COMPATIBILITY_SPECS = tuple(_compatibility(slug) for slug in ("Gamma", "ZAGamma"))


# A structural method rides the single-valued ``posthoc`` calibration pool: its slug is
# the field's value, so the fixed controls pin ``posthoc`` to the method's own slug
# (never ``"none"``, which would be a different pool member). The field structurally
# excludes any light corrector. Every other paired control matches the shipped recipe.
def _yards_controls(slug: str) -> dict[str, str]:
    return {
        "dist": "SkewNormal",
        "normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": "nll",
        "hpo_selection": "loss",
        "stabilization": "None",
        "posthoc": slug,
    }


def _yards(
    slug: str,
    schema: int,
    artifact_columns: tuple[str, ...],
    *,
    applicability: dict,
    enrollments: tuple[tuple[str, str], ...] = (),
) -> dict:
    return {
        "slug": slug,
        # v2: both methods hold their routing/expert set fixed across the holdout-blind
        # calibration folds, so a v1 board row's score is not comparable to a v2 one.
        "implementation_version": 2,
        "artifact_schema_version": schema,
        "family": "SkewNormal",
        "applicability": applicability,
        "enrollments": enrollments,
        "capabilities": _SHIP,
        "axes": {},
        "fixed_controls": _yards_controls(slug),
        "cli_flags": _CONTROL_FLAGS,
        "persist": _YARDS_PERSIST,
        "fixed_persist": {},
        "required_columns": artifact_columns,
        "shared_csv_columns": tuple(
            column for column in artifact_columns if column in ("P_PrePool", "PITRecalKnots")
        ),
        "structural": True,
        "artifact_namespace_suffix": True,
        "legacy_model_key": "structural_calibration",
        "legacy_csv_identity_column": "StructuralAdapterStrategy",
        "legacy_schema_field": "schema_version",
        "legacy_status_field": "status",
        "split_fingerprint_path": ("validation_audit", "split_fingerprint_sha256"),
        "matrix_hash_path": (),
    }


# Artifact-schema versions of the two structural adapters, echoed into each fitted pickle's
# ``structural_calibration`` wrapper (group_conditional_cdf._pipeline_steps_*) so the two can
# never drift apart. Two-part is at 4 because its deterministic CSV now carries a row-specific
# ``StructuralCalibration`` payload; affine is at 2, the generalized-routing schema it has
# written since the QB/RB pilot was lifted.
TWO_PART_ARTIFACT_SCHEMA_VERSION = 4
AFFINE_ARTIFACT_SCHEMA_VERSION = 2

BUILTIN_SPEC_DATA = (
    *_BASE_SPECS,
    *_COMPATIBILITY_SPECS,
    _yards(
        TWO_PART_STRATEGY,
        TWO_PART_ARTIFACT_SCHEMA_VERSION,
        (
            "StructuralAdapterStrategy",
            "StructuralRoute",
            "StructuralFallback",
            "StructuralCalibration",
            "StructuralF0",
            "StructuralRole",
            "StructuralPosition",
            "P_PrePool",
        ),
        applicability={"distribution_classes": ("continuous",), "role_registry_gated": True},
    ),
    _yards(
        AFFINE_STRATEGY,
        AFFINE_ARTIFACT_SCHEMA_VERSION,
        (
            "StructuralAdapterStrategy",
            "StructuralRoute",
            "StructuralFallback",
            "PITRecalKnots",
            "P_PrePool",
        ),
        applicability={
            "distribution_classes": ("continuous",),
            "required_data_columns": ("Player position",),
        },
    ),
)

# Corners the sweep must evaluate on the cell's *current* matrix before it spends a sampler
# trial. Unlike CONFIRM_EVIDENCE_CORNERS these carry no independent full-HPO verdict, so
# confirmation never synthesizes one: a mandatory corner reaches a walk only through the board
# row its own evaluation writes. The three NFL cells below are the reachability-repair
# acceptance set — each names the recipe whose evidence predates the current grid.
MANDATORY_SWEEP_CORNERS: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    ("NFL", "receiving yards", TWO_PART_STRATEGY, _yards_controls(TWO_PART_STRATEGY)),
    ("NFL", "rushing yards", AFFINE_STRATEGY, _yards_controls(AFFINE_STRATEGY)),
    (
        "NFL",
        "passing yards",
        "SkewNormal",
        {
            "dist": "SkewNormal",
            "normalization": "ratio_meanyr",
            "dist_training_loss": "crps",
            "sn_param": "direct",
            "blending_loss_fn": "nll",
            "posthoc": "none",
        },
    ),
)

# Corners with an independent full-HPO six-gate pass, kept so the sweep can seed a cell with a
# known-good recipe instead of rediscovering it and so confirmation can nominate it directly.
# Registry validates each against its spec's grid at import. NFL passing TDs:
# docs/handoffs/model_improvement_track.md, the 2026-07-20 NFL popular-market re-baseline.
CONFIRM_EVIDENCE_CORNERS: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    (
        "NFL",
        "passing tds",
        "DPO",
        {
            "dist": "DPO",
            "dist_training_loss": "nll",
            "count_dispersion_objective": "pit_ks",
            "blending_loss_fn": "nll",
            "posthoc": "roe_mean",
        },
    ),
)

# Effective historical behavior of each strategy's unpersisted controls — what a cell trained
# with before the control became a swept axis, not the CLI's ``auto`` sentinel. Reconstructing an
# incumbent from these (registry.incumbent_controls) is what keeps a cell whose stat_meta predates
# the grid nominatable instead of silently absent. Structural strategies are fully described by
# their fixed controls and need no entry.
INCUMBENT_CONTROL_DEFAULTS: dict[str, dict[str, str]] = {
    "SkewNormal": {
        "normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": _DEFAULT_BLENDING,
        "posthoc": "none",
    },
    "Mixture": {
        "normalization": "ratio_meanyr",
        "blending_loss_fn": _DEFAULT_BLENDING,
        "posthoc": "none",
    },
    "ZINB": {
        "zinb_mode": "joint",
        "count_dispersion_objective": "crps",
        "blending_loss_fn": _DEFAULT_BLENDING,
        "posthoc": "none",
    },
    "NegBin": {
        "count_dispersion_objective": "crps",
        "blending_loss_fn": _DEFAULT_BLENDING,
        "posthoc": "none",
    },
    "DPO": {
        "count_dispersion_objective": "crps",
        "blending_loss_fn": _DEFAULT_BLENDING,
        "posthoc": "none",
    },
}
