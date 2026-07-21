"""Plain-data declarations for built-in model strategies."""

from __future__ import annotations

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
_YARDS_SELECTOR = {
    "key": "structural-strategy",
    "version": 1,
    "cli_flag": "--structural-strategy",
    "meta_field": "structural_strategy",
    "default_value": "none",
    "auto_value": "auto",
}


def _base(
    slug: str,
    distribution_class: str,
    axes: dict[str, tuple[str, ...]],
    persist: dict[str, str],
    *,
    capabilities: tuple[str, ...] = _SHIP,
    fixed_controls: dict[str, str] | None = None,
) -> dict:
    return {
        "slug": slug,
        "implementation_version": 1,
        "artifact_schema_version": 1,
        "family": slug,
        "applicability": {"distribution_classes": (distribution_class,)},
        "enrollments": (),
        "capabilities": capabilities,
        "axes": axes,
        "fixed_controls": fixed_controls or {},
        "cli_flags": _CONTROL_FLAGS,
        "persist": persist,
        "fixed_persist": {},
        "required_columns": (),
        "shared_csv_columns": (),
        "selector": None,
        "artifact_namespace_suffix": False,
        "legacy_model_key": None,
        "legacy_csv_identity_column": None,
        "legacy_schema_field": "schema_version",
        "legacy_status_field": "status",
        "split_fingerprint_path": (),
        "matrix_hash_path": (),
    }


_NORMS = (
    "ratio_meanyr",
    "centered_additive_mean10",
    "centered_additive_eb_meanyr_k10",
)
_BLENDING = ("crps", "nll")
_BASE_SPECS = (
    _base(
        "SkewNormal",
        "continuous",
        {
            "normalization": _NORMS,
            "dist_training_loss": ("crps", "nll"),
            "sn_param": ("direct", "centered"),
            "blending_loss_fn": _BLENDING,
        },
        {
            "dist": "dist",
            "normalization": "target_normalization",
            "dist_training_loss": "dist_training_loss",
            "sn_param": "sn_param",
            "blending_loss_fn": "blending",
        },
        fixed_controls={"dist": "SkewNormal"},
    ),
    _base(
        "Mixture",
        "continuous",
        {"dist": ("Mixture",), "normalization": _NORMS},
        {"dist": "dist", "normalization": "target_normalization"},
        capabilities=_RESEARCH,
    ),
    _base(
        "ZINB",
        "count",
        {
            "dist": ("ZINB",),
            "zinb_mode": ("joint", "hurdle"),
            "count_dispersion_objective": ("crps", "pit_ks"),
            "blending_loss_fn": _BLENDING,
        },
        {
            "dist": "dist",
            "zinb_mode": "zinb_mode",
            "count_dispersion_objective": "count_dispersion_objective",
            "blending_loss_fn": "blending",
        },
    ),
    _base(
        "NegBin",
        "count",
        {
            "dist": ("NegBin",),
            "count_dispersion_objective": ("crps", "pit_ks"),
            "blending_loss_fn": _BLENDING,
        },
        {
            "dist": "dist",
            "count_dispersion_objective": "count_dispersion_objective",
            "blending_loss_fn": "blending",
        },
    ),
    _base(
        "DPO",
        "count",
        {
            "dist": ("DPO",),
            "count_dispersion_objective": ("crps", "pit_ks"),
            "blending_loss_fn": _BLENDING,
        },
        {
            "dist": "dist",
            "count_dispersion_objective": "count_dispersion_objective",
            "blending_loss_fn": "blending",
        },
    ),
)


def _compatibility(slug: str) -> dict:
    data = _base(
        slug,
        "continuous",
        {},
        {},
        capabilities=_SERVE_ONLY,
        fixed_controls={"dist": slug},
    )
    data["applicability"] = {
        "distribution_classes": ("continuous",),
        "distributions": (slug,),
    }
    return data


_COMPATIBILITY_SPECS = tuple(_compatibility(slug) for slug in ("Gamma", "ZAGamma"))

_YARDS_CONTROLS = {
    "dist": "SkewNormal",
    "normalization": "ratio_meanyr",
    "dist_training_loss": "crps",
    "sn_param": "direct",
    "blending_loss_fn": "nll",
    "hpo_selection": "loss",
    "stabilization": "None",
    "posthoc": "none",
}
_YARDS_PERSIST = {
    "dist": "dist",
    "normalization": "target_normalization",
    "dist_training_loss": "dist_training_loss",
    "sn_param": "sn_param",
    "blending_loss_fn": "blending",
    "hpo_selection": "hpo_selection",
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
        "implementation_version": 1,
        "artifact_schema_version": schema,
        "family": "SkewNormal",
        "applicability": applicability,
        "enrollments": enrollments,
        "capabilities": _SHIP,
        "axes": {},
        "fixed_controls": _YARDS_CONTROLS,
        "cli_flags": _CONTROL_FLAGS,
        "persist": _YARDS_PERSIST,
        "fixed_persist": {"posthoc": "none"},
        "required_columns": artifact_columns,
        "shared_csv_columns": tuple(
            column for column in artifact_columns if column in ("P_PrePool", "PITRecalKnots")
        ),
        "selector": _YARDS_SELECTOR,
        "artifact_namespace_suffix": True,
        "legacy_model_key": "structural_calibration",
        "legacy_csv_identity_column": "StructuralAdapterStrategy",
        "legacy_schema_field": "schema_version",
        "legacy_status_field": "status",
        "split_fingerprint_path": ("validation_audit", "split_fingerprint_sha256"),
        "matrix_hash_path": (),
    }


BUILTIN_SPEC_DATA = (
    *_BASE_SPECS,
    *_COMPATIBILITY_SPECS,
    _yards(
        TWO_PART_STRATEGY,
        3,
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
        1,
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
