"""Serving-boundary guards for generic model-strategy identity."""

from __future__ import annotations

from copy import deepcopy

import pytest

from sportstradamus.prediction.model_prob import _resolve_serving_strategy
from sportstradamus.training.model_strategy_artifacts import (
    MODEL_STRATEGY_MODEL_KEY,
    InactiveStrategyArtifactError,
    build_artifact_identity,
)
from sportstradamus.training.model_strategy_registry import (
    BASE_STRUCTURAL_STRATEGY,
    get_strategy,
    strategy_controls,
)
from sportstradamus.training.structural_strategies import (
    ROLE_COLUMNS,
    TWO_PART_STRATEGY,
)

_MATRIX_HASH = "d" * 64


def _receiving_model(*, status: str = "active", include_split: bool = True) -> dict:
    slug = TWO_PART_STRATEGY
    spec = get_strategy(slug)
    controls = strategy_controls(spec)[0]
    adapter = {
        "slug": slug,
        "schema_version": spec.artifact_schema_version,
        "status": status,
    }
    if include_split:
        adapter["validation_audit"] = {"split_fingerprint_sha256": "split-sha256"}
    identity = build_artifact_identity(
        slug,
        "NFL",
        "receiving yards",
        controls,
        adapter,
        matrix_hash=_MATRIX_HASH,
    )
    return {
        "distribution": "SkewNormal",
        "expected_columns": [*ROLE_COLUMNS, "Player position"],
        "target_normalization": "ratio_meanyr",
        "normalized": True,
        "sn_param": "direct",
        "posthoc": "none",
        "nfl_yards_experiment": adapter,
        MODEL_STRATEGY_MODEL_KEY: identity.as_model_blob(),
    }


def test_generic_active_identity_selects_structural_dispatch():
    identity = _resolve_serving_strategy(_receiving_model(), "NFL", "receiving yards")

    assert identity.structural_strategy == (TWO_PART_STRATEGY)


@pytest.mark.parametrize("distribution", ["SkewNormal", "DPO"])
def test_identity_absent_base_legacy_model_remains_serveable(distribution):
    identity = _resolve_serving_strategy(
        {"distribution": distribution},
        "NFL",
        "passing yards",
    )

    assert identity.strategy_slug == distribution
    assert identity.structural_strategy == BASE_STRUCTURAL_STRATEGY
    assert identity.corner_fingerprint is None


def test_identity_absent_research_only_legacy_model_cannot_serve():
    with pytest.raises(ValueError, match="lacks serve capability"):
        _resolve_serving_strategy(
            {"distribution": "Mixture"},
            "WNBA",
            "points",
        )


def test_structural_adapter_without_generic_identity_fails_closed():
    model = _receiving_model()
    model.pop(MODEL_STRATEGY_MODEL_KEY)

    with pytest.raises(ValueError, match="requires canonical generic strategy identity"):
        _resolve_serving_strategy(model, "NFL", "receiving yards")


def test_generic_identity_for_wrong_cell_fails_closed():
    with pytest.raises(ValueError, match="wrong-cell strategy identity"):
        _resolve_serving_strategy(_receiving_model(), "NFL", "rushing yards")


def test_inactive_generic_identity_cannot_serve():
    with pytest.raises(InactiveStrategyArtifactError, match="cannot serve"):
        _resolve_serving_strategy(
            _receiving_model(status="killed_fallback"),
            "NFL",
            "receiving yards",
        )


def test_structural_identity_without_split_fingerprint_fails_closed():
    with pytest.raises(ValueError, match="split fingerprint"):
        _resolve_serving_strategy(
            _receiving_model(include_split=False),
            "NFL",
            "receiving yards",
        )


@pytest.mark.parametrize("field", ["controls_json", "corner_fingerprint"])
def test_tampered_corner_identity_fails_closed(field):
    model = deepcopy(_receiving_model())
    model[MODEL_STRATEGY_MODEL_KEY][field] = "{}" if field == "controls_json" else "tampered"

    with pytest.raises(ValueError, match=r"strategy controls|corner fingerprint"):
        _resolve_serving_strategy(model, "NFL", "receiving yards")


def test_generic_strategy_requires_its_registered_feature_columns():
    model = _receiving_model()
    model["expected_columns"].remove(ROLE_COLUMNS[0])

    with pytest.raises(ValueError, match="not applicable/capable"):
        _resolve_serving_strategy(model, "NFL", "receiving yards")


def test_generic_strategy_distribution_must_match_its_corner():
    model = _receiving_model()
    model["distribution"] = "DPO"

    with pytest.raises(ValueError, match="model recipe mismatch: distribution"):
        _resolve_serving_strategy(model, "NFL", "receiving yards")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_normalization", "centered_additive_mean10", "target_normalization"),
        ("sn_param", "centered", "sn_param"),
        ("normalized", False, "normalized"),
        ("posthoc", "roe_mean", "posthoc"),
    ],
)
def test_generic_runtime_transform_must_match_signed_controls(field, value, message):
    model = _receiving_model()
    model[field] = value

    with pytest.raises(ValueError, match=message):
        _resolve_serving_strategy(model, "NFL", "receiving yards")


def test_generic_count_strategy_requires_raw_target_semantics():
    spec = get_strategy("DPO")
    controls = strategy_controls(spec)[0]
    model = {
        "distribution": "DPO",
        "expected_columns": [],
        "target_normalization": "none",
        "normalized": False,
        MODEL_STRATEGY_MODEL_KEY: build_artifact_identity(
            spec.slug,
            "NFL",
            "passing tds",
            controls,
            matrix_hash=_MATRIX_HASH,
        ).as_model_blob(),
    }

    assert _resolve_serving_strategy(model, "NFL", "passing tds").strategy_slug == "DPO"
    model["normalized"] = True
    with pytest.raises(ValueError, match="normalized"):
        _resolve_serving_strategy(model, "NFL", "passing tds")
    model["normalized"] = False
    model["target_normalization"] = "ratio_meanyr"
    with pytest.raises(ValueError, match="target_normalization"):
        _resolve_serving_strategy(model, "NFL", "passing tds")


def test_generic_zinb_mode_must_match_signed_controls():
    spec = get_strategy("ZINB")
    controls = next(c for c in strategy_controls(spec) if c["zinb_mode"] == "hurdle")
    model = {
        "distribution": "ZINB",
        "expected_columns": [],
        "target_normalization": "none",
        "normalized": False,
        "zinb_mode": "joint",
        MODEL_STRATEGY_MODEL_KEY: build_artifact_identity(
            spec.slug,
            "NFL",
            "passing tds",
            controls,
            matrix_hash=_MATRIX_HASH,
        ).as_model_blob(),
    }

    with pytest.raises(ValueError, match="zinb_mode"):
        _resolve_serving_strategy(model, "NFL", "passing tds")


def test_generic_research_only_strategy_cannot_serve():
    spec = get_strategy("Mixture")
    controls = strategy_controls(spec)[0]
    model = {
        "distribution": "Mixture",
        "expected_columns": [],
        "target_normalization": controls["normalization"],
        "normalized": True,
        MODEL_STRATEGY_MODEL_KEY: build_artifact_identity(
            spec.slug,
            "WNBA",
            "points",
            controls,
            matrix_hash=_MATRIX_HASH,
        ).as_model_blob(),
    }

    with pytest.raises(ValueError, match="not applicable/capable"):
        _resolve_serving_strategy(model, "WNBA", "points")
