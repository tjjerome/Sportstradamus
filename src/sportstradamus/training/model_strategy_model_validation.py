"""Validate persisted strategy-model recipe and structural adapter state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.model_strategy_registry import StrategySpec, registered_strategies

if TYPE_CHECKING:
    from sportstradamus.training.model_strategy_artifacts import ArtifactIdentity


def _nested(payload: Mapping[str, object], path: tuple[str, ...]) -> object | None:
    if not path:
        return None
    value: object = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def validate_model_recipe(
    model: Mapping[str, object], spec: StrategySpec, controls: Mapping[str, str]
) -> None:
    """Reject drift between signed strategy controls and persisted runtime fields."""
    expected = {
        field: controls[control]
        for control, field in (
            ("dist", "distribution"),
            ("sn_param", "sn_param"),
            ("zinb_mode", "zinb_mode"),
            ("posthoc", "posthoc"),
        )
        if control in controls
    }
    expected.update(spec.fixed_persist)
    normalization = controls.get("normalization")
    expected_normalized = None
    if normalization is not None:
        expected["target_normalization"] = normalization
        expected_normalized = (
            get_target_normalization(normalization).start_mode_flag == "normalized"
        )
    elif "count" in spec.applicability.distribution_classes:
        expected["target_normalization"] = "none"
        expected_normalized = False
    drift = [field for field, value in expected.items() if model.get(field) != value]
    if expected_normalized is not None and model.get("normalized") is not expected_normalized:
        drift.append("normalized")
    if drift:
        raise ValueError(f"{spec.slug}: signed model recipe mismatch: {', '.join(drift)}")


def _adapter_keys(model: Mapping[str, object]) -> set[str]:
    return {
        str(spec.legacy_model_key)
        for spec in registered_strategies()
        if spec.legacy_model_key and spec.legacy_model_key in model
    }


def _validate_adapter(
    spec: StrategySpec, identity: ArtifactIdentity, model: Mapping[str, object]
) -> None:
    keys = _adapter_keys(model)
    if not spec.legacy_model_key:
        if keys:
            raise ValueError(f"{spec.slug}: unexpected adapter identity")
        return
    if keys != {spec.legacy_model_key}:
        raise ValueError(f"{spec.slug}: missing or ambiguous adapter identity")
    legacy = model[spec.legacy_model_key]
    if not isinstance(legacy, Mapping):
        raise ValueError(f"{spec.slug}: malformed adapter identity")
    split = _nested(legacy, spec.split_fingerprint_path)
    if (
        legacy.get("slug") != spec.slug
        or legacy.get(spec.legacy_status_field) != identity.status
        or legacy.get(spec.legacy_schema_field) != spec.artifact_schema_version
        or (str(split) if split is not None else None) != identity.split_fingerprint
    ):
        raise ValueError(f"{spec.slug}: mismatched adapter identity")
