"""Resolve canonical strategy identity from current and legacy model artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from sportstradamus.training.model_strategy_artifacts import (
    MODEL_STRATEGY_MODEL_KEY,
    ArtifactIdentity,
    _identity_from_blob,
    _validate_controls,
    _validate_core,
)
from sportstradamus.training.model_strategy_model_validation import (
    _adapter_keys,
    _nested,
    _validate_adapter,
    validate_model_recipe,
)
from sportstradamus.training.model_strategy_registry import StrategySpec, get_strategy


def _legacy_identity(
    spec: StrategySpec, league: str, market: str, legacy: Mapping[str, object] | None = None
) -> ArtifactIdentity:
    payload = legacy or {}
    split = _nested(payload, spec.split_fingerprint_path)
    return ArtifactIdentity.from_spec(
        spec,
        league,
        market,
        str(payload.get(spec.legacy_status_field, "active")),
        split_fingerprint=str(split) if split is not None else None,
    )


def resolve_report_identity(
    model: Mapping[str, object], league: str, market: str
) -> ArtifactIdentity:
    if MODEL_STRATEGY_MODEL_KEY in model:
        payload = model[MODEL_STRATEGY_MODEL_KEY]
        if not isinstance(payload, Mapping):
            raise ValueError("malformed model_strategy identity")
        identity = _identity_from_blob(payload)
        spec = _validate_core(identity, league, market)
        controls = _validate_controls(identity, spec)
        validate_model_recipe(model, spec, controls)
        _validate_adapter(spec, identity, model)
        return identity
    keys = _adapter_keys(model)
    if keys:
        if len(keys) != 1 or not isinstance(model[next(iter(keys))], Mapping):
            raise ValueError("malformed legacy strategy identity")
        legacy = model[next(iter(keys))]
        slug = legacy.get("slug")
        if not isinstance(slug, str):
            raise ValueError("legacy strategy identity requires a slug")
        spec = get_strategy(slug)
        identity = _legacy_identity(spec, league, market, legacy)
        _validate_core(identity, league, market)
        _validate_adapter(spec, identity, model)
        return identity
    distribution = model.get("distribution")
    if not isinstance(distribution, str):
        raise ValueError("identity-absent legacy model requires a registered distribution")
    spec = get_strategy(distribution)
    if spec.is_structural:
        raise ValueError("structural legacy model requires its adapter identity")
    identity = _legacy_identity(spec, league, market)
    _validate_core(identity, league, market)
    return identity
