"""Canonical, spec-, corner-, input-, and cell-bound strategy artifact identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import pandas as pd

from sportstradamus.training.model_strategy_model_validation import (
    _nested,
    _validate_adapter,
    validate_model_recipe,
)
from sportstradamus.training.model_strategy_registry import (
    BASE_STRUCTURAL_STRATEGY,
    StrategySpec,
    controls_json,
    corner_fingerprint,
    get_strategy,
    parse_controls,
    strategy_controls,
)

MODEL_STRATEGY_MODEL_KEY = "model_strategy"
MODEL_STRATEGY_CSV_COLUMN = "ModelStrategy"
STRUCTURAL_STRATEGY_CSV_COLUMN = "StructuralStrategy"
STRATEGY_SIGNATURE_CSV_COLUMN = "StrategySignature"
STRATEGY_IMPLEMENTATION_CSV_COLUMN = "StrategyImplementationVersion"
ARTIFACT_SCHEMA_CSV_COLUMN = "StrategyArtifactSchemaVersion"
STRATEGY_STATUS_CSV_COLUMN = "StrategyStatus"
STRATEGY_LEAGUE_CSV_COLUMN = "StrategyLeague"
STRATEGY_MARKET_CSV_COLUMN = "StrategyMarket"
STRATEGY_CONTROLS_CSV_COLUMN = "StrategyControlsJSON"
CORNER_FINGERPRINT_CSV_COLUMN = "StrategyCornerFingerprint"
MATRIX_HASH_CSV_COLUMN = "StrategyMatrixHash"
SPLIT_FINGERPRINT_CSV_COLUMN = "StrategySplitFingerprint"


class InactiveStrategyArtifactError(ValueError):
    """An explicit strategy produced its declared safe fallback, not an active candidate."""


@dataclass(frozen=True)
class ArtifactIdentity:
    strategy_slug: str
    structural_strategy: str
    signature: str
    implementation_version: int
    artifact_schema_version: int
    league: str
    market: str
    status: str
    controls_json: str | None = None
    corner_fingerprint: str | None = None
    matrix_hash: str | None = None
    split_fingerprint: str | None = None

    @classmethod
    def from_spec(
        cls,
        spec: StrategySpec,
        league: str,
        market: str,
        status: str,
        *,
        controls_json: str | None = None,
        corner_fingerprint: str | None = None,
        matrix_hash: str | None = None,
        split_fingerprint: str | None = None,
    ) -> ArtifactIdentity:
        """Build the canonical identity fields owned by a registered strategy spec."""
        return cls(
            strategy_slug=spec.slug,
            structural_strategy=(spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY),
            signature=spec.canonical_signature,
            implementation_version=spec.implementation_version,
            artifact_schema_version=spec.artifact_schema_version,
            league=league,
            market=market,
            status=status,
            controls_json=controls_json,
            corner_fingerprint=corner_fingerprint,
            matrix_hash=matrix_hash,
            split_fingerprint=split_fingerprint,
        )

    def as_model_blob(self) -> dict[str, object]:
        return asdict(self)


def build_artifact_identity(
    strategy_slug: str,
    league: str,
    market: str,
    controls: Mapping[str, str],
    legacy_payload: Mapping[str, object] | None = None,
    *,
    matrix_hash: str,
) -> ArtifactIdentity:
    spec = get_strategy(strategy_slug)
    if spec.enrollments and (league, market) not in spec.enrollments:
        raise ValueError(f"{spec.slug}: strategy is not enrolled for this cell")
    canonical_controls = dict(controls)
    if canonical_controls not in strategy_controls(spec):
        raise ValueError(f"{spec.slug}: controls are not a registered strategy corner")
    if not isinstance(matrix_hash, str) or not matrix_hash:
        raise ValueError(f"{spec.slug}: matrix hash is required")
    payload = legacy_payload or {}
    status = payload.get(spec.legacy_status_field, "active")
    if not isinstance(status, str) or not status:
        raise ValueError(f"{spec.slug}: invalid strategy status")
    split = _nested(payload, spec.split_fingerprint_path)
    split_text = str(split) if split is not None else None
    if (
        legacy_payload is not None
        and spec.split_fingerprint_path
        and status == "active"
        and not split_text
    ):
        raise ValueError(f"{spec.slug}: active strategy artifact requires a split fingerprint")
    control_blob = controls_json(canonical_controls)
    return ArtifactIdentity.from_spec(
        spec,
        league,
        market,
        status,
        controls_json=control_blob,
        corner_fingerprint=corner_fingerprint(spec, canonical_controls, matrix_hash),
        matrix_hash=matrix_hash,
        split_fingerprint=split_text,
    )


_STRATEGY_IDENTITY_CSV_FIELDS = (
    (MODEL_STRATEGY_CSV_COLUMN, "strategy_slug"),
    (STRUCTURAL_STRATEGY_CSV_COLUMN, "structural_strategy"),
    (STRATEGY_SIGNATURE_CSV_COLUMN, "signature"),
    (STRATEGY_IMPLEMENTATION_CSV_COLUMN, "implementation_version"),
    (ARTIFACT_SCHEMA_CSV_COLUMN, "artifact_schema_version"),
    (STRATEGY_LEAGUE_CSV_COLUMN, "league"),
    (STRATEGY_MARKET_CSV_COLUMN, "market"),
    (STRATEGY_STATUS_CSV_COLUMN, "status"),
    (STRATEGY_CONTROLS_CSV_COLUMN, "controls_json"),
    (CORNER_FINGERPRINT_CSV_COLUMN, "corner_fingerprint"),
    (MATRIX_HASH_CSV_COLUMN, "matrix_hash"),
    (SPLIT_FINGERPRINT_CSV_COLUMN, "split_fingerprint"),
)
STRATEGY_IDENTITY_CSV_COLUMNS = tuple(column for column, _key in _STRATEGY_IDENTITY_CSV_FIELDS)


def artifact_identity_columns(identity: Mapping[str, object]) -> dict[str, object]:
    return {
        column: identity[key]
        for column, key in _STRATEGY_IDENTITY_CSV_FIELDS
        if identity.get(key) is not None
    }


def _constant(frame: pd.DataFrame, column: str) -> object:
    if column not in frame or frame[column].isna().any() or frame[column].nunique() != 1:
        raise ValueError(f"strategy artifact requires one constant nonmissing {column}")
    return frame[column].iloc[0]


def _text(payload: Mapping[str, object], key: str, *, optional: bool = False) -> str | None:
    value = payload.get(key)
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"model-strategy identity requires nonempty string {key}")
    return value


def _identity_from_blob(payload: Mapping[str, object]) -> ArtifactIdentity:
    implementation = payload.get("implementation_version")
    schema = payload.get("artifact_schema_version")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (implementation, schema)
    ):
        raise ValueError("model-strategy identity versions must be integers")
    return ArtifactIdentity(
        strategy_slug=str(_text(payload, "strategy_slug")),
        structural_strategy=str(_text(payload, "structural_strategy")),
        signature=str(_text(payload, "signature")),
        implementation_version=implementation,
        artifact_schema_version=schema,
        league=str(_text(payload, "league")),
        market=str(_text(payload, "market")),
        status=str(_text(payload, "status")),
        controls_json=_text(payload, "controls_json", optional=True),
        corner_fingerprint=_text(payload, "corner_fingerprint", optional=True),
        matrix_hash=_text(payload, "matrix_hash", optional=True),
        split_fingerprint=_text(payload, "split_fingerprint", optional=True),
    )


def _validate_core(identity: ArtifactIdentity, league: str, market: str) -> StrategySpec:
    spec = get_strategy(identity.strategy_slug)
    structural = spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY
    if spec.enrollments and (identity.league, identity.market) not in spec.enrollments:
        raise ValueError(f"{spec.slug}: wrong-cell strategy identity is not enrolled for this cell")
    if (
        identity.structural_strategy != structural
        or identity.signature != spec.canonical_signature
        or identity.implementation_version != spec.implementation_version
        or identity.artifact_schema_version != spec.artifact_schema_version
        or (identity.league, identity.market) != (league, market)
    ):
        raise ValueError(f"{spec.slug}: stale, mismatched, or wrong-cell strategy identity")
    if (
        spec.split_fingerprint_path
        and identity.status == "active"
        and not identity.split_fingerprint
    ):
        raise ValueError(f"{spec.slug}: active strategy identity requires a split fingerprint")
    if not spec.split_fingerprint_path and identity.split_fingerprint is not None:
        raise ValueError(f"{spec.slug}: split fingerprint contract mismatch")
    return spec


def _validate_controls(identity: ArtifactIdentity, spec: StrategySpec) -> dict[str, str]:
    if any(
        value is None
        for value in (identity.controls_json, identity.corner_fingerprint, identity.matrix_hash)
    ):
        raise ValueError(f"{spec.slug}: strategy corner/input identity is missing")
    controls = parse_controls(identity.controls_json)
    if controls_json(controls) != identity.controls_json or controls not in strategy_controls(spec):
        raise ValueError(f"{spec.slug}: stale or noncanonical strategy controls")
    if identity.corner_fingerprint != corner_fingerprint(spec, controls, identity.matrix_hash):
        raise ValueError(f"{spec.slug}: stale strategy corner fingerprint")
    return controls


def validate_strategy_artifacts(
    spec: StrategySpec,
    controls: Mapping[str, str],
    frame: pd.DataFrame,
    model: Mapping[str, object],
    *,
    league: str,
    market: str,
    matrix_hash: str,
) -> ArtifactIdentity:
    from sportstradamus.training.model_strategy_frame import validate_strategy_frame

    frame_identity, frame_spec = validate_strategy_frame(frame, league=league, market=market)
    payload = model.get(MODEL_STRATEGY_MODEL_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{spec.slug}: missing or malformed generic strategy identity")
    identity = _identity_from_blob(payload)
    model_spec = _validate_core(identity, league, market)
    model_controls = _validate_controls(identity, model_spec)
    validate_model_recipe(model, model_spec, model_controls)
    _validate_adapter(model_spec, identity, model)
    if identity.status != "active":
        raise InactiveStrategyArtifactError(f"{spec.slug}: inactive strategy artifact")
    expected_controls = controls_json(dict(controls))
    expected_corner = corner_fingerprint(spec, controls, matrix_hash)
    if (
        model_spec != spec
        or frame_spec != spec
        or identity.matrix_hash != matrix_hash
        or identity.controls_json != expected_controls
        or identity.corner_fingerprint != expected_corner
    ):
        raise ValueError(f"{spec.slug}: mismatched strategy artifact")
    if identity != frame_identity:
        raise ValueError(f"{spec.slug}: model/CSV strategy identity mismatch")
    return identity
