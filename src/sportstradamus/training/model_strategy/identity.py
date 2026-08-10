"""Canonical strategy artifact identity: build, validate, and resolve.

One module owns the whole identity contract that used to be split across four files:

* the immutable :class:`ArtifactIdentity` schema and its spec-/corner-/input-/cell-bound
  construction (:func:`build_artifact_identity`);
* the persisted-recipe and structural-adapter validators (:func:`validate_model_recipe`,
  :func:`_validate_adapter`);
* CSV-frame identity validation (:func:`validate_strategy_frame`) and full model+CSV
  cross-validation (:func:`validate_strategy_artifacts`);
* resolution of a single authoritative identity from current or legacy artifacts
  (:func:`resolve_report_identity`).

The blob→core→controls validation sequence is shared by all three read paths through
:func:`_resolve_and_validate_identity`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import pandas as pd

from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.model_strategy.registry import (
    BASE_STRUCTURAL_STRATEGY,
    StrategySpec,
    controls_json,
    corner_fingerprint,
    get_strategy,
    parse_controls,
    registered_strategies,
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


def _legacy_status_and_split(
    spec: StrategySpec, legacy_payload: Mapping[str, object] | None
) -> tuple[str, str | None]:
    """Resolve the artifact status and split fingerprint from a legacy payload, failing loud."""
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
    return status, split_text


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
    status, split_text = _legacy_status_and_split(spec, legacy_payload)
    return ArtifactIdentity.from_spec(
        spec,
        league,
        market,
        status,
        controls_json=controls_json(canonical_controls),
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


def _core_fields_match(
    identity: ArtifactIdentity, spec: StrategySpec, league: str, market: str
) -> bool:
    """Whether the identity's spec-owned fields and cell match the registered spec."""
    structural = spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY
    return (
        identity.structural_strategy == structural
        and identity.signature == spec.canonical_signature
        and identity.implementation_version == spec.implementation_version
        and identity.artifact_schema_version == spec.artifact_schema_version
        and (identity.league, identity.market) == (league, market)
    )


def _validate_core(identity: ArtifactIdentity, league: str, market: str) -> StrategySpec:
    spec = get_strategy(identity.strategy_slug)
    if spec.enrollments and (identity.league, identity.market) not in spec.enrollments:
        raise ValueError(f"{spec.slug}: wrong-cell strategy identity is not enrolled for this cell")
    if not _core_fields_match(identity, spec, league, market):
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


def _resolve_and_validate_identity(
    payload: Mapping[str, object], league: str | None, market: str | None
) -> tuple[ArtifactIdentity, StrategySpec, dict[str, str]]:
    """Deserialize an identity blob and validate its core + controls; the shared read path."""
    identity = _identity_from_blob(payload)
    spec = _validate_core(identity, league or identity.league, market or identity.market)
    controls = _validate_controls(identity, spec)
    return identity, spec, controls


_CSV_FIELDS = {
    "strategy_slug": MODEL_STRATEGY_CSV_COLUMN,
    "structural_strategy": STRUCTURAL_STRATEGY_CSV_COLUMN,
    "signature": STRATEGY_SIGNATURE_CSV_COLUMN,
    "implementation_version": STRATEGY_IMPLEMENTATION_CSV_COLUMN,
    "artifact_schema_version": ARTIFACT_SCHEMA_CSV_COLUMN,
    "league": STRATEGY_LEAGUE_CSV_COLUMN,
    "market": STRATEGY_MARKET_CSV_COLUMN,
    "status": STRATEGY_STATUS_CSV_COLUMN,
    "controls_json": STRATEGY_CONTROLS_CSV_COLUMN,
    "corner_fingerprint": CORNER_FINGERPRINT_CSV_COLUMN,
    "matrix_hash": MATRIX_HASH_CSV_COLUMN,
}


def _adapter_columns() -> set[str]:
    """Structural-adapter-only CSV columns that may never appear without a generic identity."""
    columns: set[str] = set()
    for registered in registered_strategies():
        if not registered.is_structural:
            continue
        columns.update(set(registered.required_columns) - set(registered.shared_csv_columns))
        if registered.legacy_csv_identity_column:
            columns.add(registered.legacy_csv_identity_column)
    return columns


def _coerce_int_field(value: object, key: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strategy artifact requires integral {key}") from exc
    if isinstance(value, bool) or float(value) != integer:
        raise ValueError(f"strategy artifact requires integral {key}")
    return integer


def _csv_identity_payload(frame: pd.DataFrame) -> dict[str, object] | None:
    """Constant CSV identity fields as a blob, or ``None`` for an identity-absent legacy frame."""
    if MODEL_STRATEGY_CSV_COLUMN not in frame:
        generic_columns = {*_CSV_FIELDS.values(), SPLIT_FINGERPRINT_CSV_COLUMN}
        if any(column in frame for column in generic_columns):
            raise ValueError("partial generic model-strategy identity")
        if any(column in frame for column in _adapter_columns()):
            raise ValueError("adapter strategy columns require generic model-strategy identity")
        return None
    payload = {key: _constant(frame, column) for key, column in _CSV_FIELDS.items()}
    for key in ("implementation_version", "artifact_schema_version"):
        payload[key] = _coerce_int_field(payload[key], key)
    payload["split_fingerprint"] = (
        _constant(frame, SPLIT_FINGERPRINT_CSV_COLUMN)
        if SPLIT_FINGERPRINT_CSV_COLUMN in frame
        else None
    )
    return payload


def validate_strategy_frame(
    frame: pd.DataFrame, *, league: str | None = None, market: str | None = None
) -> tuple[ArtifactIdentity | None, StrategySpec | None]:
    """Validate generic CSV identity; identity-absent non-adapter legacy frames return ``None``."""
    payload = _csv_identity_payload(frame)
    if payload is None:
        return None, None
    identity, spec, _controls = _resolve_and_validate_identity(payload, league, market)
    if identity.status != "active":
        raise InactiveStrategyArtifactError(f"{spec.slug}: inactive strategy artifact")
    if spec.legacy_csv_identity_column and (
        str(_constant(frame, spec.legacy_csv_identity_column)) != spec.slug
    ):
        raise ValueError(f"{spec.slug}: adapter CSV identity mismatch")
    missing = sorted(set(spec.required_columns) - set(frame))
    if missing:
        raise ValueError(f"{spec.slug}: artifact missing required columns {missing}")
    return identity, spec


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
    frame_identity, frame_spec = validate_strategy_frame(frame, league=league, market=market)
    payload = model.get(MODEL_STRATEGY_MODEL_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{spec.slug}: missing or malformed generic strategy identity")
    identity, model_spec, model_controls = _resolve_and_validate_identity(payload, league, market)
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
        identity, spec, controls = _resolve_and_validate_identity(payload, league, market)
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
