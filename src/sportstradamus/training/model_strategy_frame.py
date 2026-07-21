"""CSV-only validation for canonical model-strategy identities."""

from __future__ import annotations

import pandas as pd

from sportstradamus.training.model_strategy_artifacts import (
    ARTIFACT_SCHEMA_CSV_COLUMN,
    CORNER_FINGERPRINT_CSV_COLUMN,
    MATRIX_HASH_CSV_COLUMN,
    MODEL_STRATEGY_CSV_COLUMN,
    SPLIT_FINGERPRINT_CSV_COLUMN,
    STRATEGY_CONTROLS_CSV_COLUMN,
    STRATEGY_IMPLEMENTATION_CSV_COLUMN,
    STRATEGY_LEAGUE_CSV_COLUMN,
    STRATEGY_MARKET_CSV_COLUMN,
    STRATEGY_SIGNATURE_CSV_COLUMN,
    STRATEGY_STATUS_CSV_COLUMN,
    STRUCTURAL_STRATEGY_CSV_COLUMN,
    ArtifactIdentity,
    InactiveStrategyArtifactError,
    _constant,
    _identity_from_blob,
    _validate_controls,
    _validate_core,
)
from sportstradamus.training.model_strategy_registry import StrategySpec, registered_strategies

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


def validate_strategy_frame(
    frame: pd.DataFrame, *, league: str | None = None, market: str | None = None
) -> tuple[ArtifactIdentity | None, StrategySpec | None]:
    """Validate generic CSV identity; identity-absent non-adapter legacy frames return ``None``."""
    adapter_columns: set[str] = set()
    for registered in registered_strategies():
        if not registered.is_structural:
            continue
        adapter_columns.update(
            set(registered.required_columns) - set(registered.shared_csv_columns)
        )
        if registered.legacy_csv_identity_column:
            adapter_columns.add(registered.legacy_csv_identity_column)
    if MODEL_STRATEGY_CSV_COLUMN not in frame:
        generic_columns = {*_CSV_FIELDS.values(), SPLIT_FINGERPRINT_CSV_COLUMN}
        if any(column in frame for column in generic_columns):
            raise ValueError("partial generic model-strategy identity")
        if any(column in frame for column in adapter_columns):
            raise ValueError("adapter strategy columns require generic model-strategy identity")
        return None, None
    payload = {key: _constant(frame, column) for key, column in _CSV_FIELDS.items()}
    for key in ("implementation_version", "artifact_schema_version"):
        value = payload[key]
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"strategy artifact requires integral {key}") from exc
        if isinstance(value, bool) or float(value) != integer:
            raise ValueError(f"strategy artifact requires integral {key}")
        payload[key] = integer
    payload["split_fingerprint"] = (
        _constant(frame, SPLIT_FINGERPRINT_CSV_COLUMN)
        if SPLIT_FINGERPRINT_CSV_COLUMN in frame
        else None
    )
    identity = _identity_from_blob(payload)
    spec = _validate_core(identity, league or identity.league, market or identity.market)
    _validate_controls(identity, spec)
    if identity.status != "active":
        raise InactiveStrategyArtifactError(f"{spec.slug}: inactive strategy artifact")
    if spec.legacy_csv_identity_column:
        if str(_constant(frame, spec.legacy_csv_identity_column)) != spec.slug:
            raise ValueError(f"{spec.slug}: adapter CSV identity mismatch")
    missing = sorted(set(spec.required_columns) - set(frame))
    if missing:
        raise ValueError(f"{spec.slug}: artifact missing required columns {missing}")
    return identity, spec
