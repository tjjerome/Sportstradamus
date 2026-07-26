"""Versioned, serving-independent model dependencies for matrix reconstruction."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

from sportstradamus.helpers.io import market_file_slug

DEPENDENCY_SCHEMA_VERSION = 1
DEPENDENCY_NAMESPACE = "volume-v1"


@dataclass(frozen=True)
class ModelDependency:
    """Validated dependency artifact and the lineage fields that identify it."""

    path: Path
    payload: dict
    sha256: str


def _validate_namespace(namespace: str) -> str:
    """Return a safe single-directory dependency namespace."""
    if not namespace or Path(namespace).name != namespace or namespace in {".", ".."}:
        raise ValueError(f"invalid dependency namespace: {namespace!r}")
    return namespace


def dependency_path(
    root: Path,
    league: str,
    market: str,
    *,
    namespace: str = DEPENDENCY_NAMESPACE,
) -> Path:
    """Resolve a dependency path without consulting the prunable serving model root."""
    return root / _validate_namespace(namespace) / f"{market_file_slug(league, market)}.mdl"


def load_model_dependency(
    root: Path,
    league: str,
    market: str,
    *,
    namespace: str = DEPENDENCY_NAMESPACE,
) -> ModelDependency:
    """Load and validate one artifact from the versioned dependency namespace."""
    namespace = _validate_namespace(namespace)
    path = dependency_path(root, league, market, namespace=namespace)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing matrix dependency {league} {market}: {path}; "
            "serving models and old_models are not valid dependency registries"
        )
    raw = path.read_bytes()
    payload = pickle.loads(raw)
    identity = payload.get("dependency_identity")
    expected_identity = {
        "namespace": namespace,
        "schema_version": DEPENDENCY_SCHEMA_VERSION,
        "league": league,
        "market": market,
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError(f"invalid dependency identity for {league} {market}: {path}")
    if not identity.get("training_cutoff"):
        raise ValueError(f"dependency {league} {market} has no training cutoff")
    if not identity.get("matrix_sha256"):
        raise ValueError(f"dependency {league} {market} has no matrix SHA")
    if payload.get("distribution") != "SkewNormal":
        raise ValueError(f"dependency {league} {market} must be SkewNormal")
    columns = payload.get("expected_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"dependency {league} {market} has no feature schema")
    if not payload.get("model_version"):
        raise ValueError(f"dependency {league} {market} has no model version")
    if not payload.get("trained_at"):
        raise ValueError(f"dependency {league} {market} has no training timestamp")
    return ModelDependency(path, payload, hashlib.sha256(raw).hexdigest())
