"""Public declarative registry for model-strategy research and confirmation."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from sportstradamus.training.model_strategy_specs import BUILTIN_SPEC_DATA
from sportstradamus.training.role_specs import role_spec_for

BASE_STRUCTURAL_STRATEGY = "none"
AUTO_STRUCTURAL_STRATEGY = "auto"
BASE_ARTIFACT_SCHEMA_VERSION = 1

CAP_DETERMINISTIC_TRAIN = "deterministic_train"
CAP_FULL_HPO = "full_hpo"
CAP_SCORE = "score"
CAP_SERVE = "serve"
CAP_CONFIRM = "confirm"
SWEEP_CAPABILITIES = frozenset({CAP_DETERMINISTIC_TRAIN, CAP_SCORE})


@dataclass(frozen=True)
class CellContext:
    league: str
    market: str
    distribution: str
    distribution_class: str
    data_columns: frozenset[str] | None = None
    matrix_sha256: str | None = None


@dataclass(frozen=True)
class Applicability:
    distribution_classes: tuple[str, ...]
    distributions: tuple[str, ...] = ()
    required_data_columns: tuple[str, ...] = ()
    role_registry_gated: bool = False

    def matches(self, cell: CellContext) -> bool:
        if self.distribution_classes and cell.distribution_class not in self.distribution_classes:
            return False
        if self.distributions and cell.distribution not in self.distributions:
            return False
        required = set(self.required_data_columns)
        if self.role_registry_gated:
            spec = role_spec_for(cell.league, cell.market)
            if spec is None:
                return False
            required |= set(spec.all_columns)
        if required and cell.data_columns is None:
            return False
        return required <= (cell.data_columns or frozenset())


@dataclass(frozen=True)
class StrategySpec:
    slug: str
    implementation_version: int
    artifact_schema_version: int
    family: str
    applicability: Applicability
    enrollments: tuple[tuple[str, str], ...]
    capabilities: frozenset[str]
    axes: Mapping[str, tuple[str, ...]]
    fixed_controls: Mapping[str, str]
    cli_flags: Mapping[str, str]
    persist: Mapping[str, str]
    fixed_persist: Mapping[str, str]
    required_columns: tuple[str, ...]
    shared_csv_columns: tuple[str, ...]
    artifact_namespace_suffix: bool
    legacy_model_key: str | None
    legacy_csv_identity_column: str | None
    legacy_schema_field: str
    legacy_status_field: str
    split_fingerprint_path: tuple[str, ...]
    matrix_hash_path: tuple[str, ...]
    selector_key: str | None = None
    selector_version: int | None = None
    selector_cli_flag: str | None = None
    selector_meta_field: str | None = None
    selector_default_value: str | None = None
    selector_auto_value: str | None = None

    @property
    def is_structural(self) -> bool:
        return self.selector_key is not None

    @property
    def canonical_signature(self) -> str:
        return _signature(asdict(self))

    def enrolled_for(self, cell: CellContext) -> bool:
        enrolled = not self.enrollments or (cell.league, cell.market) in self.enrollments
        return enrolled and self.applicability.matches(cell)


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signature(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _build_spec(data: Mapping[str, object]) -> StrategySpec:
    values = dict(data)
    values["applicability"] = Applicability(**values["applicability"])
    values["capabilities"] = frozenset(values["capabilities"])
    selector = values.pop("selector")
    if selector:
        values.update({f"selector_{key}": value for key, value in selector.items()})
    return StrategySpec(**values)


_BUILTIN_SLUGS = [str(data["slug"]) for data in BUILTIN_SPEC_DATA]
if len(_BUILTIN_SLUGS) != len(set(_BUILTIN_SLUGS)):
    raise ValueError("duplicate built-in model-strategy slug")
_STRATEGIES = {spec.slug: spec for spec in map(_build_spec, BUILTIN_SPEC_DATA)}
_CONFIRM_DEPENDENCIES = frozenset({CAP_FULL_HPO, CAP_SCORE, CAP_SERVE})
for _spec in _STRATEGIES.values():
    if CAP_CONFIRM in _spec.capabilities and not _spec.capabilities >= _CONFIRM_DEPENDENCIES:
        raise ValueError(f"{_spec.slug}: confirm requires full_hpo, score, and serve capabilities")


def get_strategy(slug: str) -> StrategySpec:
    try:
        return _STRATEGIES[slug]
    except KeyError as exc:
        raise ValueError(f"unknown model strategy {slug!r}") from exc


def registered_strategies() -> tuple[StrategySpec, ...]:
    return tuple(_STRATEGIES.values())


def strategies_for_cell(
    cell: CellContext, *, required_capabilities: Iterable[str] = ()
) -> tuple[StrategySpec, ...]:
    required = frozenset(required_capabilities)
    return tuple(
        spec
        for spec in _STRATEGIES.values()
        if required <= spec.capabilities and spec.enrolled_for(cell)
    )


def validate_strategy_selection(
    cell: CellContext,
    spec: StrategySpec,
    controls: Mapping[str, str],
    *,
    required_capabilities: Iterable[str] = (),
) -> None:
    if spec not in strategies_for_cell(cell, required_capabilities=required_capabilities):
        raise ValueError(f"{spec.slug}: strategy is not applicable/capable for this cell")
    if dict(controls) not in strategy_controls(spec):
        raise ValueError(f"{spec.slug}: controls are not a registered strategy corner")


def strategy_controls(spec: StrategySpec) -> tuple[dict[str, str], ...]:
    if not spec.axes:
        return (dict(spec.fixed_controls),)
    names = tuple(spec.axes)
    return tuple(
        {**spec.fixed_controls, **dict(zip(names, values, strict=True))}
        for values in itertools.product(*(spec.axes[name] for name in names))
    )


def controls_json(controls: Mapping[str, object]) -> str:
    return canonical_json(dict(controls))


def parse_controls(payload: object) -> dict[str, str]:
    if not isinstance(payload, str) or not payload:
        raise ValueError("strategy controls_json is missing")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise ValueError("strategy controls_json must encode an object")
    if any(value is None or not isinstance(value, str) for value in decoded.values()):
        raise ValueError("strategy controls_json values must be nonmissing strings")
    return decoded


def corner_fingerprint(
    spec: StrategySpec,
    controls: Mapping[str, object],
    matrix_sha256: str,
) -> str:
    if not isinstance(matrix_sha256, str) or not matrix_sha256:
        raise ValueError("corner fingerprint requires a cached-matrix SHA256")
    return _signature(
        {
            "strategy_signature": spec.canonical_signature,
            "family": spec.family,
            "controls": dict(controls),
            "matrix_sha256": matrix_sha256,
        }
    )


def distribution_class(distribution: str) -> str:
    if distribution in {"SkewNormal", "Mixture"}:
        return "continuous"
    if distribution in {"ZINB", "NegBin", "DPO"}:
        return "count"
    raise ValueError(f"distribution {distribution!r} has no registered strategy class")
