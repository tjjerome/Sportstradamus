"""Public declarative registry for model-strategy research and confirmation.

Owns the strategy catalog (built from :mod:`sportstradamus.training.model_strategy.specs`),
the applicability/capability queries the sweep and confirm loops filter on, the canonical
spec/control fingerprints that make a board resumable, and the CLI-arg / persistence /
namespace builders that turn a selected corner into a ``meditate`` invocation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from sportstradamus.training.model_strategy.specs import (
    BUILTIN_SPEC_DATA,
    CONFIRM_EVIDENCE_CORNERS,
    INCUMBENT_CONTROL_DEFAULTS,
    MANDATORY_SWEEP_CORNERS,
)
from sportstradamus.training.role_specs import role_spec_for

BASE_STRUCTURAL_STRATEGY = "none"
BASE_ARTIFACT_SCHEMA_VERSION = 1

CAP_DETERMINISTIC_TRAIN = "deterministic_train"
CAP_FULL_HPO = "full_hpo"
CAP_SCORE = "score"
CAP_SERVE = "serve"
CAP_CONFIRM = "confirm"
# A family must also be confirmable to be worth a trial: the sweep's product is a recipe the
# nomination lane can retrain at full HPO and ship. Mixture ranks best on most boards and can do
# neither, so ranking it only spends budget on a corner that is skipped at nomination.
SWEEP_CAPABILITIES = frozenset({CAP_DETERMINISTIC_TRAIN, CAP_SCORE, CAP_CONFIRM})


@dataclass(frozen=True)
class CellContext:
    league: str
    market: str
    distribution: str
    distribution_class: str
    data_columns: frozenset[str] | None = None
    matrix_sha256: str | None = None
    target_is_integer: bool | None = None
    global_mean: float | None = None


@dataclass(frozen=True)
class Applicability:
    distribution_classes: tuple[str, ...]
    distributions: tuple[str, ...] = ()
    required_data_columns: tuple[str, ...] = ()
    role_registry_gated: bool = False
    requires_integer_target: bool = False
    max_global_mean: float | None = None

    def _target_admits(self, cell: CellContext) -> bool:
        """Whether the cell's target shape admits this family.

        Both gates fail OPEN on an absent fact, unlike ``required_data_columns``: they exist to
        prune the sweep's search space, whereas ``validate_strategy_selection`` runs them again on
        a corner an operator already forced and signed — from a context carrying no matrix facts —
        and must not start rejecting it.
        """
        if self.requires_integer_target and cell.target_is_integer is False:
            return False
        return (
            self.max_global_mean is None
            or cell.global_mean is None
            or cell.global_mean <= self.max_global_mean
        )

    def matches(self, cell: CellContext) -> bool:
        if self.distribution_classes and cell.distribution_class not in self.distribution_classes:
            return False
        if self.distributions and cell.distribution not in self.distributions:
            return False
        if not self._target_admits(cell):
            return False
        required = set(self.required_data_columns)
        if self.role_registry_gated:
            spec = role_spec_for(cell.league, cell.market)
            if spec is None:
                return False
            required |= set(spec.all_columns)
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
    structural: bool = False

    @property
    def is_structural(self) -> bool:
        """Whether this method reshapes the target/CDF earlier than the corrector stage.

        Load-bearing beyond selection: it gates the persisted ``structural_calibration``
        adapter blob, the split-fingerprint contract, the artifact namespace suffix, and
        the serve-side group-conditional-CDF dispatch. Structural methods are selected
        through the ``posthoc`` calibration pool, not a separate axis.
        """
        return self.structural

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


for _league, _market, _slug, _controls in (*MANDATORY_SWEEP_CORNERS, *CONFIRM_EVIDENCE_CORNERS):
    if _controls not in strategy_controls(get_strategy(_slug)):
        raise ValueError(f"{_slug}: seed corner for {_league} {_market} is not a registered corner")

for _slug, _defaults in INCUMBENT_CONTROL_DEFAULTS.items():
    _spec = get_strategy(_slug)
    _reachable = {name: set(values) for name, values in _spec.axes.items()}
    if any(
        value not in _reachable.get(name, {_spec.fixed_controls.get(name)})
        for name, value in _defaults.items()
    ):
        raise ValueError(f"{_slug}: incumbent control default is not a reachable control value")


def incumbent_controls(
    spec: StrategySpec, persisted: Mapping[str, object]
) -> dict[str, str] | None:
    """A cell's persisted recipe completed into a registered corner, or ``None`` if it isn't one.

    Resolution order is fixed controls, then explicit persisted values, then the strategy's
    historical effective defaults (:data:`specs.INCUMBENT_CONTROL_DEFAULTS`) for whatever the cell
    never wrote down. An explicit value is never coerced: a cell carrying one the current grid
    dropped yields no incumbent rather than a silently rewritten recipe.
    """
    controls = dict(spec.fixed_controls)
    controls.update(
        {
            control: str(persisted[field])
            for control, field in spec.persist.items()
            if persisted.get(field) is not None
        }
    )
    defaults = INCUMBENT_CONTROL_DEFAULTS.get(spec.slug, {})
    for name in (*spec.fixed_controls, *spec.axes):
        if name not in controls and name in defaults:
            controls[name] = defaults[name]
    return controls if controls in strategy_controls(spec) else None


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


def meditate_command(league: str, market: str, *args: str) -> list[str]:
    """Build the canonical cell-bound ``meditate`` subprocess command."""
    return [
        "poetry",
        "run",
        "meditate",
        "--league",
        league,
        "--market",
        market,
        *args,
    ]


def strategy_cli_args(
    cell: CellContext, spec: StrategySpec, controls: Mapping[str, str]
) -> list[str]:
    """Forward each control as its ``--flag value``; a structural method's slug rides ``--posthoc``.

    A structural spec pins its own slug in the ``posthoc`` control, so this loop emits
    ``--posthoc <slug>`` — the calibration pool is the selector, no separate axis.
    """
    del cell
    args: list[str] = []
    for name, value in controls.items():
        if name not in spec.cli_flags:
            raise ValueError(f"{spec.slug}: no CLI flag declared for control {name!r}")
        args.extend((spec.cli_flags[name], value))
    return args


def strategy_full_hpo_cli_args(
    cell: CellContext, spec: StrategySpec, controls: Mapping[str, str]
) -> list[str]:
    if not spec.enrolled_for(cell) or CAP_FULL_HPO not in spec.capabilities:
        raise ValueError(f"{spec.slug}: strategy cannot run full-HPO for this cell")
    args: list[str] = []
    for name, value in controls.items():
        if name in spec.persist:
            continue
        if name not in spec.cli_flags:
            raise ValueError(f"{spec.slug}: no CLI flag declared for control {name!r}")
        args.extend((spec.cli_flags[name], value))
    return args


def strategy_persistence_edits(
    cell: CellContext, spec: StrategySpec, controls: Mapping[str, str]
) -> dict[str, str]:
    del cell
    missing = set(spec.persist) - set(controls)
    if missing:
        raise ValueError(f"{spec.slug}: missing persisted controls {sorted(missing)}")
    edits = {field: controls[name] for name, field in spec.persist.items()}
    edits.update(spec.fixed_persist)
    return edits


def artifact_namespace(base_namespace: str, spec: StrategySpec) -> str:
    return f"{base_namespace}__{spec.slug}" if spec.artifact_namespace_suffix else base_namespace
