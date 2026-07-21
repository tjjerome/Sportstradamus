"""CLI and persistence mapping for declarative model strategies."""

from __future__ import annotations

from collections.abc import Mapping

from sportstradamus.training.model_strategy_registry import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_FULL_HPO,
    CellContext,
    StrategySpec,
    strategies_for_cell,
)


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


def _selector_contract(spec: StrategySpec) -> tuple[object, ...] | None:
    if spec.selector_key is None:
        return None
    return (
        spec.selector_key,
        spec.selector_version,
        spec.selector_cli_flag,
        spec.selector_meta_field,
        spec.selector_default_value,
        spec.selector_auto_value,
    )


def _selectors(cell: CellContext) -> dict[str, StrategySpec]:
    selectors: dict[str, StrategySpec] = {}
    for spec in strategies_for_cell(cell):
        if spec.selector_key is None:
            continue
        previous = selectors.setdefault(spec.selector_key, spec)
        if _selector_contract(previous) != _selector_contract(spec):
            raise ValueError(f"conflicting selector contract {spec.selector_key!r}")
    return selectors


def strategy_cli_args(
    cell: CellContext, spec: StrategySpec, controls: Mapping[str, str]
) -> list[str]:
    args: list[str] = []
    for selector in _selectors(cell).values():
        selected = (
            spec.slug
            if spec.selector_key == selector.selector_key
            else selector.selector_default_value
        )
        args.extend((str(selector.selector_cli_flag), str(selected)))
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
    args = [
        item
        for selector in _selectors(cell).values()
        for item in (str(selector.selector_cli_flag), str(selector.selector_auto_value))
    ]
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
    missing = set(spec.persist) - set(controls)
    if missing:
        raise ValueError(f"{spec.slug}: missing persisted controls {sorted(missing)}")
    edits = {field: controls[name] for name, field in spec.persist.items()}
    edits.update(spec.fixed_persist)
    edits["structural_strategy"] = spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY
    for selector in _selectors(cell).values():
        selected = (
            spec.slug
            if spec.selector_key == selector.selector_key
            else selector.selector_default_value
        )
        edits[str(selector.selector_meta_field)] = str(selected)
    return edits


def artifact_namespace(base_namespace: str, spec: StrategySpec) -> str:
    return f"{base_namespace}__{spec.slug}" if spec.artifact_namespace_suffix else base_namespace
