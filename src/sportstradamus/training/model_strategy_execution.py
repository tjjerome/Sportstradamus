"""CLI and persistence mapping for declarative model strategies."""

from __future__ import annotations

from collections.abc import Mapping

from sportstradamus.training.model_strategy_registry import (
    CAP_FULL_HPO,
    CellContext,
    StrategySpec,
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
