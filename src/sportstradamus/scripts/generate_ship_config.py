#!/usr/bin/env python3
"""Generate an exhaustive, gate-driven ship_config.json for one branch.

Reads the canonical ``gate1_decisions.json`` (which cells passed Gate 1 + their
strategy) and writes ``ship_config.json`` over **all** ``ALL_MARKETS`` cells:
active cells get their decisions strategy, every other cell gets ``"withheld"``.
This is default-deny serving control — only gate-passing cells keep a pickle and
are served by prophecize.

Active set by branch:

* ``devel`` — every cell in the decisions file (Gate-1 passers).
* ``main`` — decisions cells that are also live-``graduated`` (Gate 2), per
  ``training.graduation``. Dormant (empty) until live metrics exist.

Usage
-----
    poetry run generate-ship-config --branch devel
    poetry run generate-ship-config --branch main --dry-run
    poetry run generate-ship-config --branch devel --prune
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from sportstradamus.helpers.io import (
    LIVE_METRICS_PATH,
    MODEL_STATS_PATH,
    prune_model_pickle,
)
from sportstradamus.training.baselines import STRATEGY_SLUGS
from sportstradamus.training.graduation import graduated_cells
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.ship_config import (
    GATE1_DECISIONS_PATH,
    SHIP_CONFIG_PATH,
    WITHHELD,
    ShipConfig,
)


def load_decisions(path: Path) -> ShipConfig:
    """Load and validate ``gate1_decisions.json``.

    Args:
        path: Path to the decisions JSON.

    Returns:
        Nested ``{league: {market: strategy}}`` map.

    Raises:
        ValueError: If any value is not a known strategy slug. The decisions
            file records real strategies only — ``"withheld"`` is a generated
            ship_config value, never a decision.
    """
    with open(path) as fh:
        decisions: ShipConfig = json.load(fh)
    for league, markets in decisions.items():
        for market, strategy in markets.items():
            if strategy not in STRATEGY_SLUGS:
                raise ValueError(
                    f"gate1_decisions.json: {league}/{market} has non-strategy "
                    f"value {strategy!r}; valid: {STRATEGY_SLUGS}"
                )
    return decisions


def active_cells(
    branch: str,
    decisions: ShipConfig,
    model_stats_path: Path,
    live_metrics_path: Path,
) -> set[tuple[str, str]]:
    """Return the set of cells that serve on ``branch``.

    Args:
        branch: ``"devel"`` (all decisions) or ``"main"`` (decisions that are
            also live-graduated).
        decisions: Loaded decisions map.
        model_stats_path: Gate-1 parquet path (only read for ``main``).
        live_metrics_path: Gate-2 parquet path (only read for ``main``).

    Returns:
        Set of active ``(league, market)`` tuples.

    Raises:
        ValueError: If ``branch`` is neither ``"devel"`` nor ``"main"``.
    """
    decision_cells = {
        (league, market) for league, markets in decisions.items() for market in markets
    }
    if branch == "devel":
        return decision_cells
    if branch == "main":
        return decision_cells & graduated_cells(model_stats_path, live_metrics_path)
    raise ValueError(f"unknown branch {branch!r}; expected 'devel' or 'main'")


def build_ship_config(decisions: ShipConfig, active: set[tuple[str, str]]) -> ShipConfig:
    """Build an exhaustive ship_config over ``ALL_MARKETS``.

    Active cells get their decisions strategy; every other ``ALL_MARKETS`` cell
    gets ``"withheld"``. Output is deterministic (leagues and markets sorted).

    Args:
        decisions: Loaded decisions map (its strategies fill the active cells).
        active: The set of active ``(league, market)`` tuples.

    Returns:
        Nested ``{league: {market: strategy-or-withheld}}`` over all 96 cells.

    Raises:
        ValueError: If a decisions cell is not in ``ALL_MARKETS`` (typo guard).
    """
    for league, markets in decisions.items():
        for market in markets:
            if league not in ALL_MARKETS or market not in ALL_MARKETS[league]:
                raise ValueError(f"decisions cell {league}/{market} not in ALL_MARKETS")
    config: ShipConfig = {}
    for league in sorted(ALL_MARKETS):
        cell: dict[str, str] = {}
        for market in sorted(ALL_MARKETS[league]):
            if (league, market) in active:
                cell[market] = decisions[league][market]
            else:
                cell[market] = WITHHELD
        config[league] = cell
    return config


@click.command()
@click.option(
    "--branch",
    type=click.Choice(["devel", "main"]),
    required=True,
    help="Which branch's gate to apply: 'devel' = Gate 1 passers, 'main' = graduated.",
)
@click.option(
    "--prune/--no-prune",
    default=False,
    help="Also delete every non-active cell's model pickle (immediate dark-out on this machine).",
)
@click.option(
    "--decisions",
    type=click.Path(path_type=Path),
    default=None,
    help="Decisions JSON path (defaults to data/gate1_decisions.json).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output ship_config.json path (defaults to data/ship_config.json).",
)
@click.option(
    "--model-stats",
    type=click.Path(path_type=Path),
    default=None,
    help="Gate 1 parquet path (defaults to data/model_stats.parquet). Only read for --branch main.",
)
@click.option(
    "--live-metrics",
    type=click.Path(path_type=Path),
    default=None,
    help="Gate 2 parquet path (defaults to data/live_metrics_per_market.parquet). main only.",
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print the config; do not write or prune."
)
def main(branch, prune, decisions, out, model_stats, live_metrics, dry_run) -> None:
    """Write an exhaustive, gate-driven ship_config.json for one branch."""
    decisions_path = Path(decisions) if decisions else Path(str(GATE1_DECISIONS_PATH))
    out_path = Path(out) if out else Path(str(SHIP_CONFIG_PATH))
    model_stats_path = Path(model_stats) if model_stats else Path(str(MODEL_STATS_PATH))
    live_metrics_path = Path(live_metrics) if live_metrics else Path(str(LIVE_METRICS_PATH))

    decisions_map = load_decisions(decisions_path)
    active = active_cells(branch, decisions_map, model_stats_path, live_metrics_path)
    config = build_ship_config(decisions_map, active)

    n_active = sum(1 for lg in config for mk in config[lg] if config[lg][mk] != WITHHELD)
    n_withheld = sum(1 for lg in config for mk in config[lg] if config[lg][mk] == WITHHELD)
    payload = json.dumps(config, indent=4, sort_keys=True)

    if dry_run:
        click.echo(payload)
        click.echo(
            f"# branch={branch} active={n_active} withheld={n_withheld} (dry-run, not written)"
        )
        return

    out_path.write_text(payload + "\n")
    click.echo(f"wrote {out_path}: active={n_active} withheld={n_withheld} (branch={branch})")

    if prune:
        pruned = 0
        for league in ALL_MARKETS:
            for market in ALL_MARKETS[league]:
                if (league, market) not in active and prune_model_pickle(league, market):
                    pruned += 1
        click.echo(f"pruned {pruned} non-active pickles")


if __name__ == "__main__":
    main()
