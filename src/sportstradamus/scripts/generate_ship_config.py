#!/usr/bin/env python3
"""Promote / demote cells in ``stat_meta.json`` based on gate state.

``stat_meta.json`` (committed) holds the canonical per-cell shipping
record. Humans edit ``shipped`` from ``"withheld"`` to ``"devel"`` when a
cell clears Gate 1. This CLI handles the ``main`` promotion mechanically:
when Gate-2 graduation runs, cells that pass move ``shipped: "devel"`` →
``shipped: "main"`` and cells that fail demote back.

Usage
-----
    poetry run generate-ship-config --branch main          # mutate stat_meta in place
    poetry run generate-ship-config --branch main --dry-run
    poetry run generate-ship-config --branch devel         # validate + summarize (no mutation)
    poetry run generate-ship-config --branch devel --prune
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import tabulate

from sportstradamus.helpers.io import (
    LIVE_METRICS_PATH,
    MODEL_STATS_PATH,
    prune_model_pickle,
)
from sportstradamus.training.graduation import (
    free_passer_cells,
    graduated_cells,
    served_cells_failing_ship,
)
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.ship_config import (
    STAT_META_PATH,
    WITHHELD,
    load_ship_config,
)


def _load_meta(path: Path) -> dict[str, dict[str, dict]]:
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _write_meta(path: Path, meta: dict) -> None:
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=4)
        fh.write("\n")


def _print_changes(changes: list[tuple[str, str, str, str]], n_cells: int) -> None:
    """Render the promotion/demotion change log as a table plus a one-line tally.

    Table stays plain so ``tabulate`` keeps its columns aligned; only the tally
    is colored (green when anything promoted, per check_graduation's state map).
    """
    if changes:
        rows = [
            [f"{league} {market}", old, new, "promoted" if new == "main" else "demoted"]
            for league, market, old, new in changes
        ]
        click.echo(
            tabulate.tabulate(rows, headers=["cell", "from", "to", "reason"], tablefmt="github")
        )
    n_promoted = sum(1 for *_, new in changes if new == "main")
    n_demoted = len(changes) - n_promoted
    tally = f"{n_promoted} promoted, {n_demoted} demoted, {n_cells - len(changes)} unchanged"
    click.secho(tally, fg="green" if n_promoted else "yellow")


def promote_for_main(
    meta: dict[str, dict[str, dict]],
    graduated: set[tuple[str, str]],
) -> tuple[dict[str, dict[str, dict]], list[tuple[str, str, str, str]]]:
    """Apply Gate-2 graduation to a stat_meta map.

    Cells in ``graduated`` move ``shipped: "devel"`` → ``shipped: "main"``.
    Cells currently ``shipped: "main"`` that fall out of ``graduated``
    demote back to ``shipped: "devel"``. ``"withheld"`` cells are
    untouched (they have not passed Gate 1).

    Args:
        meta: Loaded stat_meta map (mutated in-place + returned).
        graduated: Set of ``(league, market)`` tuples currently passing
            Gate 2 per ``training.graduation``.

    Returns:
        ``(meta, changes)`` — the mutated map and the per-cell change log
        as ``[(league, market, old_shipped, new_shipped), ...]``.
    """
    changes: list[tuple[str, str, str, str]] = []
    for league, markets in meta.items():
        for market, cell in markets.items():
            current = cell.get("shipped")
            new = current
            if (league, market) in graduated and current == "devel":
                new = "main"
            elif (league, market) not in graduated and current == "main":
                new = "devel"
            if new != current:
                cell["shipped"] = new
                changes.append((league, market, current, new))
    return meta, changes


@click.command()
@click.option(
    "--branch",
    type=click.Choice(["devel", "main"]),
    required=True,
    help=(
        "Gate to apply. 'main' promotes graduated cells (shipped:devel→main) "
        "and demotes failed cells (shipped:main→devel) in stat_meta.json. "
        "'devel' validates the current stat_meta + prints a summary; the "
        "human curates devel promotions by hand."
    ),
)
@click.option(
    "--prune/--no-prune",
    default=False,
    help=(
        "Delete every withheld cell's model pickle (immediate dark-out on "
        "this machine). Honors the resolved branch policy."
    ),
)
@click.option(
    "--meta",
    type=click.Path(path_type=Path),
    default=None,
    help="stat_meta.json path (defaults to data/config/stat_meta.json).",
)
@click.option(
    "--model-stats",
    type=click.Path(path_type=Path),
    default=None,
    help="Gate 1 parquet path (defaults to data/model_stats.parquet). Read for both branches.",
)
@click.option(
    "--live-metrics",
    type=click.Path(path_type=Path),
    default=None,
    help="Gate 2 parquet path (defaults to data/live_metrics_per_market.parquet). main only.",
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print the changes; do not mutate or prune."
)
def main(branch, prune, meta, model_stats, live_metrics, dry_run) -> None:
    """Promote / demote stat_meta entries based on gate state."""
    meta_path = Path(meta) if meta else Path(str(STAT_META_PATH))
    model_stats_path = Path(model_stats) if model_stats else Path(str(MODEL_STATS_PATH))
    live_metrics_path = Path(live_metrics) if live_metrics else Path(str(LIVE_METRICS_PATH))

    meta_map = _load_meta(meta_path)

    if branch == "main":
        graduated = graduated_cells(model_stats_path, live_metrics_path)
        meta_map, changes = promote_for_main(meta_map, graduated)
        n_cells = sum(len(markets) for markets in meta_map.values())
        click.secho(f"# branch=main (graduated={len(graduated)} cells)", bold=True)
        _print_changes(changes, n_cells)
        if not dry_run and changes:
            _write_meta(meta_path, meta_map)
            click.echo(f"wrote {meta_path}")
        elif dry_run:
            click.echo("# (dry-run, not written)")
    else:
        config = load_ship_config(branch="devel", path=meta_path)
        n_active = sum(1 for lg in config for mk in config[lg] if config[lg][mk] != WITHHELD)
        n_withheld = sum(1 for lg in config for mk in config[lg] if config[lg][mk] == WITHHELD)
        click.echo(
            f"# branch=devel: active={n_active} withheld={n_withheld} "
            f"(stat_meta.json validated; no mutation)"
        )
        # Report-only, like the free-passer sweep below: the weekly meditate warns on the
        # same cells and the human owns the cull decision (flip shipped=withheld, or --prune).
        failing = served_cells_failing_ship(meta_map, model_stats_path)
        if failing:
            for lg, mk in failing:
                click.echo(
                    f"  WARN {lg}/{mk}: shipped on devel but ship=False (offline gates fail)"
                )
            click.secho(
                f"# {len(failing)} served cell(s) fail the ship gate — cull by flipping "
                "shipped=withheld in stat_meta.json (add --prune for immediate dark-out)",
                fg="yellow",
            )

    # Free-passer sweep (report-only): cells the offline gate now passes that are
    # still withheld. Surfaced for a manual flip after a scorecard re-confirm; never
    # auto-flipped (a sweep pass disagreeing with the scorecard is a scorer bug).
    free_passers = free_passer_cells(meta_map, model_stats_path)
    for lg, mk in free_passers:
        click.echo(
            f"  FREE-PASSER {lg}/{mk}: ship=True but shipped=withheld "
            "(re-confirm on official scorecard, then flip to devel)"
        )
    if free_passers:
        click.echo(
            f"# {len(free_passers)} free-passer cell(s) — manual flip after scorecard re-confirm"
        )

    if prune:
        # Use the resolved (post-mutation) config to decide what to prune.
        config = load_ship_config(branch=branch, path=meta_path)
        pruned = 0
        for league in ALL_MARKETS:
            for market in ALL_MARKETS[league]:
                cell = config.get(league, {}).get(market, WITHHELD)
                if cell == WITHHELD and prune_model_pickle(league, market):
                    pruned += 1
        click.echo(f"pruned {pruned} withheld pickles")


if __name__ == "__main__":
    main()
