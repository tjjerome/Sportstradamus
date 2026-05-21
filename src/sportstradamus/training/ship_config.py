"""Per-cell ship config — which strategy each ``(league, market)`` trains with.

``data/ship_config.json`` is a git-tracked, nested ``{league: {market: strategy}}``
map (mirroring :data:`training.markets.ALL_MARKETS`) that drives ``meditate``'s
per-cell training decision and, with it, what is live on the production server
(which tracks ``devel``). Each cell is in one of three states:

- a real strategy slug (one of :data:`baselines.STRATEGY_SLUGS`) — **shipped**:
  train this cell with that strategy;
- :data:`WITHHELD` (``"withheld"``) — **under rework**: ``meditate`` skips training
  and prunes the cell's production pickle so inference dark-outs the market;
- **absent from the map** — **untouched**: train with the run's default strategy.
  An empty or missing file is therefore a strict no-op.

Shipping a cell that has cleared Gate 1 is a one-line edit to ``ship_config.json``
merged to ``devel`` (see docs/gbdt_mean_regression_plan.md, "Ship mechanism —
per-cell strategy config on devel"). Inference never reads this file — it decodes
the strategy from the self-describing pickle — so training config and inference
cannot drift.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from pathlib import Path

from sportstradamus import data
from sportstradamus.training.baselines import STRATEGY_SLUGS

# Reserved config value: skip training this cell and prune its production pickle.
WITHHELD = "withheld"

SHIP_CONFIG_PATH = pkg_resources.files(data) / "ship_config.json"

# Nested {league: {market: strategy}}, where strategy is a slug or WITHHELD.
ShipConfig = dict[str, dict[str, str]]


def load_ship_config(path: Path | None = None) -> ShipConfig:
    """Load and validate the per-cell ship config.

    Args:
        path: Config path; defaults to :data:`SHIP_CONFIG_PATH`. A missing file
            yields an empty map (no cell shipped or withheld — a strict no-op).

    Returns:
        Nested ``{league: {market: strategy}}`` map. Every value is a known
        strategy slug or :data:`WITHHELD`.

    Raises:
        ValueError: If any value is neither a known slug nor ``"withheld"``, so a
            typo or an unbuilt strategy fails at startup rather than mid-run.
    """
    path = Path(str(SHIP_CONFIG_PATH)) if path is None else Path(path)
    if not path.exists():
        return {}
    with open(path) as infile:
        config: ShipConfig = json.load(infile)
    valid = set(STRATEGY_SLUGS) | {WITHHELD}
    for league, markets in config.items():
        for market, strategy in markets.items():
            if strategy not in valid:
                raise ValueError(
                    f"ship_config.json: cell {league}/{market} has unknown "
                    f"strategy {strategy!r}; valid values: {sorted(valid)}"
                )
    return config


def resolve_cell_strategy(
    league: str,
    market: str,
    flag_strategy: str,
    config: ShipConfig,
) -> str:
    """Resolve the training strategy for one cell.

    The map is authoritative when it lists the cell; otherwise the run's
    ``--target-strategy`` flag value fills the gap, so an empty map reproduces
    today's behavior exactly.

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market stem.
        flag_strategy: The run-wide ``--target-strategy`` value (the fallback).
        config: A loaded :func:`load_ship_config` map.

    Returns:
        A strategy slug to train with, or :data:`WITHHELD` if the cell is marked
        for rework (the caller prunes the pickle and skips training).
    """
    mapped = config.get(league, {}).get(market)
    return mapped if mapped is not None else flag_strategy
