"""Per-cell ship config — which strategy each ``(league, market)`` trains with.

``data/ship_config.json`` is a git-tracked, nested ``{league: {market: strategy}}``
map (mirroring :data:`training.markets.ALL_MARKETS`) that drives ``meditate``'s
per-cell training decision and, with it, what is live on the production server
(which tracks ``devel``). Each cell is in one of three states:

- a real strategy slug (one of :data:`baselines.STRATEGY_SLUGS`) — **shipped**:
  train this cell with that strategy under the run's default ZINB architecture;
- an object ``{"strategy": <slug>, "zinb_mode": <mode>}`` — **shipped with an
  explicit ZINB architecture**: pins the cell's :data:`baselines.ZINB_MODES`
  (``"joint"`` / ``"hurdle"``) independent of the run's ``--zinb-mode`` flag — the
  only way to express e.g. a count market that ships as ``ratio_meanyr`` under the
  hurdle architecture. ``zinb_mode`` is optional and falls back to the flag;
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
from sportstradamus.training.baselines import STRATEGY_SLUGS, ZINB_MODES

# Reserved config value: skip training this cell and prune its production pickle.
WITHHELD = "withheld"

SHIP_CONFIG_PATH = pkg_resources.files(data) / "ship_config.json"

# Each cell is a bare strategy string (slug or WITHHELD) or a strategy object
# {"strategy": slug, "zinb_mode": mode} that also pins the ZINB architecture.
ShipCell = str | dict[str, str]
ShipConfig = dict[str, dict[str, ShipCell]]

# Strategy values a cell may name in either form. Computed once at import.
_VALID_STRATEGIES = frozenset(STRATEGY_SLUGS) | {WITHHELD}


def _validate_cell(league: str, market: str, cell: ShipCell) -> None:
    """Raise ``ValueError`` unless ``cell`` is a valid bare slug or strategy object.

    A bare string must be a known strategy or :data:`WITHHELD`. An object must
    carry a known ``"strategy"`` and may carry a ``"zinb_mode"`` from
    :data:`baselines.ZINB_MODES`; any other key is a typo and fails fast.
    """
    if isinstance(cell, str):
        if cell not in _VALID_STRATEGIES:
            raise ValueError(
                f"ship_config.json: cell {league}/{market} has unknown strategy "
                f"{cell!r}; valid values: {sorted(_VALID_STRATEGIES)}"
            )
        return
    if isinstance(cell, dict):
        strategy = cell.get("strategy")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"ship_config.json: cell {league}/{market} object has unknown or "
                f"missing strategy {strategy!r}; valid values: {sorted(_VALID_STRATEGIES)}"
            )
        mode = cell.get("zinb_mode")
        if mode is not None and mode not in ZINB_MODES:
            raise ValueError(
                f"ship_config.json: cell {league}/{market} has unknown zinb_mode "
                f"{mode!r}; valid values: {sorted(ZINB_MODES)}"
            )
        extra = set(cell) - {"strategy", "zinb_mode"}
        if extra:
            raise ValueError(
                f"ship_config.json: cell {league}/{market} object has unexpected "
                f"key(s) {sorted(extra)}; allowed: ['strategy', 'zinb_mode']"
            )
        return
    raise ValueError(
        f"ship_config.json: cell {league}/{market} must be a strategy string or a "
        f"{{'strategy', 'zinb_mode'}} object, got {type(cell).__name__}"
    )


def load_ship_config(path: Path | None = None) -> ShipConfig:
    """Load and validate the per-cell ship config.

    Args:
        path: Config path; defaults to :data:`SHIP_CONFIG_PATH`. A missing file
            yields an empty map (no cell shipped or withheld — a strict no-op).

    Returns:
        Nested ``{league: {market: cell}}`` map, as authored (no normalizing).
        Each cell is a known strategy slug, :data:`WITHHELD`, or a validated
        ``{"strategy", "zinb_mode"}`` object.

    Raises:
        ValueError: If any cell is not a valid slug, ``"withheld"``, or strategy
            object, so a typo or an unbuilt strategy fails at startup, not mid-run.
    """
    path = Path(str(SHIP_CONFIG_PATH)) if path is None else Path(path)
    if not path.exists():
        return {}
    with open(path) as infile:
        config: ShipConfig = json.load(infile)
    for league, markets in config.items():
        for market, cell in markets.items():
            _validate_cell(league, market, cell)
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
    cell = config.get(league, {}).get(market)
    if cell is None:
        return flag_strategy
    return cell["strategy"] if isinstance(cell, dict) else cell


def resolve_cell_zinb_mode(
    league: str,
    market: str,
    flag_zinb_mode: str,
    config: ShipConfig,
) -> str:
    """Resolve the ZINB architecture (``joint`` / ``hurdle``) for one cell.

    Only an object-form cell pins ``zinb_mode``; a bare-string cell or an absent
    cell falls back to the run's ``--zinb-mode`` flag, so an empty map (and every
    existing string-valued cell) reproduces today's behavior exactly.

    Args:
        league: League code (e.g. ``"NFL"``).
        market: Market stem.
        flag_zinb_mode: The run-wide ``--zinb-mode`` value (the fallback).
        config: A loaded :func:`load_ship_config` map.

    Returns:
        One of :data:`baselines.ZINB_MODES`.
    """
    cell = config.get(league, {}).get(market)
    if isinstance(cell, dict):
        return cell.get("zinb_mode", flag_zinb_mode)
    return flag_zinb_mode
