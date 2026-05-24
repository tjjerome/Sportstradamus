"""Per-cell ship config — which strategy each ``(league, market)`` trains with.

``data/config/stat_meta.json`` (committed) holds the canonical per-cell
record:

    {
        "NBA": {
            "PTS": {"dist": "SkewNormal", "shipped": "devel", "strategy": "ratio_meanyr"},
            "FG3M": {"dist": "ZINB", "shipped": "devel", "strategy": "none"},
            "FGA": {"dist": "SkewNormal", "shipped": "withheld", "strategy": "none"},
            ...
        },
        ...
    }

Three fields per cell drive shipping + training:

* ``dist`` — distribution family the cell trains with (``SkewNormal`` /
  ``ZINB`` / ``NegBin`` / ``Gamma`` / ``ZAGamma``). Determines which
  pipeline branch consumes the strategy slug.
* ``shipped`` — release surface:

  - ``"withheld"`` — never shipped; ``meditate`` skips training and prunes
    the production pickle so inference dark-outs the market.
  - ``"devel"`` — passed Gate 1; ships on the ``devel`` branch that the
    production server tracks. Has not yet graduated through Gate 2 to
    ``main``.
  - ``"main"`` — passed Gate 1 AND Gate 2; ships on both ``devel`` and
    ``main``.

* ``strategy`` — training-pipeline strategy:

  - A ``STRATEGY_SLUGS`` value (e.g. ``"ratio_meanyr"``,
    ``"centered_additive_eb_meanyr_k10"``) — used in the SkewNormal
    training branch to pick a target transform.
  - :data:`STRATEGY_NONE` (``"none"``) — no extra processing. Every
    count-branch cell uses this today; future strategies may add
    post-processing / calibration / etc. that count cells can opt into.

Shipping a cell that has cleared Gate 1 is a one-field edit to
``stat_meta.json``: set ``shipped`` from ``"withheld"`` to ``"devel"`` (or
``"main"`` after Gate 2 graduation). Inference never reads this file — it
decodes the strategy from the self-describing pickle — so training config
and inference cannot drift.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from pathlib import Path

from sportstradamus import data
from sportstradamus.training.baselines import STRATEGY_SLUGS

# Reserved strategy value: the cell does not opt into any pipeline strategy.
# Distinct from a real slug so future strategies don't have to overload it.
STRATEGY_NONE = "none"

# Reserved shipped value: cell is not shipped on any branch.
WITHHELD = "withheld"

# Allowed shipped values per training branch — a cell is "active" on branch
# ``b`` iff ``cell["shipped"] in _ALLOWED_FOR_BRANCH[b]``.
_ALLOWED_FOR_BRANCH: dict[str, frozenset[str]] = {
    "devel": frozenset({"devel", "main"}),
    "main": frozenset({"main"}),
}

# All recognized values for the ``shipped`` field.
_SHIPPED_VALUES: frozenset[str] = frozenset({WITHHELD, "devel", "main"})

STAT_META_PATH = pkg_resources.files(data) / "config" / "stat_meta.json"

# Nested {league: {market: strategy_or_withheld}} as returned by load_ship_config.
ShipConfig = dict[str, dict[str, str]]


def _load_stat_meta(path: Path) -> dict[str, dict[str, dict]]:
    if not path.exists():
        return {}
    with open(path) as infile:
        return json.load(infile)


def _validate_cell(league: str, market: str, cell: dict) -> None:
    """Raise ValueError if a stat_meta entry is internally inconsistent."""
    shipped = cell.get("shipped")
    strategy = cell.get("strategy")
    dist = cell.get("dist")
    if shipped not in _SHIPPED_VALUES:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown shipped "
            f"value {shipped!r}; valid: {sorted(_SHIPPED_VALUES)}"
        )
    if strategy != STRATEGY_NONE and strategy not in STRATEGY_SLUGS:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown strategy "
            f"value {strategy!r}; valid: {sorted(set(STRATEGY_SLUGS) | {STRATEGY_NONE})}"
        )
    if dist == "SkewNormal" and strategy == STRATEGY_NONE and shipped != WITHHELD:
        raise ValueError(
            f"stat_meta.json: SkewNormal cell {league}/{market} cannot ship "
            f"with strategy=none (SkewNormal requires a real strategy slug)"
        )
    if dist != "SkewNormal" and strategy != STRATEGY_NONE:
        raise ValueError(
            f"stat_meta.json: non-SkewNormal cell {league}/{market} (dist={dist!r}) "
            f"cannot carry strategy={strategy!r}; the slug only applies to the "
            f"SkewNormal branch. Use {STRATEGY_NONE!r}."
        )


def load_ship_config(branch: str = "devel", path: Path | None = None) -> ShipConfig:
    """Project ``stat_meta.json`` into the legacy ship-config shape.

    Args:
        branch: Training branch — ``"devel"`` (default, includes cells
            shipped on devel or main) or ``"main"`` (only cells shipped on
            main). Determines which ``shipped`` values are considered
            active; inactive cells appear as :data:`WITHHELD` in the
            returned map.
        path: Override for the stat_meta file path. Defaults to
            :data:`STAT_META_PATH`. Missing file yields an empty map.

    Returns:
        Nested ``{league: {market: strategy_or_withheld}}`` where the
        value is the cell's strategy slug (if active on ``branch``) or
        :data:`WITHHELD` (if not active). ``meditate`` consumes this via
        :func:`resolve_cell_strategy`.

    Raises:
        ValueError: If ``branch`` is not recognized, or any cell in
            ``stat_meta.json`` violates the schema invariants (see
            :func:`_validate_cell`).
    """
    if branch not in _ALLOWED_FOR_BRANCH:
        raise ValueError(f"Unknown branch {branch!r}; valid: {sorted(_ALLOWED_FOR_BRANCH)}")
    path = Path(str(STAT_META_PATH)) if path is None else Path(path)
    meta = _load_stat_meta(path)
    allowed = _ALLOWED_FOR_BRANCH[branch]
    config: ShipConfig = {}
    for league, markets in meta.items():
        cfg_markets: dict[str, str] = {}
        for market, cell in markets.items():
            _validate_cell(league, market, cell)
            shipped = cell["shipped"]
            if shipped in allowed:
                cfg_markets[market] = cell["strategy"]
            else:
                cfg_markets[market] = WITHHELD
        config[league] = cfg_markets
    return config


def resolve_cell_strategy(
    league: str,
    market: str,
    flag_strategy: str,
    config: ShipConfig,
) -> str:
    """Resolve the training strategy for one cell.

    The map is authoritative when it lists the cell; otherwise the run's
    ``--target-strategy`` flag value fills the gap, so an empty map
    reproduces today's behavior exactly.

    A returned value of :data:`STRATEGY_NONE` means "this cell does not
    opt into any pipeline strategy." :mod:`training.cli` substitutes the
    CLI's ``--target-strategy`` value at the call site, so the underlying
    training pipeline always receives a real slug (count-branch training
    ignores the slug anyway).

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market stem.
        flag_strategy: The run-wide ``--target-strategy`` value (fallback
            for cells absent from the map).
        config: A loaded :func:`load_ship_config` map.

    Returns:
        A strategy slug, :data:`STRATEGY_NONE`, or :data:`WITHHELD` (the
        caller prunes the pickle and skips training in the last case).
    """
    mapped = config.get(league, {}).get(market)
    return mapped if mapped is not None else flag_strategy
