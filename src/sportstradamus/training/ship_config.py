"""Per-cell training **strategy** each ``(league, market)`` ships with.

A cell's *strategy* is the combination of its target-normalization choice, its
post-hoc calibration choice, and any future per-cell training knob.
``data/config/stat_meta.json`` (committed) holds the canonical per-cell record:

    {
        "NBA": {
            "PTS": {"dist": "SkewNormal", "shipped": "devel",
                    "target_normalization": "ratio_meanyr", "posthoc": "none"},
            "FG3M": {"dist": "ZINB", "shipped": "devel",
                     "target_normalization": "none", "posthoc": "none"},
            ...
        },
        ...
    }

Fields per cell drive shipping + training:

* ``dist`` — distribution family the cell trains with (``SkewNormal`` /
  ``ZINB`` / ``NegBin`` / ``Gamma`` / ``ZAGamma``). Determines which
  pipeline branch consumes the target-normalization slug.
* ``shipped`` — release surface:

  - ``"withheld"`` — never shipped; ``meditate`` skips training and prunes
    the production pickle so inference dark-outs the market.
  - ``"devel"`` — passed Gate 1; ships on the ``devel`` branch that the
    production server tracks. Has not yet graduated through Gate 2 to
    ``main``.
  - ``"main"`` — passed Gate 1 AND Gate 2; ships on both ``devel`` and
    ``main``.

* ``target_normalization`` — how the GBDT target is reshaped:

  - A :data:`TARGET_NORMALIZATION_SLUGS` value (e.g. ``"ratio_meanyr"``,
    ``"centered_additive_eb_meanyr_k10"``) — SkewNormal branch only.
  - :data:`TARGET_NORM_NONE` (``"none"``) — no target transform; every
    count-branch cell uses this.

* ``posthoc`` — post-hoc calibration applied after the distribution is formed,
  one of :data:`sportstradamus.training.posthoc.POSTHOC_SLUGS`. Orthogonal to
  ``target_normalization`` — any family may carry any ``posthoc``.

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
from sportstradamus.training.baselines import TARGET_NORMALIZATION_SLUGS
from sportstradamus.training.posthoc import POSTHOC_SLUGS

# Reserved target-normalization value: the cell applies no target transform.
# Distinct from a real slug so future normalizations don't have to overload it.
TARGET_NORM_NONE = "none"

# Reserved shipped value: cell is not shipped on any branch.
WITHHELD = "withheld"

# Distribution family name for the custom PyTorch SkewNormal; used in ship-gate
# logic to detect which cells need a real strategy slug vs. TARGET_NORM_NONE.
SKEW_NORMAL_DIST: str = "SkewNormal"

# Allowed shipped values per training branch — a cell is "active" on branch
# ``b`` iff ``cell["shipped"] in _ALLOWED_FOR_BRANCH[b]``.
_ALLOWED_FOR_BRANCH: dict[str, frozenset[str]] = {
    "devel": frozenset({"devel", "main"}),
    "main": frozenset({"main"}),
}

# All recognized values for the ``shipped`` field.
_SHIPPED_VALUES: frozenset[str] = frozenset({WITHHELD, "devel", "main"})

STAT_META_PATH = pkg_resources.files(data) / "config" / "stat_meta.json"

# Nested {league: {market: target_normalization_or_withheld}} as returned by load_ship_config.
ShipConfig = dict[str, dict[str, str]]


def load_stat_meta(path: Path) -> dict[str, dict[str, dict]]:
    if not path.exists():
        return {}
    with open(path) as infile:
        return json.load(infile)


def _validate_cell(league: str, market: str, cell: dict) -> None:
    """Raise ValueError if a stat_meta entry is internally inconsistent."""
    shipped = cell.get("shipped")
    target_norm = cell.get("target_normalization")
    posthoc = cell.get("posthoc", TARGET_NORM_NONE)
    dist = cell.get("dist")
    if shipped not in _SHIPPED_VALUES:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown shipped "
            f"value {shipped!r}; valid: {sorted(_SHIPPED_VALUES)}"
        )
    if target_norm != TARGET_NORM_NONE and target_norm not in TARGET_NORMALIZATION_SLUGS:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown target_normalization "
            f"value {target_norm!r}; valid: "
            f"{sorted(set(TARGET_NORMALIZATION_SLUGS) | {TARGET_NORM_NONE})}"
        )
    if posthoc not in POSTHOC_SLUGS:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown posthoc "
            f"value {posthoc!r}; valid: {sorted(POSTHOC_SLUGS)}"
        )
    if dist == SKEW_NORMAL_DIST and target_norm == TARGET_NORM_NONE and shipped != WITHHELD:
        raise ValueError(
            f"stat_meta.json: SkewNormal cell {league}/{market} cannot ship "
            f"with target_normalization=none (SkewNormal requires a real slug)"
        )
    if dist != SKEW_NORMAL_DIST and target_norm != TARGET_NORM_NONE:
        raise ValueError(
            f"stat_meta.json: non-SkewNormal cell {league}/{market} (dist={dist!r}) "
            f"cannot carry target_normalization={target_norm!r}; the slug only applies "
            f"to the SkewNormal branch. Use {TARGET_NORM_NONE!r}."
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
        Nested ``{league: {market: target_normalization_or_withheld}}`` where the
        value is the cell's target-normalization slug (if active on ``branch``) or
        :data:`WITHHELD` (if not active). ``meditate`` consumes this via
        :func:`resolve_cell_target_normalization`.

    Raises:
        ValueError: If ``branch`` is not recognized, or any cell in
            ``stat_meta.json`` violates the schema invariants (see
            :func:`_validate_cell`).
    """
    if branch not in _ALLOWED_FOR_BRANCH:
        raise ValueError(f"Unknown branch {branch!r}; valid: {sorted(_ALLOWED_FOR_BRANCH)}")
    path = Path(str(STAT_META_PATH)) if path is None else Path(path)
    meta = load_stat_meta(path)
    allowed = _ALLOWED_FOR_BRANCH[branch]
    config: ShipConfig = {}
    for league, markets in meta.items():
        cfg_markets: dict[str, str] = {}
        for market, cell in markets.items():
            _validate_cell(league, market, cell)
            shipped = cell["shipped"]
            if shipped in allowed:
                cfg_markets[market] = cell["target_normalization"]
            else:
                cfg_markets[market] = WITHHELD
        config[league] = cfg_markets
    return config


def resolve_cell_target_normalization(
    league: str,
    market: str,
    flag_strategy: str,
    config: ShipConfig,
) -> str:
    """Resolve the training strategy for one cell.

    The map is authoritative when it lists the cell; otherwise the run's
    ``--target-strategy`` flag value fills the gap, so an empty map
    reproduces today's behavior exactly.

    A returned value of :data:`TARGET_NORM_NONE` means "this cell does not
    opt into any pipeline strategy." :mod:`training.cli` substitutes the
    CLI's ``--target-strategy`` value at the call site, so the underlying
    training pipeline always receives a real slug (count-branch training
    ignores the slug anyway).

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market stem.
        flag_strategy: The run-wide ``--target-normalization`` value (fallback
            for cells absent from the map).
        config: A loaded :func:`load_ship_config` map.

    Returns:
        A strategy slug, :data:`TARGET_NORM_NONE`, or :data:`WITHHELD` (the
        caller prunes the pickle and skips training in the last case).
    """
    mapped = config.get(league, {}).get(market)
    return mapped if mapped is not None else flag_strategy
