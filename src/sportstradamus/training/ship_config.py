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
from sportstradamus.training.calibration import BLENDING_SLUGS, DEFAULT_BLENDING
from sportstradamus.training.posthoc import POSTHOC_SLUGS

# Reserved target-normalization value: the cell applies no target transform.
# Distinct from a real slug so future normalizations don't have to overload it.
TARGET_NORM_NONE = "none"

# Reserved shipped value: cell is not shipped on any branch.
WITHHELD = "withheld"

# Sentinel meaning "honor each cell's stat_meta value; don't override."
# Same literal as training.pipeline.LOSS_AUTO — duplicated, not imported:
# pipeline.py already imports TARGET_NORM_NONE from this module, so importing
# LOSS_AUTO back from pipeline would create an import cycle.
_AUTO_SENTINEL = "auto"

# Historic CLI default, applied only when a cell has no stat_meta entry to
# honor (an empty --deterministic config, or a market missing from
# stat_meta.json) and --target-normalization was left on _AUTO_SENTINEL.
_UNMAPPED_TARGET_NORMALIZATION_FALLBACK = "ratio_meanyr"

# Distribution family name for the custom PyTorch SkewNormal; a CONTINUOUS_DISTS member.
SKEW_NORMAL_DIST: str = "SkewNormal"

# Continuous families train on a normalized target, so their cells carry a real
# target_normalization slug; every other family requires TARGET_NORM_NONE.
CONTINUOUS_DISTS: frozenset[str] = frozenset({SKEW_NORMAL_DIST, "Mixture"})

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
    _valid_target_norms = set(TARGET_NORMALIZATION_SLUGS) | {TARGET_NORM_NONE}
    if target_norm not in _valid_target_norms:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown target_normalization "
            f"value {target_norm!r}; valid: {sorted(_valid_target_norms)}"
        )
    if posthoc not in POSTHOC_SLUGS:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown posthoc "
            f"value {posthoc!r}; valid: {sorted(POSTHOC_SLUGS)}"
        )
    blending = cell.get("blending", DEFAULT_BLENDING)
    if blending not in BLENDING_SLUGS:
        raise ValueError(
            f"stat_meta.json: cell {league}/{market} has unknown blending "
            f"value {blending!r}; valid: {sorted(BLENDING_SLUGS)}"
        )
    if dist in CONTINUOUS_DISTS and target_norm == TARGET_NORM_NONE and shipped != WITHHELD:
        raise ValueError(
            f"stat_meta.json: continuous cell {league}/{market} (dist={dist!r}) cannot "
            f"ship with target_normalization=none (the continuous branch requires a real slug)"
        )
    if dist not in CONTINUOUS_DISTS and target_norm != TARGET_NORM_NONE:
        raise ValueError(
            f"stat_meta.json: non-continuous cell {league}/{market} (dist={dist!r}) "
            f"cannot carry target_normalization={target_norm!r}; the slug only applies "
            f"to the continuous branch. Use {TARGET_NORM_NONE!r}."
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


def resolve_flag_target_normalization(flag_strategy: str) -> str:
    """Materialize ``flag_strategy`` to a concrete slug with no cell to defer to.

    Passes an explicit slug through unchanged; substitutes the historic
    production default when left on the auto sentinel. Used both by
    :func:`resolve_cell_target_normalization`'s unmapped-cell case and by
    ``training.cli.meditate``'s count-branch / bypass-withholding
    substitutions, which need the same "make it concrete" step.
    """
    return (
        flag_strategy
        if flag_strategy != _AUTO_SENTINEL
        else _UNMAPPED_TARGET_NORMALIZATION_FALLBACK
    )


def resolve_cell_target_normalization(
    league: str,
    market: str,
    flag_strategy: str,
    config: ShipConfig,
) -> str:
    """Resolve the training strategy for one cell.

    ``flag_strategy`` of ``"auto"`` (the CLI default) honors the cell's
    stat_meta ``target_normalization`` when the map lists one — the
    production cron path. An explicit slug overrides that cell's
    normalization outright, even in a real (non-``--deterministic``) run —
    mirrors ``training.cli._resolve_cell_knob``'s auto-sentinel pattern for
    the other search-axis knobs (blending/hpo_selection/etc). A cell absent
    from the map (an empty ``--deterministic`` config, or a market genuinely
    missing from ``stat_meta.json``) has no per-cell opinion to honor or
    override, so ``flag_strategy`` is materialized via
    :func:`resolve_flag_target_normalization` instead. :data:`WITHHELD` and
    :data:`TARGET_NORM_NONE` always pass through unchanged; the caller
    (``training.cli.meditate``) owns bypass and count-branch substitution.

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market stem.
        flag_strategy: The run-wide ``--target-normalization`` value.
        config: A loaded :func:`load_ship_config` map.

    Returns:
        A strategy slug, :data:`TARGET_NORM_NONE`, or :data:`WITHHELD` (the
        caller prunes the pickle and skips training in the last case).
    """
    mapped = config.get(league, {}).get(market)
    if mapped is None:
        return resolve_flag_target_normalization(flag_strategy)
    if mapped in (WITHHELD, TARGET_NORM_NONE):
        return mapped
    return mapped if flag_strategy == _AUTO_SENTINEL else flag_strategy
