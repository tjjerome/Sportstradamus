"""JSON I/O for the split per-cell config files.

Two physical files back the per-cell config:

* ``stat_meta.json`` (committed): per-cell ``{dist, shipped, strategy}``.
  Hand-curated; runtime updates from meditate (e.g. dist re-derivation)
  surface as git diffs for the human to commit.
* ``stat_calibration.json`` (gitignored): per-cell ``{cv, std, zi}``.
  Runtime-recomputed every meditate run.

The save helpers below mutate ONE field at a time so that a writer
updating ``zi`` does not clobber another writer's ``cv`` update. The
on-disk file is read fresh, the requested field is updated in place,
and the file is rewritten.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from pathlib import Path

from sportstradamus import data

_CONFIG_DIR = pkg_resources.files(data) / "config"
_STAT_META_PATH = _CONFIG_DIR / "stat_meta.json"
_STAT_CAL_PATH = _CONFIG_DIR / "stat_calibration.json"


def _load(path) -> dict:
    if not Path(str(path)).is_file():
        return {}
    with open(path) as f:
        return json.load(f)


def _write(path, data_dict: dict) -> None:
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)
        f.write("\n")


def load_distribution_config() -> dict:
    """Read ``{league: {market: dist}}`` derived from ``stat_meta.json``."""
    meta = _load(_STAT_META_PATH)
    return {lg: {m: cell["dist"] for m, cell in mkts.items()} for lg, mkts in meta.items()}


def load_shipped_config() -> dict:
    """Read ``{league: {market: shipped}}`` derived from ``stat_meta.json``.

    ``shipped`` is one of ``"withheld"`` / ``"devel"`` / ``"main"`` and drives
    the release surface for each cell. The training-stats parquet carries it
    so dashboards and downstream tooling can filter by shipping state without
    re-reading ``stat_meta.json``.
    """
    meta = _load(_STAT_META_PATH)
    return {
        lg: {m: cell.get("shipped", "withheld") for m, cell in mkts.items()}
        for lg, mkts in meta.items()
    }


def save_distribution_config(config: dict) -> None:
    """Persist a full ``{league: {market: dist}}`` map into ``stat_meta.json``.

    Overwrites the ``dist`` field on every (league, market) in ``config``;
    cells absent from ``config`` are untouched. Other fields on existing
    cells (``shipped``, ``strategy``) are preserved.
    """
    meta = _load(_STAT_META_PATH)
    for league, markets in config.items():
        for market, dist in markets.items():
            cell = meta.setdefault(league, {}).setdefault(
                market, {"dist": "Gamma", "shipped": "withheld", "strategy": "none"}
            )
            cell["dist"] = dist
    _write(_STAT_META_PATH, meta)


def load_zi_config() -> dict:
    """Read ``{league: {market: zi}}`` (non-null entries) from ``stat_calibration.json``."""
    cal = _load(_STAT_CAL_PATH)
    return {
        lg: {m: cell["zi"] for m, cell in mkts.items() if cell.get("zi") is not None}
        for lg, mkts in cal.items()
    }


def save_zi_config(config: dict) -> None:
    """Persist a full ``{league: {market: zi}}`` map into ``stat_calibration.json``.

    Mirrors :func:`save_distribution_config`'s contract: every cell in
    ``config`` has its ``zi`` field overwritten; other calibration fields
    (``cv``, ``std``) are preserved.
    """
    cal = _load(_STAT_CAL_PATH)
    for league, markets in config.items():
        for market, zi in markets.items():
            cell = cal.setdefault(league, {}).setdefault(
                market, {"cv": 1.0, "std": None, "zi": None}
            )
            cell["zi"] = zi
    _write(_STAT_CAL_PATH, cal)


def save_cv_std_config(cv_config: dict, std_config: dict) -> None:
    """Persist ``cv`` + ``std`` maps into ``stat_calibration.json``.

    Both maps share ``{league: {market: value}}`` shape and are written in
    one disk round-trip; the ``zi`` field on existing cells is preserved.
    """
    cal = _load(_STAT_CAL_PATH)
    for league, markets in cv_config.items():
        for market, cv in markets.items():
            cell = cal.setdefault(league, {}).setdefault(
                market, {"cv": 1.0, "std": None, "zi": None}
            )
            cell["cv"] = cv
    for league, markets in std_config.items():
        for market, std in markets.items():
            cell = cal.setdefault(league, {}).setdefault(
                market, {"cv": 1.0, "std": None, "zi": None}
            )
            cell["std"] = std
    _write(_STAT_CAL_PATH, cal)
