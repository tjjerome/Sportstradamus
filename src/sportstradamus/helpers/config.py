"""Configuration + credential loading for the helpers package.

Every JSON file under ``src/sportstradamus/data/config/`` and the ``odds_api`` key
from ``src/sportstradamus/creds/keys.json`` is loaded eagerly at import
time. The resulting dicts are surfaced as module-level constants so callers
can ``from sportstradamus.helpers import stat_cv`` etc. without paying a
load cost per call site.

Per-cell training metadata is split across two files:

* ``stat_meta.json`` — **committed.** Per-cell ``{dist, shipped, target_normalization, posthoc}``:
  the semi-stable ML config humans curate (distribution family, release
  state, target-normalization slug, post-hoc corrector slug).
* ``stat_calibration.json`` — **gitignored.** Per-cell ``{cv, std, zi}``:
  the calibration values that meditate recomputes every run.

Legacy module-level constants (``stat_dist`` / ``stat_cv`` / ``stat_std`` /
``stat_zi``) are derived from the union of both files so existing imports
keep working unchanged.

The eager load is deliberate: a missing config file should raise at startup
rather than silently produce wrong answers deep inside a training run.
``stat_calibration.json`` is the one exception: it may be missing on a
fresh clone before the first meditate run, in which case calibration
fields default to safe values (cv=1.0, std=None, zi missing) that the
legacy ``.get(..., default)`` call sites already tolerate.
"""

import importlib.resources as pkg_resources
import json

from sportstradamus import creds, data

# --- API keys ---------------------------------------------------------------

with (pkg_resources.files(creds) / "keys.json").open() as infile:
    _keys = json.load(infile)
    odds_api = _keys["odds_api"]

# --- Per-league / per-market config dicts ----------------------------------

_config_dir = pkg_resources.files(data) / "config"

with (_config_dir / "abbreviations.json").open() as infile:
    abbreviations = json.load(infile)

with (_config_dir / "combo_props.json").open() as infile:
    combo_props = json.load(infile)

with (_config_dir / "stat_meta.json").open() as infile:
    _stat_meta_committed: dict[str, dict[str, dict]] = json.load(infile)

# stat_calibration.json is gitignored — a fresh clone may not have it. In
# that case start from an empty calibration map; meditate will bootstrap
# values on its first run, matching the legacy stat_cv/stat_std behavior
# (those files were already gitignored).
_cal_path = _config_dir / "stat_calibration.json"
if _cal_path.is_file():
    with _cal_path.open() as infile:
        _stat_cal: dict[str, dict[str, dict]] = json.load(infile)
else:
    _stat_cal = {}

# Public union of committed meta + runtime calibration. Each cell carries
# every field; callers that want a typed view can read the field directly.
stat_meta: dict[str, dict[str, dict]] = {}
for _league, _markets in _stat_meta_committed.items():
    stat_meta[_league] = {}
    for _market, _meta_cell in _markets.items():
        _cal_cell = _stat_cal.get(_league, {}).get(_market, {})
        stat_meta[_league][_market] = {
            "dist": _meta_cell["dist"],
            "shipped": _meta_cell["shipped"],
            "target_normalization": _meta_cell["target_normalization"],
            "posthoc": _meta_cell.get("posthoc", "none"),
            "cv": _cal_cell.get("cv", 1.0),
            "std": _cal_cell.get("std"),
            "zi": _cal_cell.get("zi"),
        }

# Derived per-field views of the merged stat_meta map, kept so existing
# ``from sportstradamus.helpers import stat_*`` import sites work unchanged.
stat_dist: dict[str, dict[str, str]] = {
    league: {market: cell["dist"] for market, cell in markets.items()}
    for league, markets in stat_meta.items()
}
stat_cv: dict[str, dict[str, float]] = {
    league: {market: cell["cv"] for market, cell in markets.items()}
    for league, markets in stat_meta.items()
}
stat_std: dict[str, dict[str, float | None]] = {
    league: {market: cell["std"] for market, cell in markets.items()}
    for league, markets in stat_meta.items()
}
stat_zi: dict[str, dict[str, float]] = {
    league: {market: cell["zi"] for market, cell in markets.items() if cell.get("zi") is not None}
    for league, markets in stat_meta.items()
}

with (_config_dir / "stat_map.json").open() as infile:
    stat_map = json.load(infile)

with (_config_dir / "book_weights.json").open() as infile:
    book_weights = json.load(infile)

with (_config_dir / "prop_books.json").open() as infile:
    books = json.load(infile)

with (_config_dir / "goalies.json").open() as infile:
    nhl_goalies = json.load(infile)

with (_config_dir / "feature_filter.json").open() as infile:
    feature_filter = json.load(infile)

with (_config_dir / "banned_combos.json").open() as infile:
    banned = json.load(infile)

# Banned combos ship with player-name keys joined by " & "; rewrite as
# frozensets so parlay-leg comparisons are order-independent.
for platform in banned:
    for league in list(banned[platform].keys()):
        banned[platform][league]["team"] = {
            frozenset(k.split(" & ")): v for k, v in banned[platform][league]["team"].items()
        }
        banned[platform][league]["opponent"] = {
            frozenset(k.split(" & ")): v for k, v in banned[platform][league]["opponent"].items()
        }

with (_config_dir / "name_map.json").open() as infile:
    name_map = json.load(infile)

# Underdog contest-variant payouts feed parlay scoring + display reconciliation
# in :mod:`sportstradamus.prediction.correlation`. Keys are bet sizes encoded
# as strings; rewrite to int keys so callers can index by ``bet_size``
# directly.
with (_config_dir / "underdog_payouts.json").open() as infile:
    _raw_underdog = json.load(infile)

underdog_payouts: dict[str, dict[int, float | list[float]]] = {}
for _variant, _table in _raw_underdog.items():
    if _variant.startswith("_"):
        continue
    underdog_payouts[_variant] = {int(k): v for k, v in _table.items()}
