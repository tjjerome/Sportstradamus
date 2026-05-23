"""Configuration + credential loading for the helpers package.

Every JSON file under ``src/sportstradamus/data/config/`` and the ``odds_api`` key
from ``src/sportstradamus/creds/keys.json`` is loaded eagerly at import
time. The resulting dicts are surfaced as module-level constants so callers
can ``from sportstradamus.helpers import stat_cv`` etc. without paying a
load cost per call site.

The eager load is deliberate: a missing config file should raise at startup
rather than silently produce wrong answers deep inside a training run.
"""

import importlib.resources as pkg_resources
import json
import os

from sportstradamus import creds, data

# --- API keys ---------------------------------------------------------------

with open(pkg_resources.files(creds) / "keys.json") as infile:
    _keys = json.load(infile)
    odds_api = _keys["odds_api"]

# --- Per-league / per-market config dicts ----------------------------------

_config_dir = pkg_resources.files(data) / "config"

with open(_config_dir / "abbreviations.json") as infile:
    abbreviations = json.load(infile)

with open(_config_dir / "combo_props.json") as infile:
    combo_props = json.load(infile)

with open(_config_dir / "stat_cv.json") as infile:
    stat_cv = json.load(infile)

with open(_config_dir / "stat_dist.json") as infile:
    stat_dist = json.load(infile)

with open(_config_dir / "stat_std.json") as infile:
    stat_std = json.load(infile)

# stat_zi is optional: a league that hasn't estimated zero-inflation yet
# will fall through to a {} default and the downstream callers treat
# "missing gate" as "no zero-inflation".
_zi_path = _config_dir / "stat_zi.json"
if os.path.isfile(_zi_path):
    with open(_zi_path) as infile:
        stat_zi = json.load(infile)
else:
    stat_zi = {}

with open(_config_dir / "stat_map.json") as infile:
    stat_map = json.load(infile)

with open(_config_dir / "book_weights.json") as infile:
    book_weights = json.load(infile)

with open(_config_dir / "prop_books.json") as infile:
    books = json.load(infile)

with open(_config_dir / "goalies.json") as infile:
    nhl_goalies = json.load(infile)

with open(_config_dir / "feature_filter.json") as infile:
    feature_filter = json.load(infile)

with open(_config_dir / "banned_combos.json") as infile:
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

with open(_config_dir / "name_map.json") as infile:
    name_map = json.load(infile)

# Underdog contest-variant payouts feed parlay scoring + display reconciliation
# in :mod:`sportstradamus.prediction.correlation`. Keys are bet sizes encoded
# as strings; rewrite to int keys so callers can index by ``bet_size``
# directly.
with open(_config_dir / "underdog_payouts.json") as infile:
    _raw_underdog = json.load(infile)

underdog_payouts: dict[str, dict[int, float | list[float]]] = {}
for _variant, _table in _raw_underdog.items():
    if _variant.startswith("_"):
        continue
    underdog_payouts[_variant] = {int(k): v for k, v in _table.items()}
