"""Cross-cutting utilities shared across the scraping, training, and prediction pipelines.

Split into submodules by concern:

* :mod:`sportstradamus.helpers.config` — loads every JSON in ``data/`` and
  ``creds/`` at import time and exposes the resulting dicts as module-level
  constants. Changing a JSON shape means touching this file.
* :mod:`sportstradamus.helpers.text` — name normalization and small
  collection helpers (``remove_accents``, ``merge_dict``, ``hmean``,
  ``get_trends``, ``get_mlb_pitchers``).
* :mod:`sportstradamus.helpers.distributions` — the distributional math
  that the model/bookmaker fusion runs on: ``get_ev``, ``get_odds``,
  ``fused_loc``, ``set_model_start_values``, plus the smaller odds-conversion
  and ``no_vig_odds`` helpers.
* :mod:`sportstradamus.helpers.scraping` — the ``Scrape`` HTTP client with
  ScrapeOps-managed header rotation.
* :mod:`sportstradamus.helpers.archive` — the ``Archive`` singleton that
  persists odds/EV data to klepto HDF archives, plus the ``clean_archive``
  migration helper.

Every legacy ``from sportstradamus.helpers import <name>`` keeps working
because this module re-exports the full public surface below. Prefer the
submodule path in new code.
"""

import requests

from sportstradamus.helpers.archive import Archive, clean_archive
from sportstradamus.helpers.config import (
    abbreviations,
    banned,
    book_weights,
    books,
    combo_props,
    feature_filter,
    name_map,
    nhl_goalies,
    odds_api,
    stat_cv,
    stat_dist,
    stat_map,
    stat_std,
    stat_zi,
    underdog_payouts,
)
from sportstradamus.helpers.distributions import (
    fit_distro,
    fused_loc,
    get_ev,
    get_odds,
    get_push_prob,
    no_vig_odds,
    odds_to_prob,
    prob_to_odds,
    set_model_start_values,
)
from sportstradamus.helpers.scraping import Scrape
from sportstradamus.helpers.text import (
    get_mlb_pitchers,
    get_trends,
    hmean,
    merge_dict,
    remove_accents,
)

__all__ = [
    "Archive",
    "Scrape",
    "abbreviations",
    "banned",
    "book_weights",
    "books",
    "clean_archive",
    "combo_props",
    "feature_filter",
    "fit_distro",
    "fused_loc",
    "get_ev",
    "get_mlb_pitchers",
    "get_odds",
    "get_push_prob",
    "get_trends",
    "hmean",
    "merge_dict",
    "name_map",
    "nhl_goalies",
    "no_vig_odds",
    "odds_api",
    "odds_to_prob",
    "prob_to_odds",
    "remove_accents",
    "requests",
    "set_model_start_values",
    "stat_cv",
    "stat_dist",
    "stat_map",
    "stat_std",
    "stat_zi",
    "underdog_payouts",
]
