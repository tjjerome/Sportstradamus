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
  ``fused_loc``, ``set_model_start_values``, ``decode_predictive_mean``,
  ``apply_temperature``, plus the smaller odds-conversion and ``no_vig_odds``
  helpers.
* :mod:`sportstradamus.helpers.scraping` — the ``Scrape`` HTTP client with
  ScrapeOps-managed header rotation.
* :mod:`sportstradamus.helpers.archive` — the ``Archive`` singleton that
  persists odds/EV data to a DuckDB archive, plus the ``clean_archive``
  housekeeping helper.
* :mod:`sportstradamus.helpers.market_display` — ``market_display_name``, the
  render-only market-slug-to-prose lookup.

Every legacy ``from sportstradamus.helpers import <name>`` keeps working
because this module re-exports the full public surface below. Prefer the
submodule path in new code.
"""

import requests

from sportstradamus.helpers.archive import Archive, LazyArchive, clean_archive
from sportstradamus.helpers.config import (
    abbreviations,
    banned,
    book_gate,
    book_skewnormal_shape,
    book_weights,
    books,
    combo_props,
    feature_filter,
    name_map,
    nhl_goalies,
    odds_api,
    sleeper_payouts,
    stat_cv,
    stat_dist,
    stat_map,
    stat_meta,
    stat_std,
    stat_zi,
    underdog_payouts,
)
from sportstradamus.helpers.distributions import (
    GATE_PUBLISH_THRESHOLD,
    NONZERO_DENOM_GATE,
    UNDERDOG_BOOST_BASELINE,
    DecodedParams,
    apply_cdf_recal,
    apply_temperature,
    decode_predictive_mean,
    dp_crps,
    dp_pmf_curve,
    fit_distro,
    fused_loc,
    gamma_crps,
    get_ev,
    get_odds,
    get_push_prob,
    negbin_crps,
    no_vig_odds,
    odds_to_prob,
    predictive_std,
    prob_to_odds,
    set_model_start_values,
    shape_from_moments,
    skewnorm_crps,
    skewnormal_loc_from_mean,
)
from sportstradamus.helpers.logging import JsonFormatter, get_logger
from sportstradamus.helpers.market_display import market_display_name
from sportstradamus.helpers.scraping import Scrape
from sportstradamus.helpers.text import (
    get_mlb_pitchers,
    get_trends,
    hmean,
    merge_dict,
    remove_accents,
)

__all__ = [
    "GATE_PUBLISH_THRESHOLD",
    "NONZERO_DENOM_GATE",
    "UNDERDOG_BOOST_BASELINE",
    "Archive",
    "DecodedParams",
    "JsonFormatter",
    "LazyArchive",
    "Scrape",
    "abbreviations",
    "apply_cdf_recal",
    "apply_temperature",
    "banned",
    "book_gate",
    "book_skewnormal_shape",
    "book_weights",
    "books",
    "clean_archive",
    "combo_props",
    "decode_predictive_mean",
    "dp_crps",
    "dp_pmf_curve",
    "feature_filter",
    "fit_distro",
    "fused_loc",
    "gamma_crps",
    "get_ev",
    "get_logger",
    "get_mlb_pitchers",
    "get_odds",
    "get_push_prob",
    "get_trends",
    "hmean",
    "market_display_name",
    "merge_dict",
    "name_map",
    "negbin_crps",
    "nhl_goalies",
    "no_vig_odds",
    "odds_api",
    "odds_to_prob",
    "predictive_std",
    "prob_to_odds",
    "remove_accents",
    "requests",
    "set_model_start_values",
    "shape_from_moments",
    "skewnorm_crps",
    "skewnormal_loc_from_mean",
    "sleeper_payouts",
    "stat_cv",
    "stat_dist",
    "stat_map",
    "stat_meta",
    "stat_std",
    "stat_zi",
    "underdog_payouts",
]
