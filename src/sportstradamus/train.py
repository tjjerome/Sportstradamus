"""Back-compat shim: ``sportstradamus.train`` → ``training`` package.

The training pipeline was split into ``sportstradamus.training.*`` in
Phase 4c of the maintainability refactor. Any external code that imports
from this module continues to work via these re-exports.
"""

from sportstradamus.training import (
    compute_market_importance,
    correlate,
    filter_features,
    filter_market,
    fit_book_weights,
    fit_model_weight,
    meditate,
    report,
    see_features,
    select_distribution,
)
from sportstradamus.training.config import (
    load_distribution_config,
    load_zi_config,
    save_distribution_config,
    save_zi_config,
)
from sportstradamus.training.data import count_training_rows, trim_matrix
from sportstradamus.training.hyperparams import _BoundedResponseFn, warm_start_hyper_opt

__all__ = [
    "_BoundedResponseFn",
    "compute_market_importance",
    "correlate",
    "count_training_rows",
    "filter_features",
    "filter_market",
    "fit_book_weights",
    "fit_model_weight",
    "load_distribution_config",
    "load_zi_config",
    "meditate",
    "report",
    "save_distribution_config",
    "save_zi_config",
    "see_features",
    "select_distribution",
    "trim_matrix",
    "warm_start_hyper_opt",
]
