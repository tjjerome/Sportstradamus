"""Training package: LightGBMLSS model training pipeline.

Public API re-exports for back-compat with any code that imported from
``sportstradamus.train`` directly.
"""

from sportstradamus.training.calibration import (
    fit_book_weights,
    fit_model_weight,
    select_distribution,
)
from sportstradamus.training.cli import meditate
from sportstradamus.training.correlate import correlate
from sportstradamus.training.report import report
from sportstradamus.training.shap import (
    compute_market_importance,
    filter_features,
    filter_market,
    see_features,
)

__all__ = [
    "compute_market_importance",
    "correlate",
    "filter_features",
    "filter_market",
    "fit_book_weights",
    "fit_model_weight",
    "meditate",
    "report",
    "see_features",
    "select_distribution",
]
