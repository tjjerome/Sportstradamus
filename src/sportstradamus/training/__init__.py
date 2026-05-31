"""Training package: LightGBMLSS model training pipeline.

Re-exports the public API so callers can do
``from sportstradamus.training import train_market`` without knowing the
internal module layout.
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
    see_features,
)

__all__ = [
    "compute_market_importance",
    "correlate",
    "fit_book_weights",
    "fit_model_weight",
    "meditate",
    "report",
    "see_features",
    "select_distribution",
]
