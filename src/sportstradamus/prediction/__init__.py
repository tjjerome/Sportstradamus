"""Prediction package: distributional scoring and parlay construction.

Public API re-exports for back-compat with any code that imported from
``sportstradamus.prediction`` directly.
"""

from sportstradamus.prediction.cli import main
from sportstradamus.prediction.correlation import find_correlation
from sportstradamus.prediction.model_prob import model_prob
from sportstradamus.prediction.scoring import match_offers, process_offers
from sportstradamus.prediction.sheets import save_data

__all__ = [
    "find_correlation",
    "main",
    "match_offers",
    "model_prob",
    "process_offers",
    "save_data",
]
