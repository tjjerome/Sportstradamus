"""Prediction package: distributional scoring and parlay construction."""

from sportstradamus.prediction.cli import main
from sportstradamus.prediction.correlation import find_correlation
from sportstradamus.prediction.model_prob import model_prob
from sportstradamus.prediction.persist import write_current_offers
from sportstradamus.prediction.scoring import match_offers, process_offers

__all__ = [
    "find_correlation",
    "main",
    "match_offers",
    "model_prob",
    "process_offers",
    "write_current_offers",
]
