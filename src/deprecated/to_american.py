# ARCHIVED 2026-07-05 from src/sportstradamus/dashboard/components/deep_dive.py
# Reason: The P8 Phase C rev-3 context strip replaced the Moneyline American-odds
#         metric with a plain favorite-win-probability display (ml_fav_prob, already
#         a [0,1] probability), removing this formatter's only call site.
# Last live SHA: 8c4447d
# Original imports (now unresolved here):
#   import numpy as np


def to_american(p: float) -> str:
    """Format a win probability as American odds, or ``"N/A"`` outside (0, 1)."""
    if not isinstance(p, float) or np.isnan(p) or p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    return f"+{round((1 - p) / p * 100)}"
