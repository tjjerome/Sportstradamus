# ARCHIVED 2026-11-26 from src/sportstradamus/prediction/stories/pricing.py
# Reason: The story menu ranks both presets by the fitted value (ev), so Bankroll
#         Builder's full-Kelly log-growth objective and the stake printed beside it
#         lost their last callers. Their closed-form test was deleted from
#         tests/golden/test_story_menu.py (git keeps it).
# Last live SHA: dfe57da2
# Original imports (now unresolved here):
#   import math

# From src/sportstradamus/prediction/stories/pricing.py.
def _kelly_fraction(p: float, payout: float) -> float:
    """Full-Kelly fraction of bankroll for a single bet; 0 when there's no edge."""
    b = payout - 1.0
    if b <= 0.0:
        return 0.0
    return max(0.0, (p * (b + 1.0) - 1.0) / b)


def _log_growth(p: float, payout: float) -> float:
    """Expected log-growth of a single parlay bet at its own full-Kelly fraction."""
    f = _kelly_fraction(p, payout)
    if f <= 0.0:
        return 0.0
    b = payout - 1.0
    return p * math.log1p(b * f) + (1.0 - p) * math.log1p(-f)
