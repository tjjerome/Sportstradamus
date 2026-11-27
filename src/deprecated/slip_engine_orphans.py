# ARCHIVED 2026-11-26 from src/sportstradamus/dashboard/slip_engine.py
# Reason: The Correlated tab now ranks every partner by its pair's fitted EV and reads the
#         pair's SlipScore directly (deep_dive_tabs._ranked_partners), removing this
#         model-EV lift's only call site.
# Last live SHA: af6988b6
# Original imports (now unresolved here):
#   from collections.abc import Mapping
#   from sportstradamus.dashboard.slip_engine import PairValues, score_slip
#   from sportstradamus.prediction.stories.context import GameCtx


def ev_lift(
    focus: Mapping,
    candidate: Mapping,
    ctxs: Mapping[str, GameCtx],
    mods: Mapping[str, PairValues],
    *,
    platform: str,
    bankroll: float = 0.0,
    shrinkage: float = 1.0,
) -> float:
    """EV of {focus + candidate} minus focus alone (spec §4.1 Correlated tab).

    Both scored through the same copula path as the slip rail; ~milliseconds
    per candidate, so the Details tab computes live (no precompute). A candidate
    the platform refuses beside the focus prices at $0, so it lifts nothing.
    """
    pair = score_slip(
        [focus, candidate], ctxs, mods, platform=platform, bankroll=bankroll, shrinkage=shrinkage
    )
    solo = score_slip(
        [focus], ctxs, mods, platform=platform, bankroll=bankroll, shrinkage=shrinkage
    )
    return pair.model_ev - solo.model_ev
