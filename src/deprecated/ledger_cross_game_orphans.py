# ARCHIVED 2026-11-26 from src/sportstradamus/strategies/_ledger_cross_game.py
# Reason: Every ledger pool prices a leg at its fitted chance (policy_v5,
#         docs/handoffs/sim-bettor-ledger.md §10), so the trust cut on the served read
#         lost its only caller. Its test was deleted from
#         tests/golden/test_ledger_cross_game.py (git keeps it).
# Last live SHA: 30dc28fc
# Original imports (now unresolved here):
#   from dataclasses import replace
#   import pandas as pd
#   from sportstradamus.strategies._ledger_cross_game import _ScoredLeg, _served_legs
#   from sportstradamus.strategies.underdog_pickem import resolve_market_shrinkage


def _score_legs(eligible: pd.DataFrame, platform: str) -> list[_ScoredLeg]:
    """``eligible``'s legs, each read pulled toward a coin flip by its cell's trust."""
    out = []
    for leg in _served_legs(eligible, platform):
        shrinkage, _source = resolve_market_shrinkage(leg.canonical_leg["league"], leg.cell_market)
        adj_p = 0.5 + (leg.win_prob - 0.5) * max(0.0, min(1.0, shrinkage))
        out.append(replace(leg, win_prob=adj_p))
    return out
