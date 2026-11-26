# ARCHIVED 2026-11-26 from src/sportstradamus/prediction/correlation.py
# Reason: The same-game entry search stakes no entry (docs/handoffs/profit-engine.md 5.3),
#         so nothing read the per-leg Kelly shrinkage it laid on ``GameArrays``.
# Last live SHA: 38e9177b
# Original imports (now unresolved here):
#   import numpy as np
#   from sportstradamus.helpers import cell_market


def _leg_shrinkage(game_df, platform, league, shrinkage_cache):
    """Per-leg Kelly shrinkage weight, cached by ``(league, cell market)``.

    ``shrinkage_cache`` is caller-owned and shared across every game in one
    league's processing loop — ``resolve_market_shrinkage`` hits training/CLV
    I/O per call, and markets repeat heavily across a slate's legs, so this
    avoids re-resolving the same cell hundreds of times per league. Legs
    whose ``Market`` has no ``stat_map`` entry for this platform keep ``1.0``:
    those are game legs (moneylines, totals) with no per-market calibration
    concept, not no-evidence player cells — ``resolve_shrinkage``'s own
    no-evidence default is the smaller :data:`NO_EVIDENCE_SHRINKAGE`.
    """
    from sportstradamus.strategies.underdog_pickem import resolve_market_shrinkage

    out = np.ones(len(game_df), dtype=float)
    for i, label in enumerate(game_df["Market"]):
        cell = cell_market(league, platform, label)
        if cell is None:
            continue
        key = (league, cell)
        if key not in shrinkage_cache:
            shrinkage_cache[key] = resolve_market_shrinkage(*key)[0]
        out[i] = shrinkage_cache[key]
    return out
