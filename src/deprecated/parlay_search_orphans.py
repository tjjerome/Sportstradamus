# ARCHIVED 2026-11-26 from src/sportstradamus/prediction/parlay.py
# Reason: The same-game entry search keeps each size's best entries by fitted value and
#         stakes none (docs/handoffs/profit-engine.md 5.3), so the Kelly-units score
#         behind ``Rec Bet`` and the ``Fun`` sort view lost their only callers.
# Last live SHA: c5593014
# Original imports (now unresolved here):
#   import numpy as np
#   from sportstradamus.helpers import stat_cv, stat_std
#   from sportstradamus.prediction.parlay import resolve_leg_stat
#   from sportstradamus.strategies.kelly import (
#       DEFAULT_KELLY_FRACTION,
#       MAX_FRACTION_OF_BANKROLL,
#       kelly_edge,
#   )


def _parlay_fun(bet, league, stat_by_label):
    """Heuristic 'fun' score: line distance in std units, only Over / H2H legs count.

    A market's spread is filed under the league's own key, not the label the leg
    carries. A cell with no stored spread scores on a unit one.
    """
    stds, cvs = stat_std.get(league, {}), stat_cv.get(league, {})
    fun = 0.0
    for leg in bet:
        stat = resolve_leg_stat(leg["Market"], stat_by_label)
        # stat_std holds None for a cell meditate has not calibrated yet.
        line_in_stds = leg["Line"] / (stds.get(stat) or 1)
        if "H2H" in leg["Market"]:
            fun += 3 - abs(line_in_stds)
        elif leg["Bet"] == "Over":
            fun += 2 - 1 / cvs.get(stat, 1) + line_in_stds
    return fun


def _parlay_kelly_units(g, bet_id, p, payout):
    """Shrinkage-aware Kelly ranking score for one evaluated candidate.

    win_prob is exact on the no-push analytical fast path (p = payout *
    P(all hit) there, so division recovers P(all hit) exactly) but only an
    approximation on the push-aware MC path. units is a ranking/filter score,
    not a final stake, so the approximation is acceptable here.
    """
    parlay_shrinkage = float(np.min(g.shrinkage[np.ix_(bet_id)]))
    win_prob = p / payout
    raw_kelly = kelly_edge(win_prob, payout, parlay_shrinkage)
    if raw_kelly <= 0:
        return -1.0
    return raw_kelly * DEFAULT_KELLY_FRACTION / MAX_FRACTION_OF_BANKROLL
