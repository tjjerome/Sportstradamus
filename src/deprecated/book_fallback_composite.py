# ARCHIVED 2026-08-29 from src/sportstradamus/prediction/model_prob.py
# Reason: book_fallback_prob now resolves modal-line cohort quotes through
#         helpers.training_quotes.resolve_training_quote (the cross-line weighted-EV
#         average this composited was a Jensen error a DFS platform's reposted
#         50/50 could poison), and the boost devig now goes through
#         helpers.distributions.dfs_boost_probs — so the composite EV reader and
#         the half-payout boost devig lost their last callers.
# Last live SHA: e1246576
# Original imports (now unresolved here):
#   import numpy as np
#   from sportstradamus.prediction.model_prob import archive

# Coin-flip prior used when no bookmaker price was available for an offer;
# retired along with these functions (unquoted rows now price at the
# payout-implied probability of the chosen side).
_BOOK_PRIOR_PROB: float = 0.5


def _composite_book_evs(players, league: str, market: str, date_map: dict, stat_data) -> dict:
    """Composite (book-weighted, vig-free) EV per player, with combo fallback.

    Reads ``archive.get_ev`` for each player; when the archive has no direct
    price, falls back to the convolved consensus from ``check_combo_markets``
    (qb-yards / qb-tds). Players with no book price carry NaN.
    """
    evs = {}
    for player in players:
        date = date_map.get(player, "")
        ev = archive.get_ev(league, market, date, player)
        if np.isnan(ev):
            ev = stat_data.check_combo_markets(market, player, date)
        evs[player] = ev
    return evs


def _odds_from_boost(o: dict) -> np.ndarray:
    p = [
        _BOOK_PRIOR_PROB / o.get("Boost_Under", 1)
        if o.get("Boost_Under", 1) > 0
        else 1 - _BOOK_PRIOR_PROB / o.get("Boost_Over", 1),
        _BOOK_PRIOR_PROB / o.get("Boost_Over", 1)
        if o.get("Boost_Over", 1) > 0
        else 1 - _BOOK_PRIOR_PROB / o.get("Boost_Under", 1),
    ]
    return p / np.sum(p)
