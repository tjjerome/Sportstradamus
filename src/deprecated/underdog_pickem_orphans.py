# ARCHIVED 2026-08-29 from src/sportstradamus/strategies/underdog_pickem.py
# Reason: _parlay_shrinkage now maps each parlay's canonical ``legs`` records
#         (raw platform ``market`` -> stat_map cell key) directly, so the
#         player->markets map built from the filtered offer universe lost its
#         only caller.
# Last live SHA: e1246576
# Original imports (now unresolved here):
#   import pandas as pd
#   from sportstradamus.helpers import stat_map


def _canonical_markets_by_player(filtered_offers, platform="Underdog"):
    """Map each offer's player to the canonical market(s) they were offered on.

    Offer ``Market`` is the raw platform name (a ``stat_map[platform]`` key);
    mapping it gives the canonical cell key (``REB``, ``BLST``, …) that
    ``resolve_market_shrinkage`` resolves against. The leg display strings
    can't drive this — they carry a third, lossy namespace (``blocks_and_steals``).
    """
    if filtered_offers.empty or not {"Player", "Market"}.issubset(filtered_offers.columns):
        return {}
    out = {}
    for player, market in zip(
        filtered_offers["Player"],
        filtered_offers["Market"].map(stat_map.get(platform, {})),
        strict=True,
    ):
        if isinstance(market, str):
            out.setdefault(str(player), set()).add(market)
    return out


# ARCHIVED 2026-11-26 from src/sportstradamus/strategies/underdog_pickem.py
# Reason: Pick'em entries are valued at the search row's fitted EV and carry no stake
#         until the entries' money record supports one (docs/handoffs/profit-engine.md
#         5.3), so the Kelly call this trust figure sized lost its only caller. Its four
#         tests were deleted from tests/golden/test_underdog_pickem.py (git keeps them).
# Last live SHA: 2ac492b7
# Original imports (now unresolved here):
#   import pandas as pd
#   from sportstradamus.helpers import cell_market
#   from sportstradamus.strategies.kelly import NO_EVIDENCE_SHRINKAGE


def _parlay_shrinkage(row: pd.Series, league: str, platform: str) -> tuple[float, str]:
    """Kelly shrinkage for one parlay: the most conservative of its leg markets.

    A parlay cashes only if every leg hits, so it inherits the trust of its
    least-calibrated market — resolve per distinct leg market and keep the min.
    Leg ``market`` is the raw platform name (a ``stat_map[platform]`` key);
    ``cell_market`` gives the cell key (``REB``, ``BLST``, …) that
    ``resolve_market_shrinkage`` resolves against. Combo / H2H names with no
    mapping are skipped; a parlay with none left has no cell to vouch for it and
    sizes on the no-evidence prior.
    """
    cells = {cell_market(league, platform, leg["market"]) for leg in row["legs"]} - {None}
    if not cells:
        return NO_EVIDENCE_SHRINKAGE, "fallback"
    return min((resolve_market_shrinkage(league, cell) for cell in cells), key=lambda r: r[0])
