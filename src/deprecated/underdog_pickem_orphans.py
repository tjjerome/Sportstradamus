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
