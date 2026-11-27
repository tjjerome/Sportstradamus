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


# ARCHIVED 2026-11-26 from src/sportstradamus/strategies/underdog_pickem.py
# Reason: Every ledger pool prices a leg at its fitted chance (policy_v5), so the
#         ledger's trust cut, this lookup's last caller, was archived
#         (ledger_cross_game_orphans.py). Its test was deleted from
#         tests/golden/test_kelly.py (git keeps it).
# Last live SHA: 30dc28fc
# Original imports (now unresolved here):
#   import pandas as pd
#   from sportstradamus.strategies.kelly import resolve_shrinkage


def resolve_market_shrinkage(league: str, market: str) -> tuple[float, str]:
    """Resolve Kelly shrinkage for one ``(league, market)`` cell.

    Blends live CLV-segment calibration with offline training calibration per
    ``kelly.resolve_shrinkage``'s explicit>blended>training>clv_segment>fallback
    chain; the returned source label tells a caller which rung of that chain
    fired for this cell.

    ``market`` is the league's own cell key (``helpers.cell_market`` resolves a
    platform's label to it), and the training reader has a row under no other
    name. The live reader matches history's ``Market``, which is that key except
    for Underdog's NBA and WNBA fantasy points: history files those rows as
    ``fantasy points underdog``, so their segment is not read under the cell key.

    Returns:
        ``(shrinkage, source)`` where ``source`` is one of ``"blended"``,
        ``"training"``, ``"clv_segment"``, or ``"fallback"``.
    """
    # Lazy imports keep torch (via training.report) out of the dashboard import
    # path — do NOT hoist clv or get_market_calibration to module level.
    from sportstradamus import clv as _clv
    from sportstradamus.training.report import get_market_calibration

    live_bss, live_n = _clv.get_segment_calibration(league, market)
    train_bss = get_market_calibration(league, market)["brier_skill_score"]

    shrinkage = resolve_shrinkage(training_bss=train_bss, live_bss=live_bss, live_n=live_n)
    has_train = not pd.isna(train_bss)
    has_live = not pd.isna(live_bss) and live_n > 0
    if not has_train and not has_live:
        return shrinkage, "fallback"
    if has_train and has_live:
        return shrinkage, "blended"
    return shrinkage, "training" if has_train else "clv_segment"


# ARCHIVED 2026-11-26 from src/sportstradamus/strategies/underdog_pickem.py
# Reason: Pick'em hands every scored record to the search, as the board's own search
#         does, and the ledger's shared cross-game pool takes the board's picks
#         (policy_v5), so this leg gate lost its last caller. PickemConfig's
#         min_model_edge, min_sharp_edge and disagreement_threshold went with it, and
#         its three tests were deleted from tests/golden/test_underdog_pickem.py (git
#         keeps them).
# Last live SHA: dde02525
# Original imports (now unresolved here):
#   import pandas as pd
#   from sportstradamus.strategies.underdog_pickem import PickemConfig


def filter_legs(offers: pd.DataFrame, config: PickemConfig) -> pd.DataFrame:
    """Apply leg-level filters: model & sharp coverage + edge + disagreement."""
    if offers.empty:
        return offers
    required = {"Win Prob", "Market Prob"}
    if not required.issubset(offers.columns):
        msg = f"filter_legs needs columns {required}; got {set(offers.columns)}"
        raise ValueError(msg)

    df = offers.dropna(subset=["Win Prob", "Market Prob"]).copy()
    keep = (
        ((df["Win Prob"] - 0.5) >= config.min_model_edge)
        & ((df["Market Prob"] - 0.5) >= config.min_sharp_edge)
        & ((df["Win Prob"] - df["Market Prob"]).abs() <= config.disagreement_threshold)
    )
    return df.loc[keep].copy()
