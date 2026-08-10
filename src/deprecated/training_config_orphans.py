# ARCHIVED 2026-07-18 from src/sportstradamus/training/config.py
# Reason: report() was its only caller; the pickle-derived dist writeback it served
#   reverted a pending stat_meta family flip off a stale pickle (NBA STL, W5) —
#   stat_meta dist is an input, never derived from artifacts.
# Last live SHA: 46d49cd
# Original imports (now unresolved here):
#   (none — pure function, no top-level imports; it called the module-private
#   `_load` / `_write` / `_STAT_META_PATH` still defined in training/config.py)


def save_distribution_config(config: dict) -> None:
    """Persist a full ``{league: {market: dist}}`` map into ``stat_meta.json``.

    Overwrites the ``dist`` field on every (league, market) in ``config``;
    cells absent from ``config`` are untouched. Other fields on existing
    cells (``shipped``, ``target_normalization``, ``posthoc``) are preserved.
    """
    meta = _load(_STAT_META_PATH)
    for league, markets in config.items():
        for market, dist in markets.items():
            cell = meta.setdefault(league, {}).setdefault(
                market,
                {
                    "dist": "Gamma",
                    "shipped": "withheld",
                    "target_normalization": "none",
                    "posthoc": "none",
                },
            )
            cell["dist"] = dist
    _write(_STAT_META_PATH, meta)
