# ARCHIVED 2026-10-11 from src/sportstradamus/prediction/cli.py
# Reason: Nothing read data/runtime/current_pickem.parquet (no dashboard page, no
#         CLI), so prophecize stopped building it: a parlay search per contest
#         variant per platform on every run. The hook is archived here with the
#         writer, the path and the two strategies helpers only it called; each
#         block names the file it came from.
# Last live SHA: 515c9805
# Original imports (now unresolved here):
#   from decimal import Decimal
#   from typing import Any
#   import pandas as pd
#   from sportstradamus.helpers.io import _RUNTIME_DIR, _atomic_write_parquet
#   from sportstradamus.spiderLogger import logger
#   from sportstradamus.strategies.underdog_pickem import (
#       PickemConfig, RecommendedEntry, _parlays_per_variant, construct_entries,
#   )

# From src/sportstradamus/prediction/cli.py. main() copied each platform's frame
# as process_offers returned it, before the Market remap and the Underdog Boost
# rescale (find_correlation expects that shape), kept the copy only when the
# platform's block finished, and called this after write_current_offer_details
# with {"Underdog": <copy or None>, "Sleeper": <copy or None>}.
def _write_pickem_snapshot(scored_by_platform: dict[str, pd.DataFrame | None], stats: dict) -> None:
    """Build the per-platform Pick'em entries snapshot from already-scored offers.

    Reuses prophecize's scored frames — no second scrape, no extra archive
    lock — and writes one combined ``current_pickem.parquet`` for the
    dashboard, tagged by ``platform``. Each platform's entry construction is
    guarded independently so one platform's failure logs and continues
    rather than blanking the other's entries.
    """
    from sportstradamus.strategies._pickem_emit import entries_to_frame
    from sportstradamus.strategies.underdog_pickem import (
        REFERENCE_BANKROLL,
        build_entries_from_scored,
    )

    all_entries = []
    for platform, scored in scored_by_platform.items():
        if scored is None or scored.empty:
            continue
        try:
            all_entries.extend(
                build_entries_from_scored(
                    REFERENCE_BANKROLL,
                    scored,
                    stats,
                    platform=platform,
                )
            )
        except Exception:
            logger.exception("Failed to build %s pickem snapshot", platform)
    write_current_pickem(entries_to_frame(all_entries))


# From src/sportstradamus/helpers/io.py.
CURRENT_PICKEM_PATH = _RUNTIME_DIR / "current_pickem.parquet"


# From src/sportstradamus/prediction/persist.py.
def write_current_pickem(entries: pd.DataFrame) -> None:
    """Write the current-run Underdog Pick'em entries snapshot atomically.

    `entries` is the frame from `strategies._pickem_emit.entries_to_frame`
    (bankroll-independent fields; the dashboard sizes stakes from a user
    bankroll). Written even when empty so the dashboard reflects the latest run.
    """
    _atomic_write_parquet(entries, CURRENT_PICKEM_PATH)


# From src/sportstradamus/strategies/underdog_pickem.py.
#
# prophecize's hourly snapshot sizes stakes at this nominal bankroll purely to
# drive rank_and_dedupe ordering (bankroll-invariant below the per-bet cap); the
# dashboard re-sizes every entry against the user's actual bankroll.
REFERENCE_BANKROLL = Decimal("1000")


def build_entries_from_scored(
    bankroll: Decimal,
    scored_offers_df: pd.DataFrame,
    stats: dict[str, Any],
    config: PickemConfig | None = None,
    *,
    platform: str = "Underdog",
) -> list[RecommendedEntry]:
    """Build ranked entries from offers already scored by ``process_offers``.

    Skips the scrape and model scoring — the caller supplies ``scored_offers_df``
    — and runs only the per-variant correlation search before ranking and
    sizing. The ``prophecize`` snapshot hook uses this so the hourly run feeds
    the dashboard without a second scrape or archive lock.
    """
    config = config or PickemConfig()
    parlay_dfs = _parlays_per_variant(scored_offers_df, stats, config, platform)
    return construct_entries(
        bankroll,
        config,
        parlay_dfs=parlay_dfs,
        platform=platform,
    )


# From src/sportstradamus/strategies/_pickem_emit.py.
_PICKEM_FRAME_COLS = [
    "id",
    "contest_variant",
    "entry_size",
    "legs",
    "joint_prob",
    "payout_multiplier",
    "ev",
    "shrinkage",
    "shrinkage_source",
    "league",
    "game",
    "platform",
]


def entries_to_frame(entries: list[RecommendedEntry]) -> pd.DataFrame:
    """Flatten ranked entries into the dashboard pickem snapshot frame.

    Bankroll-independent fields only: the dashboard sizes stakes live from a
    user-entered bankroll, so ``recommended_stake`` is intentionally dropped.
    """
    if not entries:
        return pd.DataFrame(columns=_PICKEM_FRAME_COLS)
    return pd.DataFrame(
        [
            {
                "id": e.id,
                "contest_variant": e.contest_variant,
                "entry_size": e.entry_size,
                "legs": list(e.legs),
                "joint_prob": float(e.joint_prob),
                "payout_multiplier": float(e.payout_multiplier),
                "ev": float(e.ev),
                "shrinkage": float(e.shrinkage),
                "shrinkage_source": e.shrinkage_source,
                "league": e.extras.get("league", ""),
                "game": e.extras.get("game", ""),
                "platform": e.platform,
            }
            for e in entries
        ]
    )
