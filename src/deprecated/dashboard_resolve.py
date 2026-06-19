# ARCHIVED 2026-06-11 from src/sportstradamus/dashboard/data.py
# Reason: Dashboard-side resolution superseded by nightly.py (`poetry run
#         reflect`), which owns history/parlay resolve + CLV fill; the
#         dashboard reads pre-resolved parquets only, so these three (and
#         the LEAGUE_CLASSES dict only load_stats consumed) had no caller.
# Last live SHA: 2eed419
# Original imports (now unresolved here):
#   import numpy as np
#   import pandas as pd
#   import streamlit as st
#   from sportstradamus.analysis import backfill_crps, check_bet, resolve_history
#   from sportstradamus.helpers.io import write_history, write_parlay_hist
#   from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

LEAGUE_CLASSES = {
    "NBA": StatsNBA,
    "WNBA": StatsWNBA,
    "MLB": StatsMLB,
    "NFL": StatsNFL,
    "NHL": StatsNHL,
}


@st.cache_resource(show_spinner="Loading league stats...")
def load_stats() -> dict:
    """Load Stats objects from cached pickle files (no API calls).

    API updates are handled by the nightly script (poetry run reflect),
    not at dashboard startup. This keeps the dashboard load fast (~2 s).
    """
    stats = {}
    for lg, cls in LEAGUE_CLASSES.items():
        try:
            obj = cls()
            obj.load()
            if not obj.gamelog.empty:
                stats[lg] = obj
        except Exception:
            pass
    return stats


def resolve_and_save(history: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Resolve pending predictions (fill Actual + CRPS) and save back to parquet."""
    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    pending_count = history["Actual"].isna().sum()
    if pending_count:
        history = resolve_history(history, stats)

    # CRPS is precomputed here (not in the dashboard) so the diagnostics page only
    # aggregates it; this also backfills history that predates the column.
    crps_added = backfill_crps(history)

    if pending_count or crps_added:
        write_history(history)
    return history


def resolve_parlays_and_save(parlays: pd.DataFrame, stats: dict, stat_map: dict) -> pd.DataFrame:
    """Resolve pending parlays and save back to parquet."""
    if parlays.empty or "Legs" not in parlays.columns:
        return parlays

    unresolved = parlays.loc[parlays["Legs"].isna()]
    if len(unresolved) == 0:
        return parlays

    from tqdm import tqdm

    tqdm.pandas()
    results = unresolved.progress_apply(
        lambda bet: check_bet(bet, stats, stat_map), axis=1
    ).to_list()
    parlays.loc[parlays["Legs"].isna(), ["Legs", "Misses"]] = results
    write_parlay_hist(parlays)
    return parlays
