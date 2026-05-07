"""Nightly resolution script for Sportstradamus.

Runs after games finish to:
1. Fetch latest stats from league APIs (update)
2. Fill in Actual column in history (parquet primary, .dat fallback)
3. Fill in Legs/Misses columns in parlay_hist
4. Fill in Close Books P / Market CLV / Model CLV per offer
5. Write resolve_meta.json with last-run timestamp

Schedule with cron after games finish, e.g.:
    0 2 * * * cd /home/trevor/Sportstradamus && poetry run reflect
"""

import importlib.resources as pkg_resources
import json
import logging
from datetime import datetime

import click
import numpy as np
from tqdm import tqdm

from sportstradamus import clv, data
from sportstradamus.analysis import check_bet, resolve_history
from sportstradamus.helpers import Archive
from sportstradamus.helpers.io import (
    read_history,
    read_parlay_hist,
    write_history,
    write_parlay_hist,
)
from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LEAGUE_CLASSES = {
    "NBA": StatsNBA,
    "WNBA": StatsWNBA,
    "MLB": StatsMLB,
    "NFL": StatsNFL,
    "NHL": StatsNHL,
}


@click.command()
@click.option("--league", default=None, help="Resolve only this league (default: all).")
@click.option(
    "--skip-update",
    is_flag=True,
    default=False,
    help="Skip stats.update() API calls — use cached gamelogs only.",
)
@click.option(
    "--history-only", is_flag=True, default=False, help="Skip parlay resolution (much faster)."
)
def run(league, skip_update, history_only):
    """Nightly resolution: update stats, fill Actual/Legs/Misses, save."""
    # ------------------------------------------------------------------
    # 1. Load league Stats objects (optionally with update)
    # ------------------------------------------------------------------
    stats = {}
    leagues_to_load = [league] if league else list(LEAGUE_CLASSES.keys())

    for lg in tqdm(leagues_to_load, desc="Loading league stats"):
        cls = LEAGUE_CLASSES[lg]
        try:
            obj = cls()
            obj.load()
            if not skip_update:
                try:
                    obj.update()
                    logger.info(f"{lg}: stats updated")
                except Exception:
                    logger.warning(f"{lg}: update() failed, using cached gamelog")
            if hasattr(obj, "gamelog") and not obj.gamelog.empty:
                stats[lg] = obj
            else:
                logger.warning(f"{lg}: gamelog empty, skipping")
        except Exception:
            logger.warning(f"{lg}: load() failed, skipping")

    if not stats:
        logger.error("No league stats loaded. Aborting.")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 2. Resolve history (fill Actual column + CLV trio)
    # ------------------------------------------------------------------
    history = read_history()
    if history.empty:
        logger.error("history.parquet/.dat is empty or missing. Aborting.")
        raise SystemExit(1)

    if "Offers" not in history.columns:
        from sportstradamus.analysis import _migrate_flat_history

        logger.info("Migrating history to normalized schema")
        history = _migrate_flat_history(history)

    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    n_before_hist = int(history["Actual"].isna().sum())
    logger.info(f"Resolving {n_before_hist} pending history rows")
    history = resolve_history(history, stats)
    n_resolved_hist = n_before_hist - int(history["Actual"].isna().sum())
    logger.info(f"History: resolved {n_resolved_hist} / {n_before_hist} pending rows")

    # Load Archive once and fold closing-line values into every offer.
    archive = Archive()
    history = clv.fill_from_archive(history, archive)
    write_history(history)

    clv_summary = clv.summarize(history)
    if clv_summary["n"]:
        logger.info(
            "CLV legs: %d  Market CLV mean: %+.3f  Model CLV mean: %+.3f  beat-close: %.1f%%",
            clv_summary["n"],
            clv_summary["market_clv_mean"],
            clv_summary["model_clv_mean"],
            100.0 * clv_summary["frac_beat_close"],
        )
        if not clv_summary["segments"].empty:
            logger.info(
                "CLV segments (n>=%d):\n%s",
                clv.CLV_SEGMENT_MIN_N,
                clv_summary["segments"].to_string(index=False),
            )
    else:
        logger.info("CLV: no resolved legs with closing-line data")

    # ------------------------------------------------------------------
    # 3. Resolve parlay_hist (fill Legs/Misses columns)
    # ------------------------------------------------------------------
    n_resolved_parl = 0
    if not history_only:
        parlays = read_parlay_hist()
        stat_map = json.loads((pkg_resources.files(data) / "stat_map.json").read_text())

        unresolved = parlays.loc[parlays["Legs"].isna()]
        n_before_parl = len(unresolved)
        logger.info(f"Resolving {n_before_parl} pending parlay rows")

        if n_before_parl > 0:
            tqdm.pandas(desc="Resolving parlays")
            results = unresolved.progress_apply(
                lambda bet: check_bet(bet, stats, stat_map), axis=1
            ).tolist()
            parlays.loc[parlays["Legs"].isna(), ["Legs", "Misses"]] = results
            write_parlay_hist(parlays)
            n_resolved_parl = sum(
                1 for legs, _ in results if not (isinstance(legs, float) and np.isnan(legs))
            )
            logger.info(f"Parlays: resolved {n_resolved_parl} / {n_before_parl} pending rows")

    # ------------------------------------------------------------------
    # 4. Write metadata
    # ------------------------------------------------------------------
    meta = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "history_resolved": n_resolved_hist,
        "parlays_resolved": n_resolved_parl,
        "history_total": len(history),
        "history_pending": int(history["Actual"].isna().sum()),
    }
    meta_path = pkg_resources.files(data) / "resolve_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    click.echo(
        f"Done. History: {n_resolved_hist} resolved. "
        f"Parlays: {n_resolved_parl} resolved. "
        f"Last run: {meta['last_run']}"
    )
