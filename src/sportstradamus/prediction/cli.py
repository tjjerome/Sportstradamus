"""Main ``prophecize`` CLI entrypoint.

Orchestrates the full prediction pipeline:

1. Load and update ``Stats`` objects for active leagues
2. Scrape DFS offers (Underdog, Sleeper)
3. Score via :func:`process_offers` (feature extraction + distributional model)
4. Snapshot scored offers + parlays as parquet for the Streamlit dashboard
5. Persist a rolling year of predictions to ``data/history.parquet`` and
   the new parlays to ``data/parlay_hist.parquet``
"""

from __future__ import annotations

import datetime
import os
from functools import partialmethod

import click
import line_profiler
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import creds
from sportstradamus.books import get_sleeper, get_ud
from sportstradamus.helpers import Archive, get_logger, stat_map
from sportstradamus.helpers.io import (
    read_history,
    read_parlay_hist,
    write_history,
    write_parlay_hist,
)
from sportstradamus.prediction.persist import write_current_offers
from sportstradamus.prediction.scoring import process_offers
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA

pd.set_option("mode.chained_assignment", None)
pd.set_option("future.no_silent_downcasting", True)
os.environ["LINE_PROFILE"] = "0"

archive = Archive()

_HISTORY_RETENTION_DAYS = 365


@click.command()
@click.option("--progress/--no-progress", default=True, help="Display progress bars")
@click.option(
    "--legacy-correlation/--no-legacy-correlation",
    default=False,
    help=(
        "Reproduce the pre-2026.05 parlay pipeline verbatim — no PSD repair, "
        "no push-aware EV, mixed insurance/power Boost overwrite. Removed "
        "next release; provided as a one-cycle escape hatch."
    ),
)
@click.option(
    "--contest-variant",
    type=click.Choice(["pooled", "power", "flex", "insurance", "rivals"]),
    default="pooled",
    help=(
        "Underdog payout pool for parlay scoring. Default 'pooled' "
        "combines power (2-3 legs) and flex (4+ legs) into one pool; "
        "single-variant names are kept for the pickem-build path."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Verbosity for the structured JSONL log.",
)
@line_profiler.profile
def main(progress, legacy_correlation, contest_variant, log_level):
    """Run the full prediction pipeline and write results to Google Sheets."""
    cli_log = get_logger("prophecize")
    cli_log.setLevel(log_level)
    cli_log.info(
        "prophecize invoked",
        extra={"contest_variant": contest_variant, "legacy_correlation": legacy_correlation},
    )
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not progress))

    sports = []
    nba = StatsNBA()
    nba.load()
    if datetime.datetime.today().date() > (nba.season_start - datetime.timedelta(days=7)):
        sports.append("NBA")
    nfl = StatsNFL()
    nfl.load()
    if datetime.datetime.today().date() > (nfl.season_start - datetime.timedelta(days=7)):
        sports.append("NFL")
    wnba = StatsWNBA()
    wnba.load()
    if datetime.datetime.today().date() > (wnba.season_start - datetime.timedelta(days=7)):
        sports.append("WNBA")

    stats = {}
    if "NBA" in sports:
        nba.update()
        stats.update({"NBA": nba})
    if "NFL" in sports:
        nfl.update()
        stats.update({"NFL": nfl})
    if "WNBA" in sports:
        wnba.update()
        stats.update({"WNBA": wnba})

    all_offers: list[pd.DataFrame] = []
    parlay_df = pd.DataFrame()
    platforms_run: list[str] = []

    try:
        ud_dict = get_ud()
        ud_offers, ud5 = process_offers(
            ud_dict,
            "Underdog",
            stats,
            contest_variant=contest_variant,
            legacy=legacy_correlation,
        )
        parlay_df = pd.concat([parlay_df, ud5])
        ud_offers["Stat"] = ud_offers["Market"]  # preserve gamelog key before remap
        ud_offers["Market"] = ud_offers["Market"].map(stat_map["Underdog"])
        ud_offers.loc[ud_offers["Bet"] == "Over", "Boost"] = (
            1.78 * ud_offers.loc[ud_offers["Bet"] == "Over", "Boost"]
        )
        ud_offers.loc[ud_offers["Bet"] == "Under", "Boost"] = (
            1.78 / ud_offers.loc[ud_offers["Bet"] == "Under", "Boost"]
        )
        ud_offers["Platform"] = "Underdog"
        all_offers.append(ud_offers)
        platforms_run.append("Underdog")
    except Exception:
        logger.exception("Failed to get Underdog")

    try:
        sl_dict = get_sleeper()
        sl_offers, sl5 = process_offers(
            sl_dict,
            "Sleeper",
            stats,
            contest_variant=contest_variant,
            legacy=legacy_correlation,
        )
        parlay_df = pd.concat([parlay_df, sl5])
        sl_offers["Stat"] = sl_offers["Market"]  # preserve gamelog key before remap
        sl_offers["Market"] = sl_offers["Market"].map(stat_map["Sleeper"])
        sl_offers.loc[sl_offers["Bet"] == "Under", "Boost"] = (
            1.78 * 1.78 / sl_offers.loc[sl_offers["Bet"] == "Under", "Boost"]
        )
        sl_offers["Platform"] = "Sleeper"
        all_offers.append(sl_offers)
        platforms_run.append("Sleeper")
    except Exception:
        logger.exception("Failed to get Sleeper")

    snapshot_offers = pd.concat(all_offers) if all_offers else pd.DataFrame()
    if not parlay_df.empty:
        parlay_df.sort_values("Model EV", ascending=False, inplace=True)
        parlay_df.drop_duplicates(inplace=True)
        parlay_df.reset_index(drop=True, inplace=True)
        parlay_df[["Legs", "Misses", "Profit"]] = np.nan

    # Overwrite O/U percentage with raw game total from Archive
    if not snapshot_offers.empty and "O/U" in snapshot_offers.columns:
        snapshot_offers["O/U"] = snapshot_offers.apply(
            lambda r: archive.get_total(r["League"], r["Date"], r["Team"]) or r["O/U"],
            axis=1,
        )

    write_current_offers(
        snapshot_offers,
        parlay_df,
        sports,
        platforms_run,
        contest_variant=contest_variant,
        stats_dict=stats,
    )

    # --- Append to rolling parlay history ---
    if not parlay_df.empty:
        old_parlays = read_parlay_hist()
        if not old_parlays.empty:
            combined = pd.concat([parlay_df, old_parlays], ignore_index=True).drop_duplicates(
                subset=["Model EV", "Books EV"], ignore_index=True
            )
        else:
            combined = parlay_df
        write_parlay_hist(combined)

    archive.write()
    logger.info("Checking historical predictions")
    from sportstradamus.analysis import _merge_offers, _migrate_flat_history

    history = read_history()
    if not history.empty:
        if "Offers" not in history.columns:
            history = _migrate_flat_history(history)
    else:
        history = pd.DataFrame(
            columns=[
                "Player",
                "League",
                "Team",
                "Date",
                "Market",
                "Model EV",
                "Books EV",
                "Dist",
                "CV",
                "Model Param",
                "Gate",
                "Temperature",
                "Disp Cal",
                "Step",
                "Offers",
                "Actual",
            ]
        )

    pred_key = ["Player", "League", "Date", "Market"]
    pred_level_cols = [
        "Team",
        "Model EV",
        "Books EV",
        "Dist",
        "CV",
        "Model Param",
        "Gate",
        "Temperature",
        "Disp Cal",
        "Step",
    ]

    if all_offers:
        all_df = pd.concat(all_offers)
        all_df.loc[(all_df["Market"] == "AST") & (all_df["League"] == "NHL"), "Market"] = "assists"
        all_df.loc[(all_df["Market"] == "PTS") & (all_df["League"] == "NHL"), "Market"] = "points"
        all_df.loc[(all_df["Market"] == "BLK") & (all_df["League"] == "NHL"), "Market"] = "blocked"
        all_df.dropna(subset="Market", inplace=True, ignore_index=True)
    else:
        all_df = pd.DataFrame()

    new_preds = []
    if not all_df.empty:
        for key, grp in all_df.groupby(pred_key):
            player, league, date, market = key
            latest = grp.iloc[-1]
            offers = []
            for _, r in grp.iterrows():
                if pd.isna(r.get("Line")):
                    continue
                offers.append(
                    (
                        float(r["Line"]),
                        float(r.get("Boost", 1)),
                        str(r.get("Platform", "")),
                        str(r.get("Bet", "")),
                        float(r["Model P"]) if pd.notna(r.get("Model P")) else np.nan,
                        float(r["Books P"]) if pd.notna(r.get("Books P")) else np.nan,
                        # Closing fields are filled by `reflect` once the game
                        # locks; left NaN at prediction time.
                        np.nan,
                        np.nan,
                        np.nan,
                    )
                )
            row = {
                "Player": player,
                "League": league,
                "Date": date,
                "Market": market,
                "Offers": offers,
            }
            for col in pred_level_cols:
                row[col] = latest.get(col, np.nan)
            new_preds.append(row)

    new_df = pd.DataFrame(new_preds)

    if not history.empty and not new_df.empty:
        history = history.set_index(pred_key)
        new_df = new_df.set_index(pred_key)

        for idx in new_df.index:
            if idx in history.index:
                old_offers = history.at[idx, "Offers"]
                if not isinstance(old_offers, list):
                    old_offers = []
                history.at[idx, "Offers"] = _merge_offers(old_offers, new_df.at[idx, "Offers"])
                for col in pred_level_cols:
                    val = new_df.at[idx, col]
                    if pd.notna(val):
                        history.at[idx, col] = val
            else:
                history.loc[idx] = new_df.loc[idx]

        history = history.reset_index()
    elif not new_df.empty:
        history = new_df

    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    gameDates = pd.to_datetime(history.Date).dt.date
    history = history.loc[
        (datetime.datetime.today().date() - datetime.timedelta(days=_HISTORY_RETENTION_DAYS))
        <= gameDates
    ]
    write_history(history)

    logger.info("Success!")


if __name__ == "__main__":
    main()
