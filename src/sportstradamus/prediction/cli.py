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

from sportstradamus.helpers import Archive, stat_map
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
@line_profiler.profile
def main(progress):
    """Run the full prediction pipeline and snapshot results to parquet."""
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
        from sportstradamus.books import get_ud

        ud_dict = get_ud()
        ud_offers, ud5 = process_offers(ud_dict, "Underdog", stats)
        parlay_df = pd.concat([parlay_df, ud5])
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
        from sportstradamus.books import get_sleeper

        sl_dict = get_sleeper()
        sl_offers, sl5 = process_offers(sl_dict, "Sleeper", stats)
        parlay_df = pd.concat([parlay_df, sl5])
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

    write_current_offers(snapshot_offers, parlay_df, sports, platforms_run)

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
    from sportstradamus.analysis import _merge_offers

    history = read_history()
    if history.empty:
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
                    )
                )
            row = {"Player": player, "League": league, "Date": date, "Market": market, "Offers": offers}
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
