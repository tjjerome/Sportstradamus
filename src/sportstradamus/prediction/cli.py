"""Main ``prophecize`` CLI entrypoint.

Orchestrates the full prediction pipeline:

1. Google Sheets OAuth
2. Load and update ``Stats`` objects for active leagues
3. Scrape DFS offers (Underdog, Sleeper)
4. Score via :func:`process_offers` (feature extraction + distributional model)
5. Write per-platform worksheets and the ``Projections`` / ``All Parlays`` sheets
6. Persist a rolling history of predictions to ``data/history.dat``
"""

from __future__ import annotations

import datetime
import importlib.resources as pkg_resources
import os
import os.path
from functools import partialmethod

import click
import gspread
import line_profiler
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from tqdm import tqdm

from sportstradamus import creds, data
from sportstradamus.books import get_sleeper, get_ud
from sportstradamus.helpers import Archive, stat_map
from sportstradamus.prediction.scoring import process_offers
from sportstradamus.prediction.sheets import save_data
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA

pd.set_option("mode.chained_assignment", None)
pd.set_option("future.no_silent_downcasting", True)
os.environ["LINE_PROFILE"] = "0"

archive = Archive()

_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
]

_EXPORT_DROP_COLS = [
    "Model EV",
    "Model Param",
    "Books EV",
    "Model P",
    "Books P",
    "Dist",
    "CV",
    "Gate",
    "Temperature",
    "Disp Cal",
    "Step",
    "Player position",
    "K",
]


def _authorize_sheets():
    """Return an authorized gspread client, refreshing or re-acquiring creds."""
    token_path = pkg_resources.files(creds) / "token.json"
    cred = None
    if os.path.exists(token_path):
        cred = Credentials.from_authorized_user_file(token_path, _SHEETS_SCOPES)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                pkg_resources.files(creds) / "credentials.json", _SHEETS_SCOPES
            )
            cred = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(cred.to_json())
    return gspread.authorize(cred)


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
    type=click.Choice(["power", "flex", "insurance", "rivals"]),
    default="power",
    help=(
        "Underdog contest variant for parlay scoring. Default 'power' "
        "matches the displayed Boost column historically; 'insurance' "
        "matches the legacy ranking line."
    ),
)
@line_profiler.profile
def main(progress, legacy_correlation, contest_variant):
    """Run the full prediction pipeline and write results to Google Sheets."""
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not progress))

    gc = _authorize_sheets()

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

    all_offers = []
    parlay_df = pd.DataFrame()

    try:
        ud_dict = get_ud()
        ud_offers, ud5 = process_offers(
            ud_dict,
            "Underdog",
            stats,
            contest_variant=contest_variant,
            legacy=legacy_correlation,
        )
        save_data(
            ud_offers.drop(columns=_EXPORT_DROP_COLS, errors="ignore"),
            ud5.drop(columns=["P", "PB"]),
            "Underdog",
            gc,
        )
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
        save_data(
            sl_offers.drop(
                columns=[c for c in _EXPORT_DROP_COLS if c != "Player position"],
                errors="ignore",
            ),
            sl5.drop(columns=["P", "PB"]),
            "Sleeper",
            gc,
        )
        parlay_df = pd.concat([parlay_df, sl5])
        sl_offers["Market"] = sl_offers["Market"].map(stat_map["Sleeper"])
        sl_offers.loc[sl_offers["Bet"] == "Under", "Boost"] = (
            1.78 * 1.78 / sl_offers.loc[sl_offers["Bet"] == "Under", "Boost"]
        )
        sl_offers["Platform"] = "Sleeper"
        all_offers.append(sl_offers)
    except Exception:
        logger.exception("Failed to get Sleeper")

    if len(all_offers) > 0:
        df = pd.concat(all_offers)
        df = df[
            [
                "League",
                "Date",
                "Team",
                "Opponent",
                "Player",
                "Market",
                "Model EV",
                "Model Param",
                "Line",
                "Boost",
                "Bet",
            ]
        ]
        df["Bet"] = "Over"
        df = (
            df.sort_values(["League", "Date", "Team", "Player", "Market"])
            .drop_duplicates(["Player", "League", "Date", "Market"], ignore_index=True, keep="last")
            .dropna()
        )
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        wks = gc.open("Sportstradamus").worksheet("Projections")
        wks.batch_clear(["A:K"])
        wks.update([df.columns.values.tolist(), *df.values.tolist()])

    if not parlay_df.empty:
        parlay_df.sort_values("Model EV", ascending=False, inplace=True)
        parlay_df.drop_duplicates(inplace=True)
        parlay_df.reset_index(drop=True, inplace=True)
        parlay_df[["Legs", "Misses", "Profit"]] = np.nan

        wks = gc.open("Sportstradamus").worksheet("Parlay Search")
        wks.batch_clear(["J7:J50"])

        filepath = pkg_resources.files(data) / "parlay_hist.dat"
        if os.path.isfile(filepath):
            old5 = pd.read_pickle(filepath)
            parlay_df = pd.concat([parlay_df, old5], ignore_index=True).drop_duplicates(
                subset=["Model EV", "Books EV"], ignore_index=True
            )

        parlay_df.to_pickle(filepath)

    archive.write()
    logger.info("Checking historical predictions")
    from sportstradamus.analysis import _merge_offers, _migrate_flat_history

    filepath = pkg_resources.files(data) / "history.dat"
    if os.path.isfile(filepath):
        history = pd.read_pickle(filepath)
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

    all_df = pd.concat(all_offers)
    all_df.loc[(all_df["Market"] == "AST") & (all_df["League"] == "NHL"), "Market"] = "assists"
    all_df.loc[(all_df["Market"] == "PTS") & (all_df["League"] == "NHL"), "Market"] = "points"
    all_df.loc[(all_df["Market"] == "BLK") & (all_df["League"] == "NHL"), "Market"] = "blocked"
    all_df.dropna(subset="Market", inplace=True, ignore_index=True)

    new_preds = []
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
        (datetime.datetime.today().date() - datetime.timedelta(days=365)) <= gameDates
    ]
    history.to_pickle(filepath)

    logger.info("Success!")


if __name__ == "__main__":
    main()
