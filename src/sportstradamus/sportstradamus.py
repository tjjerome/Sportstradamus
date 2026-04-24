from sportstradamus.books import get_sleeper, get_ud
from sportstradamus.helpers import (
    Archive,
    banned,
    fused_loc,
    get_ev,
    get_odds,
    set_model_start_values,
    stat_cv,
    stat_map,
    stat_std,
    stat_zi,
)
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA

archive = Archive()
import datetime
import importlib.resources as pkg_resources
import os.path
import pickle
import re
import warnings
from functools import partialmethod
from itertools import combinations
from math import comb
from operator import itemgetter

import click
import gspread
import line_profiler
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import expit, logit
from scipy.stats import gamma, hmean, multivariate_normal, nbinom, norm, poisson, skellam
from tqdm import tqdm

from sportstradamus import creds, data

pd.set_option("mode.chained_assignment", None)
pd.set_option("future.no_silent_downcasting", True)
os.environ["LINE_PROFILE"] = "0"


@click.command()
@click.option("--progress/--no-progress", default=True, help="Display progress bars")
@line_profiler.profile
def main(progress):
    # Initialize tqdm based on the value of 'progress' flag
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not progress))

    # Authorize the gspread API
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]
    cred = None

    # Check if token.json file exists and load credentials
    if os.path.exists(pkg_resources.files(creds) / "token.json"):
        cred = Credentials.from_authorized_user_file(
            (pkg_resources.files(creds) / "token.json"), SCOPES
        )

    # If no valid credentials found, let the user log in
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                (pkg_resources.files(creds) / "credentials.json"), SCOPES
            )
            cred = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open((pkg_resources.files(creds) / "token.json"), "w") as token:
            token.write(cred.to_json())

    gc = gspread.authorize(cred)

    sports = []
    nba = StatsNBA()
    nba.load()
    if datetime.datetime.today().date() > (nba.season_start - datetime.timedelta(days=7)):
        sports.append("NBA")
    # mlb = StatsMLB()
    # mlb.load()
    # if datetime.datetime.today().date() > (mlb.season_start - datetime.timedelta(days=7)):
    #     sports.append("MLB")
    # # nhl = StatsNHL()
    # nhl.load()
    # if datetime.datetime.today().date() > (nhl.season_start - datetime.timedelta(days=7)):
    #     sports.append("NHL")
    nfl = StatsNFL()
    nfl.load()
    if datetime.datetime.today().date() > (nfl.season_start - datetime.timedelta(days=7)):
        sports.append("NFL")
    wnba = StatsWNBA()
    wnba.load()
    if datetime.datetime.today().date() > (wnba.season_start - datetime.timedelta(days=7)):
        sports.append("WNBA")

    """
    Start gathering player stats
    """
    stats = {}
    if "NBA" in sports:
        nba.update()
        stats.update({"NBA": nba})
    # if "MLB" in sports:
    #     mlb.update()
    #     stats.update({"MLB": mlb})
    # if "NHL" in sports:
    #     nhl.update()
    #     stats.update({"NHL": nhl})
    if "NFL" in sports:
        nfl.update()
        stats.update({"NFL": nfl})
    if "WNBA" in sports:
        wnba.update()
        stats.update({"WNBA": wnba})

    all_offers = []
    parlay_df = pd.DataFrame()

    # Underdog

    try:
        ud_dict = get_ud()
        ud_offers, ud5 = process_offers(ud_dict, "Underdog", stats)
        save_data(
            ud_offers.drop(
                columns=[
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
                ],
                errors="ignore",
            ),
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

    # Sleeper

    try:
        sl_dict = get_sleeper()
        sl_offers, sl5 = process_offers(sl_dict, "Sleeper", stats)
        save_data(
            sl_offers.drop(
                columns=[
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
                ],
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
        # Drop rows with non-finite floats (Inf/-Inf) so gspread can JSON-serialize
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
        # Migrate old flat schema to normalized schema if needed
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

    # Build new predictions from current run
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
    # NHL market normalization
    all_df.loc[(all_df["Market"] == "AST") & (all_df["League"] == "NHL"), "Market"] = "assists"
    all_df.loc[(all_df["Market"] == "PTS") & (all_df["League"] == "NHL"), "Market"] = "points"
    all_df.loc[(all_df["Market"] == "BLK") & (all_df["League"] == "NHL"), "Market"] = "blocked"
    all_df.dropna(subset="Market", inplace=True, ignore_index=True)

    # Group current run into normalized predictions
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

    # Merge with existing history
    if not history.empty and not new_df.empty:
        history = history.set_index(pred_key)
        new_df = new_df.set_index(pred_key)

        # Update existing predictions: overwrite pred-level cols, merge offers
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


@line_profiler.profile
def process_offers(offer_dict, book, stats):
    """Process the offers from the given offer dictionary and match them with player statistics.

    Args:
        offer_dict (dict): Dictionary containing the offers to be processed.
        book (str): Name of the book or platform.
        stats (dict): Dictionary containing player stats.

    Returns:
        list: List of processed offers.

    """
    new_offers = []
    logger.info(f"Processing {book} offers")
    if len(offer_dict) > 0:
        # Calculate the total number of offers to process
        total = sum(sum(len(i) for i in v.values()) for v in offer_dict.values())

        # Display a progress bar
        with tqdm(total=total, desc=f"Matching {book} Offers", unit="offer") as pbar:
            for league, markets in offer_dict.items():
                if league in stats:
                    stat_data = stats.get(league)
                    if stat_data.season_start > datetime.datetime.today().date() - datetime.timedelta(days=14):
                        logger.info(f"{league} season has not started, skipping stat matching")
                        continue

                    all_offers = {}
                    for offers in markets.values():
                        all_offers.update({v["Player"]: v for v in offers})

                    all_offers = list(all_offers.values())
                    stat_data.get_depth(all_offers)
                    stat_data.get_volume_stats(all_offers)
                    if league == "MLB":
                        stat_data.get_volume_stats(all_offers, pitcher=True)
                else:
                    # Handle untapped markets where the league is not supported
                    for market, offers in markets.items():
                        archive.add_dfs(offers, book, stat_map[book])
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    archive.add_dfs(offers, book, stat_map[book])
                    # Match the offers with player statistics
                    playerStats = match_offers(offers, league, market, book, stat_data)
                    pbar.update(len(offers))
                    if len(playerStats) == 0:
                        # No matched offers found for the market
                        logger.info(f"{league}, {market} offers not matched")
                    else:
                        modeled_offers = model_prob(
                            offers, league, market, book, stat_data, playerStats
                        )
                        # Add the matched offers to the new_offers list
                        new_offers.extend(modeled_offers)

    offer_df, parlays = find_correlation(new_offers, stats, book)

    logger.info(str(len(offer_df)) + " offers processed")
    return offer_df, parlays


@line_profiler.profile
def find_correlation(offers, stats, platform):
    logger.info("Finding Correlations")

    new_map = stat_map[platform].copy()
    warnings.simplefilter("ignore")

    df = pd.DataFrame(offers)
    versus_mask = df["Player"].str.contains(" vs. ")
    if not df.loc[versus_mask].empty:
        df.loc[versus_mask, "Team"] = df.loc[versus_mask].apply(
            lambda x: x["Team"].split("/")[0 if x["Bet"] == "Over" else 1], axis=1
        )
        df.loc[versus_mask, "Opponent"] = df.loc[versus_mask].apply(
            lambda x: x["Opponent"].split("/")[0 if x["Bet"] == "Over" else 1], axis=1
        )

    combo_mask = df["Team"].apply(lambda x: len(set(x.split("/"))) == 1)
    df.loc[combo_mask, "Team"] = df.loc[combo_mask, "Team"].apply(lambda x: x.split("/")[0])
    df.loc[combo_mask, "Opponent"] = df.loc[combo_mask, "Opponent"].apply(lambda x: x.split("/")[0])

    df["Team Correlation"] = ""
    df["Opp Correlation"] = ""
    parlay_df = pd.DataFrame(
        columns=[
            "Game",
            "Date",
            "League",
            "Platform",
            "Model EV",
            "Books EV",
            "Boost",
            "Rec Bet",
            "Leg 1",
            "Leg 2",
            "Leg 3",
            "Leg 4",
            "Leg 5",
            "Leg 6",
            "Legs",
            "P",
            "PB",
            "Fun",
            "Bet Size",
        ]
    )
    usage_str = {"NBA": "MIN", "WNBA": "MIN", "NFL": "snap pct", "NHL": "TimeShare"}
    tiebreaker_str = {
        "NBA": "USG_PCT short",
        "WNBA": "USG_PCT short",
        "NFL": "route participation short",
        "NHL": "Fenwick short",
    }
    positions = {
        "NBA": ["P", "C", "F", "W", "B"],
        "NFL": ["QB", "WR", "RB", "TE"],
        "NHL": ["C", "W", "D", "G"],
        "WNBA": ["G", "F", "C"],
    }
    payout_table = {  # using equivalent payouts when insured picks are better
        "Underdog": [3.5, 6.5, 10.9, 20.2, 39.9],
        "PrizePicks": [3, 5.3, 10, 20.8, 38.8],
        "Sleeper": [1, 1, 1, 1, 1],
        "ParlayPlay": [1, 1, 1, 1, 1],
        "Chalkboard": [1, 1, 1, 1, 1],
    }

    for league in ["NFL", "NBA", "WNBA", "MLB", "NHL"]:
        league_df = df.loc[df["League"] == league]
        if league_df.empty:
            continue
        c = pd.read_csv(pkg_resources.files(data) / (f"{league}_corr.csv"), index_col=[0, 1, 2])
        c.rename_axis(["team", "market", "correlation"], inplace=True)
        c.columns = ["R"]
        stat_data = stats.get(league)
        team_mod_map = banned[platform][league]["team"]
        opp_mod_map = banned[platform][league]["opponent"]
        if platform == "Underdog":
            league_df["Boost"] = league_df["Boost"] / 1.78

        if league != "MLB":
            league_df["Player position"] = league_df["Player position"].apply(
                lambda x: positions[league][x - 1]
                if isinstance(x, int)
                else [positions[league][i - 1] for i in x]
            )
            combo_df = league_df.loc[league_df.Player.str.contains(r"\+|vs.")]
            league_df = league_df.loc[~league_df.index.isin(combo_df.index)]
            player_df = league_df[["Player", "Team", "Player position"]]
            # for i, row in combo_df.iterrows():
            #     players = row.Player.replace("vs.", "+").split(" + ")
            #     teams = row.Team.split("/")
            #     if len(teams) == 1:
            #         teams = teams*len(players)
            #     pos = row["Player position"]
            #     entries = []
            #     for j in np.arange(len(players)):
            #         entries.append(
            #             {"Player": players[j], "Team": teams[j], "Player position": pos[j]})

            #     player_df = pd.concat([player_df, pd.DataFrame(entries)])

            player_df.drop_duplicates(inplace=True)
            stat_data.profile_market(usage_str[league])
            usage = pd.DataFrame(
                stat_data.playerProfile[[usage_str[league] + " short", tiebreaker_str[league]]]
            )
            usage.reset_index(inplace=True)
            usage.rename(
                columns={
                    "player display name": "Player",
                    "playerName": "Player",
                    "PLAYER_NAME": "Player",
                },
                inplace=True,
            )
            player_df = player_df.merge(usage, how="left").fillna(0).infer_objects(copy=False)
            ranks = (
                player_df.sort_values(tiebreaker_str[league], ascending=False)
                .groupby(["Team", "Player position"])
                .rank(ascending=False, method="first")[usage_str[league] + " short"]
                .astype(int)
            )
            player_df["Player position"] = player_df["Player position"] + ranks.astype(str)
            player_df.index = player_df.Player
            player_df = player_df["Player position"].to_dict()
            league_df["Player position"] = league_df.Player.map(player_df)
            # combo_df["Player position"] = combo_df.Player.apply(
            #     lambda x: [player_df.get(p) for p in x.replace("vs.", "+").split(" + ")])
            # league_df = pd.concat([league_df, combo_df])
        else:
            league_df["Player position"] = league_df["Player position"].apply(
                lambda x: ("B" + str(x) if x > 0 else "P")
                if isinstance(x, int)
                else ["B" + str(i) if i > 0 else "P" for i in x]
            )

        if league == "NHL":
            new_map.update({"Points": "points", "Blocked Shots": "blocked", "Assists": "assists"})
        if league in ("NBA", "WNBA"):
            new_map.update({"Fantasy Points": "fantasy points prizepicks"})

        league_df["cMarket"] = league_df.apply(
            lambda x: [
                x["Player position"]
                + "."
                + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", ""))
            ]
            if isinstance(x["Player position"], str)
            else [
                p
                + "."
                + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", ""))
                for p in x["Player position"]
            ],
            axis=1,
        )

        league_df["Desc"] = (
            league_df[["Player", "Bet", "Line", "Market"]].astype(str).agg(" ".join, axis=1)
        )

        league_df["Desc"] = (
            league_df["Desc"]
            + " - "
            + league_df["Model P"].multiply(100).round(1).astype(str)
            + "%, "
            + league_df["Boost"].round(2).astype(str)
            + "x"
        )

        checked_teams = []
        teams = [team for team in league_df.Team.unique() if "/" not in team]
        for team in tqdm(teams, desc=f"Checking {league} games", unit="game"):
            if team in checked_teams:
                continue
            team_df = league_df.loc[league_df["Team"] == team]
            opp = team_df.Opponent.mode().values[0]
            date = team_df.Date.mode().values[0]
            opp_df = league_df.loc[league_df["Team"] == opp]
            if not opp_df.empty:
                opp_df["cMarket"] = opp_df.apply(
                    lambda x: ["_OPP_" + c for c in x["cMarket"]], axis=1
                )
            split_df = league_df.loc[
                league_df["Team"].str.contains("/")
                & (league_df["Team"].str.contains(team) | league_df["Team"].str.contains(opp))
            ]
            if not split_df.empty:
                split_df["cMarket"] = split_df.apply(
                    lambda x: [
                        ("_OPP_" + c) if (x["Team"].split("/")[d] == opp) else c
                        for d, c in enumerate(x["cMarket"])
                    ],
                    axis=1,
                )
            game_df = pd.concat([team_df, opp_df, split_df])
            game_df.drop_duplicates(subset="Desc", inplace=True)
            checked_teams.append(team)
            checked_teams.append(opp)

            off_weight = 0.75
            team_c = c.loc[team] * 0.5
            def_mask = ["_OPP_" in x and "_OPP_" in y for x, y in team_c.index]
            off_mask = [not x for x in def_mask]
            team_c.loc[off_mask] = team_c.loc[off_mask] * 2 * off_weight
            team_c.loc[def_mask] = team_c.loc[def_mask] * 2 * (1 - off_weight)
            opp_c = c.loc[opp] * 0.5
            def_mask = ["_OPP_" in x and "_OPP_" in y for x, y in opp_c.index]
            off_mask = [not x for x in def_mask]
            opp_c.loc[off_mask] = opp_c.loc[off_mask] * 2 * off_weight
            opp_c.loc[def_mask] = opp_c.loc[def_mask] * 2 * (1 - off_weight)
            opp_c.index = pd.MultiIndex.from_tuples(
                [
                    (f"_OPP_{x}".replace("_OPP__OPP_", ""), f"_OPP_{y}".replace("_OPP__OPP_", ""))
                    for x, y in opp_c.index
                ],
                names=("market", "correlation"),
            )
            c_map = team_c["R"].add(opp_c["R"], fill_value=0).to_dict()

            game_df.reset_index(drop=True, inplace=True)
            game_dict = game_df.to_dict("index")

            idx = game_df.loc[
                (game_df["Books"] > 0.85)
                & (game_df["Books P"] >= 0.25)
                & (game_df["Model"] > 1)
                & (game_df["Model P"] >= 0.3)
            ].sort_values("K", ascending=False)
            idx = idx.drop_duplicates(subset=["Player", "Team", "Market", "Line"])
            idx = idx.groupby("Player").head(6)
            idx = (
                idx.sort_values(["Model", "Books"], ascending=False)
                .groupby("Team")
                .head(30)
                .sort_values(["Team", "Player"])
            )
            idx = (
                idx.sort_values(["Model", "Books"], ascending=False)
                .head(40)
                .sort_values(["Team", "Player"])
            )
            # idx = idx.sort_values(['Model', 'Books', 'Team', 'Player'], ascending=False).head(50)
            bet_df = idx.to_dict("index")

            C = np.eye(len(game_dict))
            M = np.zeros([len(game_dict), len(game_dict)])
            p_model = game_df["Model P"].to_numpy()
            p_books = game_df["Books P"].to_numpy()
            boosts = game_df["Boost"].to_numpy()
            V = p_model * (1 - p_model)
            V = V.reshape(len(game_dict), 1) * V
            V = np.sqrt(V)
            P = p_model.reshape(len(p_model), 1) * p_model
            Vb = p_books * (1 - p_books)
            Vb = Vb.reshape(len(game_dict), 1) * Vb
            Vb = np.sqrt(Vb)
            Pb = p_books.reshape(len(p_books), 1) * p_books
            for i, j in combinations(range(len(game_dict)), 2):
                leg1 = game_dict[i]
                leg2 = game_dict[j]
                cm1 = leg1["cMarket"]
                cm2 = leg2["cMarket"]
                b1 = [leg1["Bet"]] * len(cm1)
                b2 = [leg2["Bet"]] * len(cm2)
                n1 = leg1["Player"]
                n2 = leg2["Player"]

                if "vs." in n1:
                    b1[1] = "Under" if b1[0] == "Over" else "Over"

                if "vs." in n2:
                    b2[1] = "Under" if b2[0] == "Over" else "Over"

                rho = 0
                boost = 0 if (n1 in n2 or n2 in n1) else 1
                for xi, x in enumerate(cm1):
                    for yi, y in enumerate(cm2):
                        increment = c_map.get((x, y), c_map.get((y, x), 0))
                        if b1[xi] != b2[yi]:
                            increment = -increment
                        rho += increment

                        # Modify boost based on conditions
                        mod_map = team_mod_map if ("_OPP_" in x) == ("_OPP_" in y) else opp_mod_map

                        x_key = re.sub(r"[0-9]", "", x)
                        y_key = re.sub(r"[0-9]", "", y)
                        x_key = x_key.replace("_OPP_", "")
                        y_key = y_key.replace("_OPP_", "")

                        modifier = mod_map.get(frozenset([x_key, y_key]), [1, 1])
                        boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]

                C[i, j] = C[j, i] = rho / len(cm1) / len(cm2)
                M[i, j] = M[j, i] = boost

            EV = (
                np.multiply(
                    np.multiply(np.exp(np.multiply(C, V)), P),
                    boosts.reshape(len(boosts), 1) * M * boosts,
                )
                * payout_table[platform][0]
            )
            EVb = (
                np.multiply(
                    np.multiply(np.exp(np.multiply(C, Vb)), Pb),
                    boosts.reshape(len(boosts), 1) * M * boosts,
                )
                * payout_table[platform][0]
            )

            for i, offer in game_df.iterrows():
                indices = (EV[:, i] > 0.95) & (C[:, i] > 0.05) & (EVb[:, i] > 0.9)
                corr = game_df.loc[indices].copy()
                corr["Corr Mult"] = np.exp(C[indices, i] * V[indices, i])
                corr = corr.sort_values("Corr Mult", ascending=False).groupby("Player").head(1)
                same = corr.loc[corr["Team"] == offer["Team"]]
                other = corr.loc[corr["Team"] != offer["Team"]]
                df.loc[
                    (df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"]),
                    "Team Correlation",
                ] = ", ".join(
                    (same["Desc"] + " (" + same["Corr Mult"].round(2).astype(str) + "x)").to_list()
                )
                df.loc[
                    (df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"]),
                    "Opp Correlation",
                ] = ", ".join(
                    (
                        other["Desc"] + " (" + other["Corr Mult"].round(2).astype(str) + "x)"
                    ).to_list()
                )

            info = {
                "Game": "/".join(sorted([team, opp])),
                "Date": date,
                "League": league,
                "Platform": platform,
            }
            best_bets = []
            if not (platform in ["Chalkboard", "ParlayPlay"] and league == "MLB"):
                payouts = payout_table[platform]
                max_boost = 2.5 if platform == "Underdog" else 60
                best_bets = beam_search_parlays(
                    idx,
                    EV,
                    C,
                    M,
                    p_model,
                    p_books,
                    boosts,
                    payouts,
                    max_boost,
                    bet_df,
                    info,
                    team,
                    opp,
                )

                if len(best_bets) > 0:
                    bets = pd.DataFrame(best_bets)

                    df5 = (
                        pd.concat(
                            [
                                bets.sort_values("Model EV", ascending=False).head(300),
                                bets.sort_values("Rec Bet", ascending=False).head(300),
                                bets.sort_values("Fun", ascending=False).head(300),
                            ]
                        )
                        .drop_duplicates()
                        .sort_values("Model EV", ascending=False)
                    )

                    if len(df5) > 5:
                        rho_matrix = np.zeros([len(df5), len(df5)])
                        bets = df5["Bet ID"].to_list()
                        for i, j in tqdm(
                            combinations(range(len(df5)), 2),
                            desc="Filtering...",
                            leave=False,
                            total=comb(len(df5), 2),
                        ):
                            bet1 = bets[i]
                            bet2 = bets[j]
                            rho_cross = np.mean(C[np.ix_(bet1, bet2)])
                            rho_bet1 = np.mean(C[np.ix_(bet1, bet1)])
                            rho_bet2 = np.mean(C[np.ix_(bet2, bet2)])

                            rho_matrix[i, j] = np.clip(
                                rho_cross / np.sqrt(rho_bet1) / np.sqrt(rho_bet2), -1, 0.999
                            )

                        X = np.concatenate([row[i + 1 :] for i, row in enumerate(1 - rho_matrix)])
                        Z = linkage(X, "ward")
                        df5["Family"] = fcluster(Z, 3, criterion="maxclust")

                    else:
                        df5["Family"] = 1

                    parlay_df = pd.concat(
                        [parlay_df, df5.drop(columns="Bet ID")], ignore_index=True
                    )

    if platform == "Underdog":
        payouts = [0, 0, 3.5, 6.5, 6, 10, 25]
        parlay_df["Boost"] = parlay_df["Bet Size"].apply(lambda x: payouts[x]) * parlay_df["Boost"]

    return df.dropna(subset="Model").sort_values("Model", ascending=False), parlay_df


def beam_search_parlays(
    idx, EV, C, M, p_model, p_books, boosts, payouts, max_boost, bet_df, info, team, opp
):
    K = 1000
    max_bet_size = len(payouts) + 1
    leg_indices = sorted(idx.index.to_numpy())
    leg_players = {i: bet_df[i]["Player"] for i in leg_indices}
    leg_teams = {i: bet_df[i]["Team"] for i in leg_indices}

    # Seed: each individual leg as a 1-element tuple
    candidates = [(i,) for i in leg_indices]
    all_results = []

    for target_size in tqdm(
        range(2, max_bet_size + 1), desc=f"{info['League']}, {team}/{opp} Parlays", leave=False
    ):
        next_candidates = []

        for parlay in candidates:
            used_players = {leg_players[i] for i in parlay}
            last_idx = parlay[-1]

            for new_leg in leg_indices:
                if new_leg <= last_idx:
                    continue
                new_player = leg_players[new_leg]
                if new_player in used_players:
                    continue
                if any(new_player in p or p in new_player for p in used_players):
                    continue

                extended = (*parlay, new_leg)

                # Fast screening: geometric mean of pairwise EVs
                n_pairs = target_size * (target_size - 1) // 2
                ev_prod = np.prod(EV[np.ix_(extended, extended)][np.triu_indices(target_size, 1)])
                geo_mean = ev_prod ** (1 / n_pairs)
                if geo_mean < 1.05:
                    continue

                next_candidates.append((extended, geo_mean))

        # Sort and keep top K
        next_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = next_candidates[:K]

        # Full evaluation on candidates at this bet size
        payout_base = payouts[target_size - 2]

        for parlay, _ in top_candidates:
            bet_id = parlay
            bet_size = target_size

            # Must cover both teams
            covers_team = any(team in leg_teams[i] for i in bet_id)
            covers_opp = any(opp in leg_teams[i] for i in bet_id)
            if not (covers_team and covers_opp):
                continue

            boost = np.prod(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]) * np.prod(
                boosts[np.ix_(bet_id)]
            )
            if boost <= 0.7 or boost > max_boost:
                continue

            pb = p_books[np.ix_(bet_id)]
            prev_pb = np.prod(pb) * boost * payout_base
            if prev_pb < 0.9:
                continue

            p = p_model[np.ix_(bet_id)]
            prev_p = np.prod(p) * boost * payout_base
            if prev_p < 1.5:
                continue

            SIG = C[np.ix_(bet_id, bet_id)]
            if any(np.linalg.eigvalsh(SIG) < 0.0001):
                continue

            payout = np.clip(payout_base * boost, 1, 100)
            p = payout * multivariate_normal.cdf(norm.ppf(p), np.zeros(bet_size), SIG)
            pb = p / prev_p * prev_pb
            units = (p - 1) / (payout - 1) / 0.05

            if units < 0.5 or p < 2 or pb < 0.9:
                continue

            bet = itemgetter(*bet_id)(bet_df)
            parlay_dict = info | {
                "Model EV": p,
                "Books EV": pb,
                "Boost": boost,
                "Rec Bet": units,
                "Leg 1": "",
                "Leg 2": "",
                "Leg 3": "",
                "Leg 4": "",
                "Leg 5": "",
                "Leg 6": "",
                "Legs": ", ".join([leg["Desc"] for leg in bet]),
                "Bet ID": bet_id,
                "P": prev_p,
                "PB": prev_pb,
                "Fun": np.sum(
                    [
                        3
                        - (
                            np.abs(leg["Line"])
                            / stat_std.get(info["League"], {}).get(leg["Market"], 1)
                        )
                        if ("H2H" in leg["Desc"])
                        else 2
                        - 1 / stat_cv.get(info["League"], {}).get(leg["Market"], 1)
                        + leg["Line"] / stat_std.get(info["League"], {}).get(leg["Market"], 1)
                        for leg in bet
                        if (leg["Bet"] == "Over") or ("H2H" in leg["Desc"])
                    ]
                ),
                "Bet Size": bet_size,
                "Leg Probs": tuple(bet_df[i]["Model P"] for i in bet_id),
                "Corr Pairs": tuple(SIG[np.triu_indices(bet_size, 1)]),
                "Boost Pairs": tuple(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]),
                "Indep P": float(np.prod(p_model[np.ix_(bet_id)]) * payout),
                "Indep PB": float(np.prod(p_books[np.ix_(bet_id)]) * payout),
            }
            for j in range(bet_size):
                parlay_dict["Leg " + str(j + 1)] = bet[j]["Desc"]

            all_results.append(parlay_dict)

        # Carry forward for next extension round
        candidates = [p for p, _ in top_candidates]

    return all_results


def save_data(df, parlay_df, book, gc):
    """Save offers data to a Google Sheets worksheet.

    Args:
        offers (list): List of offer data.
        book (str): Name of the DFS book.
        gc (gspread.client.Client): Google Sheets client.

    Raises:
        Exception: If there is an error writing the offers to the worksheet.
    """
    if len(df) > 0:
        try:
            df.sort_values("Model", ascending=False, inplace=True)
            mask = (df.Books > 0.95) & (df.Model > 1.02) & (df.Boost <= 2.5)
            if book == "Underdog":
                df["Boost"] = df["Boost"] / 1.78
            # Access the Google Sheets worksheet and update its contents
            wks = gc.open("Sportstradamus").worksheet(book)
            wks.clear()
            wks.update(
                [
                    df.columns.values.tolist(),
                    *df.loc[mask].values.tolist(),
                    *df.loc[~mask].values.tolist(),
                ]
            )
            wks.set_basic_filter()
            # Apply number formatting to the relevant columns
            wks.format("J:K", {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
            wks.format("N:P", {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})
            wks.update_cell(
                1,
                19,
                "Last Updated: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            )

            wks = gc.open("Sportstradamus").worksheet("All Parlays")
            sheet_df = pd.DataFrame(wks.get_all_records())
            if not sheet_df.empty:
                sheet_df = sheet_df.loc[sheet_df.Platform != book]
            if not parlay_df.empty:
                bet_ranks = parlay_df.groupby(["Platform", "Game", "Family"]).rank("first", False)[
                    ["Model EV", "Rec Bet", "Fun"]
                ]
                parlay_df = parlay_df.join(bet_ranks, rsuffix=" Rank")
                sheet_df = pd.concat([sheet_df, parlay_df]).sort_values("Model EV", ascending=False)

            wks.clear()
            sheet_df = sheet_df.drop(
                columns=["Leg Probs", "Corr Pairs", "Boost Pairs", "Indep P", "Indep PB"],
                errors="ignore",
            )
            sheet_df = sheet_df.replace([np.inf, -np.inf], np.nan).fillna("")
            wks.update([sheet_df.columns.values.tolist(), *sheet_df.values.tolist()])

            if not sheet_df.empty:
                wks = gc.open("Sportstradamus").worksheet("Parlay Search")
                wks.update_cell(1, 5, sheet_df.iloc[0]["Platform"])
                wks.update_cell(2, 5, sheet_df.iloc[0]["League"])
                wks.update_cell(3, 5, sheet_df.iloc[0]["Game"])
                wks.update_cell(4, 5, "Highest EV")
                wks.update_cell(7, 2, 1)
                wks.update_cell(7, 5, 1)
                wks.update_cell(7, 8, 1)
                wks.update_cell(
                    1, 10, "Last Updated: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                )

        except Exception:
            # Log the exception if there is an error writing the offers to the worksheet
            logger.exception(f"Error writing {book} offers")


@line_profiler.profile
def match_offers(offers, league, market, platform, stat_data):
    """Matches offers with statistical data and applies various calculations and transformations.

    Args:
        offers (list): List of offers to match.
        league (str): League name.
        market (str): Market name.
        platform (str): Platform name.
        stat_data (obj): Statistical data object.

    Returns:
        list: List of matched offers.
    """
    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
    if league in ("NBA", "WNBA"):
        market = market.replace("underdog", "prizepicks")
    if market in stat_data.gamelog.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            playerStats = stat_data.get_stats(market, offers)
            if playerStats.empty:
                return playerStats

            # Slice to the model's trained schema. expected_columns is embedded
            # in the pickle at training time and is the source of truth.
            filename = "_".join([league, market]).replace(" ", "-")
            filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
            with open(filepath, "rb") as infile:
                expected_cols = pickle.load(infile)["expected_columns"]
            playerStats = playerStats[expected_cols]

            return (
                playerStats[~playerStats.index.duplicated(keep="first")]
                .fillna(0)
                .infer_objects(copy=False)
            )
    else:
        return pd.DataFrame()


def model_prob(offers, league, market, platform, stat_data, playerStats):
    """Matches offers with statistical data and applies various calculations and transformations.

    Args:
        offers (list): List of offers to match.
        league (str): League name.
        market (str): Market name.
        platform (str): Platform name.
        datasets (dict): Dictionary of datasets.
        stat_data (obj): Statistical data object.

    Returns:
        list: List of matched offers.
    """

    def odds_from_boost(o):
        p = [
            0.5 / o.get("Boost_Under", 1)
            if o.get("Boost_Under", 1) > 0
            else 1 - 0.5 / o.get("Boost_Over", 1),
            0.5 / o.get("Boost_Over", 1)
            if o.get("Boost_Over", 1) > 0
            else 1 - 0.5 / o.get("Boost_Under", 1),
        ]
        return p / np.sum(p)

    totals_map = archive.default_totals
    dateMap = {x["Player"]: x["Date"] for x in offers}

    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
    if league in ("NBA", "WNBA"):
        market = market.replace("underdog", "prizepicks")
    filename = "_".join([league, market]).replace(" ", "-")
    filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
    offer_df = pd.DataFrame(offers)
    offer_df.index = offer_df.Player
    if "yards" in market:
        offer_df = offer_df.loc[(offer_df.Player.str.contains("vs.")) | (offer_df.Line > 8)]
    if os.path.isfile(filepath):
        with open(filepath, "rb") as infile:
            filedict = pickle.load(infile)
        cv = filedict["cv"]
        model_weight = filedict["weight"]
        r_book = filedict.get("r_book", None)
        temperature = filedict.get("temperature", None)
        dispersion_cal = filedict.get("dispersion_cal", 1.0)
        shape_ceiling = filedict.get("shape_ceiling")
        dist = filedict["distribution"]
        step = filedict["step"]
        normalized = filedict.get("normalized", False)
        hist_gate = (
            stat_zi.get(league, {}).get(market, 0)
            if dist in ("ZINB", "ZAGamma", "SkewNormal")
            else 0
        )

        if market in stat_data.volume_stats:
            prob_params = pd.DataFrame(index=playerStats.index)
            prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} mean"])
            if f"proj {market} std" in stat_data.playerProfile.columns:
                prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} std"])

            prob_params.rename(
                columns={f"proj {market} mean": "Model EV", f"proj {market} std": "Model Param"},
                inplace=True,
            )

        else:
            model = filedict["model"]

            categories = ["Home", "Player position"]
            if "Player position" not in playerStats.columns:
                categories.remove("Player position")
            for c in categories:
                playerStats[c] = playerStats[c].astype("category")

            set_model_start_values(model, dist, playerStats, normalized=normalized)

            prob_params = model.predict(playerStats, pred_type="parameters")
            prob_params.index = playerStats.index

        prob_params.sort_index(inplace=True)
        playerStats.sort_index(inplace=True)

        if "Defense position" not in playerStats:
            playerStats["Defense position"] = playerStats["Defense avg"]

        evs = []
        for player in playerStats.index:
            ev = archive.get_ev(stat_data.league, market, dateMap.get(player, ""), player)
            line = archive.get_line(stat_data.league, market, dateMap.get(player, ""), player)
            if np.isnan(ev):
                ev = stat_data.check_combo_markets(market, player, dateMap.get(player, ""))
            if line <= 0:
                line = np.max([playerStats.loc[player, "Avg10"], 0.5])
            if (ev <= 0 or np.isnan(ev)) and player in offer_df.index:
                o = offer_df.loc[player]
                if isinstance(o, pd.DataFrame):
                    o = o.iloc[0]
                # get_ev with gate returns the base mean directly
                ev = get_ev(
                    line,
                    odds_from_boost(o.to_dict())[0],
                    stat_cv[stat_data.league].get(market, 1),
                    dist=dist,
                    gate=hist_gate or None,
                )
            elif hist_gate and ev > 0:
                # Archive EVs are already base means (get_ev with gate)
                pass

            evs.append(ev)

        playerStats["Books EV"] = evs
        playerStats["Books STD"] = cv * np.array(evs)

        # NegBin/ZINB parameters need explicit mean computation from total_count and probs
        # PyTorch NegBin: probs = success probability → mean = r*p/(1-p)
        # For ZI dists, Model EV is the BASE mean (not deflated by gate)
        # so fused_loc blends base means from model and book consistently.
        if dist in ("NegBin", "ZINB") and "total_count" in prob_params.columns:
            base_ev = prob_params["total_count"] * prob_params["probs"] / (1 - prob_params["probs"])
            prob_params["Model EV"] = base_ev
            if dist == "ZINB":
                prob_params["Model Gate"] = prob_params["gate"]
            prob_params["Model R"] = prob_params["total_count"]

        # Gamma/ZAGamma parameters need explicit mean computation from alpha (concentration) and beta (rate)
        if dist in ("Gamma", "ZAGamma") and "concentration" in prob_params.columns:
            base_ev = prob_params["concentration"] / prob_params["rate"]
            prob_params["Model EV"] = base_ev
            if dist == "ZAGamma":
                prob_params["Model Gate"] = prob_params["gate"]
            prob_params["Model Alpha"] = prob_params["concentration"]

        # SkewNormal parameters: denormalize and compute EV
        if dist == "SkewNormal" and "loc" in prob_params.columns:
            # Denormalize: loc and scale were normalized by the same denom used in training.
            # Hurdle-filtered markets (hist_gate > 0.05) trained on MeanYr_nonzero; others on MeanYr.
            denom_col = (
                "MeanYr_nonzero"
                if (hist_gate > 0.05 and "MeanYr_nonzero" in playerStats.columns)
                else "MeanYr"
            )
            meanyr_vals = playerStats[denom_col].clip(lower=0.5).values
            loc_abs = prob_params["loc"].values * meanyr_vals
            scale_abs = prob_params["scale"].values * meanyr_vals
            alpha_sn = prob_params["alpha"].values

            # Compute EV from SkewNormal: EV = loc + sigma * delta * sqrt(2/pi)
            delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
            base_ev = loc_abs + scale_abs * delta * np.sqrt(2 / np.pi)

            prob_params["Model EV"] = base_ev
            prob_params["Model Sigma"] = scale_abs
            prob_params["Model Skew"] = alpha_sn
            if hist_gate > 0.02:
                prob_params["Model Gate"] = hist_gate

        offer_df = offer_df.join(playerStats).join(prob_params).reset_index(drop=True)
        offer_df = offer_df.loc[~offer_df[["Books EV", "Model EV"]].isna().all(axis=1)]
        if offer_df.empty:
            return []
        # Compute book-side probability
        if dist == "SkewNormal":
            offer_df["Books"] = offer_df.apply(
                lambda x: 1
                - get_odds(
                    x["Line"],
                    x["Books EV"],
                    dist,
                    cv=cv,
                    step=step,
                    sigma=x["Books EV"] * cv,
                    skew_alpha=0,
                    gate=hist_gate or None,
                ),
                axis=1,
            )
        else:
            offer_df["Books"] = offer_df.apply(
                lambda x: 1
                - get_odds(x["Line"], x["Books EV"], dist, cv, step=step, gate=hist_gate or None),
                axis=1,
            )

        # Clamp model shape to training-time ceiling (safety net)
        if shape_ceiling is not None:
            if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
                offer_df["Model R"] = np.minimum(offer_df["Model R"], shape_ceiling)
            elif dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
                offer_df["Model Alpha"] = np.minimum(offer_df["Model Alpha"], shape_ceiling)

        # Blend model and book predictions via fused_loc (uses base dist type)
        # For ZI/hurdle dists, pass gate_model + hist_gate so fused_loc blends the gate.
        if dist == "SkewNormal":
            _zi_kw = dict(gate_book=hist_gate) if hist_gate > 0.02 else {}
            blended_base_mean, sigma_blend, skew_blend, gate_blend = fused_loc(
                model_weight,
                offer_df["Model EV"].to_numpy(),
                offer_df["Books EV"].fillna(offer_df["Model EV"]).to_numpy(),
                cv,
                "SkewNormal",
                sigma=offer_df["Model Sigma"].to_numpy()
                if "Model Sigma" in offer_df.columns
                else None,
                skew_alpha=offer_df["Model Skew"].to_numpy()
                if "Model Skew" in offer_df.columns
                else None,
                **_zi_kw,
            )
            if gate_blend is not None:
                offer_df["Model EV"] = (1 - gate_blend) * blended_base_mean
                offer_df["Model Gate"] = gate_blend
            else:
                offer_df["Model EV"] = blended_base_mean
            offer_df["Model Sigma"] = sigma_blend
            offer_df["Model Skew"] = skew_blend

        elif dist in ("NegBin", "ZINB"):
            _zi_kw = (
                dict(gate_model=offer_df["Model Gate"].to_numpy(), gate_book=hist_gate)
                if dist == "ZINB" and "Model Gate" in offer_df.columns
                else {}
            )
            r_blend, p_blend, gate_blend = fused_loc(
                model_weight,
                offer_df["Model EV"].to_numpy(),
                offer_df["Books EV"].fillna(offer_df["Model EV"]).to_numpy(),
                cv,
                "NegBin",
                r=offer_df["Model R"].to_numpy() if "Model R" in offer_df.columns else None,
                **_zi_kw,
            )
            blended_base_mean = r_blend * (1 - p_blend) / p_blend
            # Display EV = overall mean (deflated by blended gate for ZI dists)
            if gate_blend is not None:
                offer_df["Model EV"] = (1 - gate_blend) * blended_base_mean
                offer_df["Model Gate"] = gate_blend
            else:
                offer_df["Model EV"] = blended_base_mean
            offer_df["Model R"] = r_blend
        else:
            _zi_kw = (
                dict(gate_model=offer_df["Model Gate"].to_numpy(), gate_book=hist_gate)
                if dist == "ZAGamma" and "Model Gate" in offer_df.columns
                else {}
            )
            alpha_blend, beta_blend, gate_blend = fused_loc(
                model_weight,
                offer_df["Model EV"].to_numpy(),
                offer_df["Books EV"].fillna(offer_df["Model EV"]).to_numpy(),
                cv,
                "Gamma",
                alpha=offer_df["Model Alpha"].to_numpy()
                if "Model Alpha" in offer_df.columns
                else None,
                **_zi_kw,
            )
            blended_base_mean = alpha_blend / beta_blend
            if gate_blend is not None:
                offer_df["Model EV"] = (1 - gate_blend) * blended_base_mean
                offer_df["Model Gate"] = gate_blend
            else:
                offer_df["Model EV"] = blended_base_mean
            offer_df["Model Alpha"] = alpha_blend

        # For display, convert Books EV from base mean to overall mean for ZI/hurdle dists
        if hist_gate and dist in ("ZINB", "ZAGamma", "SkewNormal"):
            offer_df["Books EV"] = (1 - hist_gate) * offer_df["Books EV"]

        # Apply dispersion calibration (keeps mean fixed, adjusts concentration)
        # SkewNormal: CRPS loss handles dispersion — skip post-hoc calibration
        if dispersion_cal != 1.0 and dist != "SkewNormal":
            if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
                offer_df["Model R"] = offer_df["Model R"] * dispersion_cal
            elif dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
                offer_df["Model Alpha"] = offer_df["Model Alpha"] * dispersion_cal

        # Raw distributional probability, then temperature scaling calibration
        _r = (
            offer_df["Model R"].to_numpy()
            if (dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns)
            else None
        )
        _alpha = (
            offer_df["Model Alpha"].to_numpy()
            if (dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns)
            else None
        )
        _sigma = (
            offer_df["Model Sigma"].to_numpy()
            if (dist == "SkewNormal" and "Model Sigma" in offer_df.columns)
            else None
        )
        _skew = (
            offer_df["Model Skew"].to_numpy()
            if (dist == "SkewNormal" and "Model Skew" in offer_df.columns)
            else None
        )
        _gate = (
            offer_df["Model Gate"].to_numpy()
            if (dist in ("ZINB", "ZAGamma", "SkewNormal") and "Model Gate" in offer_df.columns)
            else None
        )
        _model_ev = blended_base_mean  # pass base mean to get_odds (gate handled separately)

        if dist == "SkewNormal":
            _raw_under = get_odds(
                offer_df["Line"].to_numpy(),
                _model_ev,
                dist,
                cv=cv,
                step=step,
                sigma=_sigma,
                skew_alpha=_skew,
                gate=_gate,
            )
        else:
            _raw_under = get_odds(
                offer_df["Line"].to_numpy(),
                _model_ev,
                dist,
                cv,
                alpha=_alpha,
                step=step,
                r=_r,
                gate=_gate,
            )
        _raw_over = 1 - _raw_under
        if temperature is not None:
            _raw_over_clipped = np.clip(_raw_over, 1e-6, 1 - 1e-6)
            _cal_over = expit(logit(_raw_over_clipped) / temperature)
            offer_df["Model Under"] = 1 - _cal_over
        else:
            offer_df["Model Under"] = _raw_under

        offer_df["Model Over"] = 1 - offer_df["Model Under"]

        # Hard cap on confidence before boost
        MAX_CONFIDENCE = 0.90
        offer_df["Model Over"] = offer_df["Model Over"].clip(upper=MAX_CONFIDENCE)
        offer_df["Model Under"] = offer_df["Model Under"].clip(upper=MAX_CONFIDENCE)

        # Keep Model Over/Under as pure probabilities; boost only for Kelly
        offer_df["Model P"] = offer_df[["Model Over", "Model Under"]].max(axis=1)
        offer_df["Bet"] = offer_df[["Model Over", "Model Under"]].idxmax(axis=1).str[6:]

        # Resolve per-direction boost
        if "Boost" in offer_df.columns:
            offer_df.loc[offer_df["Boost"] == 1, ["Boost_Under", "Boost_Over"]] = 1
        # TODO handle combo props here
        offer_df[["Boost_Under", "Boost_Over"]] = offer_df[["Boost_Under", "Boost_Over"]].fillna(
            0
        ).infer_objects(copy=False) * (1.78 if platform == "Underdog" else 1)
        offer_df["Boost"] = offer_df.apply(
            lambda x: (x["Boost_Over"] if x["Bet"] == "Over" else x["Boost_Under"])
            if not np.isnan(x["Boost_Over"])
            else x["Boost"],
            axis=1,
        )

        # Boosted values only for Kelly criterion
        offer_df["Model"] = offer_df["Model P"] * offer_df["Boost"]
        offer_df.loc[(offer_df["Bet"] == "Under"), "Books"] = (
            1 - offer_df.loc[(offer_df["Bet"] == "Under"), "Books"]
        )
        offer_df["Books P"] = offer_df["Books"].fillna(0.5)
        offer_df["Books"] = offer_df["Books P"] * offer_df["Boost"]
        offer_df["K"] = (offer_df["Model"] - 1) / (offer_df["Boost"] - 1)
        offer_df["Distance"] = offer_df["Boost"] / 1.78
        offer_df.loc[offer_df["Distance"] < 1, "Distance"] = (
            1 / offer_df.loc[offer_df["Distance"] < 1, "Distance"]
        )
        offer_df = (
            offer_df.loc[offer_df["Boost"] <= 3.65]
            .sort_values("Distance", ascending=True)
            .groupby("Player")
            .head(3)
        )

        offer_df["Avg 5"] = offer_df["Avg5"] - offer_df["Line"]
        offer_df["Avg H2H"] = offer_df["AvgH2H"] - offer_df["Line"]
        offer_df.loc[offer_df["H2HPlayed"] == 0, "Avg H2H"] = 0
        offer_df["O/U"] = offer_df["Total"] / totals_map.get(league, 1)
        offer_df["DVPOA"] = offer_df["Defense position"]
        if "Player position" not in offer_df:
            offer_df["Player position"] = -1

        offer_df["Player position"] = offer_df["Player position"].astype("category")
        offer_df["Player position"] = (
            offer_df["Player position"].cat.set_categories(range(-1, 5)).fillna(-1).astype(int)
        )
        if dist in ("NegBin", "ZINB"):
            offer_df["Model Param"] = offer_df["Model R"]
        elif dist == "SkewNormal":
            offer_df["Model Param"] = offer_df["Model Sigma"]
        else:
            offer_df["Model Param"] = offer_df["Model Alpha"]

        # Distribution parameters for dashboard reconstruction
        offer_df["Dist"] = dist
        offer_df["CV"] = cv
        offer_df["Gate"] = offer_df.get("Model Gate", np.nan)
        offer_df["Temperature"] = temperature
        offer_df["Disp Cal"] = dispersion_cal
        offer_df["Step"] = step

        return offer_df[
            [
                "League",
                "Date",
                "Team",
                "Opponent",
                "Player",
                "Market",
                "Line",
                "Boost",
                "Bet",
                "Books",
                "Model",
                "Avg 5",
                "Avg H2H",
                "Moneyline",
                "O/U",
                "DVPOA",
                "Player position",
                "Model EV",
                "Model Param",
                "Model P",
                "Books EV",
                "Books P",
                "K",
                "Dist",
                "CV",
                "Gate",
                "Temperature",
                "Disp Cal",
                "Step",
            ]
        ].to_dict("records")

    else:
        logger.warning(f"{filename} missing")
        return []

    for o in tqdm(offers, leave=False):  # TODO try this with pd array funcs
        players = [o["Player"]]
        if "+" in o["Player"] or "vs." in o["Player"]:
            players = o["Player"].replace("vs.", "+").split("+")
            players = [player.strip() for player in players]
            stats = []
            for player in players:
                # if len(player.split(" ")[0].replace(".", "")) <= 2:
                #     if league == "NFL":
                #         nameStr = 'player display name'
                #         namePos = 1
                #     elif league == "NBA":
                #         nameStr = 'PLAYER_NAME'
                #         namePos = 2
                #     elif league == "MLB":
                #         nameStr = "playerName"
                #         namePos = 3
                #     elif league == "NHL":
                #         nameStr = "playerName"
                #         namePos = 7
                #     name_df = stat_data.gamelog.loc[stat_data.gamelog[nameStr].str.contains(player.split(
                #         " ")[1]) & stat_data.gamelog[nameStr].str.startswith(player.split(" ")[0][0])]
                #     if name_df.empty:
                #         pass
                #     else:
                #         players[i] = name_df.iloc[0, namePos]
                #         player = name_df.iloc[0, namePos]

                if player not in playerStats.index:
                    stats.append(0)
                else:
                    stats.append(playerStats.loc[player])

            if any([type(s) is int for s in stats]):
                logger.warning(f"{o['Player']}, {market} stat error")
                continue
        else:
            stats = playerStats.loc[o["Player"]] if o["Player"] in playerStats.index else 0
            if type(stats) is int:
                logger.warning(f"{o['Player']}, {market} stat error")
                continue

        if all([player in playerStats.index for player in players]):
            if "+" in o["Player"]:
                ev1 = archive.get_ev(o["League"], market, o["Date"], players[0])
                ev2 = archive.get_ev(o["League"], market, o["Date"], players[1])

                if not np.isnan(ev1) and not np.isnan(ev2):
                    ev = ev1 + ev2
                    if dist == "Poisson":
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [poisson.cdf(line[0], ev), poisson.sf(line[1], ev)]
                    else:
                        line = o["Line"]
                        p = [norm.cdf(line, ev, ev * cv), norm.sf(line, ev, ev * cv)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2

                params = []
                for player in players:
                    params.append(prob_params.loc[player])

                if dist in ("NegBin", "ZINB"):
                    r1_raw, p1_raw = params[0]["total_count"], params[0]["probs"]
                    r2_raw, p2_raw = params[1]["total_count"], params[1]["probs"]
                    # Blend each player's r with r_book via geometric mean
                    if r_book is not None:
                        r1 = np.exp(
                            model_weight * np.log(r1_raw) + (1 - model_weight) * np.log(r_book)
                        )
                        r2 = np.exp(
                            model_weight * np.log(r2_raw) + (1 - model_weight) * np.log(r_book)
                        )
                    else:
                        r1, r2 = r1_raw, r2_raw
                    r_sum = r1 + r2
                    # Convert PyTorch probs to scipy convention (1 - probs) before averaging
                    p1_scipy = 1 - p1_raw
                    p2_scipy = 1 - p2_raw
                    p_sum = (r1 * p1_scipy + r2 * p2_scipy) / (r1 + r2)
                    under = nbinom.cdf(int(o["Line"]), r_sum, p_sum)
                    push = nbinom.pmf(int(o["Line"]), r_sum, p_sum)
                    under -= push / 2
                elif dist in ("Gamma", "ZAGamma"):
                    # Gamma combo: sum of two independent Gammas
                    alpha1 = params[0]["concentration"]
                    alpha2 = params[1]["concentration"]
                    beta1 = params[0]["rate"]
                    beta2 = params[1]["rate"]
                    ev_sum = alpha1 / beta1 + alpha2 / beta2
                    var_sum = alpha1 / beta1**2 + alpha2 / beta2**2
                    alpha_sum = ev_sum**2 / var_sum
                    high = np.floor((o["Line"] + step) / step) * step
                    low = np.ceil((o["Line"] - step) / step) * step
                    under = gamma.cdf(high, alpha_sum, scale=ev_sum / alpha_sum)
                    push = under - gamma.cdf(low, alpha_sum, scale=ev_sum / alpha_sum)
                    under = under - push / 2

            elif "vs." in o["Player"]:
                ev1 = archive.get_ev(o["League"], market, o["Date"], players[0])
                ev2 = archive.get_ev(o["League"], market, o["Date"], players[1])
                if not np.isnan(ev1) and not np.isnan(ev2):
                    if dist == "Poisson":
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [skellam.cdf(line[0], ev2, ev1), skellam.sf(line[1], ev2, ev1)]
                    else:
                        line = o["Line"]
                        p = [
                            norm.cdf(-line, ev1 - ev2, (ev1 + ev2) * cv),
                            norm.sf(-line, ev1 - ev2, (ev1 + ev2) * cv),
                        ]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2

                params = []
                for player in players:
                    params.append(prob_params.loc[player])

                if dist in ("NegBin", "ZINB"):
                    # PyTorch NegBin: mean = r * probs / (1 - probs)
                    mu1 = params[0]["total_count"] * params[0]["probs"] / (1 - params[0]["probs"])
                    mu2 = params[1]["total_count"] * params[1]["probs"] / (1 - params[1]["probs"])
                    under = skellam.cdf(o["Line"], mu2, mu1)
                    push = skellam.pmf(o["Line"], mu2, mu1)
                    under -= push / 2
                elif dist in ("Gamma", "ZAGamma"):
                    # Gamma vs: difference of two independent Gammas approximated
                    alpha1 = params[0]["concentration"]
                    beta1 = params[0]["rate"]
                    alpha2 = params[1]["concentration"]
                    beta2 = params[1]["rate"]
                    diff_mean = alpha1 / beta1 - alpha2 / beta2
                    diff_std = np.sqrt(alpha1 / beta1**2 + alpha2 / beta2**2)
                    high = np.floor((-o["Line"] + step) / step) * step
                    low = np.ceil((-o["Line"] - step) / step) * step
                    under = norm.cdf(high, diff_mean, diff_std)
                    push = under - norm.cdf(low, diff_mean, diff_std)
                    under = under - push / 2

            else:
                if stats["Line"] != o["Line"]:
                    ev = get_ev(stats["Line"], 1 - stats["Odds"], cv, dist=dist)
                    p = 1 - get_odds(o["Line"], ev, dist, cv, step=step)
                    if np.isnan(p):
                        stats["Odds"] = 0
                    else:
                        stats["Odds"] = p

                if (stats["Odds"] == 0) or (stats["Odds"] == 0.5):
                    p = [
                        0.5 / o.get("Boost_Under", 1)
                        if o.get("Boost_Under", 1) > 0
                        else 1 - 0.5 / o.get("Boost_Over", 1),
                        0.5 / o.get("Boost_Over", 1)
                        if o.get("Boost_Over", 1) > 0
                        else 1 - 0.5 / o.get("Boost_Under", 1),
                    ]
                    p = p / np.sum(p)
                    stats["Odds"] = p[1]
                else:
                    p = [1 - stats["Odds"], stats["Odds"]]

                params = prob_params.loc[o["Player"]]
                if dist in ("NegBin", "ZINB"):
                    r = params["total_count"]
                    # Convert PyTorch probs to scipy convention: p_scipy = 1 - probs_torch
                    p_scipy = 1 - params["probs"]
                    under = nbinom.cdf(int(o["Line"]), r, p_scipy)
                    push = nbinom.pmf(int(o["Line"]), r, p_scipy)
                    under -= push / 2
                elif dist in ("Gamma", "ZAGamma"):
                    alpha = params["concentration"]
                    beta = params["rate"]
                    high = np.floor((o["Line"] + step) / step) * step
                    low = np.ceil((o["Line"] - step) / step) * step
                    under = gamma.cdf(high, alpha, scale=1 / beta)
                    push = under - gamma.cdf(low, alpha, scale=1 / beta)
                    under = under - push / 2

            proba = filt.predict_proba([[2 * (1 - under) - 1, 2 * p[1] - 1]])[0]

        else:
            if "+" in o["Player"]:
                ev1 = (
                    get_ev(stats[0]["Line"], 1 - stats[0]["Odds"], cv, dist=dist)
                    if stats[0]["Odds"] != 0.5
                    else None
                )
                ev2 = (
                    get_ev(stats[1]["Line"], 1 - stats[1]["Odds"], cv, dist=dist)
                    if stats[1]["Odds"] != 0.5
                    else None
                )

                if ev1 and ev2:
                    ev = ev1 + ev2
                    if dist == "Poisson":
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [poisson.cdf(line[0], ev), poisson.sf(line[1], ev)]
                    else:
                        line = o["Line"]
                        p = [norm.cdf(line, ev, ev * cv), norm.sf(line, ev, ev * cv)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2
            elif "vs." in o["Player"]:
                ev1 = (
                    get_ev(stats[0]["Line"], 1 - stats[0]["Odds"], cv, dist=dist)
                    if stats[0]["Odds"] != 0.5
                    else None
                )
                ev2 = (
                    get_ev(stats[1]["Line"], 1 - stats[1]["Odds"], cv, dist=dist)
                    if stats[1]["Odds"] != 0.5
                    else None
                )
                if ev1 and ev2:
                    if dist == "Poisson":
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [skellam.cdf(line[0], ev2, ev1), skellam.sf(line[1], ev2, ev1)]
                    else:
                        line = o["Line"]
                        p = [
                            norm.cdf(-line, ev1 - ev2, (ev1 + ev2) * cv),
                            norm.sf(-line, ev1 - ev2, (ev1 + ev2) * cv),
                        ]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2
            else:
                if stats["Line"] != o["Line"]:
                    ev = get_ev(stats["Line"], 1 - stats["Odds"], cv, dist=dist)
                    stats["Odds"] = get_odds(o["Line"], ev, dist, cv, step)

                if (stats["Odds"] == 0) or (stats["Odds"] == 0.5):
                    p = [
                        0.5 / o.get("Boost_Under", 1)
                        if o.get("Boost_Under", 1) > 0
                        else 1 - 0.5 / o.get("Boost_Over", 1),
                        0.5 / o.get("Boost_Over", 1)
                        if o.get("Boost_Over", 1) > 0
                        else 1 - 0.5 / o.get("Boost_Under", 1),
                    ]
                    p = p / np.sum(p)
                    stats["Odds"] = p[1]
                else:
                    p = [1 - stats["Odds"], stats["Odds"]]

            proba = p

        if (
            proba[1] * o.get("Boost_Over", 1) > proba[0] * o.get("Boost_Under", 1)
            or o.get("Boost", 1) > 1
        ):
            o["Boost"] = o.get("Boost_Over", 1)
            o["Bet"] = "Over"
            o["Books"] = p[1]
            o["Model"] = proba[1]
        else:
            o["Boost"] = o.get("Boost_Under", 1)
            o["Bet"] = "Under"
            o["Books"] = p[0]
            o["Model"] = proba[0]

        o.pop("Boost_Over", None)
        o.pop("Boost_Under", None)

        if "+" in o["Player"]:
            avg5 = np.sum([s["Avg5"] for s in stats])
            o["Avg 5"] = avg5 - o["Line"] if avg5 != 0 else 0
            avgh2h = np.sum([s["AvgH2H"] for s in stats])
            o["Avg H2H"] = avgh2h - o["Line"] if avgh2h != 0 else 0
            o["Moneyline"] = np.mean([s["Moneyline"] for s in stats])
            o["O/U"] = np.mean([s["Total"] for s in stats]) / totals_map.get(o["League"], 1)
            o["DVPOA"] = hmean([s["Defense position"] + 1 for s in stats]) - 1
            if ("Player position" in stats[0]) and ("Player position" in stats[1]):
                o["Player position"] = (
                    int(stats[0]["Player position"]),
                    int(stats[1]["Player position"]),
                )
            else:
                o["Player position"] = (-1, -1)
        elif "vs." in o["Player"]:
            avg5 = stats[0]["Avg5"] - stats[1]["Avg5"]
            o["Avg 5"] = avg5 + o["Line"]
            avgh2h = stats[0]["AvgH2H"] - stats[1]["AvgH2H"]
            o["Avg H2H"] = avgh2h + o["Line"]
            o["Moneyline"] = np.mean([stats[0]["Moneyline"], 1 - stats[1]["Moneyline"]])
            o["O/U"] = np.mean([s["Total"] for s in stats]) / totals_map.get(o["League"], 1)
            o["DVPOA"] = (
                hmean([stats[0]["Defense position"] + 1, 1 - stats[1]["Defense position"]]) - 1
            )
            if ("Player position" in stats[0]) and ("Player position" in stats[1]):
                o["Player position"] = (
                    int(stats[0]["Player position"]),
                    int(stats[1]["Player position"]),
                )
            else:
                o["Player position"] = (-1, -1)
        else:
            o["Avg 5"] = stats["Avg5"] - o["Line"] if stats["Avg5"] != 0 else 0
            o["Avg H2H"] = stats["AvgH2H"] - o["Line"] if stats["AvgH2H"] != 0 else 0
            o["Moneyline"] = stats["Moneyline"]
            o["O/U"] = stats["Total"] / totals_map.get(o["League"], 1)
            o["DVPOA"] = stats["Defense position"]
            if "Player position" in stats:
                o["Player position"] = int(stats["Player position"])
            else:
                o["Player position"] = -1

        if 2 > o.get("Boost", 1) > 0.5:
            new_offers.append(o)

    return new_offers


if __name__ == "__main__":
    main()
