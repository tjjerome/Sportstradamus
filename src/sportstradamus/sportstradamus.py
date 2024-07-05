from sportstradamus.spiderLogger import logger
from sportstradamus.stats import StatsNBA, StatsMLB, StatsNHL, StatsNFL, StatsWNBA
from sportstradamus.books import get_pp, get_ud, get_sleeper, get_parp
from sportstradamus.helpers import archive, get_ev, get_odds, stat_cv, stat_std, accel_asc, banned
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread
import click
import re
from scipy.stats import poisson, skellam, norm, hmean, multivariate_normal
from scipy.cluster.hierarchy import fcluster, linkage
from math import comb
import numpy as np
import pickle
import json
from sportstradamus import creds, data
from functools import partialmethod
from tqdm import tqdm
import pandas as pd
import os.path
import datetime
import importlib.resources as pkg_resources
import warnings
from itertools import combinations, permutations, product
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import time
import line_profiler

pd.set_option('mode.chained_assignment', None)
os.environ["LINE_PROFILE"] = "0"

@click.command()
@click.option("--progress/--no-progress", default=True, help="Display progress bars")
@line_profiler.profile
def main(progress):
    global untapped_markets
    global stat_map
    # Initialize tqdm based on the value of 'progress' flag
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not progress))

    # Authorize the gspread API
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]
    cred = None

    # Check if token.json file exists and load credentials
    if os.path.exists((pkg_resources.files(creds) / "token.json")):
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

    with open((pkg_resources.files(data) / "stat_map.json"), "r") as infile:
        stat_map = json.load(infile)

    sports = []
    nba = StatsNBA()
    nba.load()
    if datetime.datetime.today().date() > (nba.season_start - datetime.timedelta(days=7)):
        sports.append("NBA")
    mlb = StatsMLB()
    mlb.load()
    if datetime.datetime.today().date() > (mlb.season_start - datetime.timedelta(days=7)):
        sports.append("MLB")
    nhl = StatsNHL()
    nhl.load()
    if datetime.datetime.today().date() > (nhl.season_start - datetime.timedelta(days=7)):
        sports.append("NHL")
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
    if "MLB" in sports:
        mlb.update()
        stats.update({"MLB": mlb})
    if "NHL" in sports:
        nhl.update()
        stats.update({"NHL": nhl})
    if "NFL" in sports:
        nfl.update()
        stats.update({"NFL": nfl})
    if "WNBA" in sports:
        wnba.update()
        stats.update({"WNBA": wnba})

    untapped_markets = []

    all_offers = []
    parlay_df = pd.DataFrame()

    # PrizePicks

    # try:
    #     pp_dict = get_pp()
    #     pp_offers, pp5 = process_offers(
    #         pp_dict, "PrizePicks", stats)
    #     save_data(pp_offers, pp5.drop(columns=["P", "PB"]), "PrizePicks", gc)
    #     parlay_df = pd.concat([parlay_df, pp5])
    #     pp_offers["Market"] = pp_offers["Market"].map(stat_map["PrizePicks"])
        # all_offers.append(pp_offers)
    # except Exception as exc:
    #     logger.exception("Failed to get PrizePicks")

    # Underdog

    try:
        ud_dict = get_ud()
        ud_offers, ud5 = process_offers(
            ud_dict, "Underdog", stats)
        save_data(ud_offers, ud5.drop(columns=["P", "PB"]), "Underdog", gc)
        parlay_df = pd.concat([parlay_df, ud5])
        ud_offers["Market"] = ud_offers["Market"].map(stat_map["Underdog"])
        all_offers.append(ud_offers)
    except Exception as exc:
        logger.exception("Failed to get Underdog")

    # Sleeper

    try:
        sl_dict = get_sleeper()
        sl_offers, sl5 = process_offers(
            sl_dict, "Sleeper", stats)
        save_data(sl_offers, sl5.drop(columns=["P", "PB"]), "Sleeper", gc)
        parlay_df = pd.concat([parlay_df, sl5])
        sl_offers["Market"] = sl_offers["Market"].map(stat_map["Sleeper"])
        all_offers.append(sl_offers)
    except Exception as exc:
        logger.exception("Failed to get Sleeper")

    # ParlayPlay

    try:
        parp_dict = get_parp()
        parp_offers, parp5 = process_offers(
            parp_dict, "ParlayPlay", stats)
        save_data(parp_offers, parp5.drop(columns=["P", "PB"]), "ParlayPlay", gc)
        parlay_df = pd.concat([parlay_df, parp5])
        parp_offers["Market"] = parp_offers["Market"].map(stat_map["ParlayPlay"])
        all_offers.append(parp_offers)
    except Exception as exc:
        logger.exception("Failed to get ParlayPlay")


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
            parlay_df = pd.concat([parlay_df, old5], ignore_index=True).drop_duplicates(subset=["Model EV", "Books EV"], ignore_index=True)

        parlay_df.to_pickle(filepath)

    archive.write()
    logger.info("Checking historical predictions")
    filepath = pkg_resources.files(data) / "history.dat"
    if os.path.isfile(filepath):
        history = pd.read_pickle(filepath)
    else:
        history = pd.DataFrame(
            columns=["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Boost", "Books", "Model", "Result"])

    df = pd.concat(all_offers).drop_duplicates(["Player", "League", "Date", "Market"],
                                                           ignore_index=True)[["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Boost", "Books", "Model"]]
    df.loc[(df['Market'] == 'AST') & (
        df['League'] == 'NHL'), 'Market'] = "assists"
    df.loc[(df['Market'] == 'PTS') & (
        df['League'] == 'NHL'), 'Market'] = "points"
    df.loc[(df['Market'] == 'BLK') & (
        df['League'] == 'NHL'), 'Market'] = "blocked"
    df.dropna(subset='Market', inplace=True, ignore_index=True)
    history = pd.concat([df, history]).drop_duplicates(["Player", "League", "Date", "Market"],
                                                       ignore_index=True)
    if "Result" not in history.columns:
        history["Result"] = np.nan
    gameDates = pd.to_datetime(history.Date).dt.date
    history = history.loc[(datetime.datetime.today(
    ).date() - datetime.timedelta(days=28)) <= gameDates]
    history.to_pickle(filepath)

    if len(untapped_markets) > 0:
        untapped_df = pd.DataFrame(untapped_markets).drop_duplicates()
        wks = gc.open("Sportstradamus").worksheet("Untapped Markets")
        wks.clear()
        wks.update([untapped_df.columns.values.tolist()] +
                   untapped_df.values.tolist())
        wks.set_basic_filter()

    if "NFL" in sports:
        logger.info("Getting NFL Fantasy Rankings")

        filepath = pkg_resources.files(data) / "models/NFL_fantasy-points-underdog.mdl"
        with open(filepath, "rb") as infile:
            filedict = pickle.load(infile)
        model = filedict["model"]

        playerStats, playerData = nfl.get_fantasy()
        categories = ["Home", "Position"]
        for c in categories:
            playerStats[c] = playerStats[c].astype('category')

        prob_params = pd.DataFrame()
        preds = model.predict(
            playerStats, pred_type="parameters")
        preds.index = playerStats.index
        prob_params = pd.concat([prob_params, preds])

        prob_params = prob_params.loc[playerStats.index]
        prob_params['Player'] = playerStats.index
        positions = {0: "QB", 1: "WR", 2: "RB", 3: "TE"}
        prob_params['Position'] = playerStats.Position.map(positions)
        prob_params = prob_params.join(playerData)
        prob_params['Projection'] = prob_params['loc'].round(1)
        prob_params['Floor'] = norm.ppf(.1395, loc=prob_params['loc'],
                                        scale=prob_params['scale'])
        prob_params['Floor'] = prob_params['Floor'].clip(0).round(1)
        prob_params['Ceiling'] = norm.ppf(.9561, loc=prob_params['loc'],
                                        scale=prob_params['scale'])
        prob_params['Ceiling'] = prob_params['Ceiling'].clip(0).round(1)
        prob_params['Rank'] = prob_params.groupby('Position').rank(
            ascending=False, method='dense')['Ceiling']
        prob_params.loc[prob_params['Position'] == "QB", 'VORP12'] = prob_params.loc[prob_params['Position'] == "QB",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "QB") & (prob_params["Rank"] == 13), 'Ceiling'].mean()
        prob_params.loc[prob_params['Position'] == "WR", 'VORP12'] = prob_params.loc[prob_params['Position'] == "WR",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "WR") & (prob_params["Rank"] == 31), 'Ceiling'].mean()
        prob_params.loc[prob_params['Position'] == "RB", 'VORP12'] = prob_params.loc[prob_params['Position'] == "RB",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "RB") & (prob_params["Rank"] == 19), 'Ceiling'].mean()
        prob_params.loc[prob_params['Position'] == "TE", 'VORP12'] = prob_params.loc[prob_params['Position'] == "TE",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "TE") & (prob_params["Rank"] == 13), 'Ceiling'].mean()
        prob_params.loc[prob_params['Position'] == "QB", 'VORP6'] = prob_params.loc[prob_params['Position'] == "QB",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "QB") & (prob_params["Rank"] == 7), 'Ceiling'].mean()
        prob_params.loc[prob_params['Position'] == "WR", 'VORP6'] = prob_params.loc[prob_params['Position'] == "WR",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "WR") & (prob_params["Rank"] == 19), 'Ceiling'].mean()
        prob_params.loc[prob_params['Position'] == "RB", 'VORP6'] = prob_params.loc[prob_params['Position'] == "RB",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "RB") & (prob_params["Rank"] == 10), 'Ceiling'].mean()
        prob_params.loc[prob_params['Position'] == "TE", 'VORP6'] = prob_params.loc[prob_params['Position'] == "TE",
                                                                                    'Ceiling'] - prob_params.loc[(prob_params['Position'] == "TE") & (prob_params["Rank"] == 7), 'Ceiling'].mean()

        prob_params = prob_params[['Player', 'Team', 'Game', 'Position', 'Rank',
                                'Projection', 'Floor', 'Ceiling',
                                'VORP12', 'VORP6']].sort_values("VORP6", ascending=False)

        if len(prob_params) > 0:
            wks = gc.open("Sportstradamus").worksheet("Fantasy")
            wks.clear()
            wks.update([prob_params.columns.values.tolist()] +
                    prob_params.values.tolist())
            wks.set_basic_filter()

    logger.info("Success!")


@line_profiler.profile
def process_offers(offer_dict, book, stats):
    """
    Process the offers from the given offer dictionary and match them with player statistics.

    Args:
        offer_dict (dict): Dictionary containing the offers to be processed.
        book (str): Name of the book or platform.
        stats (dict): Dictionary containing player stats.

    Returns:
        list: List of processed offers.

    """
    global untapped_markets
    global stat_map
    new_offers = []
    logger.info(f"Processing {book} offers")
    if len(offer_dict) > 0:
        # Calculate the total number of offers to process
        total = sum(sum(len(i) for i in v.values())
                    for v in offer_dict.values())

        # Display a progress bar
        with tqdm(total=total, desc=f"Matching {book} Offers", unit="offer") as pbar:
            for league, markets in offer_dict.items():
                if league in stats:
                    stat_data = stats.get(league)
                else:
                    # Handle untapped markets where the league is not supported
                    for market, offers in markets.items():
                        archive.add_dfs(offers, stat_map[book])
                        untapped_markets.append(
                            {"Platform": book, "League": league, "Market": market}
                        )
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    archive.add_dfs(offers, stat_map[book])
                    # Match the offers with player statistics
                    playerStats = match_offers(
                        offers, league, market, book, stat_data, pbar
                    )

                    if len(playerStats) == 0:
                        # No matched offers found for the market
                        logger.info(f"{league}, {market} offers not matched")
                        untapped_markets.append(
                            {"Platform": book, "League": league, "Market": market}
                        )
                    else:
                        modeled_offers = model_prob(
                            offers, league, market, book, stat_data, playerStats
                        )
                        # Add the matched offers to the new_offers list
                        new_offers.extend(modeled_offers)

    new_offers, best_fives = find_correlation(new_offers, stats, book)

    logger.info(str(len(new_offers)) + " offers processed")
    return new_offers, best_fives


@line_profiler.profile
def find_correlation(offers, stats, platform):
    global stat_map
    logger.info("Finding Correlations")

    new_map = stat_map[platform].copy()
    warnings.simplefilter('ignore')

    df = pd.DataFrame(offers)
    versus_mask = df["Player"].str.contains(" vs. ")
    if not df.loc[versus_mask].empty:
        df.loc[versus_mask, "Team"] = df.loc[versus_mask].apply(lambda x: x["Team"].split("/")[0 if x["Bet"]=="Over" else 1], axis=1)
        df.loc[versus_mask, "Opponent"] = df.loc[versus_mask].apply(lambda x: x["Opponent"].split("/")[0 if x["Bet"]=="Over" else 1], axis=1)

    combo_mask = df["Team"].apply(lambda x: len(set(x.split("/"))) == 1)
    df.loc[combo_mask, "Team"] = df.loc[combo_mask, "Team"].apply(lambda x: x.split("/")[0])
    df.loc[combo_mask, "Opponent"] = df.loc[combo_mask, "Opponent"].apply(lambda x: x.split("/")[0])

    df["Correlated Bets"] = ""
    parlay_df = pd.DataFrame(columns=["Game", "Date", "League", "Platform", "Model EV", "Books EV", "Boost", "Rec Bet", "Leg 1", "Leg 2", "Leg 3", "Leg 4", "Leg 5", "Leg 6", "Legs", "P", "PB", "Fun", "Bet Size"])
    usage_str = {
        "NBA": "MIN",
        "WNBA": "MIN",
        "NFL": "snap pct",
        "NHL": "TimeShare"
    }
    tiebreaker_str = {
        "NBA": "USG_PCT short",
        "WNBA": "USG_PCT short",
        "NFL": "route participation short",
        "NHL": "Fenwick short"
    }
    positions = {
        "NBA": ["P", "C", "F", "W", "B"],
        "NFL": ["QB", "WR", "RB", "TE"],
        "NHL": ["C", "W", "D", "G"],
        "WNBA": ['G', 'F', 'C']
    }
    payout_table = { # using equivalent payouts when insured picks are better
        "Underdog": [3, 6, 10.9, 20.2, 39.9],
        "PrizePicks": [3, 5.3, 10, 20.8, 38.8],
        "Sleeper": [1, 1, 1, 1, 1],
        "ParlayPlay": [1, 1, 1, 1, 1],
        "Chalkboard": [1, 1, 1, 1, 1]
    }
    # cutoff_values = { # (m, b)
    #     "Model": (0, 1.1),
    #     "Books": (-0.124, 0.936)
    # }

    for league in ["NFL", "NBA", "WNBA", "MLB", "NHL"]:
        league_df = df.loc[df["League"] == league]
        if league_df.empty:
            continue
        c = pd.read_csv(pkg_resources.files(data) / (f"{league}_corr.csv"), index_col = [0,1,2])
        c.rename_axis(["team", "market", "correlation"], inplace=True)
        c.columns = ["R"]
        # cutoff_model = cutoff_values["Model"]
        # cutoff_books = cutoff_values["Books"]
        stat_data = stats.get(league)
        team_mod_map = banned[platform][league]['team']
        opp_mod_map = banned[platform][league]['opponent']

        if league != "MLB":
            league_df.Position = league_df.Position.apply(lambda x: positions[league][x] if isinstance(
                x, int) else [positions[league][i] for i in x])
            combo_df = league_df.loc[league_df.Player.str.contains("\+|vs.")]
            league_df = league_df.loc[~league_df.index.isin(combo_df.index)]
            player_df = league_df[["Player", "Team", "Position"]]
            for i, row in combo_df.iterrows():
                players = row.Player.replace("vs.", "+").split(" + ")
                teams = row.Team.split("/")
                if len(teams) == 1:
                    teams = teams*len(players)
                pos = row.Position
                entries = []
                for j in np.arange(len(players)):
                    entries.append(
                        {"Player": players[j], "Team": teams[j], "Position": pos[j]})

                player_df = pd.concat([player_df, pd.DataFrame(entries)])

            player_df.drop_duplicates(inplace=True)
            stat_data.profile_market(usage_str[league])
            usage = pd.DataFrame(
                stat_data.playerProfile[[usage_str[league] + " short", tiebreaker_str[league]]])
            usage.reset_index(inplace=True)
            usage.rename(
                columns={"player display name": "Player", "playerName": "Player", "PLAYER_NAME": "Player"}, inplace=True)
            player_df = player_df.merge(usage, how='left').fillna(0)
            ranks = player_df.sort_values(tiebreaker_str[league], ascending=False).groupby(
                ["Team", "Position"]).rank(ascending=False, method='first')[usage_str[league] + " short"].astype(int)
            player_df.Position = player_df.Position + ranks.astype(str)
            player_df.index = player_df.Player
            player_df = player_df.Position.to_dict()
            league_df.Position = league_df.Player.map(player_df)
            combo_df.Position = combo_df.Player.apply(
                lambda x: [player_df.get(p) for p in x.replace("vs.", "+").split(" + ")])
            league_df = pd.concat([league_df, combo_df])
        else:
            league_df.Position = league_df.Position.apply(lambda x: ("B"+str(x) if x>0 else "P") if isinstance(x, int) else ["B"+str(i) if i>0 else "P" for i in x])

        if league == "NHL":
            new_map.update({
                "Points": "points",
                "Blocked Shots": "blocked",
                "Assists": "assists"
            })
        if league == "NBA" or league == "WNBA":
            new_map.update({
                "Fantasy Points": "fantasy points prizepicks"
            })

        league_df["cMarket"] = league_df.apply(lambda x: [x["Position"] + "." + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", ""))] if isinstance(
            x["Position"], str) else [p + "." + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", "")) for p in x["Position"]], axis=1)

        league_df["Desc"] = league_df[[
            "Player", "Bet", "Line", "Market"]].astype(str).agg(" ".join, axis=1)

        league_df["Desc"] = league_df["Desc"] + " - " + \
            league_df["Model"].multiply(100).round(1).astype(str) + "%, " + \
            league_df["Boost"].astype(str) + "x"

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
                    lambda x: ["_OPP_" + c for c in x["cMarket"]], axis=1)
            split_df = league_df.loc[league_df["Team"].str.contains("/") & (league_df["Team"].str.contains(team) | league_df["Team"].str.contains(opp))]
            if not split_df.empty:
                split_df["cMarket"] = split_df.apply(lambda x: [("_OPP_" + c) if (
                    x["Team"].split("/")[d] == opp) else c for d, c in enumerate(x["cMarket"])], axis=1)
            game_df = pd.concat([team_df, opp_df, split_df])
            game_df.drop_duplicates(subset="Desc", inplace=True)
            checked_teams.append(team)
            checked_teams.append(opp)

            off_weight = .75
            team_c = c.loc[team]*.5
            def_mask = ["_OPP_" in x and "_OPP_" in y for x,y in team_c.index]
            off_mask = [not x for x in def_mask]
            team_c.loc[off_mask] = team_c.loc[off_mask]*2*off_weight
            team_c.loc[def_mask] = team_c.loc[def_mask]*2*(1-off_weight)
            opp_c = c.loc[opp]*.5
            def_mask = ["_OPP_" in x and "_OPP_" in y for x,y in opp_c.index]
            off_mask = [not x for x in def_mask]
            opp_c.loc[off_mask] = opp_c.loc[off_mask]*2*off_weight
            opp_c.loc[def_mask] = opp_c.loc[def_mask]*2*(1-off_weight)
            opp_c.index = pd.MultiIndex.from_tuples([(f"_OPP_{x}".replace("_OPP__OPP_", ""), f"_OPP_{y}".replace("_OPP__OPP_", "")) for x, y in opp_c.index], names=("market", "correlation"))
            c_map = team_c["R"].add(opp_c["R"], fill_value=0).to_dict()

            game_df.loc[:, 'Model'] = game_df['Model'].clip(upper=0.65)
            game_df.loc[:, 'Boosted Model'] = game_df['Model'] * game_df["Boost"]
            game_df.loc[:, 'Boosted Books'] = game_df['Books'] * game_df["Boost"]
            game_df.reset_index(drop=True, inplace=True)
            game_dict = game_df.to_dict('index')

            idx = game_df.loc[(game_df["Boosted Books"] > .49) & (game_df["Books"] >= .33) & (game_df["Boosted Model"] > .54) & (game_df["Model"] >= .4)].sort_values(['Boosted Model', 'Boosted Books'], ascending=False)
            idx = idx.drop_duplicates(subset=["Player", "Team", "Market"])
            idx = idx.groupby(['Player', 'Bet']).head(3)
            idx = idx.sort_values(['Boosted Model', 'Boosted Books'], ascending=False).groupby('Team').head(20).sort_values(['Team', 'Player'])
            idx = idx.sort_values(['Boosted Model', 'Boosted Books'], ascending=False).head(38).sort_values(['Team', 'Player'])
            bet_df = idx.to_dict('index')
            team_players = idx.loc[idx.Team == team, 'Player'].unique()
            opp_players = idx.loc[idx.Team == opp, 'Player'].unique()
            combo_players = idx.loc[idx.Team.str.contains("/"), 'Player'].unique()

            C = np.eye(len(game_dict))
            M = np.zeros([len(game_dict), len(game_dict)])
            p_model = game_df.Model.to_numpy()
            p_books = game_df.Books.to_numpy()
            boosts = game_df.Boost.to_numpy()
            V = p_model*(1-p_model)
            V = V.reshape(len(game_dict),1)*V
            V = np.sqrt(V)
            P = p_model.reshape(len(p_model),1)*p_model
            Vb = p_books*(1-p_books)
            Vb = Vb.reshape(len(game_dict),1)*Vb
            Vb = np.sqrt(Vb)
            Pb = p_books.reshape(len(p_books),1)*p_books
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
                        rho += c_map.get((x, y), c_map.get((y, x), 0))
                        if b1[xi] != b2[yi]:
                            rho -= 2*rho
                            
                        # Modify boost based on conditions
                        if ("_OPP_" in x) == ("_OPP_" in y):
                            mod_map = team_mod_map
                        else:
                            mod_map = opp_mod_map

                        x_key = re.sub(r'[0-9]', '', x)
                        y_key = re.sub(r'[0-9]', '', y)
                        x_key = x_key.replace("_OPP_", "")
                        y_key = y_key.replace("_OPP_", "")

                        modifier = mod_map.get(frozenset([x_key, y_key]), [1,1])
                        boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]

                C[i, j] = C[j, i] = rho / len(cm1) / len(cm2)
                M[i, j] = M[j, i] = boost

            EV = np.multiply(np.multiply(np.exp(np.multiply(C,V)),P),boosts.reshape(len(boosts),1)*M*boosts)*payout_table[platform][0]
            EVb = np.multiply(np.multiply(np.exp(np.multiply(C,Vb)),Pb),boosts.reshape(len(boosts),1)*M*boosts)*payout_table[platform][0]

            for i, offer in game_df.iterrows():
                indices = np.logical_and(EV[:,i]>.95, C[:,i]>.05, EVb[:,i]>.9)
                corr = game_df.loc[indices]
                corr["P"] = EV[indices, i]
                corr = corr.sort_values("P", ascending=False).groupby("Player").head(1)
                df.loc[(df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"]), 'Correlated Bets'] = ", ".join(
                    (corr["Desc"] + " (" + corr["P"].round(2).astype(str) + ")").to_list())
            
            player_array = idx['Player'].to_numpy()
            index_array = idx.index.to_numpy()
            player_indices = {player: index_array[player_array == player] for player in set(player_array)}

            info = {
                    "Game": "/".join(sorted([team, opp])),
                    "Date": date,
                    "League": league,
                    "Platform": platform
                    }
            best_bets = []
            if not (platform == "PrizePicks" and league in ["MLB", "NFL", "NHL"]) and not (platform in ["Chalkboard", "ParlayPlay"] and league == "MLB"):
                combos = []
                for bet_size in np.arange(2, len(payout_table[platform]) + 2):
                    team_splits = [x if len(x)==3 else x+[0] for x in accel_asc(bet_size) if 2 <= len(x) <= 3]
                    team_splits = set.union(*[set(permutations(x)) for x in team_splits])
                    for split in team_splits:
                        if split[2] > len(combo_players):
                            continue
                        
                        for i in combinations(team_players, split[0]):
                            for j in combinations(opp_players, split[1]):
                                selected_players = i + j
                                if split[2] != 0:
                                    for k in combinations(combo_players, split[2]):
                                        if len(set([item for row in [l.replace(" vs. ", " + ").split(" + ") for l in k] for item in row]).union(selected_players)) != bet_size+split[2]:
                                            continue
                                        combos.extend(product(*[player_indices[player] for player in selected_players+k]))
                                else:
                                    combos.extend(product(*[player_indices[player] for player in selected_players]))

                thresholds = payout_table[platform]
                max_boost = 60 if platform in ["Sleeper", "ParlayPlay", "Chalkboard"] else 3

                with Pool(processes=4) as p:
                    chunk_size = len(combos) // 4
                    if chunk_size > 0:
                        combos_chunks = [(combos[i:i + chunk_size], p_model, p_books, boosts, M, C, bet_df, info, thresholds, max_boost) for i in range(0, len(combos), chunk_size)]

                        for result in tqdm(p.imap_unordered(compute_bets, combos_chunks), total=len(combos_chunks), desc=f"{league}, {team}/{opp} {bet_size}-Leg Parlays", leave=False):
                            best_bets.extend(result)

                if len(best_bets) > 0:
                    bets = pd.DataFrame(best_bets)
                    
                    df5 = pd.concat([bets.sort_values('Model EV', ascending=False).head(300),
                                     bets.sort_values('Rec Bet', ascending=False).head(300),
                                     bets.sort_values('Fun', ascending=False).head(300)]).drop_duplicates().sort_values('Model EV', ascending=False)
                    
                    if len(df5) > 5:
                        rho_matrix = np.zeros([len(df5), len(df5)])
                        bets = df5["Bet ID"].to_list()
                        for i, j in tqdm(combinations(range(len(df5)), 2), desc="Filtering...", leave=False, total=comb(len(df5),2)):
                            bet1 = bets[i]
                            bet2 = bets[j]
                            rho_cross = np.mean(C[np.ix_(bet1, bet2)])
                            rho_bet1 = np.mean(C[np.ix_(bet1, bet1)])
                            rho_bet2 = np.mean(C[np.ix_(bet2, bet2)])

                            rho_matrix[i, j] = np.clip(rho_cross/np.sqrt(rho_bet1)/np.sqrt(rho_bet2),-1,0.999)

                        X = np.concatenate([row[i+1:] for i, row in enumerate(1-rho_matrix)])
                        Z = linkage(X, 'ward')
                        df5["Family"] = fcluster(Z, 3, criterion='maxclust')
                        
                    else:
                        df5["Family"] = 1

                    parlay_df = pd.concat([parlay_df, df5.drop(columns="Bet ID")], ignore_index=True)
                
    # test_df=pd.concat([parlay_df.sort_values("Model EV", ascending=False).groupby(["Game", "Family"]).head(50),
    #                    parlay_df.sort_values("Fun", ascending=False).groupby(["Game", "Family"]).head(50), 
    #                    parlay_df.sort_values("Rec Bet", ascending=False).groupby(["Game", "Family"]).head(50)]).drop_duplicates()
    # np.polyfit(np.arange(3,7),test_df.groupby("Bet Size")["P"].min(),1)
    # np.polyfit(np.arange(3,7),test_df.groupby("Bet Size")["P"].min(),1)
    return df.drop(columns='Position').dropna().sort_values("Model", ascending=False), parlay_df

def compute_bets(args):
    combos, p_model, p_books, boosts, M, C, bet_df, info, thresholds, max_boost = args
    results = []
    for bet_id in tqdm(combos, leave=False):
        bet_size = len(bet_id)
        threshold = thresholds[bet_size-2]
        boost = np.product(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size,1)])*np.product(boosts[np.ix_(bet_id)])
        if boost <= 0.7 or boost > max_boost:
            continue

        pb = p_books[np.ix_(bet_id)]
        prev_pb = np.product(pb)*boost*threshold
        if prev_pb < .9:
            continue

        p = p_model[np.ix_(bet_id)]
        prev_p = np.product(p)*boost*threshold
        if prev_p < 1.15:
            continue

        SIG = C[np.ix_(bet_id, bet_id)]
        if any(np.linalg.eigvals(SIG)<0.0001):
            continue
        
        payout = np.clip(threshold*boost, 1, 100)
        pb = payout*multivariate_normal.cdf(norm.ppf(pb), np.zeros(bet_size), SIG)
        if pb < 1.01:
            continue
        
        p = payout*multivariate_normal.cdf(norm.ppf(p), np.zeros(bet_size), SIG)
        units = (p - 1)/(payout - 1)/0.05
        
        if units < 0.9 or p < 1.5:
            continue
        
        bet = itemgetter(*bet_id)(bet_df)
        parlay = info | {
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
            "Fun": np.sum([3-(np.abs(leg["Line"])/stat_std.get(info["League"], {}).get(leg["Market"], 1)) if ("H2H" in leg["Desc"]) else 2 - 1/stat_cv.get(info["League"], {}).get(leg["Market"], 1) + leg["Line"]/stat_std.get(info["League"], {}).get(leg["Market"], 1) for leg in bet if (leg["Bet"] == "Over") or ("H2H" in leg["Desc"])]),
            "Bet Size": bet_size
        }
        for i in np.arange(bet_size):
            parlay["Leg " + str(i+1)] = bet[i]["Desc"]
        
        results.append(parlay)

    return results

def save_data(df, parlay_df, book, gc):
    """
    Save offers data to a Google Sheets worksheet.

    Args:
        offers (list): List of offer data.
        book (str): Name of the DFS book.
        gc (gspread.client.Client): Google Sheets client.

    Raises:
        Exception: If there is an error writing the offers to the worksheet.
    """
    if len(df) > 0:

        try:
            df["Books"] = df["Books"]*df["Boost"]
            df["Model"] = df["Model"]*df["Boost"]
            df.sort_values("Model", ascending=False, inplace=True)
            if book in ["Sleeper", "ParlayPlay", "Chalkboard"]:
                mask = (df.Books > .99) & (df.Model > 1.07) & (2.5 >= df.Boost)
            else:
                mask = (df.Books > .54) & (df.Model > .58) & (1.5 >= df.Boost) & (df.Boost >= .75)
            # Access the Google Sheets worksheet and update its contents
            wks = gc.open("Sportstradamus").worksheet(book)
            wks.clear()
            wks.update([df.columns.values.tolist()] + df.loc[mask].values.tolist() + df.loc[~mask].values.tolist())
            wks.set_basic_filter()
            df["Books"] = df["Books"]/df["Boost"]
            df["Model"] = df["Model"]/df["Boost"]

            # Apply number formatting to the relevant columns
            if book in ["Sleeper", "ParlayPlay", "Chalkboard"]:
                wks.format(
                    "J:K", {"numberFormat": {
                        "type": "NUMBER", "pattern": "0.00"}}
                )
            else:
                wks.format(
                    "J:K", {"numberFormat": {
                        "type": "PERCENT", "pattern": "0.00%"}}
                )
            wks.format(
                "N:P", {"numberFormat": {
                    "type": "PERCENT", "pattern": "0.00%"}}
            )
            wks.update_cell(
                1, 18,
                "Last Updated: "
                + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            )

            wks = gc.open("Sportstradamus").worksheet("All Parlays")
            sheet_df = pd.DataFrame(wks.get_all_records())
            if not sheet_df.empty:
                sheet_df = sheet_df.loc[sheet_df.Platform != book]
            if not parlay_df.empty:
                bet_ranks = parlay_df.groupby(["Platform", "Game", "Family"]).rank('first', False)[["Model EV", "Rec Bet", "Fun"]]
                parlay_df = parlay_df.join(bet_ranks, rsuffix=" Rank")
                sheet_df = pd.concat([sheet_df,parlay_df]).sort_values("Model EV", ascending=False)
                
            wks.clear()
            wks.update([sheet_df.columns.values.tolist()] +
                        sheet_df.values.tolist())
            
            wks = gc.open("Sportstradamus").worksheet("Parlay Search")
            wks.update_cell(1, 5, sheet_df.iloc[0]["Platform"])
            wks.update_cell(2, 5, sheet_df.iloc[0]["League"])
            wks.update_cell(3, 5, sheet_df.iloc[0]["Game"])
            wks.update_cell(4, 5, "Highest EV")
            wks.update_cell(7, 2, 1)
            wks.update_cell(7, 5, 1)
            wks.update_cell(7, 8, 1)
            wks.update_cell(
                1, 10, "Last Updated: "
                + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            )
            
        except Exception:
            # Log the exception if there is an error writing the offers to the worksheet
            logger.exception(f"Error writing {book} offers")


@line_profiler.profile
def match_offers(offers, league, market, platform, stat_data, pbar):
    """
    Matches offers with statistical data and applies various calculations and transformations.

    Args:
        offers (list): List of offers to match.
        league (str): League name.
        market (str): Market name.
        platform (str): Platform name.
        stat_data (obj): Statistical data object.
        pbar (obj): Progress bar object.

    Returns:
        list: List of matched offers.
    """
    global stat_map

    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points",
                  "BLK": "blocked"}.get(market, market)
    if league == "NBA":
        market = market.replace("underdog", "prizepicks")
    if market in stat_data.gamelog.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat_data.profile_market(market)
    else:
        return []

    playerStats = []
    playerNames = []

    for o in tqdm(offers, leave=False, disable=not pbar):
        if o["Team"] == "":
            continue

        if "+" in o["Player"] or "vs." in o["Player"]:
            players = o["Player"].replace("vs.", "+").split("+")
            players = [player.strip() for player in players]
            teams = o["Team"].replace(" VS ", "/").split("/")
            teams = [i for i in teams if i]
            if len(teams) < len(players):
                teams = teams*len(players)
            opponents = o["Opponent"].split("/")
            if len(opponents[0]) > 3:
                opponents = list(map(''.join, zip(*[iter(o["Opponent"])]*3)))
            opponents = [i for i in opponents if i]
            if len(opponents) < len(players):
                opponents = opponents*len(players)
            for i, player in enumerate(players):
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
                #         if teams[i] == "":
                #             continue  # TODO: finish this

                lines = list(archive.archive.get(league, {}).get(
                    market, {}).get(o["Date"], {}).get(player, {}).get("Lines", []))
                if len(lines) > 0:
                    line = lines[-1]
                else:
                    line = 0.5

                this_o = o | {
                    "Player": player,
                    "Line": line,
                    "Market": market,
                    "Team": teams[i],
                    "Opponent": opponents[i]
                }

                lines = []

                archive.add_dfs([this_o], stat_map[platform])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    stats = stat_data.get_stats(this_o, date=o["Date"])

                if type(stats) is int:
                    logger.warning(f"{o['Player']}, {market} stat error")
                    pbar.update()
                    continue
                playerStats.append(stats)
                playerNames.append(player)

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stats = stat_data.get_stats(
                    o | {"Market": market}, date=o["Date"])
            if type(stats) is int:
                logger.warning(f"{o['Player']}, {market} stat error")
                pbar.update()
                continue
            playerStats.append(stats)
            playerNames.append(o["Player"])

        pbar.update()

    playerStats = pd.DataFrame(playerStats, index=playerNames)
    return playerStats[~playerStats.index.duplicated(keep='first')].fillna(0)


def model_prob(offers, league, market, platform, stat_data, playerStats):
    """
    Matches offers with statistical data and applies various calculations and transformations.

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
    global stat_map

    totals_map = archive.default_totals

    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points",
                  "BLK": "blocked"}.get(market, market)
    if league == "NBA":
        market = market.replace("underdog", "prizepicks")
    filename = "_".join([league, market]).replace(" ", "-")
    filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
    new_offers = []
    if os.path.isfile(filepath):
        with open(filepath, "rb") as infile:
            filedict = pickle.load(infile)
        model = filedict["model"]
        dist = filedict["distribution"]
        filt = filedict["filter"]
        step = filedict["step"]
        cv = filedict["cv"]

        categories = ["Home", "Position"]
        if "Position" not in playerStats.columns:
            categories.remove("Position")
        for c in categories:
            playerStats[c] = playerStats[c].astype('category')

        sv = playerStats[["MeanYr", "STDYr"]].to_numpy()
        if dist == "Poisson":
            sv = sv[:,0]
            sv.shape = (len(sv),1)

        model.start_values = sv
        prob_params = pd.DataFrame()
        preds = model.predict(
            playerStats, pred_type="parameters")
        preds.index = playerStats.index
        prob_params = pd.concat([prob_params, preds])

        prob_params.sort_index(inplace=True)
        playerStats.sort_index(inplace=True)

    elif market not in stat_data.gamelog.columns:
        return []
    else:
        cv = stat_cv.get(league,{}).get(market,1)
        logger.warning(f"{filename} missing")
        return []

    for o in tqdm(offers, leave=False):
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
            if o["Player"] in playerStats.index:
                stats = playerStats.loc[o["Player"]]
            else:
                stats = 0
            if type(stats) is int:
                logger.warning(f"{o['Player']}, {market} stat error")
                continue

        if all([player in playerStats.index for player in players]):
            if "+" in o["Player"]:
                ev1 = archive.get_ev(o["League"], market, o["Date"], players[0])
                ev2 = archive.get_ev(o["League"], market, o["Date"], players[1])

                if not np.isnan(ev1) and not np.isnan(ev2):
                    ev = ev1 + ev2
                    if cv == 1:
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [poisson.cdf(line[0], ev), poisson.sf(line[1], ev)]
                    else:
                        line = o["Line"]
                        p = [norm.cdf(line, ev, ev*cv),
                            norm.sf(line, ev, ev*cv)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2

                params = []
                for player in players:
                    params.append(prob_params.loc[player])

                if dist == "Poisson":
                    under = poisson.cdf(
                        o["Line"], params[1]["rate"] + params[0]["rate"])
                    push = poisson.pmf(
                        o["Line"], params[1]["rate"] + params[0]["rate"])
                    under -= push/2
                elif dist == "Gaussian":
                    high = np.floor((o["Line"]+step)/step)*step
                    low = np.ceil((o["Line"]-step)/step)*step
                    under = norm.cdf(high,
                                    params[1]["loc"] +
                                    params[0]["loc"],
                                    params[1]["scale"] +
                                    params[0]["scale"])
                    push = under - norm.cdf(low,
                                            params[1]["loc"] +
                                            params[0]["loc"],
                                            params[1]["scale"] +
                                            params[0]["scale"])
                    under = under - push/2

            elif "vs." in o["Player"]:
                ev1 = archive.get_ev(o["League"], market, o["Date"], players[0])
                ev2 = archive.get_ev(o["League"], market, o["Date"], players[1])
                if not np.isnan(ev1) and not np.isnan(ev2):
                    if cv == 1:
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [skellam.cdf(line[0], ev2, ev1),
                            skellam.sf(line[1], ev2, ev1)]
                    else:
                        line = o["Line"]
                        p = [norm.cdf(-line, ev1 - ev2, (ev1 + ev2)*cv),
                            norm.sf(-line, ev1 - ev2, (ev1 + ev2)*cv)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2

                params = []
                for player in players:
                    params.append(prob_params.loc[player])

                if dist == "Poisson":
                    under = skellam.cdf(
                        o["Line"], params[1]["rate"], params[0]["rate"])
                    push = skellam.pmf(
                        o["Line"], params[1]["rate"], params[0]["rate"])
                    under -= push/2
                elif dist == "Gaussian":
                    high = np.floor((-o["Line"]+step)/step)*step
                    low = np.ceil((-o["Line"]-step)/step)*step
                    under = norm.cdf(high,
                                    params[0]["loc"] -
                                    params[1]["loc"],
                                    params[0]["scale"] +
                                    params[1]["scale"])
                    push = under - norm.cdf(low,
                                            params[0]["loc"] -
                                            params[1]["loc"],
                                            params[0]["scale"] +
                                            params[1]["scale"])
                    under = under - push/2

            else:
                if stats["Line"] != o["Line"]:
                    ev = get_ev(stats["Line"], 1-stats["Odds"], cv)
                    p = 1-get_odds(o["Line"], ev, cv, step=step)
                    if np.isnan(p):
                        stats["Odds"] = 0
                    else:
                        stats["Odds"] = p

                if (stats["Odds"] == 0) or (stats["Odds"] == 0.5):
                    p = [0.5/o.get("Boost_Under", 1) if o.get("Boost_Under", 1) > 0 else 1-0.5/o.get("Boost_Over", 1),
                         0.5/o.get("Boost_Over", 1) if o.get("Boost_Over", 1) > 0 else 1-0.5/o.get("Boost_Under", 1)]
                    p = p/np.sum(p)
                    stats["Odds"] = p[1]
                else:
                    p = [1-stats["Odds"], stats["Odds"]]

                params = prob_params.loc[o["Player"]]
                if dist == "Poisson":
                    under = poisson.cdf(o["Line"], params["rate"])
                    push = poisson.pmf(o["Line"], params["rate"])
                    under -= push/2
                elif dist == "Gaussian":
                    high = np.floor((o["Line"]+step)/step)*step
                    low = np.ceil((o["Line"]-step)/step)*step
                    under = norm.cdf(
                        high, params["loc"], params["scale"])
                    push = under - norm.cdf(
                        low, params["loc"], params["scale"])
                    under = under - push/2

            proba = filt.predict_proba([[2*(1-under)-1, 2*p[1]-1]])[0]

        else:
            if "+" in o["Player"]:
                ev1 = get_ev(stats[0]["Line"], 1-stats[0]
                            ["Odds"], cv) if stats[0]["Odds"] != 0.5 else None
                ev2 = get_ev(stats[1]["Line"], 1-stats[1]
                            ["Odds"], cv) if stats[1]["Odds"] != 0.5 else None

                if ev1 and ev2:
                    ev = ev1 + ev2
                    if cv == 1:
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [poisson.cdf(line[0], ev), poisson.sf(line[1], ev)]
                    else:
                        line = o["Line"]
                        p = [norm.cdf(line, ev, ev*cv),
                            norm.sf(line, ev, ev*cv)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2
            elif "vs." in o["Player"]:
                ev1 = get_ev(stats[0]["Line"], 1-stats[0]
                            ["Odds"], cv) if stats[0]["Odds"] != 0.5 else None
                ev2 = get_ev(stats[1]["Line"], 1-stats[1]
                            ["Odds"], cv) if stats[1]["Odds"] != 0.5 else None
                if ev1 and ev2:
                    if cv == 1:
                        line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                        p = [skellam.cdf(line[0], ev2, ev1),
                            skellam.sf(line[1], ev2, ev1)]
                    else:
                        line = o["Line"]
                        p = [norm.cdf(-line, ev1 - ev2, (ev1 + ev2)*cv),
                            norm.sf(-line, ev1 - ev2, (ev1 + ev2)*cv)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                else:
                    p = [0.5] * 2
            else:
                if stats["Line"] != o["Line"]:
                    ev = get_ev(stats["Line"], 1-stats["Odds"], cv)
                    stats["Odds"] = get_odds(o["Line"], ev, cv, step)

                if (stats["Odds"] == 0) or (stats["Odds"] == 0.5):
                    p = [0.5/o.get("Boost_Under", 1) if o.get("Boost_Under", 1) > 0 else 1-0.5/o.get("Boost_Over", 1),
                         0.5/o.get("Boost_Over", 1) if o.get("Boost_Over", 1) > 0 else 1-0.5/o.get("Boost_Under", 1)]
                    p = p/np.sum(p)
                    stats["Odds"] = p[1]
                else:
                    p = [1-stats["Odds"], stats["Odds"]]
            
            proba = p

        if proba[1]*o.get("Boost_Over", 1) > proba[0]*o.get("Boost_Under", 1) or o.get("Boost", 1) > 1:
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
            o["O/U"] = np.mean([s["Total"] for s in stats]) / \
                totals_map.get(o["League"], 1)
            o["DVPOA"] = hmean([s["DVPOA"]+1 for s in stats])-1
            if ("Position" in stats[0]) and ("Position" in stats[1]):
                o["Position"] = (int(stats[0]["Position"]),
                                    int(stats[1]["Position"]))
            else:
                o["Position"] = (-1, -1)
        elif "vs." in o["Player"]:
            avg5 = stats[0]["Avg5"] - stats[1]["Avg5"]
            o["Avg 5"] = avg5 + o["Line"]
            avgh2h = stats[0]["AvgH2H"] - stats[1]["AvgH2H"]
            o["Avg H2H"] = avgh2h + o["Line"]
            o["Moneyline"] = np.mean(
                [stats[0]["Moneyline"], 1-stats[1]["Moneyline"]])
            o["O/U"] = np.mean([s["Total"] for s in stats]) / \
                totals_map.get(o["League"], 1)
            o["DVPOA"] = hmean(
                [stats[0]["DVPOA"]+1, 1-stats[1]["DVPOA"]])-1
            if ("Position" in stats[0]) and ("Position" in stats[1]):
                o["Position"] = (int(stats[0]["Position"]),
                                    int(stats[1]["Position"]))
            else:
                o["Position"] = (-1, -1)
        else:
            o["Avg 5"] = stats["Avg5"] - \
                o["Line"] if stats["Avg5"] != 0 else 0
            o["Avg H2H"] = stats["AvgH2H"] - \
                o["Line"] if stats["AvgH2H"] != 0 else 0
            o["Moneyline"] = stats["Moneyline"]
            o["O/U"] = stats["Total"]/totals_map.get(o["League"], 1)
            o["DVPOA"] = stats["DVPOA"]
            if "Position" in stats:
                o["Position"] = int(stats["Position"])
            else:
                o["Position"] = -1

        if 3 > o.get("Boost", 1) > .5:
            new_offers.append(o)

    return new_offers


if __name__ == "__main__":
    main()
