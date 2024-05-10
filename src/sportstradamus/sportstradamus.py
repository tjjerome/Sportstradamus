from sportstradamus.spiderLogger import logger
from sportstradamus.stats import StatsNBA, StatsMLB, StatsNHL, StatsNFL
from sportstradamus.books import (
    get_caesars,
    get_fd,
    get_pinnacle,
    get_dk,
    get_pp,
    get_ud,
)
from sportstradamus.helpers import archive, get_ev, get_odds, get_active_sports, stat_cv, stat_std, accel_asc
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread
import click
import re
from scipy.stats import poisson, skellam, norm, hmean, multivariate_normal
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
from scipy.cluster.hierarchy import fcluster, linkage
from operator import itemgetter

pd.set_option('mode.chained_assignment', None)

@click.command()
@click.option("--progress/--no-progress", default=True, help="Display progress bars")
@click.option("--books/--no-books", default=False, help="Get data from sportsbooks")
@click.option("--parlays/--no-parlays", default=True, help="Find best 5 leg parlays")
def main(progress, books, parlays):
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

    """"
    Start gathering sportsbook data
    """
    dk_data = []
    fd_data = []
    pin_data = []
    csb_data = []
    if books:
        if "MLB" in sports:
            logger.info("Getting DraftKings MLB lines")
            try:
                dk_data.extend(get_dk(84240, [743, 1024, 1031], "MLB"))  # MLB
            except Exception as exc:
                logger.exception("Failed to get DraftKings MLB lines")

        if "NBA" in sports:
            logger.info("Getting DraftKings NBA lines")
            try:
                dk_data.extend(
                    get_dk(42648, [583, 1215, 1216, 1217, 1218, 1293], "NBA"))  # NBA
            except Exception as exc:
                logger.exception("Failed to get DraftKings NBA lines")

        if "NHL" in sports:
            logger.info("Getting DraftKings NHL lines")
            try:
                dk_data.extend(
                    get_dk(42133, [550, 1064, 1189, 1190], "NHL"))  # NHL
            except Exception as exc:
                logger.exception("Failed to get DraftKings NHL lines")

        if "NFL" in sports:
            logger.info("Getting DraftKings NFL lines")
            try:
                dk_data.extend(
                    get_dk(88808, [1000, 1001, 1003, 1002], "NFL"))  # NFL
            except Exception as exc:
                logger.exception("Failed to get DraftKings NFL lines")

        logger.info(str(len(dk_data)) + " offers found")

        archive.add_books(dk_data, 0, stat_map["DraftKings"])
        archive.write()

        if "MLB" in sports:
            logger.info("Getting FanDuel MLB lines")
            try:
                fd_data.extend(
                    get_fd("mlb", ["pitcher-props", "innings", "batter-props",]))
            except Exception as exc:
                logger.exception("Failed to get FanDuel MLB lines")

        if "NBA" in sports:
            logger.info("Getting FanDuel NBA lines")
            try:
                fd_data.extend(get_fd('nba', ['player-points', 'player-combos', 'player-rebounds',
                                              'player-assists', 'player-threes', 'player-defense']))
            except Exception as exc:
                logger.exception("Failed to get FanDuel NBA lines")

        if "NHL" in sports:
            logger.info("Getting FanDuel NHL lines")
            try:
                fd_data.extend(
                    get_fd('nhl', ['goalie-props', 'shots', 'points-assists', 'goal-scorer']))
            except Exception as exc:
                logger.exception("Failed to get FanDuel NHL lines")

        if "NFL" in sports:
            logger.info("Getting FanDuel NFL lines")
            try:
                fd_data.extend(
                    get_fd('nfl', ['passing-props', 'receiving-props', 'rushing-props', 'td-scorer-props']))
            except Exception as exc:
                logger.exception("Failed to get FanDuel NFL lines")

        logger.info(str(len(fd_data)) + " offers found")

        archive.add_books(fd_data, 1, stat_map["FanDuel"])
        archive.write()

        if "MLB" in sports:
            logger.info("Getting Pinnacle MLB lines")
            try:
                pin_data.extend(get_pinnacle(246))  # MLB
            except Exception as exc:
                logger.exception("Failed to get Pinnacle MLB lines")

        if "NBA" in sports:
            logger.info("Getting Pinnacle NBA lines")
            try:
                pin_data.extend(get_pinnacle(487))  # NBA
            except Exception as exc:
                logger.exception("Failed to get Pinnacle NBA lines")

        if "NHL" in sports:
            logger.info("Getting Pinnacle NHL lines")
            try:
                pin_data.extend(get_pinnacle(1456))  # NHL
            except Exception as exc:
                logger.exception("Failed to get Pinnacle NHL lines")

        if "NFL" in sports:
            logger.info("Getting Pinnacle NFL lines")
            try:
                pin_data.extend(get_pinnacle(889))  # NFL
            except Exception as exc:
                logger.exception("Failed to get Pinnacle NFL lines")

        logger.info(str(len(pin_data)) + " offers found")

        archive.add_books(pin_data, 2, stat_map["Pinnacle"])
        archive.write()

        if "MLB" in sports:
            logger.info("Getting Caesars MLB Lines")
            try:
                sport = "baseball"
                league = "04f90892-3afa-4e84-acce-5b89f151063d"
                csb_data.extend(get_caesars(sport, league))
            except Exception as exc:
                logger.exception("Failed to get Caesars MLB lines")

        if "NBA" in sports:
            logger.info("Getting Caesars NBA Lines")
            try:
                sport = "basketball"
                league = "5806c896-4eec-4de1-874f-afed93114b8c"  # NBA
                csb_data.extend(get_caesars(sport, league))
            except Exception as exc:
                logger.exception("Failed to get Caesars NBA lines")

        if "NHL" in sports:
            logger.info("Getting Caesars NHL Lines")
            try:
                sport = "icehockey"
                league = "b7b715a9-c7e8-4c47-af0a-77385b525e09"
                csb_data.extend(get_caesars(sport, league))
            except Exception as exc:
                logger.exception("Failed to get Caesars NHL lines")

        if "NFL" in sports:
            logger.info("Getting Caesars NFL Lines")
            try:
                sport = "americanfootball"
                league = "007d7c61-07a7-4e18-bb40-15104b6eac92"
                csb_data.extend(get_caesars(sport, league))
            except Exception as exc:
                logger.exception("Failed to get Caesars NFL lines")

        logger.info(str(len(csb_data)) + " offers found")

        archive.add_books(csb_data, 3, stat_map["Caesars"])
        archive.write()

    untapped_markets = []

    pp_offers = pd.DataFrame(columns=["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Books", "Model", "Correct"])
    ud_offers = pd.DataFrame(columns=["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Books", "Model", "Correct"])
    best5 = pd.DataFrame()

    # PrizePicks

    try:
        pp_dict = get_pp(books)
        pp_offers, pp5 = process_offers(
            pp_dict, "PrizePicks", stats, parlays)
        save_data(pp_offers, "PrizePicks", gc)
        best5 = pd.concat([best5, pp5])
        pp_offers["Market"] = pp_offers["Market"].map(stat_map["PrizePicks"])
    except Exception as exc:
        logger.exception("Failed to get PrizePicks")

    # Underdog

    try:
        ud_dict = get_ud()
        ud_offers, ud5 = process_offers(
            ud_dict, "Underdog", stats, parlays)
        save_data(ud_offers, "Underdog", gc)
        best5 = pd.concat([best5, ud5])
        ud_offers["Market"] = ud_offers["Market"].map(stat_map["Underdog"])
    except Exception as exc:
        logger.exception("Failed to get Underdog")

    archive.write()

    if parlays and not best5.empty:
        best5.sort_values("Model EV", ascending=False, inplace=True)
        best5.drop_duplicates(subset=[col for col in best5.columns if col != "Markets"], inplace=True)
        best5.reset_index(drop=True, inplace=True)
        parlay_df = best5.copy()
        best5.drop(columns="Markets", inplace=True)
        bet_ranks = best5.groupby(["Platform", "Game", "Family"]).rank('first', False)[["Model EV", "Rec Bet", "Fun"]]
        best5 = best5.join(bet_ranks, rsuffix=" Rank")

        wks = gc.open("Sportstradamus").worksheet("All Parlays")
        wks.clear()
        wks.update([best5.columns.values.tolist()] +
                    best5.values.tolist())
        
        wks = gc.open("Sportstradamus").worksheet("Parlay Search")
        wks.update_cell(1, 5, best5.iloc[0]["Platform"])
        wks.update_cell(2, 5, best5.iloc[0]["League"])
        wks.update_cell(3, 5, best5.iloc[0]["Game"])
        wks.update_cell(4, 5, "Highest EV")
        wks.update_cell(7, 2, 1)
        wks.update_cell(7, 5, 1)
        wks.update_cell(7, 8, 1)
        wks.update_cell(
            1, 10, "Last Updated: "
            + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        )
        wks.batch_clear(["J7:J50"])


        best5 = pd.concat([best5.loc[best5["Model EV Rank"] == 1],
                           best5.loc[best5["Rec Bet Rank"] == 1], 
                           best5.loc[best5["Fun Rank"] == 1]]).\
                           drop(columns=["Legs", "Family", "Fun", "Bet Size", "Model EV Rank", "Rec Bet Rank", "Fun Rank"]).\
                           sort_values("Model EV", ascending=False)
        best5 = best5.iloc[np.unique(best5.index, return_index=True)[1]]

        wks = gc.open("Sportstradamus").worksheet("Best Parlays")
        wks.clear()
        wks.update([best5.columns.values.tolist()] +
                    best5.values.tolist())
        wks.set_basic_filter()

        parlay_df[["Legs", "Misses", "Profit"]] = np.nan

        filepath = pkg_resources.files(data) / "parlay_hist.dat"
        if os.path.isfile(filepath):
            old5 = pd.read_pickle(filepath)
            parlay_df = pd.concat([parlay_df, old5], ignore_index=True).drop_duplicates(subset=["Model EV", "Books EV"], ignore_index=True)

        parlay_df.to_pickle(filepath)

    logger.info("Checking historical predictions")
    filepath = pkg_resources.files(data) / "history.dat"
    if os.path.isfile(filepath):
        history = pd.read_pickle(filepath)
    else:
        history = pd.DataFrame(
            columns=["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Boost", "Books", "Model", "Result"])

    df = pd.concat([ud_offers, pp_offers]).drop_duplicates(["Player", "League", "Date", "Market"],
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


def process_offers(offer_dict, book, stats, parlays):
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

    new_offers, best_fives = find_correlation(new_offers, stats, book, parlays)

    logger.info(str(len(new_offers)) + " offers processed")
    return new_offers, best_fives


def find_correlation(offers, stats, platform, parlays):
    global stat_map
    logger.info("Finding Correlations")

    filepath = pkg_resources.files(data) / "banned_combos.json"
    with open(filepath, "r") as infile:
        banned = json.load(infile)

    new_map = stat_map[platform].copy()
    warnings.simplefilter('ignore')

    df = pd.DataFrame(offers)
    parlay_df = pd.DataFrame()
    df["Correlated Bets"] = ""
    usage_str = {
        "NBA": "MIN",
        "NFL": "snap pct",
        "NHL": "TimeShare"
    }
    tiebreaker_str = {
        "NBA": "USG_PCT short",
        "NFL": "route participation short",
        "NHL": "Fenwick short"
    }
    positions = {
        "NBA": ["P", "C", "F", "W", "B"],
        "NFL": ["QB", "WR", "RB", "TE"],
        "NHL": ["C", "W", "D", "G"]
    }
    payout_table = { # using equivalent payouts when insured picks are better
        "Underdog": [3, 6, 10.9, 20.2],
        "PrizePicks": [3, 5.3, 10, 20.8, 38.8]
    }
    league_cutoff_values = { # (m, b)
        "NBA": {
            "Model": (0.156, 0.828),
            "Books": (-0.061, 1.144)
        },
        "MLB": {
            "Model": (0.109, 0.861),
            "Books": (-0.060, 1.159)
        },
        "NHL": {
            "Model": (0.135, 0.793),
            "Books": (-0.048, 0.992)
        },
        "NFL": {
            "Model": (0.101, 0.878),
            "Books": (-0.074, 1.142)
        }
    }

    for league in ["NFL", "NBA", "MLB", "NHL"]:
        league_df = df.loc[df["League"] == league]
        if league_df.empty:
            continue
        c = pd.read_csv(pkg_resources.files(data) / (f"{league}_corr.csv"), index_col = [0,1,2])
        c.rename_axis(["team", "market", "correlation"], inplace=True)
        c.columns = ["R"]
        league_cutoff_model = league_cutoff_values[league]["Model"]
        league_cutoff_books = league_cutoff_values[league]["Books"]
        # team_pairs = c.apply(lambda x: [x.market.split(".")[1], x.correlation.split(
        #     ".")[1]] if "_OPP_" not in x.correlation else ["", ""], axis=1).to_list()
        # opp_pairs = c.apply(lambda x: [x.market.split(".")[1], x.correlation.split(
        #     ".")[1]] if "_OPP_" in x.correlation else ["", ""], axis=1).to_list()
        # mask1 = [pair not in banned[platform][league]['team'] and pair[::-1]
        #          not in banned[platform][league]['team'] for pair in team_pairs]
        # mask2 = [pair not in banned[platform][league]['opponent'] and pair[::-1]
        #          not in banned[platform][league]['opponent'] for pair in opp_pairs]

        # c = c.loc[[a and b for a, b in zip(mask1, mask2)]]
        stat_data = stats.get(league)
        banned_team_markets = banned[platform][league]['team']
        banned_opponent_markets = banned[platform][league]['opponent']
        mod_map = banned[platform][league]['modified']

        if league != "MLB":
            league_df.Position = league_df.Position.apply(lambda x: positions[league][x] if isinstance(
                x, int) else [positions[league][i] for i in x])
            combo_df = league_df.loc[league_df.Position.apply(
                lambda x: isinstance(x, list))]
            league_df = league_df.loc[league_df.Position.apply(
                lambda x: not isinstance(x, list))]
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
        if league == "NBA":
            new_map.update({
                "Fantasy Points": "fantasy points prizepicks"
            })

        league_df["cMarket"] = league_df.apply(lambda x: [x["Position"] + "." + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", ""))] if isinstance(
            x["Position"], str) else [p + "." + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", "")) for p in x["Position"]], axis=1)

        league_df["Desc"] = league_df[[
            "Player", "Bet", "Line", "Market"]].astype(str).agg(" ".join, axis=1)

        league_df["Desc"] = league_df["Desc"] + " - " + \
            league_df["Model"].multiply(100).round(1).astype(str) + "% (" + \
            league_df["Boost"].astype(str) + ")"

        checked_teams = []
        teams = [team for team in league_df.Team.unique() if "/" not in team]
        for team in tqdm(teams, desc=f"Checking {league} games", unit="game"):
            if team in checked_teams:
                continue
            team_df = league_df.loc[(league_df["Team"] == team) | (league_df["Team"] == f"{team}/{team}")]
            opp = team_df.Opponent.mode().values[0]
            date = team_df.Date.mode().values[0]
            opp_df = league_df.loc[(league_df["Team"] == opp) | (league_df["Team"] == f"{opp}/{opp}")]
            if not opp_df.empty:
                opp_df["cMarket"] = opp_df.apply(
                    lambda x: ["_OPP_" + c for c in x["cMarket"]], axis=1)
            split_df = league_df.loc[league_df["Team"].str.contains(
                "/") & league_df["Team"].str.contains(team) & league_df["Team"].str.contains(opp)]
            if not split_df.empty:
                split_df["cMarket"] = split_df.apply(lambda x: [("_OPP_" + c) if (
                    x["Team"].split("/")[d] == opp) else c for d, c in enumerate(x["cMarket"])], axis=1)
            game_df = pd.concat([team_df, opp_df, split_df])
            checked_teams.append(team)
            checked_teams.append(opp)
            if platform != "Underdog":
                game_df.loc[:, 'Boost'] = 1

            team_c = c.loc[team]
            opp_c = c.loc[opp]
            opp_c.index = pd.MultiIndex.from_tuples([(f"_OPP_{x}".replace("_OPP__OPP_", ""), f"_OPP_{y}".replace("_OPP__OPP_", "")) for x, y in opp_c.index], names=("market", "correlation"))
            c_map = team_c["R"].add(opp_c["R"], fill_value=0).div(2).to_dict()

            game_df.loc[:, 'Model'] = game_df['Model'].clip(upper=0.65)
            game_df.loc[:, 'Boosted Model'] = game_df['Model'] * game_df["Boost"]
            game_df.loc[:, 'Boosted Books'] = game_df['Books'] * game_df["Boost"]

            idx = game_df.loc[(game_df["Boosted Books"] > .495) & (game_df["Books"] >= .3) & (game_df["Model"] >= .35)].sort_values(['Boosted Model', 'Boosted Books'], ascending=False).groupby(['Player', 'Bet']).head(3)
            idx = idx.sort_values(['Boosted Model', 'Boosted Books'], ascending=False).groupby('Team').head(15)
            idx = idx.sort_values(['Boosted Model', 'Boosted Books'], ascending=False).head(28).sort_values(['Team', 'Player'])
            bet_df = idx.to_dict('index')
            team_players = idx.loc[idx.Team == team, 'Player'].unique()
            opp_players = idx.loc[idx.Team == opp, 'Player'].unique()
            combo_players = idx.loc[idx.Team.apply(lambda x: len(set(x.split("/")))>1), 'Player'].unique()

            best_bets = []
            for bet_size in np.arange(2, len(payout_table[platform]) + 2):
                team_splits = [x if len(x)==3 else x+[0] for x in accel_asc(bet_size) if 2 <= len(x) <= 3]
                team_splits = set.union(*[set(permutations(x)) for x in team_splits])
                combos = []
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
                                    combos.extend(product(*[idx.loc[idx.Player == player].index for player in selected_players+k]))
                            else:
                                combos.extend(product(*[idx.loc[idx.Player == player].index for player in selected_players]))

                threshold = payout_table[platform][bet_size-2]

                for bet_id in tqdm(combos, desc=f"{league}, {team}/{opp} {bet_size}-Leg Parlays", leave=False):
                    bet = itemgetter(*bet_id)(bet_df)

                    p = np.product([leg["Boosted Model"] for leg in bet])
                    pb = np.product([leg["Boosted Books"] for leg in bet])

                    if p*threshold < (league_cutoff_model[0]*bet_size+league_cutoff_model[1]) or pb*threshold < (league_cutoff_books[0]*bet_size+league_cutoff_books[1]):
                        continue

                    markets = [i for j in [leg['cMarket'] for leg in bet] for i in j]
                    team1_markets = [leg.split(".")[1]
                                     for leg in markets if "_OPP_" not in leg]
                    team2_markets = [leg.split(".")[1]
                                     for leg in markets if "_OPP_" in leg]

                    if (any([bc[0] in team1_markets and bc[1] in team1_markets for bc in banned_team_markets]) or
                        any([bc[0] in team2_markets and bc[1] in team2_markets for bc in banned_team_markets]) or
                        any([bc[0] in team1_markets and bc[1] in team2_markets for bc in banned_opponent_markets]) or
                        any([bc[0] in team2_markets and bc[1] in team1_markets for bc in banned_opponent_markets])):
                        continue

                    boost = 1

                    # get correlation matrix
                    bet_indices = np.arange(bet_size)
                    SIG = np.zeros([bet_size, bet_size])

                    # Iterate over combinations of bets
                    for i, j in combinations(bet_indices, 2):
                        cm1 = bet[i]['cMarket']
                        b1 = [bet[i]['Bet']] * len(cm1)
                        if "vs." in bet[i]["Player"]:
                            b1[1] = "Under" if b1[0] == "Over" else "Over"

                        cm2 = bet[j]['cMarket']
                        b2 = [bet[j]['Bet']] * len(cm2)
                        if "vs." in bet[j]["Player"]:
                            b2[1] = "Under" if b2[0] == "Over" else "Over"

                        # Vectorized operation to compute rho for all combinations of x and y
                        rho_matrix = np.zeros((len(cm1), len(cm2)))
                        for xi, x in enumerate(cm1):
                            for yi, y in enumerate(cm2):
                                rho = c_map.get((x, y), c_map.get((y, x), 0))
                                if b1[xi] != b2[yi]:
                                    rho = -rho
                                rho_matrix[xi, yi] = rho

                                # Modify boost based on conditions
                                x_key = re.sub(r'[0-9]', '', x)
                                y_key = re.sub(r'[0-9]', '', y)
                                if "_OPP_" in x_key:
                                    x_key = x_key.replace("_OPP_", "")
                                    if "_OPP_" in y_key:
                                        y_key = y_key.replace("_OPP_", "")
                                    else:
                                        y_key = "_OPP_" + y_key
                                if x_key in mod_map:
                                    modifier = mod_map[x_key].get(y_key, [1,1])
                                    boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]
                                # elif y_key in mod_map:
                                #     modifier = mod_map[y_key].get(x_key, [1,1])
                                #     boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]

                        # Sum rho_matrix to update SIG
                        SIG[i, j] = np.sum(rho_matrix)

                    SIG = SIG + SIG.T + np.eye(bet_size)

                    boost = np.clip(boost, .667, 1.6125) * np.product([leg["Boost"] for leg in bet])
                    payout = np.clip(payout_table[platform][bet_size-2]*boost, 1, 100)

                    try:
                        p = payout*multivariate_normal.cdf([norm.ppf(leg["Model"]) for leg in bet], np.zeros(bet_size), SIG)
                        pb = payout*(3*multivariate_normal.cdf([norm.ppf(leg["Books"]) for leg in bet], np.zeros(bet_size), SIG)+np.product([leg["Books"] for leg in bet]))/4
                    except:
                        continue
                    
                    units = (p - 1)/(payout - 1)/0.05
                    if units >= 0.9 and pb > 1.01:
                        parlay = {
                            "Game": "/".join(sorted([team, opp])),
                            "Date": date,
                            "League": league,
                            "Platform": platform,
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
                            # "Players": {f"{leg['Player']} {leg['Bet']}" for leg in bet},
                            "Markets": [(market, "Under" if leg["Bet"] == "Over" else "Over") if i == 2 and "vs." in leg["Player"] else (market, leg["Bet"]) for leg in bet for i, market in enumerate(leg["cMarket"])],
                            "Fun": np.sum([3-(np.abs(leg["Line"])/stat_std.get(league, {}).get(leg["Market"], 1)) if ("H2H" in leg["Desc"]) else 2 - 1/stat_cv.get(league, {}).get(leg["Market"], 1) + leg["Line"]/stat_std.get(league, {}).get(leg["Market"], 1) for leg in bet if (leg["Bet"] == "Over") or ("H2H" in leg["Desc"])]),
                            "Bet Size": bet_size
                        }
                        for i in np.arange(bet_size):
                            parlay["Leg " + str(i+1)] = bet[i]["Desc"]
                        best_bets.append(parlay)

            df5 = pd.DataFrame(best_bets, columns=['Game', 'Date', 'League', 'Platform', 'Model EV', 'Books EV', 'Boost', 'Rec Bet', 'Leg 1', 'Leg 2', 'Leg 3', 'Leg 4', 'Leg 5', 'Leg 6', 'Legs', 'Markets', 'Fun', 'Bet Size'])
            
            df5.sort_values('Model EV', ascending=False, inplace=True)
            # df5.drop_duplicates('Players', inplace=True)
            # df5.drop(columns="Players", inplace=True)
            df5 = df5.groupby('Bet Size').head(100)
            
            if len(df5) > 5:

                rho_matrix = np.zeros([len(df5), len(df5)])
                bets = df5["Markets"].to_list()
                for i, j in tqdm(combinations(range(len(df5)), 2), desc="Filtering...", leave=False, total=comb(len(df5),2)):
                    bet1 = bets[i]
                    bet2 = bets[j]
                    rho = []
                    for leg1, leg2 in product(bet1, bet2):
                        rho.append(c_map.get((leg1[0], leg2[0]), 0) * (1 if leg1[1]==leg2[1] else -1))

                    rho_matrix[i, j] = np.mean(rho)

                X = np.concatenate([row[i+1:] for i, row in enumerate(1-rho_matrix)])
                Z = linkage(X, 'ward')
                df5["Family"] = fcluster(Z, 3, criterion='maxclust')
            elif len(df5) > 0:
                df5["Family"] = 1

            # df5 = pd.concat([df5.groupby("Family").head(1),
            #                     df5.sort_values("Rec Bet", ascending=False).groupby("Family").head(1), 
            #                     df5.sort_values(["Fun", "Model EV"], ascending=False).groupby("Family").head(1)]).\
            #                     drop(columns=["Players", "Family"])
            # parlay_df = pd.concat([parlay_df, df5.drop_duplicates(subset=["Model EV", "Books EV"])])

            parlay_df = pd.concat([parlay_df, df5], ignore_index=True)

            # Find best pairs
            c_map = pd.Series(c_map)
            for i, offer in tqdm(game_df.iterrows(), desc=f"Correlating {team}/{opp} bets...", leave=False, total=len(game_df)):
                R_map = pd.Series(0, index=c_map.index.levels[1])
                for market in offer.cMarket:
                    if market in c_map:
                        R_map = R_map.add(c_map.loc[market], fill_value=0)

                for market in offer.cMarket:
                    position, stat = tuple(market.split("."))
                    R_map.loc[[x.split(".")[0] == position for x in R_map.index]] = 0
                    if "_OPP_" in position:
                        if len(banned_team_markets) > 0:
                            R_map.loc[R_map.index.str.contains("|".join([x[0] if x[1] == stat else x[1] for x in banned_team_markets if stat in x])) & R_map.index.str.contains("_OPP_")] = 0
                        if len(banned_opponent_markets) > 0:
                            R_map.loc[R_map.index.str.contains("|".join([x[0] if x[1] == stat else x[1] for x in banned_opponent_markets if stat in x])) & ~R_map.index.str.contains("_OPP_")] = 0
                    else:
                        if len(banned_opponent_markets) > 0:
                            R_map.loc[R_map.index.str.contains("|".join([x[0] if x[1] == stat else x[1] for x in banned_opponent_markets if stat in x])) & ~R_map.index.str.contains("_OPP_")] = 0
                        if len(banned_team_markets) > 0:
                            R_map.loc[R_map.index.str.contains("|".join([x[0] if x[1] == stat else x[1] for x in banned_team_markets if stat in x])) & R_map.index.str.contains("_OPP_")] = 0

                pos = R_map.loc[R_map > 0.1].index.to_list()
                neg = R_map.loc[R_map < -0.1].index.to_list()
                R_map = R_map.abs().to_dict()
                pos_df = game_df.loc[game_df.apply(lambda x: x["cMarket"][0] in pos if len(x["cMarket"]) == 1 else False, axis=1) & (game_df.Bet == offer.Bet)]
                neg_df = game_df.loc[game_df.apply(lambda x: x["cMarket"][0] in neg if len(x["cMarket"]) == 1 else False, axis=1) & (game_df.Bet != offer.Bet)]
                if platform == "Underdog":
                    for cm in offer["cMarket"]:
                        pos_df["Boost"] *= pos_df["cMarket"].apply(lambda x: np.product([mod_map.get(re.sub(r'[0-9]', '', i).replace("_OPP_", ""), {}).get("_OPP_"+re.sub(r'[0-9]', '', cm) if "_OPP_" in i else re.sub(r'[0-9]', '', cm), [1,1])[0] if mod_map.get(re.sub(r'[0-9]', '', i).replace("_OPP_", "")) else mod_map.get("_OPP_"+re.sub(r'[0-9]', '', cm) if "_OPP_" in i else re.sub(r'[0-9]', '', cm), {}).get(re.sub(r'[0-9]', '', i).replace("_OPP_", ""), [1,1])[0] for i in x]))
                        neg_df["Boost"] *= neg_df["cMarket"].apply(lambda x: np.product([mod_map.get(re.sub(r'[0-9]', '', i).replace("_OPP_", ""), {}).get("_OPP_"+re.sub(r'[0-9]', '', cm) if "_OPP_" in i else re.sub(r'[0-9]', '', cm), [1,1])[1] if mod_map.get(re.sub(r'[0-9]', '', i).replace("_OPP_", "")) else mod_map.get("_OPP_"+re.sub(r'[0-9]', '', cm) if "_OPP_" in i else re.sub(r'[0-9]', '', cm), {}).get(re.sub(r'[0-9]', '', i).replace("_OPP_", ""), [1,1])[1] for i in x]))
                
                corr = pd.concat([pos_df, neg_df])
                corr["R"] = corr.cMarket.apply(lambda x: R_map[x[0]])
                corr["P"] = (np.exp(corr["R"]*np.sqrt(offer["Model"]*(1-offer["Model"])
                                                      * corr["Model"]*(1-corr["Model"])))*offer["Model"]*corr["Model"])*3
                corr["P"] = corr["P"]*offer["Boost"]*corr["Boost"]
                corr.sort_values("P", ascending=False, inplace=True)
                corr.drop_duplicates("Player", inplace=True)
                corr = corr.loc[corr["P"] > 0.9]
                df.loc[(df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"]), 'Correlated Bets'] = ", ".join(
                    (corr["Desc"] + " (" + corr["P"].round(2).astype(str) + ")").to_list())

    return df.drop(columns='Position').dropna().sort_values("Model", ascending=False), parlay_df


def save_data(df, book, gc):
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

        with open((pkg_resources.files(data) / f"results/{book}.csv"), "w") as outfile:
            df.to_csv(outfile)
        try:
            df["Books"] = df["Books"]*df["Boost"]
            df["Model"] = df["Model"]*df["Boost"]
            df.sort_values("Model", ascending=False, inplace=True)
            mask = (df.Books > .54) & (df.Model > .58) & (2 >= df.Boost) & (df.Boost >= .75)
            # Access the Google Sheets worksheet and update its contents
            wks = gc.open("Sportstradamus").worksheet(book)
            wks.clear()
            wks.update([df.columns.values.tolist()] + df.loc[mask].values.tolist() + df.loc[~mask].values.tolist())
            wks.set_basic_filter()
            df["Books"] = df["Books"]/df["Boost"]
            df["Model"] = df["Model"]/df["Boost"]

            # Apply number formatting to the relevant columns
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
        except Exception:
            # Log the exception if there is an error writing the offers to the worksheet
            logger.exception(f"Error writing {book} offers")


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

        if os.path.isfile(filepath) and all([player in playerStats.index for player in players]):
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

            if ("+" in o["Player"]) or ("vs." in o["Player"]):
                probb = []
                for stat in stats:
                    probb.append(filt.predict_proba([[2*(1-under)-1, 2*stat["Odds"]-1]])[0][1])

                proba = np.mean(probb)
                proba = [1-proba, proba]
            else:
                proba = filt.predict_proba([[2*(1-under)-1, 2*stats["Odds"]-1]])[0]

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

        new_offers.append(o)

    return new_offers


if __name__ == "__main__":
    main()
