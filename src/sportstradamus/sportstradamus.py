from sportstradamus.spiderLogger import logger
from sportstradamus.stats import StatsNBA, StatsMLB, StatsNHL, StatsNFL
from sportstradamus.books import (
    get_caesars,
    get_fd,
    get_pinnacle,
    get_dk,
    get_pp,
    get_ud,
    get_thrive,
)
from sportstradamus.helpers import archive, get_ev
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread
import click
from scipy.stats import poisson, skellam, norm, hmean, gamma, nbinom
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
from itertools import combinations


@click.command()
@click.option("--progress/--no-progress", default=True, help="Display progress bars")
@click.option("--books/--no-books", default=True, help="Get data from sportsbooks")
def main(progress, books):
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

    """"
    Start gathering sportsbook data
    """
    dk_data = {}
    fd_data = {}
    pin_data = {}
    csb_data = {}
    if books:
        logger.info("Getting DraftKings MLB lines")
        try:
            dk_data.update(get_dk(84240, [743, 1024, 1031]))  # MLB
        except Exception as exc:
            logger.exception("Failed to get DraftKings MLB lines")
        logger.info("Getting DraftKings NBA lines")
        try:
            dk_data.update(
                get_dk(42648, [583, 1215, 1216, 1217, 1218, 1219, 1220]))  # NBA
        except Exception as exc:
            logger.exception("Failed to get DraftKings NBA lines")
        logger.info("Getting DraftKings NHL lines")
        try:
            dk_data.update(get_dk(42133, [550, 1064, 1189, 1190]))  # NHL
        except Exception as exc:
            logger.exception("Failed to get DraftKings NHL lines")
        logger.info("Getting DraftKings NFL lines")
        try:
            dk_data.update(get_dk(88808, [1000, 1001, 1003]))  # NFL
        except Exception as exc:
            logger.exception("Failed to get DraftKings NFL lines")
        logger.info(str(len(dk_data)) + " offers found")

        logger.info("Getting FanDuel MLB lines")
        try:
            fd_data.update(
                get_fd("mlb", ["pitcher-props", "innings", "batter-props",]))
        except Exception as exc:
            logger.exception("Failed to get FanDuel MLB lines")
        logger.info("Getting FanDuel NBA lines")
        try:
            fd_data.update(get_fd('nba', ['player-points', 'player-combos', 'player-rebounds',
                                          'player-assists', 'player-threes', 'player-defense']))
        except Exception as exc:
            logger.exception("Failed to get FanDuel NBA lines")
        logger.info("Getting FanDuel NHL lines")
        try:
            fd_data.update(
                get_fd('nhl', ['goalie-props', 'shots', 'points-assists', 'goal-scorer']))
        except Exception as exc:
            logger.exception("Failed to get FanDuel NHL lines")
        logger.info("Getting FanDuel NFL lines")
        try:
            fd_data.update(
                get_fd('nfl', ['passing-props', 'receiving-props', 'rushing-props', 'td-scorer-props']))
        except Exception as exc:
            logger.exception("Failed to get FanDuel NFL lines")
        logger.info(str(len(fd_data)) + " offers found")

        logger.info("Getting Pinnacle MLB lines")
        try:
            pin_data.update(get_pinnacle(246))  # MLB
        except Exception as exc:
            logger.exception("Failed to get Pinnacle MLB lines")
        logger.info("Getting Pinnacle NBA lines")
        try:
            pin_data.update(get_pinnacle(487))  # NBA
        except Exception as exc:
            logger.exception("Failed to get Pinnacle NBA lines")
        logger.info("Getting Pinnacle NHL lines")
        try:
            pin_data.update(get_pinnacle(1456))  # NHL
        except Exception as exc:
            logger.exception("Failed to get Pinnacle NHL lines")
        logger.info("Getting Pinnacle NFL lines")
        try:
            pin_data.update(get_pinnacle(889))  # NFL
        except Exception as exc:
            logger.exception("Failed to get Pinnacle NFL lines")
        logger.info(str(len(pin_data)) + " offers found")

        logger.info("Getting Caesars MLB Lines")
        try:
            sport = "baseball"
            league = "04f90892-3afa-4e84-acce-5b89f151063d"
            csb_data.update(get_caesars(sport, league))
        except Exception as exc:
            logger.exception("Failed to get Caesars MLB lines")
        logger.info("Getting Caesars NBA Lines")
        try:
            sport = "basketball"
            league = "5806c896-4eec-4de1-874f-afed93114b8c"  # NBA
            csb_data.update(get_caesars(sport, league))
        except Exception as exc:
            logger.exception("Failed to get Caesars NBA lines")
        logger.info("Getting Caesars NHL Lines")
        try:
            sport = "icehockey"
            league = "b7b715a9-c7e8-4c47-af0a-77385b525e09"
            csb_data.update(get_caesars(sport, league))
        except Exception as exc:
            logger.exception("Failed to get Caesars NHL lines")
        logger.info("Getting Caesars NFL Lines")
        try:
            sport = "americanfootball"
            league = "007d7c61-07a7-4e18-bb40-15104b6eac92"
            csb_data.update(get_caesars(sport, league))
        except Exception as exc:
            logger.exception("Failed to get Caesars NFL lines")
        logger.info(str(len(csb_data)) + " offers found")

    datasets = {
        "DraftKings": dk_data,
        "FanDuel": fd_data,
        "Pinnacle": pin_data,
        "Caesars": csb_data,
    }

    """
    Start gathering player stats
    """
    nba = StatsNBA()
    nba.load()
    nba.update()
    mlb = StatsMLB()
    mlb.load()
    mlb.update()
    nhl = StatsNHL()
    nhl.load()
    nhl.update()
    nfl = StatsNFL()
    nfl.load()
    nfl.update()

    stats = {"NBA": nba, "MLB": mlb, "NHL": nhl, "NFL": nfl}

    untapped_markets = []

    with open((pkg_resources.files(data) / "stat_map.json"), "r") as infile:
        stat_map = json.load(infile)

    pp_df = pd.DataFrame()
    ud_df = pd.DataFrame()
    th_df = pd.DataFrame()

    # PrizePicks

    try:
        pp_dict = get_pp()
        pp_offers = process_offers(pp_dict, "PrizePicks", datasets, stats)
        pp_df = pd.DataFrame(pp_offers)
        pp_df["Market"] = pp_df["Market"].map(stat_map["PrizePicks"])
        save_data(pp_offers, "PrizePicks", gc)
    except Exception as exc:
        logger.exception("Failed to get PrizePicks")

    # Underdog

    try:
        ud_dict = get_ud()
        ud_offers = process_offers(ud_dict, "Underdog", datasets, stats)
        save_data(ud_offers, "Underdog", gc)
        ud_df = pd.DataFrame(ud_offers)
        ud_df["Market"] = ud_df["Market"].map(stat_map["Underdog"])
    except Exception as exc:
        logger.exception("Failed to get Underdog")

    # ParlayPlay

    # parp_dict = get_parp()
    # parp_offers = process_offers(parp_dict, "ParlayPlay", datasets, stats)
    # save_data(parp_offers, "ParlayPlay", gc)

    # Thrive

    try:
        th_dict = get_thrive()
        th_offers = process_offers(th_dict, "Thrive", datasets, stats)
        th_df = pd.DataFrame(th_offers)
        th_df["Market"] = th_df["Market"].map(stat_map["Thrive"])
        save_data(th_offers, "Thrive", gc)
    except Exception as exc:
        logger.exception("Failed to get Thrive")

    archive.write()

    logger.info("Checking historical predictions")
    filepath = pkg_resources.files(data) / "history.dat"
    if os.path.isfile(filepath):
        history = pd.read_pickle(filepath)
    else:
        history = pd.DataFrame(
            columns=["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Model", "Correct"])

    df = pd.concat([ud_df, pp_df, th_df]).drop_duplicates(["Player", "League", "Date", "Market"],
                                                          ignore_index=True)[["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Model"]]
    df.loc[(df['Market'] == 'AST') & (
        df['League'] == 'NHL'), 'Market'] = "points"
    df.loc[(df['Market'] == 'PTS') & (
        df['League'] == 'NHL'), 'Market'] = "assists"
    df.dropna(subset='Market', inplace=True, ignore_index=True)
    history = pd.concat([df, history]).drop_duplicates(["Player", "League", "Date", "Market"],
                                                       ignore_index=True)
    history = history.loc[history["Model"] > .6]
    gameDates = pd.to_datetime(history.Date).dt.date
    history = history.loc[(datetime.datetime.today(
    ).date() - datetime.timedelta(days=28)) <= gameDates]
    nameStr = {"MLB": "playerName", "NBA": "PLAYER_NAME",
               "NFL": "player display name", "NHL": "playerName"}
    dateStr = {"MLB": "gameDate", "NBA": "GAME_DATE",
               "NFL": "gameday", "NHL": "gameDate"}
    for i, row in tqdm(history.iterrows(), desc="Checking history", total=len(history)):
        if np.isnan(row["Correct"]):
            gamelog = stats[row["League"]].gamelog
            if " + " in row["Player"]:
                players = row["Player"].split(" +")
                game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                    pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                    pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                if not game1.empty and not game2.empty:
                    history.at[i, "Correct"] = int(((game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) > row["Line"] and row["Bet"] == "Over") or (
                        (game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) < row["Line"] and row["Bet"] == "Under"))

            elif " vs. " in row["Player"]:
                players = row["Player"].split(" vs. ")
                game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                    pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                    pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                if not game1.empty and not game2.empty:
                    history.at[i, "Correct"] = int(((game1.iloc[0][row["Market"]] + row["Line"]) > game2.iloc[0][row["Market"]] and row["Bet"] == "Over") or (
                        (game1.iloc[0][row["Market"]] + row["Line"]) < game2.iloc[0][row["Market"]] and row["Bet"] == "Under"))

            else:
                game = gamelog.loc[(gamelog[nameStr[row["League"]]] == row["Player"]) & (
                    pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                if not game.empty:
                    history.at[i, "Correct"] = int((game.iloc[0][row["Market"]] > row["Line"] and row["Bet"] == "Over") or (
                        game.iloc[0][row["Market"]] < row["Line"] and row["Bet"] == "Under"))

    history.to_pickle(filepath)
    history.dropna(inplace=True, ignore_index=True)

    if len(history) > 0:
        wks = gc.open("Sportstradamus").worksheet("History")
        wks.clear()
        wks.update([history.columns.values.tolist()] +
                   history.values.tolist())
        wks.set_basic_filter()

    if len(untapped_markets) > 0:
        untapped_df = pd.DataFrame(untapped_markets).drop_duplicates()
        wks = gc.open("Sportstradamus").worksheet("Untapped Markets")
        wks.clear()
        wks.update([untapped_df.columns.values.tolist()] +
                   untapped_df.values.tolist())
        wks.set_basic_filter()

    logger.info("Getting NFL Fantasy Rankings")

    filepath = pkg_resources.files(data) / "NFL_fantasy-points-underdog.mdl"
    with open(filepath, "rb") as infile:
        filedict = pickle.load(infile)
    model = filedict["model"]

    playerStats, playerData = nfl.get_fantasy()
    categories = ["Home", "Position"]
    for c in categories:
        playerStats[c] = playerStats[c].astype('category')

    prob_params = model.predict(playerStats, pred_type="parameters")
    prob_params.index = playerStats.index
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


def process_offers(offer_dict, book, datasets, stats):
    """
    Process the offers from the given offer dictionary and match them with player statistics.

    Args:
        offer_dict (dict): Dictionary containing the offers to be processed.
        book (str): Name of the book or platform.
        datasets (dict): Dictionary containing the datasets of player prop odds.
        stats (dict): Dictionary containing player stats.

    Returns:
        list: List of processed offers.

    """
    global untapped_markets
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
                        untapped_markets.append(
                            {"Platform": book, "League": league, "Market": market}
                        )
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    # Match the offers with player statistics
                    playerStats = match_offers(
                        offers, league, market, book, datasets, stat_data, pbar
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

    new_offers = find_correlation(new_offers, stats, book)

    logger.info(str(len(new_offers)) + " offers processed")
    return new_offers


def find_correlation(offers, stats, platform):
    global stat_map
    logger.info("Finding Correlations")
    df = pd.DataFrame(offers)
    df["Correlated Bets"] = ""
    usage_str = {
        "NBA": "MIN",
        "NFL": "snap pct",
        "NHL": "TimeShare"
    }
    positions = {
        "NBA": ["G", "F", "C", "G-F", "F-C"],
        "NFL": ["QB", "WR", "RB", "TE"],
        "NHL": ["C", "R", "L", "D", "G"]
    }
    for league in ["NBA", "NFL", "MLB", "NHL"]:
        league_df = df.loc[df["League"] == league]
        league_df = league_df.loc[league_df.Position.apply(
            lambda x: not isinstance(x, tuple))]  # Remove this to correlate combo picks
        c = pd.read_csv(pkg_resources.files(data) / (league+"_corr.csv"))
        c.columns = ["market", "correlation", "R"]
        c_map = c
        c_map.index = c.correlation
        c_map = c_map.groupby('market').apply(lambda x: x)['R'].to_dict()
        stat_data = stats.get(league)

        if league != "MLB":
            league_df.Position = league_df.Position.apply(lambda x: positions[league][x] if isinstance(
                x, int) else "/".join([positions[league][i] for i in x]))
            stat_data.profile_market(usage_str[league])
            usage = pd.DataFrame(
                stat_data.playerProfile[usage_str[league] + " short"])
            usage.reset_index(inplace=True)
            usage.rename(
                columns={"player display name": "Player", "playerName": "Player", "PLAYER_NAME": "Player"}, inplace=True)
            league_df = league_df.merge(usage)
            if league == "NBA":
                new_df = league_df.loc[league_df.Position.str.contains("-")]
                new_df.Position = new_df.Position.str.split("-").str[1]
                league_df = pd.concat([league_df, new_df])
                league_df.Position = league_df.Position.str.split("-").str[0]

            ranks = league_df.groupby(["Team", "Position"]).rank(
                ascending=False, method='dense')[usage_str[league] + " short"].astype(int)
            league_df.Position = league_df.Position + ranks.astype(str)
        else:
            league_df.Position = "B" + league_df.Position.add(1).astype(str)
            league_df.loc[league_df["Market"].map(stat_map[platform]).str.contains(
                "allowed") | league_df["Market"].map(stat_map[platform]).str.contains("pitch"), "Position"] = "P"

        checked_teams = []
        best_fives = {}
        for team in list(league_df.Team.unique()):
            if team in checked_teams:
                continue
            team_df = league_df.loc[league_df["Team"] == team]
            team_df["cMarket"] = team_df["Position"] + " " + \
                team_df["Market"].map(stat_map[platform])
            opp_df = league_df.loc[league_df["Team"]
                                   == team_df.Opponent.mode().values[0]]
            opp_df["cMarket"] = "_OPP_" + opp_df["Position"] + \
                " " + opp_df["Market"].map(stat_map[platform])
            checked_teams.append(team)
            checked_teams.append(team_df.Opponent.mode().values[0])
            game_df = pd.concat([team_df, opp_df])
            parlays = []
            for bet in combinations(game_df.index.unique(), 5):
                if (len(game_df.loc[list(bet), 'Team'].unique()) != 2) or (len(game_df.loc[list(bet), 'Player'].unique()) != 5):
                    continue

                p = game_df.loc[list(bet)].drop_duplicates(
                    'Player')['Model'].product()

                # get correlation matrix
                for i in np.arange(5):
                    cm1 = game_df.loc[bet[i], 'cMarket']
                    b1 = game_df.loc[bet[i], 'Bet']
                    p1 = game_df.loc[bet[i], 'Model']
                    if len(cm1) == 2:
                        cm1 = cm1.to_list()
                        b1 = b1.iloc[0]
                        p1 = p1.iloc[0]
                    else:
                        cm1 = [cm1]
                    for j in np.arange(i, 5):
                        if i != j:
                            cm2 = game_df.loc[bet[j], 'cMarket']
                            b2 = game_df.loc[bet[j], 'Bet']
                            p2 = game_df.loc[bet[j], 'Model']
                            if len(cm2) == 2:
                                cm2 = cm2.to_list()
                                b2 = b2.iloc[0]
                                p2 = p2.iloc[0]
                            else:
                                cm2 = [cm2]

                            max_c = 0
                            for a in cm1:
                                for b in cm2:
                                    if ("_OPP_" in a) and ("_OPP_" in b):
                                        a = a.replace("_OPP_", "")
                                        b = b.replace("_OPP_", "")
                                    cc = c_map.get((a, b), 0)
                                    max_c = cc if abs(cc) > abs(
                                        max_c) else max_c
                                    cc = c_map.get((b, a), 0)
                                    max_c = cc if abs(cc) > abs(
                                        max_c) else max_c

                            if b1 != b2:
                                max_c = -max_c

                            p += np.sqrt(p1*p2*(1-p1)*(1-p2))*max_c

                p = p*20
                if platform == "Underdog":
                    p = p * \
                        game_df.loc[list(bet)].drop_duplicates(
                            'Player')['Boost'].product()
                if p > 1:
                    parlays.append(list(bet), p)

            for i, offer in team_df.iterrows():
                if len(team_df.loc[i]) == 2:
                    mask = c.market.isin(team_df.loc[i].cMarket.to_list())
                else:
                    mask = (c.market == offer.cMarket)

                pos = c.loc[mask & (c.R > 0)]
                neg = c.loc[mask & (c.R < 0)]
                R_map = c.loc[mask].drop_duplicates("correlation").set_index("correlation")[
                    "R"].abs().to_dict()
                corr = pd.concat([game_df.loc[game_df.cMarket.isin(pos.correlation.to_list()) & (game_df.Bet == offer.Bet)],
                                  game_df.loc[game_df.cMarket.isin(neg.correlation.to_list()) & (game_df.Bet != offer.Bet)]])
                corr["R"] = corr.cMarket.map(R_map)
                corr["P"] = (corr["R"]*np.sqrt(offer["Model"]*(1-offer["Model"])*corr["Model"]*(
                    1-corr["Model"]))+offer["Model"]*corr["Model"])*3
                if platform == "Underdog":
                    corr["P"] = corr["P"]*offer["Boost"]*corr["Boost"]
                corr.sort_values("P", ascending=False, inplace=True)
                corr.drop_duplicates("Player", inplace=True)
                corr.drop(corr.loc[corr["Player"] ==
                          offer["Player"]].index, inplace=True)
                df.loc[(df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"]), 'Correlated Bets'] = ", ".join(
                    (corr["Player"] + " - " + corr["Bet"] + " " + corr["Market"] + " (" + corr["P"].round(2).astype(str) + ")").to_list())

    return df.drop(columns='Position').dropna().sort_values("Model", ascending=False)


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
        try:
            # Access the Google Sheets worksheet and update its contents
            wks = gc.open("Sportstradamus").worksheet(book)
            wks.clear()
            wks.update([df.columns.values.tolist()] + df.values.tolist())
            wks.set_basic_filter()

            # Apply number formatting to the relevant columns
            if book == "ParlayPlay" or book == "Underdog":
                wks.format(
                    "J:K", {"numberFormat": {
                        "type": "PERCENT", "pattern": "0.00%"}}
                )
                wks.format(
                    "N:P", {"numberFormat": {
                        "type": "PERCENT", "pattern": "0.00%"}}
                )
                wks.update(
                    "R1",
                    "Last Updated: "
                    + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                )
            else:
                wks.format(
                    "I:J", {"numberFormat": {
                        "type": "PERCENT", "pattern": "0.00%"}}
                )
                wks.format(
                    "M:O", {"numberFormat": {
                        "type": "PERCENT", "pattern": "0.00%"}}
                )
                wks.update(
                    "Q1",
                    "Last Updated: "
                    + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                )
        except Exception:
            # Log the exception if there is an error writing the offers to the worksheet
            logger.exception(f"Error writing {book} offers")


def match_offers(offers, league, market, platform, datasets, stat_data, pbar):
    """
    Matches offers with statistical data and applies various calculations and transformations.

    Args:
        offers (list): List of offers to match.
        league (str): League name.
        market (str): Market name.
        platform (str): Platform name.
        datasets (dict): Dictionary of datasets.
        stat_data (obj): Statistical data object.
        pbar (obj): Progress bar object.

    Returns:
        list: List of matched offers.
    """
    global stat_map

    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points"}.get(market, market)
    filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
    filepath = pkg_resources.files(data) / filename
    if not os.path.isfile(filepath):
        pbar.update(len(offers))
        logger.warning(f"{filename} missing")
        return []
    with open(filepath, "rb") as infile:
        filedict = pickle.load(infile)
    cv = filedict["cv"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat_data.profile_market(market)

    playerStats = []
    playerNames = []

    for o in tqdm(offers, leave=False, disable=not pbar):
        if "+" in o["Player"] or "vs." in o["Player"]:
            players = o["Player"].replace("vs.", "+").split("+")
            players = [player.strip() for player in players]
            teams = o["Team"].split("/")
            if len(teams) < len(players):
                teams = teams*len(players)
            opponents = o["Opponent"].split("/")
            if len(opponents) < len(players):
                opponents = opponents*len(players)
            for i, player in enumerate(players):
                if len(player.split(" ")[0].replace(".", "")) <= 2:
                    if league == "NFL":
                        nameStr = 'player display name'
                        namePos = 1
                    elif league == "NBA":
                        nameStr = 'PLAYER_NAME'
                        namePos = 2
                    elif league == "MLB":
                        nameStr = "playerName"
                        namePos = 3
                    elif league == "NHL":
                        nameStr = "playerName"
                        namePos = 7
                    name_df = stat_data.gamelog.loc[stat_data.gamelog[nameStr].str.contains(player.split(
                        " ")[1]) & stat_data.gamelog[nameStr].str.startswith(player.split(" ")[0][0])]
                    if name_df.empty:
                        pass
                    else:
                        players[i] = name_df.iloc[0, namePos]
                        player = name_df.iloc[0, namePos]
                        if teams[i] == "":
                            continue  # TODO: finish this

                lines = list(archive.archive.get(league, {}).get(
                    market, {}).get(o["Date"], {}).get(player, {}).keys())
                if len(lines) > 0:
                    if "Closing Lines" in lines:
                        lines.remove("Closing Lines")
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
                for book, dataset in datasets.items():
                    codex = stat_map[book]
                    offer = dataset.get(o["Player"], {}).get(
                        codex.get(market, market)
                    )
                    lines.append(offer)

                archive.add(this_o, lines, stat_map[platform], cv)
                stats = stat_data.get_stats(this_o, date=o["Date"])

                if type(stats) is int:
                    logger.warning(f"{o['Player']}, {market} stat error")
                    pbar.update()
                    continue
                playerStats.append(stats)
                playerNames.append(player)

            archive.add(o, [None]*4, stat_map[platform], cv)
        else:
            lines = []
            for book, dataset in datasets.items():
                codex = stat_map[book]
                offer = dataset.get(o["Player"], {}).get(
                    codex.get(market, market)
                )

                lines.append(offer)

            archive.add(o, lines, stat_map[platform], cv)
            stats = stat_data.get_stats(o | {"Market": market}, date=o["Date"])
            if type(stats) is int:
                logger.warning(f"{o['Player']}, {market} stat error")
                pbar.update()
                continue
            playerStats.append(stats)
            playerNames.append(o["Player"])

        pbar.update()

    playerStats = pd.DataFrame(playerStats, index=playerNames)
    return playerStats[~playerStats.index.duplicated(keep='last')].fillna(0)


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

    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points"}.get(market, market)
    filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
    filepath = pkg_resources.files(data) / filename
    if not os.path.isfile(filepath):
        logger.warning(f"{filename} missing")
        return []

    new_offers = []
    with open(filepath, "rb") as infile:
        filedict = pickle.load(infile)
    model = filedict["model"]
    dist = filedict["distribution"]
    cv = filedict["cv"]

    categories = ["Home", "Position"]
    if "Position" not in playerStats.columns:
        categories.remove("Position")
    for c in categories:
        playerStats[c] = playerStats[c].astype('category')

    prob_params = model.predict(playerStats, pred_type="parameters")
    prob_params.index = playerStats.index

    for o in tqdm(offers, leave=False):
        if "+" in o["Player"] or "vs." in o["Player"]:
            players = o["Player"].replace("vs.", "+").split("+")
            players = [player.strip() for player in players]
            stats = []
            for i, player in enumerate(players):
                if len(player.split(" ")[0].replace(".", "")) <= 2:
                    if league == "NFL":
                        nameStr = 'player display name'
                        namePos = 1
                    elif league == "NBA":
                        nameStr = 'PLAYER_NAME'
                        namePos = 2
                    elif league == "MLB":
                        nameStr = "playerName"
                        namePos = 3
                    elif league == "NHL":
                        nameStr = "playerName"
                        namePos = 7
                    name_df = stat_data.gamelog.loc[stat_data.gamelog[nameStr].str.contains(player.split(
                        " ")[1]) & stat_data.gamelog[nameStr].str.startswith(player.split(" ")[0][0])]
                    if name_df.empty:
                        pass
                    else:
                        players[i] = name_df.iloc[0, namePos]
                        player = name_df.iloc[0, namePos]

                if player not in playerStats.index:
                    stats.append(0)
                else:
                    stats.append(playerStats.loc[player])

            if any([type(s) is int for s in stats]):
                logger.warning(f"{o['Player']}, {market} stat error")
                continue

            if "+" in o["Player"]:
                ev1 = get_ev(stats[0]["Line"], 1-stats[0]
                             ["Odds"], cv) if stats[0]["Odds"] != 0 else None
                ev2 = get_ev(stats[1]["Line"], 1-stats[1]
                             ["Odds"], cv) if stats[1]["Odds"] != 0 else None

                if ev1 is not None and ev2 is not None:
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
                    under = norm.cdf(o["Line"],
                                     params[1]["loc"] +
                                     params[0]["loc"],
                                     params[1]["scale"] +
                                     params[0]["scale"])
                elif dist == "Gamma":
                    under = gamma.cdf(o["Line"],
                                      params[1]["concentration"] +
                                      params[0]["concentration"],
                                      scale=1/(params[1]["rate"] +
                                               params[0]["rate"]))
                elif dist == "NegativeBinomial":
                    under = nbinom.cdf(o["Line"],
                                       params[1]["concentration"] +
                                       params[0]["concentration"],
                                       scale=1/(params[1]["rate"] +
                                                params[0]["rate"]))

            elif "vs." in o["Player"]:
                ev1 = get_ev(stats[0]["Line"], 1-stats[0]
                             ["Odds"], cv) if stats[0]["Odds"] != 0 else None
                ev2 = get_ev(stats[1]["Line"], 1-stats[1]
                             ["Odds"], cv) if stats[1]["Odds"] != 0 else None
                if ev1 is not None and ev2 is not None:
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
                    under = norm.cdf(-o["Line"],
                                     params[0]["loc"] -
                                     params[1]["loc"],
                                     params[0]["scale"] +
                                     params[1]["scale"])

        else:
            if o["Player"] not in playerStats.index:
                continue
            stats = playerStats.loc[o["Player"]]
            if type(stats) is int:
                logger.warning(f"{o['Player']}, {market} stat error")
                continue

            ev = get_ev(stats["Line"], 1-stats["Odds"], cv
                        ) if stats["Odds"] != 0 else None

            if ev is not None:
                if cv == 1:
                    line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                    p = [poisson.cdf(line[0], ev), poisson.sf(line[1], ev)]
                else:
                    line = o["Line"]
                    p = [norm.cdf(line, ev, ev*cv), norm.sf(line, ev, ev*cv)]
                push = 1 - p[1] - p[0]
                p[0] += push / 2
                p[1] += push / 2
            else:
                p = [0.5] * 2

            params = prob_params.loc[o["Player"]]
            if dist == "Poisson":
                under = poisson.cdf(o["Line"], params["rate"])
                push = poisson.pmf(o["Line"], params["rate"])
                under -= push/2
            elif dist == "Gaussian":
                under = norm.cdf(
                    o["Line"], params["loc"], params["scale"])

        try:
            proba = [under, 1-under]

            if proba[1] > proba[0] or o.get("Boost", 1) > 1:
                o["Bet"] = "Over"
                o["Books"] = p[1]
                o["Model"] = proba[1]
            else:
                o["Bet"] = "Under"
                o["Books"] = p[0]
                o["Model"] = proba[0]

            totals_map = {
                "NBA": 110,
                "NFL": 22.5,
                "MLB": 4.5,
                "NHL": 2.5
            }

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

            # lines = archive.archive.get(league, {}).get(market, {}).get(
            #     o["Date"], {}).get(o["Player"], {}).get("Closing Lines", [None]*4)

            # o["DraftKings"] = (
            #     lines[0]["Line"] + "/" +
            #     lines[0][o["Bet"]] if lines[0] else "N/A"
            # )
            # o["FanDuel"] = (
            #     lines[1]["Line"] + "/" +
            #     lines[1][o["Bet"]] if lines[1] else "N/A"
            # )
            # o["Pinnacle"] = (
            #     lines[2]["Line"] + "/" +
            #     lines[2][o["Bet"]] if lines[2] else "N/A"
            # )
            # o["Caesars"] = (
            #     str(lines[3]["Line"]) + "/" + str(lines[3][o["Bet"]])
            #     if lines[3]
            #     else "N/A"
            # )

            new_offers.append(o)

        except Exception:
            logger.exception(o["Player"] + ", " + o["Market"])

    return new_offers


if __name__ == "__main__":
    main()
