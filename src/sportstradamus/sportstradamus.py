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
    get_parp,
)
from sportstradamus.helpers import archive, prob_to_odds
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread
import click
from scipy.stats import poisson, skellam
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


@click.command()
@click.option("--progress/--no-progress", default=True, help="Display progress bars")
@click.option("--books/--no-books", default=True, help="Get data from sportsbooks")
def main(progress, books):
    global untapped_markets
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
        dk_data.update(get_dk(84240, [743, 1024, 1031]))  # MLB
        # logger.info("Getting DraftKings NBA lines")
        # dk_data.update(
        #    get_dk(42648, [583, 1215, 1216, 1217, 1218, 1219, 1220]))  # NBA
        # logger.info("Getting DraftKings NHL lines")
        # dk_data.update(get_dk(42133, [550, 1064, 1189]))  # NHL
        # dk_data.update(get_dk(92893, [488, 633])) # Tennis
        # dk_data.update(get_dk(91581, [488, 633])) # Tennis
        logger.info(str(len(dk_data)) + " offers found")

        logger.info("Getting FanDuel MLB lines")
        fd_data.update(
            get_fd("mlb", ["batter-props", "pitcher-props", "innings"]))
        # logger.info("Getting FanDuel NBA lines")
        # fd_data.update(get_fd('nba', ['player-points', 'player-rebounds',
        #                              'player-assists', 'player-threes', 'player-combos', 'player-defense']))
        # logger.info("Getting FanDuel NHL lines")
        # fd_data.update(
        #    get_fd('nhl', ['goal-scorer', 'shots', 'points-assists', 'goalie-props']))
        logger.info(str(len(fd_data)) + " offers found")

        logger.info("Getting Pinnacle MLB lines")
        pin_data.update(get_pinnacle(246))  # MLB
        # logger.info("Getting Pinnacle NBA lines")
        # pin_data.update(get_pinnacle(487))  # NBA
        # logger.info("Getting Pinnacle NHL lines")
        # pin_data.update(get_pinnacle(1456))  # NHL
        logger.info(str(len(pin_data)) + " offers found")

        logger.info("Getting Caesars MLB Lines")
        sport = "baseball"
        league = "04f90892-3afa-4e84-acce-5b89f151063d"
        csb_data.update(get_caesars(sport, league))
        # logger.info("Getting Caesars NBA Lines")
        # sport = "basketball"
        # league = "5806c896-4eec-4de1-874f-afed93114b8c"  # NBA
        # csb_data.update(get_caesars(sport, league))
        # logger.info("Getting Caesars NHL Lines")
        # sport = "icehockey"
        # league = "b7b715a9-c7e8-4c47-af0a-77385b525e09"
        # csb_data.update(get_caesars(sport, league))
        # logger.info("Getting Caesars NFL Lines")
        # sport = "americanfootball"
        # league = "007d7c61-07a7-4e18-bb40-15104b6eac92"
        # csb_data.update(get_caesars(sport, league))
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

    # PrizePicks

    pp_dict = get_pp()
    pp_offers = process_offers(pp_dict, "PrizePicks", datasets, stats)
    save_data(pp_offers, "PrizePicks", gc)

    # Underdog

    ud_dict = get_ud()
    ud_offers = process_offers(ud_dict, "Underdog", datasets, stats)
    save_data(ud_offers, "Underdog", gc)

    # ParlayPlay

    parp_dict = get_parp()
    parp_offers = process_offers(parp_dict, "ParlayPlay", datasets, stats)
    save_data(parp_offers, "ParlayPlay", gc)

    # Thrive

    th_dict = get_thrive()
    th_offers = process_offers(th_dict, "Thrive", datasets, stats)
    save_data(th_offers, "Thrive", gc)

    if len(untapped_markets) > 0:
        untapped_df = pd.DataFrame(untapped_markets).drop_duplicates()
        wks = gc.open("Sportstradamus").worksheet("Untapped Markets")
        wks.clear()
        wks.update([untapped_df.columns.values.tolist()] +
                   untapped_df.values.tolist())
        wks.set_basic_filter()

    archive.write()
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
                    matched_offers = match_offers(
                        offers, league, market, book, datasets, stat_data, pbar
                    )

                    if len(matched_offers) == 0:
                        # No matched offers found for the market
                        logger.info(f"{league}, {market} offers not matched")
                        untapped_markets.append(
                            {"Platform": book, "League": league, "Market": market}
                        )
                    else:
                        # Add the matched offers to the new_offers list
                        new_offers.extend(matched_offers)

    logger.info(str(len(new_offers)) + " offers processed")
    return new_offers


def save_data(offers, book, gc):
    """
    Save offers data to a Google Sheets worksheet.

    Args:
        offers (list): List of offer data.
        book (str): Name of the DFS book.
        gc (gspread.client.Client): Google Sheets client.

    Raises:
        Exception: If there is an error writing the offers to the worksheet.
    """
    if len(offers) > 0:
        try:
            # Create a DataFrame from the offers data and perform necessary operations
            df = (
                pd.DataFrame(offers)
                .dropna()
                .drop(columns="Opponent")
                .sort_values("Model", ascending=False)
            )

            # Access the Google Sheets worksheet and update its contents
            wks = gc.open("Sportstradamus").worksheet(book)
            wks.clear()
            wks.update([df.columns.values.tolist()] + df.values.tolist())
            wks.set_basic_filter()

            # Apply number formatting to the relevant columns
            if book == "ParlayPlay":
                wks.format(
                    "I:O", {"numberFormat": {
                        "type": "PERCENT", "pattern": "0.00%"}}
                )
                wks.update(
                    "T1",
                    "Last Updated: "
                    + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                )
            else:
                wks.format(
                    "H:N", {"numberFormat": {
                        "type": "PERCENT", "pattern": "0.00%"}}
                )
                wks.update(
                    "S1",
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
    with open((pkg_resources.files(data) / "stat_map.json"), "r") as infile:
        stat_map = json.load(infile)

    stat_map = stat_map[platform]
    market = stat_map["Stats"].get(market, market)
    filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
    filepath = pkg_resources.files(data) / filename
    if not os.path.isfile(filepath):
        pbar.update(len(offers))
        logger.warning(f"{filename} missing")
        return []

    new_offers = []
    with open(filepath, "rb") as infile:
        filedict = pickle.load(infile)
    model = filedict["model"]
    stat_data.profile_market(market)

    for o in tqdm(offers, leave=False, disable=not pbar):
        stats = stat_data.get_stats(o | {"Market": market}, date=o["Date"])
        if type(stats) is int:
            logger.warning(f"{o['Player']}, {market} stat error")
            pbar.update()
            continue

        try:
            v = []
            lines = []

            if " + " in o["Player"]:
                for book, dataset in datasets.items():
                    codex = stat_map[book]
                    offer = dataset.get(
                        o["Player"],
                        dataset.get(
                            " + ".join(o["Player"].split(" + ")[::-1]), {}),
                    ).get(codex.get(o["Market"], o["Market"]))

                    if offer is not None:
                        v.append(offer["EV"])
                    else:
                        players = o["Player"].split(" + ")
                        for i, player in enumerate(players):
                            if len(player.split(" ")[0].replace(".", "")) == 1:
                                if league == "NFL":
                                    nameStr = 'player display name'
                                elif league == "NBA":
                                    nameStr = 'PLAYER_NAME'
                                else:
                                    nameStr = "playerName"
                                name_df = stat_data.gamelog.loc[stat_data.gamelog[nameStr].str.contains(player.split(
                                    " ")[1]) & stat_data.gamelog[nameStr].str.startswith(player.split(" ")[0][0])]
                                if name_df.empty:
                                    continue
                                else:
                                    players[i] = name_df.iloc[0, 2]
                        ev1 = dataset.get(players[0], {players[0]: None}).get(
                            codex.get(o["Market"], o["Market"])
                        )
                        ev2 = dataset.get(players[1], {players[1]: None}).get(
                            codex.get(o["Market"], o["Market"])
                        )

                        if ev1 is not None and ev2 is not None:
                            ev = ev1["EV"] + ev2["EV"]
                            l = np.round(ev - 0.5)
                            v.append(ev)
                            offer = {
                                "Line": str(l + 0.5),
                                "Over": str(prob_to_odds(poisson.sf(l, ev))),
                                "Under": str(prob_to_odds(poisson.cdf(l, ev))),
                                "EV": ev,
                            }

                    lines.append(offer)

                if v:
                    v = np.mean(v)
                    line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                    p = [poisson.cdf(line[0], v), poisson.sf(line[1], v)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                    stats["Odds"] = p[1] - 0.5
                else:
                    p = [0.5] * 2

            elif " vs. " in o["Player"]:
                m = o["Market"].replace("H2H ", "")
                v1 = []
                v2 = []
                for book, dataset in datasets.items():
                    codex = stat_map[book]
                    offer = dataset.get(
                        o["Player"],
                        dataset.get(" vs. ".join(
                            o["Player"].split(" + ")[::-1]), {}),
                    ).get(codex.get(m, m))
                    if offer is not None:
                        v.append(offer["EV"])
                    else:
                        players = o["Player"].split(" vs. ")
                        ev1 = dataset.get(players[0], {players[0]: None}).get(
                            codex.get(m, m)
                        )
                        ev2 = dataset.get(players[1], {players[1]: None}).get(
                            codex.get(m, m)
                        )
                        if ev1 is not None and ev2 is not None:
                            v1.append(ev1["EV"])
                            v2.append(ev2["EV"])
                            l = np.round(ev2["EV"] - ev1["EV"] - 0.5)
                            offer = {
                                "Line": str(l + 0.5),
                                "Under": str(
                                    prob_to_odds(skellam.cdf(
                                        l, ev2["EV"], ev1["EV"]))
                                ),
                                "Over": str(
                                    prob_to_odds(skellam.sf(
                                        l, ev2["EV"], ev1["EV"]))
                                ),
                                "EV": (ev1["EV"], ev2["EV"]),
                            }

                    lines.append(offer)

                if v1 and v2:
                    v1 = np.mean(v1)
                    v2 = np.mean(v2)
                    line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                    p = [skellam.cdf(line[0], v2, v1),
                         skellam.sf(line[1], v2, v1)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                    stats["Odds"] = p[1] - 0.5
                else:
                    p = [0.5] * 2

            else:
                for book, dataset in datasets.items():
                    codex = stat_map[book]
                    offer = dataset.get(o["Player"], {}).get(
                        codex.get(o["Market"], o["Market"])
                    )
                    if offer is not None:
                        v.append(offer["EV"])

                    lines.append(offer)

                if v:
                    v = np.mean(v)
                    line = (np.ceil(o["Line"] - 1), np.floor(o["Line"]))
                    p = [poisson.cdf(line[0], v), poisson.sf(line[1], v)]
                    push = 1 - p[1] - p[0]
                    p[0] += push / 2
                    p[1] += push / 2
                    stats["Odds"] = p[1] - 0.5
                else:
                    p = [0.5] * 2

            proba = model.predict_proba(
                np.array(list(stats.values())).reshape(1, -1))[0, :]

            if proba[1] > proba[0]:
                o["Bet"] = "Over"
                o["Books"] = p[1]
                o["Model"] = proba[1]
            else:
                o["Bet"] = "Under"
                o["Books"] = p[0]
                o["Model"] = proba[0]

            o["Avg 10"] = (
                (stats["Avg10"]) /
                o["Line"] if " vs. " not in o["Player"] else 0
            )
            o["Last 5"] = stats["Last5"] + 0.5
            o["Last 10"] = stats["Last10"] + 0.5
            o["H2H"] = stats["H2H"] + 0.5
            o["DvPOA"] = stats["DVPOA"]

            archive.add(o, lines, stat_map["Stats"])

            o["DraftKings"] = (
                lines[0]["Line"] + "/" +
                lines[0][o["Bet"]] if lines[0] else "N/A"
            )
            o["FanDuel"] = (
                lines[1]["Line"] + "/" +
                lines[1][o["Bet"]] if lines[1] else "N/A"
            )
            o["Pinnacle"] = (
                lines[2]["Line"] + "/" +
                lines[2][o["Bet"]] if lines[2] else "N/A"
            )
            o["Caesars"] = (
                str(lines[3]["Line"]) + "/" + str(lines[3][o["Bet"]])
                if lines[3]
                else "N/A"
            )

            new_offers.append(o)

        except Exception:
            logger.exception(o["Player"] + ", " + o["Market"])

        pbar.update()

    return new_offers


if __name__ == "__main__":
    main()
