from sportsbook_spider.spiderLogger import logger
from sportsbook_spider.stats import StatsNBA, StatsMLB, StatsNHL
from sportsbook_spider.books import get_caesars, get_fd, get_pinnacle, get_dk, get_pp, get_ud, get_thrive, get_parp
from sportsbook_spider.helpers import archive, match_offers
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread
import click
from sportsbook_spider import creds
from functools import partialmethod
from tqdm import tqdm
import pandas as pd
import os.path
import datetime
import importlib.resources as pkg_resources

global nba
global mlb
global nhl
global untapped_markets


@click.command()
@click.option('--progress', default=True, help='Display progress bars')
def main(progress):
    # Initialize tqdm based on the value of 'progress' flag
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not progress))

    # Authorize the gspread API
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
    ]
    cred = None

    # Check if token.json file exists and load credentials
    if os.path.exists((pkg_resources.files(creds) / "token.json")):
        cred = Credentials.from_authorized_user_file(
            (pkg_resources.files(creds) / "token.json"), SCOPES)

    # If no valid credentials found, let the user log in
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                (pkg_resources.files(creds) / "credentials.json"), SCOPES)
            cred = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open((pkg_resources.files(creds) / "token.json"), 'w') as token:
            token.write(cred.to_json())

    gc = gspread.authorize(cred)

    """"
    Start gathering sportsbook data
    """
    dk_data = {}
    logger.info("Getting DraftKings MLB lines")
    dk_data.update(get_dk(84240, [743, 1024, 1031]))  # MLB
    logger.info("Getting DraftKings NBA lines")
    dk_data.update(
        get_dk(42648, [583, 1215, 1216, 1217, 1218, 1219, 1220]))  # NBA
    logger.info("Getting DraftKings NHL lines")
    dk_data.update(get_dk(42133, [550, 1064, 1189]))  # NHL
    # dk_data.update(get_dk(92893, [488, 633])) # Tennis
    # dk_data.update(get_dk(91581, [488, 633])) # Tennis
    logger.info(str(len(dk_data)) + " offers found")

    fd_data = {}
    logger.info("Getting FanDuel MLB lines")
    fd_data.update(get_fd('mlb', ['batter-props', 'pitcher-props', 'innings']))
    logger.info("Getting FanDuel NBA lines")
    fd_data.update(get_fd('nba', ['player-points', 'player-rebounds',
                                  'player-assists', 'player-threes', 'player-combos', 'player-defense']))
    logger.info("Getting FanDuel NHL lines")
    fd_data.update(
        get_fd('nhl', ['goal-scorer', 'shots', 'points-assists', 'goalie-props']))
    logger.info(str(len(fd_data)) + " offers found")

    pin_data = {}
    logger.info("Getting Pinnacle MLB lines")
    pin_data.update(get_pinnacle(246))  # MLB
    logger.info("Getting Pinnacle NBA lines")
    pin_data.update(get_pinnacle(487))  # NBA
    logger.info("Getting Pinnacle NHL lines")
    pin_data.update(get_pinnacle(1456))  # NHL
    logger.info(str(len(pin_data)) + " offers found")

    csb_data = {}
    logger.info("Getting Caesars MLB Lines")
    sport = "baseball"
    league = "04f90892-3afa-4e84-acce-5b89f151063d"
    csb_data.update(get_caesars(sport, league))
    logger.info("Getting Caesars NBA Lines")
    sport = "basketball"
    league = "5806c896-4eec-4de1-874f-afed93114b8c"  # NBA
    csb_data.update(get_caesars(sport, league))
    logger.info("Getting Caesars NHL Lines")
    sport = "icehockey"
    league = "b7b715a9-c7e8-4c47-af0a-77385b525e09"
    csb_data.update(get_caesars(sport, league))
    # logger.info("Getting Caesars NFL Lines")
    # sport = "americanfootball"
    # league = "007d7c61-07a7-4e18-bb40-15104b6eac92"
    # csb_data.update(get_caesars(sport, league))
    logger.info(str(len(csb_data)) + " offers found")

    datasets = {
        "DraftKings": dk_data,
        "FanDuel": fd_data,
        "Pinnacle": pin_data,
        "Caesars": csb_data
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

    stats = {
        'NBA': nba,
        'MLB': mlb,
        'NHL': nhl
    }

    untapped_markets = []

    ################# PrizePicks #################

    pp_dict = get_pp()
    pp_offers = process_offers(pp_dict, "PrizePicks", datasets, stats)

    ################# Underdog #################

    ud_dict = get_ud()
    ud_offers = process_offers(ud_dict, "Underdog", datasets, stats)

    ################# Thrive #################

    th_dict = get_thrive()
    th_offers = process_offers(th_dict, "Thrive", datasets, stats)

    ################# ParlayPlays #################

    parp_dict = get_parp()
    parp_offers = process_offers(parp_dict, "ParlayPlay", datasets, stats)

    logger.info("Writing to file...")
    offer_list = [
        ("PrizePicks", pp_offers),
        ("Underdog", ud_offers),
        ("Thrive", th_offers),
        ("ParlayPlay", parp_offers)
    ]
    for book, offers in offer_list:
        save_data(offers, book, gc)

    if len(untapped_markets) > 0:
        untapped_df = pd.DataFrame(untapped_markets).drop_duplicates()
        wks = gc.open("Sports Betting").worksheet("Untapped Markets")
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
    new_offers = []
    if len(offer_dict) > 0:
        # Calculate the total number of offers to process
        total = sum(sum(len(i) for i in v.values())
                    for v in offer_dict.values())

        # Display a progress bar
        with tqdm(total=total, desc=f"Matching {book} Offers", unit='offer') as pbar:
            for league, markets in offer_dict.items():
                if league in stats:
                    stat_data = stats.get(league)
                else:
                    # Handle untapped markets where the league is not supported
                    for market, offers in markets.items():
                        untapped_markets.append({
                            'Platform': book,
                            'League': league,
                            'Market': market
                        })
                        pbar.update(len(offers))
                    continue

                for market, offers in markets.items():
                    # Match the offers with player statistics
                    matched_offers = match_offers(
                        offers, league, market, book, datasets, stat_data, pbar)

                    if len(matched_offers) == 0:
                        # No matched offers found for the market
                        untapped_markets.append({
                            'Platform': book,
                            'League': league,
                            'Market': market
                        })
                    else:
                        # Add the matched offers to the new_offers list
                        new_offers.extend(matched_offers)

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
            df = pd.DataFrame(offers).dropna().drop(
                columns='Opponent').sort_values('Model', ascending=False)

            # Access the Google Sheets worksheet and update its contents
            wks = gc.open("Sports Betting").worksheet(book)
            wks.clear()
            wks.update([df.columns.values.tolist()] + df.values.tolist())
            wks.update("S1", "Last Updated: " +
                       datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
            wks.set_basic_filter()

            # Apply number formatting to the relevant columns
            if book == "ParlayPlay":
                wks.format("I:O", {"numberFormat": {
                           "type": "PERCENT", "pattern": "0.00%"}})
            else:
                wks.format("H:N", {"numberFormat": {
                           "type": "PERCENT", "pattern": "0.00%"}})
        except Exception as exc:
            # Log the exception if there is an error writing the offers to the worksheet
            logger.exception(f"Error writing {book} offers")


if __name__ == '__main__':

    main()
