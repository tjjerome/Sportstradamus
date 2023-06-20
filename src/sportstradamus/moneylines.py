from sportstradamus.helpers import scraper, no_vig_odds, abbreviations, remove_accents
import pickle
import json
import numpy as np
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportstradamus import creds, data
from tqdm import tqdm


def get_moneylines():
    """
    Retrieve moneyline and totals data from the odds API for NBA, MLB, and NHL.
    Process the data and store it in the archive file.
    """

    # Load archive data from file
    filepath = pkg_resources.files(data) / "archive.dat"
    with open(filepath, "rb") as infile:
        archive = pickle.load(infile)

    # Load API key
    filepath = pkg_resources.files(creds) / "odds_api.json"
    with open(filepath, "r") as infile:
        apikey = json.load(infile)["apikey"]

    # Get available sports from the API
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={apikey}"
    res = scraper.get(url)

    # Filter sports to NBA, MLB, and NHL
    sports = [
        (s["key"], s["title"])
        for s in res
        if s["title"] in ["NBA", "MLB", "NHL"] and s["active"]
    ]

    # Retrieve odds data for each sport
    for sport, league in sports:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?regions=us&markets=h2h,totals&apiKey={apikey}"
        res = scraper.get(url)

        # Process odds data for each game
        for game in tqdm(res, desc=f"Getting {league} Data", unit="game"):
            gameDate = datetime.strptime(
                game["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
            gameDate = (gameDate - timedelta(hours=5)).strftime("%Y-%m-%d")

            homeTeam = abbreviations[league][remove_accents(game["home_team"])]
            awayTeam = abbreviations[league][remove_accents(game["away_team"])]

            moneyline = []
            totals = []

            # Extract moneyline and totals data from bookmakers and markets
            for book in game["bookmakers"]:
                for market in book["markets"]:
                    if market["key"] == "h2h":
                        odds = [o["price"] for o in market["outcomes"]]
                        odds = no_vig_odds(odds[0], odds[1])
                        if market["outcomes"][0]["name"] == game["home_team"]:
                            moneyline.append(odds[0])
                        else:
                            moneyline.append(odds[1])
                    elif market["key"] == "totals":
                        totals.append(market["outcomes"][0]["point"])

            moneyline = np.mean(moneyline)
            totals = np.mean(totals)

            # Update archive data with the processed odds
            if league not in archive:
                archive[league] = {}

            if "Moneyline" not in archive[league]:
                archive[league]["Moneyline"] = {}
            if "Totals" not in archive[league]:
                archive[league]["Totals"] = {}

            if gameDate not in archive[league]["Moneyline"]:
                archive[league]["Moneyline"][gameDate] = {}
            if gameDate not in archive[league]["Totals"]:
                archive[league]["Totals"][gameDate] = {}

            archive[league]["Moneyline"][gameDate][awayTeam] = 1 - moneyline
            archive[league]["Moneyline"][gameDate][homeTeam] = moneyline

            archive[league]["Totals"][gameDate][awayTeam] = totals
            archive[league]["Totals"][gameDate][homeTeam] = totals

    # Save updated archive data to file
    filepath = pkg_resources.files(data) / "archive.dat"
    with open(filepath, "wb") as outfile:
        pickle.dump(archive, outfile, -1)


if __name__ == "__main__":
    get_moneylines()
