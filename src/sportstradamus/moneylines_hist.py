from sportstradamus.helpers import scraper, no_vig_odds, abbreviations, remove_accents
import pickle
import numpy as np
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportstradamus import data
from itertools import cycle

filepath = pkg_resources.files(data) / "archive.dat"
with open(filepath, "rb") as infile:
    archive = pickle.load(infile)

apikey = "29311eba87f90e2741845d5c70ca4c14"

sport = "icehockey_nhl"
league = "NHL"

Date = datetime.strptime("2021-10-12T08:00:00Z", "%Y-%m-%dT%H:%M:%SZ")

while Date < datetime.strptime("2022-06-27T08:00:00Z", "%Y-%m-%dT%H:%M:%SZ"):
    date = Date.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds-history/?regions=us&markets=h2h,totals&date={date}&apiKey={apikey}"
    res = scraper.get(url)["data"]

    for game in res:
        gameDate = datetime.strptime(
            game["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
        gameDate = (gameDate - timedelta(hours=5)).strftime("%Y-%m-%d")

        homeTeam = abbreviations[league][remove_accents(game["home_team"])]
        awayTeam = abbreviations[league][remove_accents(game["away_team"])]

        moneyline = []
        totals = []
        for book in game["bookmakers"]:
            for market in book["markets"]:
                if market["key"] == "h2h" and market["outcomes"][0].get("price"):
                    odds = [o["price"] for o in market["outcomes"]]
                    odds = no_vig_odds(odds[0], odds[1])
                    if market["outcomes"][0]["name"] == game["home_team"]:
                        moneyline.append(odds[0])
                    else:
                        moneyline.append(odds[1])
                elif market["key"] == "totals" and market["outcomes"][0].get("point"):
                    totals.append(market["outcomes"][0]["point"])

        moneyline = np.mean(moneyline)
        totals = np.mean(totals)

        if not league in archive:
            archive[league] = {}

        if not "Moneyline" in archive[league]:
            archive[league]["Moneyline"] = {}
        if not "Totals" in archive[league]:
            archive[league]["Totals"] = {}

        if not gameDate in archive[league]["Moneyline"]:
            archive[league]["Moneyline"][gameDate] = {}
        if not gameDate in archive[league]["Totals"]:
            archive[league]["Totals"][gameDate] = {}

        archive[league]["Moneyline"][gameDate][awayTeam] = 1 - moneyline
        archive[league]["Moneyline"][gameDate][homeTeam] = moneyline

        archive[league]["Totals"][gameDate][awayTeam] = totals
        archive[league]["Totals"][gameDate][homeTeam] = totals

    Date = Date + timedelta(days=1)

filepath = pkg_resources.files(data) / "archive.dat"
with open(filepath, "wb") as outfile:
    pickle.dump(archive, outfile, -1)
