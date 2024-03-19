from sportstradamus.helpers import get_ev, stat_cv, books, no_vig_odds, abbreviations, remove_accents, Archive
from sportstradamus.spiderLogger import logger
import pickle
import json
import requests
import numpy as np
from datetime import datetime, timedelta
import pytz
import importlib.resources as pkg_resources
from sportstradamus import creds, data
from tqdm import tqdm
from operator import itemgetter
from itertools import groupby
from time import sleep

def confer():

    # Load API key
    filepath = pkg_resources.files(creds) / "keys.json"
    with open(filepath, "r") as infile:
        keys = json.load(infile)

    archive = Archive("All")
    logger.info("Archive loaded")

    archive = get_moneylines(archive, keys["odds_api"])
    logger.info("Game data complete")
    
    # Load prop markets
    filepath = pkg_resources.files(data) / "stat_map.json"
    with open(filepath, "r") as infile:
        stat_map = json.load(infile)

    archive = get_props(archive, keys["odds_api_plus"], stat_map["Odds API"])
    logger.info("Player data complete, writing to file...")

    archive.write()
    logger.info("Success!")

def get_moneylines(archive, apikey, date=datetime.now().astimezone(pytz.timezone("America/Chicago")), sport="All", key=None):
    """
    Retrieve moneyline and totals data from the odds API for NBA, MLB, and NHL.
    Process the data and store it in the archive file.
    """

    # Get available sports from the API
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={apikey}"
    res = requests.get(url)

    historical = date.date() != datetime.today().date()

    if sport == "All":
        if historical:
            logger.warning("All sports only supported if date is today")
            return archive
        # Get available sports from the API
        res = requests.get(f"https://api.the-odds-api.com/v4/sports/?apiKey={apikey}")
        if res.status_code != 200:
            return archive

        res = res.json()

        # Filter sports
        sports = [
            (s["key"], s["title"])
            for s in res
            if s["title"] in ["NBA", "MLB", "NHL", "NFL"] and s["active"]
        ]
    elif key is None:
        logger.warning("Key needed for sports other than All")
        return archive
    else:
        sports = [(key, sport)]

    if historical:
        url = "https://api.the-odds-api.com/v4/sports/{sport}/odds-history/?regions=us&markets=h2h,totals,spreads&date={date}&apiKey={apikey}"
        dayDelta = 1
        params = {"apikey": apikey, "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    else:
        url = "https://api.the-odds-api.com/v4/sports/{sport}/odds/?regions=us&markets=h2h,totals,spreads&apiKey={apikey}"
        dayDelta = 6
        params = {"apikey": apikey}

    # Retrieve odds data for each sport
    for sport, league in sports:
        params.update({"sport": sport})
        res = requests.get(url.format(**params))
        if res.status_code == 429:
            sleep(1)
            res = requests.get(url.format(**params))
        if res.status_code != 200:
            continue

        if historical:
            res = res.json()['data']
        else:
            res = res.json()

        # Process odds data for each game
        for game in tqdm(res, desc=f"Getting {league} Data", unit="game"):
            gameDate = datetime.fromisoformat(game['commence_time']).astimezone(pytz.timezone("America/Chicago"))
            if gameDate > date+timedelta(days=dayDelta):
                continue
            gameDate = gameDate.strftime("%Y-%m-%d")

            homeTeam = abbreviations[league][remove_accents(game["home_team"])]
            awayTeam = abbreviations[league][remove_accents(game["away_team"])]

            moneyline = []
            totals = []
            spread = []

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
                    elif market["key"] == "spreads" and market["outcomes"][0].get("point"):
                        if market["outcomes"][0]["name"] == game["home_team"]:
                            spread.append(market["outcomes"][0]["point"])
                        else:
                            spread.append(market["outcomes"][1]["point"])

            moneyline = np.mean(moneyline)
            totals = np.mean(totals)
            spread = np.mean(spread)

            # Update archive data with the processed odds
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

            archive[league]["Totals"][gameDate][awayTeam] = (totals+spread)/2
            archive[league]["Totals"][gameDate][homeTeam] = (totals-spread)/2

    return archive


def get_props(archive, apikey, props, date=datetime.now().astimezone(pytz.timezone("America/Chicago")), sport="All", key=None):
    """
    Retrieve moneyline and totals data from the odds API for NBA, MLB, and NHL.
    Process the data and store it in the archive file.
    """

    historical = date.date() != datetime.today().date()

    if sport == "All":
        if historical:
            logger.warning("All sports only supported if date is today")
            return archive
        # Get available sports from the API
        res = requests.get(f"https://api.the-odds-api.com/v4/sports/?apiKey={apikey}")
        if res.status_code != 200:
            return archive

        res = res.json()
        # Filter sports
        sports = [
            (s["key"], s["title"])
            for s in res
            if s["title"] in ["NBA", "MLB", "NHL", "NFL"] and s["active"]
        ]
    elif key is None:
        logger.warning("Key needed for sports other than All")
        return
    else:
        sports = [(key, sport)]

    if historical:
        url = "https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{eventId}/odds?apiKey={apikey}&regions=us&markets={markets}&date={date}"
        event_url = "https://api.the-odds-api.com/v4/historical/sports/{sport}/events?date={date}&apiKey={apikey}"
        dayDelta = 1
        params = {
            "apikey": apikey, 
            "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            }
    else:
        url = "https://api.the-odds-api.com/v4/sports/{sport}/events/{eventId}/odds?apiKey={apikey}&regions=us&markets={markets}"
        event_url = "https://api.the-odds-api.com/v4/sports/{sport}/events?apiKey={apikey}"
        dayDelta = 6
        params = {"apikey": apikey}

    # Retrieve odds data for each sport
    for sport, league in sports:
        params.update({
            "sport": sport,
            "markets": ",".join(props[league].keys())
            # "markets": ",".join(list(props[league].keys())[:2])
            })
        events = requests.get(event_url.format(**params))
        if events.status_code == 429:
            sleep(1)
            events = requests.get(event_url.format(**params))
        if events.status_code != 200:
            continue

        if historical:
            events = events.json()['data']
        else:
            events = events.json()

        for event in events:
            gameDate = datetime.fromisoformat(event['commence_time']).astimezone(pytz.timezone("America/Chicago"))
            if gameDate > date+timedelta(days=dayDelta):
                continue
            gameDate = gameDate.strftime("%Y-%m-%d")
            params.update({"eventId": event['id']})
            res = requests.get(url.format(**params))
            if res.status_code == 429:
                sleep(1)
                res = requests.get(url.format(**params))
            if res.status_code == 404:
                return archive
            elif res.status_code != 200:
                continue

            if historical:
                game = res.json()['data']
            else:
                game = res.json()

            odds = {}

            # Extract prop data from bookmakers and markets
            for book in game["bookmakers"]:
                if book['key'] in books:
                    book_pos = books.index(book['key'])
                else:
                    continue
                for market in book["markets"]:
                    market_name = props[league].get(market["key"])

                    odds.setdefault(market_name, {})

                    outcomes = sorted(market['outcomes'], key=itemgetter('description', 'name'))
                    
                    for player, lines in groupby(outcomes, itemgetter('description')):
                        player = remove_accents(player)
                        odds[market_name].setdefault(player, {"EV": {}, "Lines":[]})
                        lines = list(lines)
                        if len(lines) > 2:
                            ev = []
                            for line, prices in groupby(sorted(lines, key=itemgetter('point')), itemgetter('point')):
                                odds[market_name][player]["Lines"].append(line)
                                prices = list(prices)
                                if len(prices) > 2:
                                    prices = [[x['price'] for x in lines if x['name']=='Over'][0],[x['price'] for x in lines if x['name']=='Under'][0]]
                                else:
                                    prices = [x['price'] for x in prices]
                                price = no_vig_odds(*prices)
                                ev.append(get_ev(line, price[1], stat_cv[league].get(market_name,1)))

                            ev = np.mean(ev)

                        else:
                            line = lines[0]['point'] if 'point' in lines[0] else 0.5
                            odds[market_name][player]["Lines"].append(line)
                            price = no_vig_odds(*[x['price'] for x in lines])
                            ev = get_ev(line, price[1], stat_cv[league].get(market_name,1))

                        odds[market_name][player]["EV"][book['key']] = ev


            # Update archive data with the processed odds
            for market in odds.keys():
                archive[league].setdefault(market, {})
                archive[league][market].setdefault(gameDate, {})
                for player in odds[market].keys():
                    archive[league][market][gameDate].setdefault(player, {"EV": {}, "Lines": []})
                    archive[league][market][gameDate][player]["EV"].update(odds[market][player]["EV"])

                    line = np.median(odds[market][player]['Lines'])
                    if line not in archive[league][market][gameDate][player]["Lines"]:
                        archive[league][market][gameDate][player]["Lines"].append(line)


    return archive

if __name__ == "__main__":
    confer()
