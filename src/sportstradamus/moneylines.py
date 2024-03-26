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

    archive = get_moneylines(archive, keys)
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
        res = requests.get(f"https://api.the-odds-api.com/v4/sports/?apiKey={apikey['odds_api']}")
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

    markets = ["h2h","totals","spreads"]

    if historical:
        url = "https://api.the-odds-api.com/v4/sports/{sport}/odds-history/?regions=us&markets={markets}&date={date}&apiKey={apikey}"
        dayDelta = 1
        params = {"apikey": apikey["odds_api_plus"], "date": date.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": ",".join(markets)}
    else:
        url = "https://api.the-odds-api.com/v4/sports/{sport}/odds/?regions=us&markets={markets}&apiKey={apikey}"
        dayDelta = 6
        params = {"apikey": apikey["odds_api"], "markets": ",".join(markets)}

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
        for game in tqdm(res, desc=f"Getting {league} Game Data", unit="game"):
            gameDate = datetime.fromisoformat(game['commence_time']).astimezone(pytz.timezone("America/Chicago"))
            if gameDate > date+timedelta(days=dayDelta):
                continue
            gameDate = gameDate.strftime("%Y-%m-%d")

            homeTeam = abbreviations[league][remove_accents(game["home_team"])]
            awayTeam = abbreviations[league][remove_accents(game["away_team"])]

            moneyline_home = {}
            moneyline_away = {}
            totals = {}
            spread_home = {}
            spread_away = {}

            # Extract moneyline and totals data from bookmakers and markets
            for book in game["bookmakers"]:
                for market in book["markets"]:
                    if market["key"] == "h2h":
                        odds = no_vig_odds(market["outcomes"][0]["price"], market["outcomes"][1]["price"])
                        if market["outcomes"][0]["name"] == game["home_team"]:
                            moneyline_home[book['key']] = odds[0]
                            moneyline_away[book['key']] = odds[1]
                        else:
                            moneyline_home[book['key']] = odds[1]
                            moneyline_away[book['key']] = odds[0]
                    elif market["key"] == "totals":
                        outcomes = sorted(market["outcomes"], key=itemgetter('name'))
                        odds = no_vig_odds(outcomes[0]['price'], outcomes[1]['price'])
                        totals[book["key"]] = get_ev(outcomes[1]["point"], odds[1])
                    elif market["key"] == "spreads" and market["outcomes"][0].get("point"):
                        outcomes = sorted(market["outcomes"], key=itemgetter('point'))
                        odds = no_vig_odds(outcomes[0]['price'], outcomes[1]['price'])
                        spread = get_ev(outcomes[1]['point'], odds[1])
                        if market["outcomes"][0]["name"] == game["home_team"]:
                            spread_home[book["key"]] = spread
                            spread_away[book["key"]] = -spread
                        else:
                            spread_home[book["key"]] = -spread
                            spread_away[book["key"]] = spread

            # Update archive data with the processed odds
            archive[league].setdefault("Moneyline", {}).setdefault(gameDate, {})
            archive[league].setdefault("Totals", {}).setdefault(gameDate, {})

            archive[league]["Moneyline"][gameDate][awayTeam] = moneyline_away
            archive[league]["Moneyline"][gameDate][homeTeam] = moneyline_home

            archive[league]["Totals"][gameDate][awayTeam] = {k:(v+spread_away[k])/2 for k, v in totals.items() if spread_away.get(k)}
            archive[league]["Totals"][gameDate][homeTeam] = {k:(v+spread_home[k])/2 for k, v in totals.items() if spread_home.get(k)}

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
        if league == "MLB":
            params['markets'] = params['markets']+",totals_1st_1_innings,spreads_1st_1_innings"
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
            totals = {}
            spread_home = {}
            spread_away = {}

            # Extract prop data from bookmakers and markets
            for book in game["bookmakers"]:
                for market in book["markets"]:
                    if "totals" in market["key"]:
                        spread_name = " ".join(market["key"].split("_")[1:])
                        outcomes = sorted(market["outcomes"], key=itemgetter('name'))
                        sub_odds = no_vig_odds(outcomes[0]['price'], outcomes[1]['price'])
                        totals.setdefault(spread_name, {})
                        totals[spread_name][book["key"]] = get_ev(outcomes[1]["point"], sub_odds[1])
                        continue
                    elif "spread" in market["key"]:
                        spread_name = " ".join(market["key"].split("_")[1:])
                        outcomes = sorted(market["outcomes"], key=itemgetter('point'))
                        sub_odds = no_vig_odds(outcomes[0]['price'], outcomes[1]['price'])
                        spread = get_ev(outcomes[1]['point'], sub_odds[1])
                        spread_home.setdefault(spread_name, {})
                        spread_away.setdefault(spread_name, {})
                        if market["outcomes"][0]["name"] == game["home_team"]:
                            spread_home[spread_name][book["key"]] = spread
                            spread_away[spread_name][book["key"]] = -spread
                        else:
                            spread_home[spread_name][book["key"]] = -spread
                            spread_away[spread_name][book["key"]] = spread
                        continue

                    market_name = props[league].get(market["key"])

                    odds.setdefault(market_name, {})

                    outcomes = sorted(market['outcomes'], key=itemgetter('description', 'name'))
                    
                    for player, lines in groupby(outcomes, itemgetter('description')):
                        player = remove_accents(player).replace(" Total", "")
                        odds[market_name].setdefault(player, {"EV": {}, "Lines":[]})
                        lines = list(lines)
                        for line in lines:
                            line.setdefault('point', 0.5)
                            line['name'] = {"Yes":"Over", "No": "Under"}.get(line['name'], line['name'])
                        if len({line['point'] for line in lines}) > 1:
                            trueline = sorted(lines, key=(lambda x: np.abs(x['price']-2)))[0]['point']
                            lines = [line for line in lines if line['point'] == trueline]
                        if len(lines) > 2:
                            lines = [[line for line in lines if line['name']=='Over'][0],[line for line in lines if line['name']=='Under'][0]]

                        line = lines[0]['point'] if 'point' in lines[0] else 0.5
                        odds[market_name][player]["Lines"].append(line)
                        price = no_vig_odds(*[x['price'] for x in lines])
                        ev = get_ev(line, price[1], stat_cv[league].get(market_name,1))

                        odds[market_name][player]["EV"][book['key']] = ev


            # Update archive data with the processed odds
            for market in odds.keys():
                archive[league].setdefault(market, {}).setdefault(gameDate, {})
                for player in odds[market].keys():
                    archive[league][market][gameDate].setdefault(player, {"EV": {}, "Lines": []})
                    archive[league][market][gameDate][player]["EV"].update(odds[market][player]["EV"])

                    line = np.median(odds[market][player]['Lines'])
                    if line not in archive[league][market][gameDate][player]["Lines"]:
                        archive[league][market][gameDate][player]["Lines"].append(line)

            for market in totals.keys():
                archive[league].setdefault(market, {}).setdefault(gameDate, {})
                
                archive[league][market][gameDate][abbreviations[league][remove_accents(game["home_team"])]] = {k:(v+spread_home.get(k,0))/2 for k, v in totals[market].items()}
                archive[league][market][gameDate][abbreviations[league][remove_accents(game["away_team"])]] = {k:(v+spread_away.get(k,0))/2 for k, v in totals[market].items()}

    return archive

if __name__ == "__main__":
    confer()
