from sportsbook_spider.helpers import scraper, no_vig_odds, abbreviations, remove_accents
import pickle
import json
import numpy as np
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportsbook_spider import creds, data
from tqdm import tqdm
from sportsbook_spider.spiderLogger import logger


def get_moneylines():
    filepath = (pkg_resources.files(data) / "archive.dat")
    with open(filepath, "rb") as infile:
        archive = pickle.load(infile)

    filepath = (pkg_resources.files(creds) / "odds_api.json")
    with open(filepath, "r") as infile:
        apikey = json.load(infile)['apikey']

    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={apikey}"

    sports = ['NBA', 'MLB', 'NHL']
    res = scraper.get(url)

    sports = [(s['key'], s['title'])
              for s in res if s['title'] in sports and s['active']]

    for sport, league in sports:
        logger.info(f"Getting {league} Data")
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?regions=us&markets=h2h,totals&apiKey={apikey}"
        res = scraper.get(url)

        for game in tqdm(res):
            gameDate = datetime.strptime(
                game['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
            gameDate = (gameDate - timedelta(hours=5)).strftime('%Y-%m-%d')

            homeTeam = abbreviations[league][remove_accents(game['home_team'])]
            awayTeam = abbreviations[league][remove_accents(game['away_team'])]

            moneyline = []
            totals = []
            for book in game['bookmakers']:
                for market in book['markets']:
                    if market['key'] == 'h2h':
                        odds = [o['price'] for o in market['outcomes']]
                        odds = no_vig_odds(odds[0], odds[1])
                        moneyline.append(odds[0])
                    elif market['key'] == 'totals':
                        totals.append(market['outcomes'][0]['point'])

            moneyline = np.mean(moneyline)
            totals = np.mean(totals)
            if not league in archive:
                archive[league] = {}

            if not 'Moneyline' in archive[league]:
                archive[league]['Moneyline'] = {}
            if not 'Totals' in archive[league]:
                archive[league]['Totals'] = {}

            if not gameDate in archive[league]['Moneyline']:
                archive[league]['Moneyline'][gameDate] = {}
            if not gameDate in archive[league]['Totals']:
                archive[league]['Totals'][gameDate] = {}

            archive[league]['Moneyline'][gameDate][awayTeam] = moneyline
            archive[league]['Moneyline'][gameDate][homeTeam] = 1-moneyline

            archive[league]['Totals'][gameDate][awayTeam] = totals
            archive[league]['Totals'][gameDate][homeTeam] = totals

    filepath = (pkg_resources.files(data) / "archive.dat")
    with open(filepath, "wb") as outfile:
        pickle.dump(archive, outfile)


if __name__ == '__main__':
    get_moneylines()
