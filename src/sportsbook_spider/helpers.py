import requests
import json
import random
import unicodedata
import datetime
import importlib.resources as pkg_resources
from . import creds
from time import sleep
from scipy.stats import poisson
from scipy.optimize import fsolve
import numpy as np
import statsapi as mlb
import logging
from scrapeops_python_requests.scrapeops_requests import ScrapeOpsRequests

logger = logging.getLogger(__name__)

with open((pkg_resources.files(creds) / "scrapeops_cred.json"), 'r') as infile:
    creds = json.load(infile)
apikey = creds['apikey']
scrapeops_logger = ScrapeOpsRequests(
    scrapeops_api_key=apikey,
    spider_name='Sportsbooks',
    job_name='Odds'
)

requests = scrapeops_logger.RequestsWrapper()


class Scrape:
    def __init__(self, apikey):
        self.headers = requests.get(
            f"http://headers.scrapeops.io/v1/browser-headers?api_key={apikey}").json()['result']

        self.header = random.choice(self.headers)

        self.weights = np.ones([len(self.headers)])

    def _new_headers(self):
        for i in range(len(self.headers)):
            if self.headers[i] == self.header:
                self.weights[i] = 0
            else:
                self.weights[i] += 1

        self.header = random.choices(self.headers, weights=self.weights)[0]

    def get(self, url, max_attempts=5, headers={}, params={}):
        for i in range(1, max_attempts+1):
            if i > 1:
                self._new_headers()
                sleep(random.uniform(3, 5))
            try:
                response = requests.get(
                    url, headers=self.header | headers, params=params)
                if response.status_code == 200:
                    return response.json()
                else:
                    logging.debug("Attempt " + str(i) +
                                  ", Error " + str(response.status_code))
            except Exception as exc:
                logger.exception("Attempt " + str(i) + ",")

        logger.warning("Max Attempts Reached")
        return None


scraper = Scrape(apikey)


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def odds_to_prob(odds):
    if odds > 0:
        return 100/(odds+100)
    else:
        odds = -odds
        return odds/(odds+100)


def prob_to_odds(p):
    if p < 0.5:
        return np.round((1-p)/p*100)
    else:
        return np.round((p/(1-p))*-100)


def no_vig_odds(over, under):
    if np.abs(over) >= 100:
        o = odds_to_prob(over)
    else:
        o = 1/over
    if np.abs(under) >= 100:
        u = odds_to_prob(under)
    else:
        u = 1/under

    juice = o+u
    return [o/juice, u/juice]


def get_ev(line, under):
    line = np.ceil(float(line) - 1)
    return fsolve(lambda x: under - poisson.cdf(line, x), line)[0]


mlb_games = mlb.schedule(start_date=datetime.date.today(),
                         end_date=datetime.date.today())
mlb_teams = mlb.get('teams', {'sportId': 1})
mlb_pitchers = {}
for game in mlb_games:
    if game['status'] != 'Final':
        awayTeam = [team['abbreviation']
                    for team in mlb_teams['teams'] if team['id'] == game['away_id']][0]
        homeTeam = [team['abbreviation']
                    for team in mlb_teams['teams'] if team['id'] == game['home_id']][0]
        if game['game_num'] == 1:
            mlb_pitchers[awayTeam] = remove_accents(
                game['away_probable_pitcher'])
            mlb_pitchers[homeTeam] = remove_accents(
                game['home_probable_pitcher'])
        elif game['game_num'] > 1:
            mlb_pitchers[awayTeam + str(game['game_num'])
                         ] = remove_accents(game['away_probable_pitcher'])
            mlb_pitchers[homeTeam + str(game['game_num'])
                         ] = remove_accents(game['home_probable_pitcher'])

mlb_pitchers['LA'] = mlb_pitchers.get('LAD')
mlb_pitchers['ANA'] = mlb_pitchers.get('LAA')
mlb_pitchers['ARI'] = mlb_pitchers.get('AZ')
mlb_pitchers['WAS'] = mlb_pitchers.get('WSH')


def get_loc(stats):
    sat = 0.035
    baseline = np.array([0, 0.5, 0.5, 0.5, 0])
    stats[stats == -1000] = baseline[stats == -1000]
    weights = np.array([0.75, 0.9, 0.5, 1, 0.75])
    weights = sat/np.dot(np.array([.1, .2, .1, .1, .06]), weights) * weights
    return np.clip(np.dot(weights, (stats-baseline)), -sat, sat)
