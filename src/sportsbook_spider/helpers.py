from sportsbook_spider.spiderLogger import logger
import requests
import json
import pickle
import os
import random
import unicodedata
import datetime
import importlib.resources as pkg_resources
from sportsbook_spider import creds, data
from time import sleep
from scipy.stats import poisson, norm, skellam
from scipy.optimize import fsolve
import numpy as np
import statsapi as mlb
from scrapeops_python_requests.scrapeops_requests import ScrapeOpsRequests
from tqdm.contrib.logging import logging_redirect_tqdm

with open((pkg_resources.files(creds) / "scrapeops_cred.json"), 'r') as infile:
    creds = json.load(infile)
apikey = creds['apikey']
scrapeops_logger = ScrapeOpsRequests(
    scrapeops_api_key=apikey,
    spider_name='Sportsbooks',
    job_name=os.uname()[1]
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
        with logging_redirect_tqdm():
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
                        logger.debug("Attempt " + str(i) +
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
        return int(np.round((1-p)/p*100))
    else:
        return int(np.round((p/(1-p))*-100))


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

mlb_pitchers['LA'] = mlb_pitchers.get('LAD', '')
mlb_pitchers['ANA'] = mlb_pitchers.get('LAA', '')
mlb_pitchers['ARI'] = mlb_pitchers.get('AZ', '')
mlb_pitchers['WAS'] = mlb_pitchers.get('WSH', '')


def likelihood(data, beta):
    l = 0
    for d in data:
        y = d['result']
        x = d['stats']
        p = norm.cdf(np.dot(x, beta))
        if p > 0 and p < 1:
            l += y*np.log(p)+(1-y)*np.log(1-p)

    return l


def get_pred(stats, weights):
    baseline = np.array([0, 0.5, 0.5, 0.5, 0])
    stats[stats == -1000] = baseline[stats == -1000]
    return norm.cdf(np.dot(stats-baseline, weights))


def get_loc(stats):
    sat = 0.035
    baseline = np.array([0, 0.5, 0.5, 0.5, 0])
    stats[stats == -1000] = baseline[stats == -1000]
    weights = np.array([0.75, 0.9, 0.5, 1, 0.75])
    weights = sat/np.dot(np.array([.1, .2, .1, .1, .06]), weights) * weights
    return np.clip(np.dot(weights, (stats-baseline)), -sat, sat)


class Archive():
    def __init__(self):
        self.archive = {}
        filepath = (pkg_resources.files(data) / "archive.dat")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.archive = pickle.load(infile)

    def __getitem__(self, item):
        return self.archive[item]

    def add(self, o, stats, lines, key):
        market = o['Market'].replace("H2H ", "")
        market = key.get(market, market)
        if not o['League'] in self.archive:
            self.archive[o['League']] = {}

        if not market in self.archive[o['League']]:
            self.archive[o['League']][market] = {}

        if not o['Date'] in self.archive[o['League']][market]:
            self.archive[o['League']][market][o['Date']] = {}

        if not o['Player'] in self.archive[o['League']][market][o['Date']]:
            self.archive[o['League']][market][o['Date']][o['Player']] = {}

        for line in lines:
            if line:
                l = np.floor(o['Line'])
                if type(line['EV']) is tuple:
                    p = skellam.sf(l, line['EV'][1], line['EV'][0])
                    if np.mod(o['Line'], 1) == 0:
                        p += skellam.pmf(l, line['EV'][1], line['EV'][0])/2
                else:
                    p = poisson.sf(l, line['EV'])
                    if np.mod(o['Line'], 1) == 0:
                        p += poisson.pmf(l, line['EV'])/2

            else:
                p = -1000

            stats = np.append(stats, p)

        self.archive[o['League']][market][o['Date']
                                          ][o['Player']][o['Line']] = stats

        self.archive[o['League']][market][o['Date']
                                          ][o['Player']]['Closing Lines'] = lines

    def write(self):
        filepath = (pkg_resources.files(data) / "archive.dat")
        with open(filepath, "wb") as outfile:
            pickle.dump(self.archive, outfile, -1)


archive = Archive()


urls = [  # "http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams",
    "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams",
    "http://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams",
    "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"]

abbreviations = {}
for url in urls:
    res = scraper.get(url)['sports'][0]['leagues'][0]
    league = res['abbreviation']
    if league not in abbreviations:
        abbreviations[league] = {}
    for team in res['teams']:
        name = remove_accents(team['team']['displayName'])
        abbr = team['team']['abbreviation']
        abbreviations[league][name] = abbr

abbreviations['NBA']['Los Angeles Clippers'] = 'LAC'
abbreviations['NHL']['St Louis Blues'] = 'STL'
# abbreviations['NFL']['Washington Football Team'] = 'WSH'


def match_offers(offers, league, market, platform, datasets, stat_data, stat_dict, pbar):
    market = stat_dict.get(market, market)
    filename = "_".join([league, market]).replace(" ", "-")+'.skl'
    filepath = (pkg_resources.files(data) / filename)
    if not os.path.isfile(filepath):
        pbar.update(len(offers))
        return []

    new_offers = []
    with open(filepath, 'rb') as infile:
        filedict = pickle.load(infile)
    model = filedict['model']
    scaler = filedict['scaler']
    threshold = filedict['threshold']
    edges = filedict['edges']
    stat_data.bucket_stats(market)
    stat_data.edges = edges
    for o in offers:
        stats = stat_data.row(o | {'Market': market}, date=o['Date'])
        if type(stats) is int:
            pbar.update()
            continue
        try:
            v = []
            lines = []
            if ' + ' in o['Player']:
                for dataset in datasets:
                    codex = dataset[platform]
                    offer = dataset.get(o['Player'], dataset.get(' + '.join(o['Player'].split(' + ')[::-1]), {})).get(
                        codex.get(o['Market'], o['Market']))
                    if offer is not None:
                        v.append(offer['EV'])
                    else:
                        players = o['Player'].split(' + ')
                        ev1 = dataset.get(players[0], {players[0]: None}).get(
                            codex.get(o['Market'], o['Market']))
                        ev2 = dataset.get(players[1], {players[1]: None}).get(
                            codex.get(o['Market'], o['Market']))
                        if ev1 is not None and ev2 is not None:
                            ev = ev1['EV']+ev2['EV']
                            l = np.round(ev-0.5)
                            v.append(ev)
                            offer = {
                                'Line': str(l+0.5),
                                'Over': str(prob_to_odds(poisson.sf(l, ev))),
                                'Under': str(prob_to_odds(poisson.cdf(l, ev))),
                                'EV': ev
                            }

                    lines.append(offer)

                if v:
                    v = np.mean(v)
                    line = (np.ceil(o['Line']-1), np.floor(o['Line']))
                    p = [poisson.cdf(line[0], v), poisson.sf(line[1], v)]
                    push = 1-p[1]-p[0]
                    p[0] += push/2
                    p[1] += push/2
                    stats['Odds'] = p[1]-0.5
                else:
                    p = [0.5]*2
            elif ' vs. ' in o['Player']:
                m = o['Market'].replace("H2H ", "")
                v1 = []
                v2 = []
                for dataset in datasets:
                    codex = dataset[platform]
                    offer = dataset.get(o['Player'], dataset.get(' vs. '.join(o['Player'].split(' + ')[::-1]), {})).get(
                        codex.get(m, m))
                    if offer is not None:
                        v.append(offer['EV'])
                    else:
                        players = o['Player'].split(' vs. ')
                        ev1 = dataset.get(players[0], {players[0]: None}).get(
                            codex.get(m, m))
                        ev2 = dataset.get(players[1], {players[1]: None}).get(
                            codex.get(m, m))
                        if ev1 is not None and ev2 is not None:
                            v1.append(ev1['EV'])
                            v2.append(ev2['EV'])
                            l = np.round(ev2['EV']-ev1['EV']-0.5)
                            offer = {
                                'Line': str(l+0.5),
                                'Under': str(prob_to_odds(skellam.cdf(l, ev2['EV'], ev1['EV']))),
                                'Over': str(prob_to_odds(skellam.sf(l, ev2['EV'], ev1['EV']))),
                                'EV': (ev1['EV'], ev2['EV'])
                            }

                    lines.append(offer)

                if v1 and v2:
                    v1 = np.mean(v1)
                    v2 = np.mean(v2)
                    line = (np.ceil(o['Line']-1), np.floor(o['Line']))
                    p = [skellam.cdf(line[0], v2, v1),
                         skellam.sf(line[1], v2, v1)]
                    push = 1-p[1]-p[0]
                    p[0] += push/2
                    p[1] += push/2
                    stats['Odds'] = p[1]-0.5
                else:
                    p = [0.5]*2
            else:
                for dataset in datasets:
                    codex = dataset[platform]
                    offer = dataset.get(o['Player'], {}).get(
                        codex.get(o['Market'], o['Market']))
                    if offer is not None:
                        v.append(offer['EV'])

                    lines.append(offer)

                if v:
                    v = np.mean(v)
                    line = (np.ceil(o['Line']-1), np.floor(o['Line']))
                    p = [poisson.cdf(line[0], v), poisson.sf(line[1], v)]
                    push = 1-p[1]-p[0]
                    p[0] += push/2
                    p[1] += push/2
                    stats['Odds'] = p[1]-0.5
                else:
                    p = [0.5]*2

            X = scaler.transform(stats)
            proba = model.predict_proba(X)[0, :]

            if proba[1] > proba[0]:
                o['Bet'] = 'Over'
                o['Books'] = p[1]
                o['Model'] = proba[1]
            else:
                o['Bet'] = 'Under'
                o['Books'] = p[0]
                o['Model'] = proba[0]

            o['Avg 10'] = (stats.loc[0, 'Avg10']) / o['Line']
            if " vs. " in o['Player']:
                o['Avg 10'] = 0.0
            o['Last 5'] = stats.loc[0, 'Last5'] + 0.5
            o['Last 10'] = stats.loc[0, 'Last10'] + 0.5
            o['H2H'] = stats.loc[0, 'H2H'] + 0.5
            o['OvP'] = stats.loc[0, 'DVPOA']

            stats = [o['Avg 10'], o['Last 5'],
                     o['Last 10'], o['H2H'], o['OvP']]
            archive.add(o, stats, lines, stat_dict)

            o['DraftKings'] = lines[0]['Line'] + "/" + \
                lines[0][o['Bet']] if lines[0] else 'N/A'
            o['FanDuel'] = lines[1]['Line'] + "/" + \
                lines[1][o['Bet']] if lines[1] else 'N/A'
            o['Pinnacle'] = lines[2]['Line'] + "/" + \
                lines[2][o['Bet']] if lines[2] else 'N/A'
            o['Caesars'] = str(lines[3]['Line']) + "/" + \
                str(lines[3][o['Bet']]) if lines[3] else 'N/A'

            new_offers.append(o)

        except:
            logger.exception(o['Player'] + ", " + o["Market"])

        pbar.update()

    return new_offers
