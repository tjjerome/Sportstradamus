from sportstradamus.spiderLogger import logger
import requests
import json
import pickle
import os
import re
import random
import unicodedata
import datetime
import importlib.resources as pkg_resources
from klepto.archives import hdfdir_archive, cache
from sportstradamus import creds, data
from time import sleep
from operator import itemgetter
from scipy.stats import poisson, skellam, norm, iqr
from scipy.optimize import fsolve, minimize
from scipy.integrate import dblquad
import numpy as np
import statsapi as mlb
import pandas as pd
from scrapeops_python_requests.scrapeops_requests import ScrapeOpsRequests
from tqdm.contrib.logging import logging_redirect_tqdm
import warnings


# Load API key
with open((pkg_resources.files(creds) / "keys.json"), "r") as infile:
    keys = json.load(infile)
    odds_api = keys["odds_api"]
    scrapeops = keys["scrapeops"]
    apikey = keys["scrapingfish"]

with open((pkg_resources.files(data) / "abbreviations.json"), "r") as infile:
    abbreviations = json.load(infile)

with open((pkg_resources.files(data) / "combo_props.json"), "r") as infile:
    combo_props = json.load(infile)

with open((pkg_resources.files(data) / "stat_cv.json"), "r") as infile:
    stat_cv = json.load(infile)

with open((pkg_resources.files(data) / "stat_std.json"), "r") as infile:
    stat_std = json.load(infile)
    
with open((pkg_resources.files(data) / "stat_map.json"), "r") as infile:
    stat_map = json.load(infile)

with open((pkg_resources.files(data) / "book_weights.json"), "r") as infile:
    book_weights = json.load(infile)

with open(pkg_resources.files(data) / "prop_books.json", "r") as infile:
    books = json.load(infile)

with open((pkg_resources.files(data) / "goalies.json"), "r") as infile:
    nhl_goalies = json.load(infile)

with open(pkg_resources.files(data) / "feature_filter.json", "r") as infile:
    feature_filter = json.load(infile)

with open(pkg_resources.files(data) / "banned_combos.json", "r") as infile:
    banned = json.load(infile)
    
for platform in banned.keys():
    for league in list(banned[platform].keys()):
        banned[platform][league]["team"] = {frozenset(k.split(" & ")): v for k,v in banned[platform][league]["team"] .items()}
        banned[platform][league]["opponent"] = {frozenset(k.split(" & ")): v for k,v in banned[platform][league]["opponent"] .items()}

def get_active_sports():
    # Get available sports from the API
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={odds_api}"
    res = requests.get(url)
    res = res.json()

    # Filter sports
    sports = [s["title"] for s in res if s["title"] in [
        "NBA", "MLB", "NHL", "NFL"] and s["active"]]

    return sports

class Scrape:
    def __init__(self, apikey):
        """
        Initialize the Scrape object with the provided API key.

        Args:
            apikey (str): API key for fetching browser headers.
        """
        self.headers = requests.get(
            f"http://headers.scrapeops.io/v1/browser-headers?api_key={scrapeops}"
        ).json()["result"]
        self.header = random.choice(self.headers)
        self.weights = np.ones([len(self.headers)])

    def _new_headers(self):
        """
        Update the weights of the headers and choose a new header based on the weights.
        """
        for i in range(len(self.headers)):
            if self.headers[i] == self.header:
                self.weights[i] = 0
            else:
                self.weights[i] += 1

        self.header = random.choices(self.headers, weights=self.weights)[0]

    def get(self, url, max_attempts=3, headers={}, params={}):
        """
        Perform a GET request to the specified URL with the provided headers and parameters.

        Args:
            url (str): The URL to fetch.
            max_attempts (int): Maximum number of attempts to make the request.
            headers (dict): Additional headers to include in the request.
            params (dict): Parameters to include in the request.

        Returns:
            dict or None: The response JSON if the request is successful (status code 200), otherwise None.

        Raises:
            Exception: If an exception occurs during the request attempts.
        """
        with logging_redirect_tqdm():
            for i in range(1, max_attempts + 1):
                if i > 1:
                    self._new_headers()
                    headers.update(self.headers)
                    sleep(random.uniform(2, 3))
                try:
                    response = requests.get(
                        url, headers=headers, params=params
                    )
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.debug(
                            "Attempt " + str(i) + ", Error " +
                            str(response.status_code)
                        )
                except Exception as exc:
                    logger.exception("Attempt " + str(i) + ",")

            logger.warning("Max Attempts Reached")
            return {}

    def get_proxy(self, url, headers={}):
        params = {
            "api_key": apikey,
            "url": url
        }

        if headers:
            headers = self.header | headers
            params["headers"] = json.dumps(headers)

        i = 0
        while (True):
            i += 1
            response = requests.get(
                "https://scraping.narf.ai/api/v1/",
                params=params
            )
            if response.status_code != 500 or i > 2:
                break

        if response.status_code == 200:
            try:
                response = response.json()
            except:
                return {}

            return response
        else:
            logger.warning("Proxy Failed")
            return {}

    def post(self, url, max_attempts=3, headers={}, params={}):
        """
        Perform a GET request to the specified URL with the provided headers and parameters.

        Args:
            url (str): The URL to fetch.
            max_attempts (int): Maximum number of attempts to make the request.
            headers (dict): Additional headers to include in the request.
            params (dict): Parameters to include in the request.

        Returns:
            dict or None: The response JSON if the request is successful (status code 200), otherwise None.

        Raises:
            Exception: If an exception occurs during the request attempts.
        """
        with logging_redirect_tqdm():
            for i in range(1, max_attempts + 1):
                if i > 1:
                    self._new_headers()
                    sleep(random.uniform(2, 3))
                try:
                    response = requests.post(
                        url, headers=self.header | headers, params=params
                    )
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.debug(
                            "Attempt " + str(i) + ", Error " +
                            str(response.status_code)
                        )
                except Exception as exc:
                    logger.exception("Attempt " + str(i) + ",")

            logger.warning("Max Attempts Reached")
            return None


with open((pkg_resources.files(data) / "name_map.json"), "r") as infile:
    name_map = json.load(infile)


def remove_accents(input_str):
    """
    Remove accents from the input string.

    Args:
        input_str (str): The input string to remove accents from.

    Returns:
        str: The input string without accents.
    """
    if input_str is None:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    out_str = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    out_str = out_str.replace(".", "")
    for substr in [" Jr", " Sr", " II", " III", " IV"]:
        if out_str.endswith(substr):
            out_str = out_str.replace(substr, "")
    out_str = out_str.replace("’", "'")
    out_str = re.sub("[\(\[].*?[\)\]]", "", out_str).strip().title()
    if "+" in out_str:
        names = out_str.split("+")
        out_str = " + ".join([name_map.get(n.strip(), n.strip())
                             for n in names])
    elif " vs " in out_str:
        names = out_str.split(" vs ")
        out_str = " vs. ".join(
            [name_map.get(n.strip(), n.strip()) for n in names])
    else:
        out_str = name_map.get(out_str, out_str)
    return out_str


def odds_to_prob(odds):
    """
    Convert odds to probability.

    Args:
        odds (float): The odds value.

    Returns:
        float: The corresponding probability value.
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        odds = -odds
        return odds / (odds + 100)


def prob_to_odds(p):
    """
    Convert probability to odds.

    Args:
        p (float): The probability value.

    Returns:
        int: The corresponding odds value.
    """
    if p < 0.5:
        return int(np.round((1 - p) / p * 100))
    else:
        return int(np.round((p / (1 - p)) * -100))


def no_vig_odds(over, under=None):
    """
    Calculate no-vig odds given over and under odds.

    Args:
        over (float): The over odds.
        under (float): The under odds.

    Returns:
        list: A list containing the no-vig odds for over and under.
    """
    if np.abs(over) >= 100:
        o = odds_to_prob(over)
    else:
        o = 1 / over
    if under is None or under <= 0:
        juice = 1.0652
        u = juice - o
    else:
        if np.abs(under) >= 100:
            u = odds_to_prob(under)
        else:
            u = 1 / under

        juice = o + u

    return [o / juice, u / juice]


def get_ev(line, under, cv=1, force_gauss=False):
    """
    Calculate the expected value (EV) given a line and under probability.

    Args:
        line (float): The line value.
        under (float): The under probability.

    Returns:
        float: The expected value (EV).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Poisson dist
        if cv == 1:
            if force_gauss:
                line = float(line)
                return fsolve(lambda x: under - norm.cdf(line, x, np.sqrt(x)), (1-under)*2*line)[0]
            else:
                line = np.ceil(float(line) - 1)
                return fsolve(lambda x: under - poisson.cdf(line, x), (1-under)*2*line)[0]
        else:
            line = float(line)
            return fsolve(lambda x: under - norm.cdf(line, x, x*cv), (1-under)*2*line)[0]

def get_odds(line, ev, cv=1, std=None, force_gauss=False, step=1, temp=1):
    high = np.floor((line+step)/step)*step
    low = np.ceil((line-step)/step)*step
    if cv == 1:
        if force_gauss:
            under = norm.cdf(high, ev, np.sqrt(ev)/temp)
            push = under - norm.cdf(low, ev, np.sqrt(ev)/temp)
            return under - push/2
        elif temp==1:
            return poisson.cdf(line, ev) - poisson.pmf(line, ev)/2
        else:
            return skellam.cdf(line, (1/(2*temp)+.5)*ev, (1/(2*temp)-.5)*ev) - skellam.pmf(line, (1/(2*temp)+.5)*ev, (1/(2*temp)-.5)*ev)/2
    else:
        if std is None:
            std = ev*cv
        under = norm.cdf(high, ev, std/temp)
        push = under - norm.cdf(low, ev, std/temp)
        return under - push/2

def fit_distro(mean, std, lower_bound, upper_bound, lower_tol=.1, upper_tol=.001):

    def objective(w, m, s):
        v = w if w >= 1 else 1/w
        if s > 0:
            return 100*max((norm.cdf(lower_bound, w*m, v*s)-lower_tol),0) + max((norm.sf(upper_bound, w*m, v*s)-upper_tol),0) + np.power(1-v, 2)
        else:
            return 100*max((poisson.cdf(lower_bound, w*m)-lower_tol),0) + max((poisson.sf(upper_bound, w*m)-upper_tol),0) + np.power(1-v, 2)
        
    res = minimize(objective, [1], args=(mean, std), bounds=[(.5, 2)], tol=1e-3, method='TNC')
    return res.x[0]

def merge_dict(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            elif np.all(a[key] == b[key]):
                pass  # same leaf value
            else:
                # raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                if key == "Line":
                    a[key].extend(b[key])
                elif key == "EV":
                    a[key].update(b[key])
                else:
                    a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def clean_archive(a, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = (datetime.datetime.today()-datetime.timedelta(days=365*4)).date()
    leagues = list(a.keys())
    for league in leagues:
        markets = list(a[league].keys())

        for market in markets:
            for date in list(a[league][market].keys()):
                if date == '' or datetime.datetime.strptime(date, "%Y-%m-%d").date() < cutoff_date:
                    a[league][market].pop(date)
                    continue

                if market not in ["Moneyline", "Totals", "1st 1 innings"]:
                    players = list(a[league][market][date].keys())
                    for player in players:
                        if player not in a[league][market][date]:
                            continue
                        if " + " in player or " vs. " in player:
                            a[league][market][date].pop(player)
                            continue
                        if "Line" in a[league][market][date][player]["EV"]:
                            a[league][market][date][player]["EV"].pop("Line")

                        player_name = remove_accents(player)
                        if player_name != player:
                            a[league][market][date][player_name] = merge_dict(a[league][market][date].get(player_name,{}), a[league][market][date].pop(player))

                        a[league][market][date][player_name]["Lines"] = [line for line in a[league][market][date][player_name]["Lines"] if line]

                        if not len(a[league][market][date][player_name]["EV"]) and not len(a[league][market][date][player_name]["Lines"]):
                            a[league][market][date].pop(player_name)

                if not len(a[league][market][date]):
                    a[league][market].pop(date)

            if not len(a[league][market]):
                a[league].pop(market)

        if not len(a[league]):
            a.pop(league)

    return a

class Archive:
    """
    A class to manage the archive of sports data.

    Attributes:
        archive (dict): The archive data.

    Methods:
        __getitem__(item): Retrieve an item from the archive.
        add(o, stats, lines, key): Add data to the archive.
        write(): Write the archive data to a file.
    """

    def __init__(self):
        """
        Initialize the Archive class.

        Loads the archive data from a file if it exists.
        """
        # filepath = pkg_resources.files(data) / "archive.dat"
        # if os.path.isfile(filepath):
        #     with open(filepath, "rb") as infile:
        #         self.archive = pickle.load(infile)

        # self.leagues = ["MLB", "NBA", "NHL", "NFL", "NCAAF", "NCAAB", "WNBA", "MISC"]
        # for league in self.leagues:
        #     filepath = pkg_resources.files(data) / f"archive_{league}.dat"
        #     if os.path.isfile(filepath):
        #         with open(filepath, 'rb') as infile:
        #             new_archive = pickle.load(infile)

        #         if type(new_archive) is dict:
        #             self.archive = merge_dict(new_archive, self.archive)

        self.archive = {}
        leagues = [f.name for f in os.scandir("archive") if f.is_dir()]
        for league in leagues:
            self.archive[league] = hdfdir_archive(f"archive/{league}", {}, protocol=-1)
            self.archive[league].load()
                        
        self.default_totals = {
            "MLB": 4.671,
            "NBA": 111.667,
            "WNBA": 81.667,
            "NFL": 22.668,
            "NHL": 2.674
            }
        
        self.changed_leagues = set()

    def __getitem__(self, item):
        """
        Retrieve an item from the archive.

        Args:
            item (str): The key to retrieve from the archive.

        Returns:
            The value associated with the given key in the archive.
        """
        return self.archive[item]

    def add(self, o, lines, key):
        """
        Add data to the archive.

        Args:
            o (dict): The data to add.
            lines (list): The list of lines.
            key (dict): A dictionary for key mapping.

        Returns:
            None
        """
        self.changed_leagues.add(o["League"])
        market = o["Market"].replace("H2H ", "")
        market = key.get(market, market)
        cv = stat_cv.get(market, 1)
        if o["League"] == "NHL":
            market_swap = {"AST": "assists",
                           "PTS": "points", "BLK": "blocked"}
            market = market_swap.get(market, market)
        if o["League"] == "NBA":
            market = market.replace("underdog", "prizepicks")

        if len(lines) < 4:
            lines = [None]*4

        self.archive.setdefault(o["League"], {}).setdefault(market, {})
        self.archive[o["League"]][market].setdefault(o["Date"], {})
        self.archive[o["League"]][market][o["Date"]
                                          ].setdefault(o["Player"], {"Lines": []})

        old_evs = self.archive[o["League"]][market][o["Date"]
                                                    ][o["Player"]].get("EV", [None]*4)
        if len(old_evs) == 0:
            old_evs = [None]*4

        evs = []
        for i, line in enumerate(lines):
            if line:
                ev = get_ev(float(line["Line"]),
                            float(line["Under"]), cv)
            else:
                ev = old_evs[i]

            evs = np.append(evs, ev)

        if o["Line"] and float(o["Line"]) not in self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"]:
            self.archive[o["League"]][market][o["Date"]
                                              ][o["Player"]]["Lines"].append(float(o["Line"]))

        self.archive[o["League"]][market][o["Date"]
                                          ][o["Player"]]["EV"] = evs

    def add_dfs(self, offers, platform, key):
        if not isinstance(offers, list):
            offers = [offers]

        df = pd.DataFrame(offers)
        if "Boost_Over" not in df.columns:
            df["Boost_Over"] = np.nan
        if "Boost" in df.columns:
            df.loc[df["Boost_Over"].isna(), "Boost_Over"] = df.loc[df["Boost_Over"].isna(), "Boost"]
        df["Boost Factor"] = np.abs(df["Boost_Over"]-1)
        df = df.loc[~df.sort_values("Boost Factor").duplicated(["Player", "Market"])]
        offers = df.to_dict(orient='records')
        for o in offers:
            if not o["Line"]:
                continue
            market = o["Market"].replace("H2H ", "")
            market = key.get(market, market)
            cv = stat_cv.get(market, 1)
            if o["League"] == "NHL":
                market_swap = {"AST": "assists",
                            "PTS": "points", "BLK": "blocked"}
                market = market_swap.get(market, market)
            if o["League"] == "NBA":
                market = market.replace("underdog", "prizepicks")

            self.archive.setdefault(o["League"], {}).setdefault(market, {}).setdefault(o["Date"], {}).setdefault(o["Player"], {"EV": {}, "Lines": []})

            if float(o["Line"]) not in self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"]:
                self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"].append(float(o["Line"]))

            over = o.get("Boost_Over", 0) if o.get("Boost_Over", 0) > 0 else o.get("Boost", 1)
            odds = no_vig_odds(over, o.get("Boost_Under"))
            self.archive[o["League"]][market][o["Date"]][o["Player"]]["EV"][platform] = get_ev(o["Line"], odds[1], cv)

    def get_moneyline(self, league, date, team):
        a = []
        w = []
        arr = self.archive.get(league, {}).get("Moneyline", {}).get(date, {}).get(team, {})
        if not arr:
            return .5
        elif type(arr) is not dict:
            self.archive.get(league, {}).get("Moneyline", {}).get(date, {}).pop(team)
            return .5
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get("Moneyline", {}).get(book, 1))

        return np.average(a, weights=w)

    def get_total(self, league, date, team):
        a = []
        w = []
        arr = self.archive.get(league, {}).get("Totals", {}).get(date, {}).get(team, {})
        if not arr:
            return self.default_totals.get(league, 1)
        elif type(arr) is not dict:
            self.archive.get(league, {}).get("Totals", {}).get(date, {}).pop(team)
            return self.default_totals.get(league, 1)
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get("Totals", {}).get(book, 1))

        return np.average(a, weights=w)

    def get_ev(self, league, market, date, player):
        a = []
        w = []
        arr = self.archive.get(league, {}).get(market, {}).get(date, {}).get(player, {}).get("EV", {})
        if not arr:
            return np.nan
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get(market, {}).get(book, 1))

        return np.average(a, weights=w)

    def get_team_market(self, league, market, date, team):
        a = []
        w = []
        arr = self.archive.get(league, {}).get(market, {}).get(date, {}).get(team, {})
        if not arr:
            return np.nan
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get(market, {}).get(book, 1))

        return np.average(a, weights=w)

    def get_line(self, league, market, date, player):
        arr = self.archive.get(league, {}).get(market, {}).get(date, {}).get(player, {}).get("Lines", [np.nan])

        line = np.floor(2*np.median(arr))/2

        return 0 if np.isnan(line) else line
    
    def to_pandas(self, league, market):
        records = {}
        if market not in self.archive[league]:
            return pd.DataFrame()
        for date in list(self.archive[league][market].keys()):
            if market not in ["Moneyline", "Total"] and datetime.datetime.strptime(date, "%Y-%m-%d").date() < datetime.datetime(2023, 5, 3).date():
                continue
            for player in list(self.archive[league][market][date].keys()):
                if "EV" in self.archive[league][market][date][player]:
                    line = self.get_line(league, market, date, player)
                    record = self.archive[league][market][date][player]["EV"].copy()
                    record["Line"] = line
                    records[(date, player)] = record
                else:
                    records[(date, player)] = self.archive[league][market][date][player]

        return pd.DataFrame(records).T

    def write(self, all=False):
        """
        Write the archive data to a file.

        Returns:
            None
        """

        leagues = list(self.changed_leagues)
        if all:
            leagues = list(self.archive.keys())

        for league in leagues:
            if type(self.archive[league]) is not cache:
                self.archive[league] = hdfdir_archive(f"archive/{league}", self.archive[league], protocol=-1)

            self.archive[league].dump()

    def clip(self, cutoff_date=None):
        if cutoff_date is None:
            cutoff_date = (datetime.datetime.today() -
                           datetime.timedelta(days=7))

        for league in list(self.archive.keys()):
            for market in list(self.archive[league].keys()):
                if market not in ['Moneyline', 'Totals']:
                    for date in list(self.archive[league][market].keys()):
                        try:
                            if datetime.datetime.strptime(date, "%Y-%m-%d") < cutoff_date:
                                self.archive[league][market].pop(date)
                        except:
                            self.archive[league][market].pop(date)
                else:
                    for date in list(self.archive[league][market].keys()):
                        try:
                            if datetime.datetime.strptime(date, "%Y-%m-%d") < (datetime.datetime.today() - datetime.timedelta(days=300)):
                                self.archive[league][market].pop(date)
                        except:
                            self.archive[league][market].pop(date)

    def merge(self, filepath):
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as infile:
                new_archive = pickle.load(infile)

            if type(new_archive) is dict:
                self.archive = merge_dict(self.archive, new_archive)

    def rename_market(self, league, old_name, new_name):
        """rename_market Rename a market in the archive"""
        self.changed_leagues.add(league)

        if new_name in self.archive[league]:
            self.archive[league][new_name] = merge_dict(
                self.archive[league][new_name], self.archive[league].pop(old_name))
        else:
            self.archive[league][new_name] = self.archive[league].pop(old_name)


scraper = Scrape(apikey)

archive = Archive()

mlb_games = mlb.schedule(
    start_date=datetime.date.today(), end_date=(datetime.date.today() + datetime.timedelta(days=7))
)
mlb_teams = mlb.get("teams", {"sportId": 1})
mlb_pitchers = {}
for game in mlb_games:
    if game["status"] in ["Pre-Game", "Scheduled"]:
        awayTeam = [
            team["abbreviation"]
            for team in mlb_teams["teams"]
            if team["id"] == game["away_id"]
        ]
        homeTeam = [
            team["abbreviation"]
            for team in mlb_teams["teams"]
            if team["id"] == game["home_id"]
        ]
        if len(awayTeam) == 1 and len(homeTeam) == 1:
            awayTeam = awayTeam[0]
            homeTeam = homeTeam[0]
        else:
            continue
        if game["game_num"] == 1:
            if "away_probable_pitcher" in game and awayTeam not in mlb_pitchers:
                mlb_pitchers[awayTeam] = remove_accents(
                    game["away_probable_pitcher"])
            if "home_probable_pitcher" in game and homeTeam not in mlb_pitchers:
                mlb_pitchers[homeTeam] = remove_accents(
                    game["home_probable_pitcher"])

mlb_pitchers["LA"] = mlb_pitchers.get("LAD", "")
mlb_pitchers["ANA"] = mlb_pitchers.get("LAA", "")
mlb_pitchers["ARI"] = mlb_pitchers.get("AZ", "")
mlb_pitchers["WAS"] = mlb_pitchers.get("WSH", "")


def prob_diff(X, Y, line):
    def joint_pdf(x, y): return X(x)*Y(y)
    return dblquad(joint_pdf, -np.inf, np.inf, lambda x: x - line, np.inf)


def prob_sum(X, Y, line):
    def joint_pdf(x, y): return X(x)*Y(y)
    return dblquad(joint_pdf, -np.inf, np.inf, -np.inf, lambda x: line - x)

def get_trends(x):
    if len(x) < 3:
        trend = np.zeros(len(x.columns))
    else:
        trend = np.polyfit(np.arange(0, len(x.tail(5))), x.tail(5), 1)[0]
    return pd.Series(trend, index=x.columns)

def hmean(items):
    total = 0
    count = 0
    for i in items:
        if (i != 0) and type(i) is not str:
            count += 1
            total += 1/i

    if total != 0:
        return count/total
    else:
        return 0

def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
