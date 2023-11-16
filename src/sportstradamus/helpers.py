from sportstradamus.spiderLogger import logger
import requests
import json
import pickle
import os
import random
import unicodedata
import datetime
import importlib.resources as pkg_resources
from sportstradamus import creds, data
from time import sleep
from scipy.stats import poisson, skellam, norm
from scipy.optimize import fsolve
from scipy.integrate import dblquad
import numpy as np
import statsapi as mlb
import pandas as pd
from scrapeops_python_requests.scrapeops_requests import ScrapeOpsRequests
from tqdm.contrib.logging import logging_redirect_tqdm


# Load API key
filepath = pkg_resources.files(creds) / "odds_api.json"
with open(filepath, "r") as infile:
    odds_api = json.load(infile)["apikey"]

with open((pkg_resources.files(data) / "abbreviations.json"), "r") as infile:
    abbreviations = json.load(infile)

with open((pkg_resources.files(data) / "combo_props.json"), "r") as infile:
    combo_props = json.load(infile)

with open((pkg_resources.files(data) / "stat_cv.json"), "r") as infile:
    stat_cv = json.load(infile)

with open((pkg_resources.files(data) / "goalies.csv"), "r") as infile:
    nhl_goalies = pd.read_csv(infile)
    nhl_goalies = list(nhl_goalies.name.unique())


def get_active_sports():
    # Get available sports from the API
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={odds_api}"
    res = requests.get(url)
    res = res.json()

    # Filter sports
    sports = [s["title"] for s in res if s["title"] in [
        "NBA", "MLB", "NHL", "NFL"] and s["active"]]

    return sports


with open((pkg_resources.files(creds) / "scrapeops_cred.json"), "r") as infile:
    creds = json.load(infile)
apikey = creds["apikey"]
scrapeops_logger = ScrapeOpsRequests(
    scrapeops_api_key=apikey, spider_name="Sportstradamus", job_name=os.uname()[1]
)

requests = scrapeops_logger.RequestsWrapper()


def fit_trendlines(group, n=5):
    trendlines = {}
    for column in group.select_dtypes(include='number').columns:
        if len(group[column].tail(n)) < 2:
            trendlines[column] = 0
        else:
            trendlines[column] = np.polyfit(
                np.arange(len(group[column].tail(n))), group[column].tail(n), 1)[0]
    return pd.Series(trendlines)


class Scrape:
    def __init__(self, apikey):
        """
        Initialize the Scrape object with the provided API key.

        Args:
            apikey (str): API key for fetching browser headers.
        """
        self.headers = requests.get(
            f"http://headers.scrapeops.io/v1/browser-headers?api_key={apikey}"
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
                    sleep(random.uniform(2, 3))
                try:
                    response = requests.get(
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
            return {}

    def get_proxy(self, url, headers={}):
        params = {
            "api_key": apikey,
            "url": url,
            "optimize_request": True
        }

        if headers:
            params["keep_headers"] = True
            headers = headers | self.header

        i = 0
        while (True):
            i += 1
            response = requests.get(
                "https://proxy.scrapeops.io/v1/",
                headers=headers,
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
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    out_str = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    if "+" in out_str:
        names = out_str.split("+")
        out_str = " + ".join([name_map.get(n.strip(), n.strip())
                             for n in names])
    elif "vs." in out_str:
        names = out_str.split("vs.")
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
    if under is None:
        juice = 1.0652
        u = juice - o
    else:
        if np.abs(under) >= 100:
            u = odds_to_prob(under)
        else:
            u = 1 / under

        juice = o + u

    return [o / juice, u / juice]


def get_ev(line, under, cv=1):
    """
    Calculate the expected value (EV) given a line and under probability.

    Args:
        line (float): The line value.
        under (float): The under probability.

    Returns:
        float: The expected value (EV).
    """
    # Poisson dist
    if cv == 1:
        line = np.ceil(float(line) - 1)
        return fsolve(lambda x: under - poisson.cdf(line, x), line)[0]
    else:
        line = float(line)
        return fsolve(lambda x: under - norm.cdf(line, x, x*cv), line)[0]


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
                    evs = b[key]
                    for i, ev in enumerate(evs):
                        if not ev:
                            evs[i] = a[key][i]

                    a[key] = evs
                else:
                    a[key] = b[key]
        else:
            a[key] = b[key]
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

    def __init__(self, league="None"):
        """
        Initialize the Archive class.

        Loads the archive data from a file if it exists.
        """
        self.archive = {}
        filepath = pkg_resources.files(data) / "archive.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.archive = pickle.load(infile)

        self.leagues = ["MLB", "NBA", "NHL", "NFL", "MISC"]
        if league != "None" and league != "All":
            self.leagues = [league]
            filepath = pkg_resources.files(data) / f"archive_{league}.dat"
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as infile:
                    new_archive = pickle.load(infile)

                if type(new_archive) is dict:
                    self.archive = merge_dict(new_archive, self.archive)
        elif league == "All":
            for league in self.leagues:
                filepath = pkg_resources.files(data) / f"archive_{league}.dat"
                if os.path.isfile(filepath):
                    with open(filepath, 'rb') as infile:
                        new_archive = pickle.load(infile)

                    if type(new_archive) is dict:
                        self.archive = merge_dict(new_archive, self.archive)

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
        market = o["Market"].replace("H2H ", "")
        market = key.get(market, market)
        cv = stat_cv.get(market, 1)
        if o["League"] == "NHL":
            market_swap = {"AST": "assists",
                           "PTS": "points", "BLK": "blocked"}
            market = market_swap.get(market, market)

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

        if float(o["Line"]) not in self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"]:
            self.archive[o["League"]][market][o["Date"]
                                              ][o["Player"]]["Lines"].append(float(o["Line"]))

        self.archive[o["League"]][market][o["Date"]
                                          ][o["Player"]]["EV"] = evs

    def add_books(self, offers, position, key):
        key = {v: k for k, v in key.items()}
        for offer in offers:
            lines = [None]*4
            lines[position] = offer
            self.add(offer, lines, key)

    def write(self, overwrite=False):
        """
        Write the archive data to a file.

        Returns:
            None
        """
        for league in self.leagues:

            filepath = pkg_resources.files(data) / f"archive_{league}.dat"
            full_archive = {}
            if os.path.isfile(filepath) and not overwrite:
                with open(filepath, "rb") as infile:
                    full_archive = pickle.load(infile)

            with open(filepath, "wb") as outfile:
                if league == "MISC":
                    misc_leagues = list(self.archive.keys())
                    for l in self.leagues:
                        if l in misc_leagues:
                            misc_leagues.remove(l)
                    for l in misc_leagues:
                        full_archive = merge_dict(
                            full_archive, {l: self.archive[l]})

                    pickle.dump(full_archive,
                                outfile, protocol=-1)
                else:
                    pickle.dump(merge_dict(full_archive, {league: self.archive[league]}),
                                outfile, protocol=-1)

        filepath = pkg_resources.files(data) / "archive.dat"
        self.clip()
        with open(filepath, "wb") as outfile:
            pickle.dump(self.archive, outfile, protocol=-1)

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

    def merge(self, filepath):
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as infile:
                new_archive = pickle.load(infile)

            if type(new_archive) is dict:
                self.archive = merge_dict(self.archive, new_archive)

    def rename_market(self, league, old_name, new_name):
        """rename_market Rename a market in the archive"""

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
