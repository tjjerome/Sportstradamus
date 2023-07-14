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
from scipy.stats import poisson, skellam
from scipy.optimize import fsolve
import numpy as np
import statsapi as mlb
from scrapeops_python_requests.scrapeops_requests import ScrapeOpsRequests
from tqdm.contrib.logging import logging_redirect_tqdm


with open((pkg_resources.files(creds) / "scrapeops_cred.json"), "r") as infile:
    creds = json.load(infile)
apikey = creds["apikey"]
scrapeops_logger = ScrapeOpsRequests(
    scrapeops_api_key=apikey, spider_name="Sportstradamus", job_name=os.uname()[1]
)

requests = scrapeops_logger.RequestsWrapper()


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

    def get(self, url, max_attempts=5, headers={}, params={}):
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
                    sleep(random.uniform(3, 5))
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
            return None


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
    if out_str == "Michael Porter":
        out_str = "Michael Porter Jr."
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


def no_vig_odds(over, under):
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
    if np.abs(under) >= 100:
        u = odds_to_prob(under)
    else:
        u = 1 / under

    juice = o + u
    return [o / juice, u / juice]


def get_ev(line, under):
    """
    Calculate the expected value (EV) given a line and under probability.

    Args:
        line (float): The line value.
        under (float): The under probability.

    Returns:
        float: The expected value (EV).
    """
    line = np.ceil(float(line) - 1)
    return fsolve(lambda x: under - poisson.cdf(line, x), line)[0]


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

    def __init__(self):
        """
        Initialize the Archive class.

        Loads the archive data from a file if it exists.
        """
        self.archive = {}
        filepath = pkg_resources.files(data) / "archive.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                self.archive = pickle.load(infile)

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
        if o["League"] == "NHL":
            market_swap = {"AST": "assists",
                           "PTS": "points", "BLK": "blockedShots"}
            market = market_swap.get(market, market)

        self.archive.setdefault(o["League"], {}).setdefault(market, {})
        self.archive[o["League"]][market].setdefault(o["Date"], {})
        self.archive[o["League"]][market][o["Date"]
                                          ].setdefault(o["Player"], {})

        odds = []
        for line in lines:
            if line:
                l = np.floor(o["Line"])
                if isinstance(line["EV"], tuple):
                    p = skellam.sf(l, line["EV"][1], line["EV"][0])
                    if np.mod(o["Line"], 1) == 0:
                        p += skellam.pmf(l, line["EV"][1], line["EV"][0]) / 2
                else:
                    p = poisson.sf(l, line["EV"])
                    if np.mod(o["Line"], 1) == 0:
                        p += poisson.pmf(l, line["EV"]) / 2
            else:
                p = None

            np.append(odds, p)

        self.archive[o["League"]][market][o["Date"]
                                          ][o["Player"]][o["Line"]] = odds
        self.archive[o["League"]][market][o["Date"]
                                          ][o["Player"]]["Closing Lines"] = lines

    def write(self):
        """
        Write the archive data to a file.

        Returns:
            None
        """
        filepath = pkg_resources.files(data) / "archive_full.dat"
        full_archive = {}
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                full_archive = pickle.load(infile)

        with open(filepath, "wb") as outfile:
            pickle.dump(merge_dict(full_archive, self.archive),
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

    def refactor(self):
        for league in self.archive.keys():
            for market in self.archive[league].keys():
                if market not in ['Moneyline', 'Totals']:
                    for date in self.archive[league][market].keys():
                        for player in self.archive[league][market][date].keys():
                            for line in self.archive[league][market][date][player].keys():
                                if line != 'Closing Lines':
                                    self.archive[league][market][date][player][line] = self.archive[league][market][date][player][line][5:]
                                    self.archive[league][market][date][player][line][self.archive[league]
                                                                                     [market][date][player][line] == -1000] = None

    def rename_market(self, league, old_name, new_name):
        """rename_market Rename a market in the archive"""

        self.archive[league][new_name] = self.archive[league].pop(old_name)


scraper = Scrape(apikey)

archive = Archive()

mlb_games = mlb.schedule(
    start_date=datetime.date.today(), end_date=datetime.date.today()
)
mlb_teams = mlb.get("teams", {"sportId": 1})
mlb_pitchers = {}
for game in mlb_games:
    if game["status"] == "Pre-Game":
        awayTeam = [
            team["abbreviation"]
            for team in mlb_teams["teams"]
            if team["id"] == game["away_id"]
        ][0]
        homeTeam = [
            team["abbreviation"]
            for team in mlb_teams["teams"]
            if team["id"] == game["home_id"]
        ][0]
        if game["game_num"] == 1:
            mlb_pitchers[awayTeam] = remove_accents(
                game["away_probable_pitcher"])
            mlb_pitchers[homeTeam] = remove_accents(
                game["home_probable_pitcher"])
        elif game["game_num"] > 1:
            mlb_pitchers[awayTeam + str(game["game_num"])] = remove_accents(
                game["away_probable_pitcher"]
            )
            mlb_pitchers[homeTeam + str(game["game_num"])] = remove_accents(
                game["home_probable_pitcher"]
            )

mlb_pitchers["LA"] = mlb_pitchers.get("LAD", "")
mlb_pitchers["ANA"] = mlb_pitchers.get("LAA", "")
mlb_pitchers["ARI"] = mlb_pitchers.get("AZ", "")
mlb_pitchers["WAS"] = mlb_pitchers.get("WSH", "")

urls = [  # "http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams",
    "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams",
    "http://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams",
    "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams",
]

abbreviations = {}
for url in urls:
    res = scraper.get(url)["sports"][0]["leagues"][0]
    league = res["abbreviation"]
    if league not in abbreviations:
        abbreviations[league] = {}
    for team in res["teams"]:
        name = remove_accents(team["team"]["displayName"])
        abbr = team["team"]["abbreviation"]
        abbreviations[league][name] = abbr

abbreviations["NBA"]["Los Angeles Clippers"] = "LAC"
abbreviations["NHL"]["St Louis Blues"] = "STL"
# abbreviations['NFL']['Washington Football Team'] = 'WSH'
