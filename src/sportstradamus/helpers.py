import datetime
import importlib.resources as pkg_resources
import json
import os
import pickle
import random
import re
import unicodedata
import warnings
from time import sleep

import numpy as np
import pandas as pd
import requests
import statsapi as mlb
from klepto.archives import cache, hdfdir_archive
from scipy.optimize import brentq, minimize
from scipy.stats import gamma, nbinom, norm, poisson, skewnorm
from tqdm.contrib.logging import logging_redirect_tqdm

from sportstradamus import creds, data
from sportstradamus.spiderLogger import logger

# Load API key
with open(pkg_resources.files(creds) / "keys.json") as infile:
    keys = json.load(infile)
    odds_api = keys["odds_api"]

with open(pkg_resources.files(data) / "abbreviations.json") as infile:
    abbreviations = json.load(infile)

with open(pkg_resources.files(data) / "combo_props.json") as infile:
    combo_props = json.load(infile)

with open(pkg_resources.files(data) / "stat_cv.json") as infile:
    stat_cv = json.load(infile)

with open(pkg_resources.files(data) / "stat_dist.json") as infile:
    stat_dist = json.load(infile)

with open(pkg_resources.files(data) / "stat_std.json") as infile:
    stat_std = json.load(infile)

_zi_path = pkg_resources.files(data) / "stat_zi.json"
if os.path.isfile(_zi_path):
    with open(_zi_path) as infile:
        stat_zi = json.load(infile)
else:
    stat_zi = {}

with open(pkg_resources.files(data) / "stat_map.json") as infile:
    stat_map = json.load(infile)

with open(pkg_resources.files(data) / "book_weights.json") as infile:
    book_weights = json.load(infile)

with open(pkg_resources.files(data) / "prop_books.json") as infile:
    books = json.load(infile)

with open(pkg_resources.files(data) / "goalies.json") as infile:
    nhl_goalies = json.load(infile)

with open(pkg_resources.files(data) / "feature_filter.json") as infile:
    feature_filter = json.load(infile)

with open(pkg_resources.files(data) / "banned_combos.json") as infile:
    banned = json.load(infile)

for platform in banned:
    for league in list(banned[platform].keys()):
        banned[platform][league]["team"] = {
            frozenset(k.split(" & ")): v for k, v in banned[platform][league]["team"].items()
        }
        banned[platform][league]["opponent"] = {
            frozenset(k.split(" & ")): v for k, v in banned[platform][league]["opponent"].items()
        }


class Scrape:
    def __init__(self):
        """Initialize the Scrape object. Loads API keys from credentials.
        Browser headers are fetched lazily on first use.
        """
        with open(pkg_resources.files(creds) / "keys.json") as f:
            _keys = json.load(f)
        self.apikey = _keys["scrapingfish"]
        self._scrapeops_key = _keys["scrapeops"]
        self._headers = None
        self._header = None
        self._weights = None

    def _ensure_headers(self):
        if self._headers is None:
            self._headers = requests.get(
                f"http://headers.scrapeops.io/v1/browser-headers?api_key={self._scrapeops_key}"
            ).json()["result"]
            self._header = random.choice(self._headers)
            self._weights = np.ones([len(self._headers)])

    @property
    def headers(self):
        self._ensure_headers()
        return self._headers

    @property
    def header(self):
        self._ensure_headers()
        return self._header

    @header.setter
    def header(self, value):
        self._header = value

    @property
    def weights(self):
        self._ensure_headers()
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def _new_headers(self):
        """Update the weights of the headers and choose a new header based on the weights."""
        self._ensure_headers()
        for i in range(len(self._headers)):
            if self._headers[i] == self._header:
                self._weights[i] = 0
            else:
                self._weights[i] += 1

        self._header = random.choices(self._headers, weights=self._weights)[0]

    def get(self, url, max_attempts=3, headers=None, params=None):
        """Perform a GET request to the specified URL with the provided headers and parameters.

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
        if params is None:
            params = {}
        if headers is None:
            headers = {}
        with logging_redirect_tqdm():
            for i in range(1, max_attempts + 1):
                if i > 1:
                    self._new_headers()
                    headers.update(self.header)
                    sleep(random.uniform(1, 3))
                try:
                    response = requests.get(url, headers=headers, params=params)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.debug("Attempt " + str(i) + ", Error " + str(response.status_code))
                except Exception:
                    logger.exception("Attempt " + str(i) + ",")

            logger.warning("Max Attempts Reached")
            return {}

    def get_proxy(self, url, headers=None):
        if headers is None:
            headers = {}
        params = {"api_key": self.apikey, "url": url}

        if headers:
            headers = self.header | headers
            params["headers"] = json.dumps(headers)

        i = 0
        while True:
            i += 1
            response = requests.get("https://scraping.narf.ai/api/v1/", params=params)
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

    def post(self, url, max_attempts=3, headers=None, params=None):
        """Perform a POST request to the specified URL with the provided headers and parameters.

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
        if params is None:
            params = {}
        if headers is None:
            headers = {}
        with logging_redirect_tqdm():
            for i in range(1, max_attempts + 1):
                if i > 1:
                    self._new_headers()
                    sleep(random.uniform(2, 3))
                try:
                    response = requests.post(url, headers=self.header | headers, params=params)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.debug("Attempt " + str(i) + ", Error " + str(response.status_code))
                except Exception:
                    logger.exception("Attempt " + str(i) + ",")

            logger.warning("Max Attempts Reached")
            return None


with open(pkg_resources.files(data) / "name_map.json") as infile:
    name_map = json.load(infile)


def remove_accents(input_str):
    """Remove accents from the input string.

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
    out_str = re.sub(r"[\(\[].*?[\)\]]", "", out_str).strip().title()
    if "+" in out_str:
        names = out_str.split("+")
        out_str = " + ".join([name_map.get(n.strip(), n.strip()) for n in names])
    elif " vs " in out_str:
        names = out_str.split(" vs ")
        out_str = " vs. ".join([name_map.get(n.strip(), n.strip()) for n in names])
    else:
        out_str = name_map.get(out_str, out_str)
    return out_str


def odds_to_prob(odds):
    """Convert odds to probability.

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
    """Convert probability to odds.

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
    """Calculate no-vig odds given over and under odds.

    Args:
        over (float): The over odds.
        under (float): The under odds.

    Returns:
        list: A list containing the no-vig odds for over and under.
    """
    o = odds_to_prob(over) if np.abs(over) >= 100 else 1 / over
    if under is None or under <= 0:
        juice = 1.0652
        u = juice - o
    else:
        u = odds_to_prob(under) if np.abs(under) >= 100 else 1 / under

        juice = o + u

    return [o / juice, u / juice]


def get_ev(line, under, cv=1, dist="Gamma", gate=None, skew_alpha=None):
    """Calculate the expected value (EV) given a line and under probability.

    For zero-inflated distributions (ZINB/ZAGamma), when gate is provided the
    book's CDF is decomposed as gate + (1-gate)*base_CDF.  The function solves
    for the base distribution mean so fused_loc receives comparable parameters.

    Args:
        line (float): The line value.
        under (float): The under probability.
        cv (float): Coefficient of variation. 1/sqrt(alpha) for Gamma, 1/r for NegBin.
        dist (str): Distribution type ("Gamma", "ZAGamma", "NegBin", "ZINB",
                     "Poisson", "SkewNormal").
        gate (float, optional): Historical zero-inflation probability.
        skew_alpha (float, optional): Skewness parameter for SkewNormal. Defaults to 0 (Normal).

    Returns:
        float: The base distribution mean (for ZI dists when gate given) or overall mean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    under = np.clip(under, 1e-6, 1 - 1e-6)

    # For ZI distributions, strip out the zero-inflation component so we solve
    # for the base distribution mean: gate + (1-gate)*base_CDF = under
    # ⇒ base_CDF = (under - gate) / (1 - gate)
    if gate is not None and gate > 0 and dist in ("ZINB", "ZAGamma", "SkewNormal"):
        under = np.clip((under - gate) / (1 - gate), 1e-6, 1 - 1e-6)

    # CDF is monotonically decreasing in mean (1→0), so a bracket is always valid:
    #   at lo≈0 the CDF≈1 > under, at hi→∞ the CDF→0 < under.
    lo = 1e-6
    hi = max(2 * line / max(1 - under, 0.01), 1.0)

    if dist in ("NegBin", "ZINB", "Poisson"):
        line = np.ceil(float(line) - 1)
        if cv == 1:

            def _pois_residual(mean):
                return float(poisson.cdf(line, mean)) - under

            while _pois_residual(hi) > 0:
                hi *= 2
            return float(brentq(_pois_residual, lo, hi, xtol=1e-8))
        r = 1.0 / cv

        def _nb_residual(mean):
            p = r / (r + mean)
            return float(nbinom.cdf(line, r, p)) - under

        while _nb_residual(hi) > 0:
            hi *= 2
        return float(brentq(_nb_residual, lo, hi, xtol=1e-8))

    elif dist == "SkewNormal":
        line = float(line)
        a = float(skew_alpha) if skew_alpha is not None else 0.0

        def _sn_residual(mean):
            sigma = mean * cv
            delta = a / np.sqrt(1 + a**2)
            loc_sn = mean - sigma * delta * np.sqrt(2 / np.pi)
            try:
                return float(skewnorm.cdf(line, a, loc=loc_sn, scale=sigma)) - under
            except (ValueError, RuntimeWarning):
                return np.nan

        # Expand hi until residual becomes negative, with safety cap
        max_hi = 1e8
        while hi < max_hi and _sn_residual(hi) > 0:
            hi = min(hi * 2, max_hi)
        # If we hit the cap and residual is still positive, return a reasonable fallback
        if _sn_residual(hi) > 0 or np.isnan(_sn_residual(hi)):
            return float(line)  # Fallback: return the line as EV
        return float(brentq(_sn_residual, lo, hi, xtol=1e-8))

    else:
        # Gamma / ZAGamma (default for all continuous distributions)
        line = float(line)
        alpha = 1.0 / (cv**2)

        def _gamma_residual(mean):
            return float(gamma.cdf(line, alpha, scale=mean / alpha)) - under

        while _gamma_residual(hi) > 0:
            hi *= 2
        return float(brentq(_gamma_residual, lo, hi, xtol=1e-8))


def get_odds(
    line, ev, dist, cv=1, alpha=None, r=None, gate=None, step=1, sigma=None, skew_alpha=None
):
    """Calculate raw probability of outcome being under the line.

    Returns the raw distributional probability. Calibration (temperature
    scaling) is applied separately at the over/under decision level.

    Parameters:
    -----------
    line : float
        The line/cutoff value
    ev : float
        Expected value (mean)
    dist : str
        Distribution type ("Poisson", "Gamma", "ZAGamma", "NegBin", "ZINB",
        "SkewNormal")
    cv : float
        Coefficient of variation. Used to derive alpha (Gamma) or r (NegBin)
        when those aren't supplied directly.
    alpha : float or np.ndarray, optional
        Gamma shape parameter. If None, derived as 1/cv².
    r : float or np.ndarray, optional
        NegBin dispersion parameter. If None for NegBin, derived as 1/cv.
    gate : float or np.ndarray, optional
        Zero-inflation probability. If None, no zero-inflation.
    step : float, optional
        Step size for binning (default: 1)
    sigma : float or np.ndarray, optional
        SkewNormal scale parameter. If None for SkewNormal, derived as ev*cv.
    skew_alpha : float or np.ndarray, optional
        SkewNormal skewness parameter. If None for SkewNormal, defaults to 0.

    Returns:
    --------
    float : Probability of outcome being under the line
    """
    high = np.floor((line + step) / step) * step
    low = np.ceil((line - step) / step) * step

    # Poisson (discrete count data)
    # NegBin without model params falls back to Poisson only when cv==1 (old encoding);
    # when cv!=1 the archive EV was Gaussian-encoded by get_ev, so fall through to the
    # Gaussian/Gamma branch for a consistent round-trip.
    if dist == "Poisson" or (dist in ("NegBin", "ZINB") and r is None and cv == 1):
        return poisson.cdf(line, ev) - poisson.pmf(line, ev) / 2

    elif dist in ("NegBin", "ZINB"):
        # NegBin / ZINB: use nbinom.cdf for overdispersed count data
        if r is None:
            r = 1 / cv
        p = r / (r + ev)
        base_cdf = nbinom.cdf(line, r, p)
        base_pmf = nbinom.pmf(line, r, p)
        if gate is not None and dist == "ZINB":
            # ZI-CDF: gate + (1 - gate) * base_CDF
            base_cdf = gate + (1 - gate) * base_cdf
            base_pmf = (1 - gate) * base_pmf
        return base_cdf - base_pmf / 2

    elif dist == "SkewNormal":
        # SkewNormal CDF via scipy.stats.skewnorm
        sigma_val = sigma if sigma is not None else ev * cv
        a = skew_alpha if skew_alpha is not None else 0.0
        delta = a / np.sqrt(1 + a**2)
        loc_sn = ev - sigma_val * delta * np.sqrt(2 / np.pi)
        cdf_high = skewnorm.cdf(high, a, loc=loc_sn, scale=sigma_val)
        cdf_low = skewnorm.cdf(low, a, loc=loc_sn, scale=sigma_val)
        if gate is not None:
            cdf_high = gate + (1 - gate) * cdf_high
            cdf_low = gate + (1 - gate) * cdf_low
        push = cdf_high - cdf_low
        return cdf_high - push / 2

    else:
        if alpha is None:
            alpha = 1 / cv**2

        # Gamma / ZAGamma distribution CDF
        cdf_high = gamma.cdf(high, alpha, scale=ev / alpha)
        cdf_low = gamma.cdf(low, alpha, scale=ev / alpha)
        if gate is not None and dist == "ZAGamma":
            # ZA-CDF: gate + (1 - gate) * base_CDF
            cdf_high = gate + (1 - gate) * cdf_high
            cdf_low = gate + (1 - gate) * cdf_low
        push = cdf_high - cdf_low
        return cdf_high - push / 2


def fit_distro(mean, std, lower_bound, upper_bound, lower_tol=0.1, upper_tol=0.001):
    def objective(w, m, s):
        v = w if w >= 1 else 1 / w
        if s > 0:
            return (
                100 * max((norm.cdf(lower_bound, w * m, v * s) - lower_tol), 0)
                + max((norm.sf(upper_bound, w * m, v * s) - upper_tol), 0)
                + np.power(1 - v, 2)
            )
        else:
            return (
                100 * max((poisson.cdf(lower_bound, w * m) - lower_tol), 0)
                + max((poisson.sf(upper_bound, w * m) - upper_tol), 0)
                + np.power(1 - v, 2)
            )

    res = minimize(objective, [1], args=(mean, std), bounds=[(0.5, 2)], tol=1e-3, method="TNC")
    return res.x[0]


def merge_dict(a, b, path=None):
    """Merges b into a."""
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], [*path, str(key)])
            elif np.all(a[key] == b[key]):
                pass  # same leaf value
            # raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            elif key == "Line":
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
        cutoff_date = (datetime.datetime.today() - datetime.timedelta(days=365 * 4)).date()
    leagues = list(a.keys())
    for league in leagues:
        markets = list(a[league].keys())

        for market in markets:
            for date in list(a[league][market].keys()):
                if date == "" or datetime.datetime.strptime(date, "%Y-%m-%d").date() < cutoff_date:
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
                            a[league][market][date][player_name] = merge_dict(
                                a[league][market][date].get(player_name, {}),
                                a[league][market][date].pop(player),
                            )

                        a[league][market][date][player_name]["Lines"] = [
                            line for line in a[league][market][date][player_name]["Lines"] if line
                        ]

                        if not len(a[league][market][date][player_name]["EV"]) and not len(
                            a[league][market][date][player_name]["Lines"]
                        ):
                            a[league][market][date].pop(player_name)

                if not len(a[league][market][date]):
                    a[league][market].pop(date)

            if not len(a[league][market]):
                a[league].pop(market)

        if not len(a[league]):
            a.pop(league)

    return a


class Archive:
    """A class to manage the archive of sports data. Uses singleton pattern
    so all consumers share one instance and one copy of loaded data.

    Attributes:
        archive (dict): The archive data.

    Methods:
        __getitem__(item): Retrieve an item from the archive.
        add(o, stats, lines, key): Add data to the archive.
        write(): Write the archive data to a file.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the Archive class.

        Loads the archive data from a file if it exists.
        """
        if self._initialized:
            return
        self._initialized = True

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
        self._loaded = set()
        leagues = [f.name for f in os.scandir("archive") if f.is_dir()]
        for league in leagues:
            self.archive[league] = hdfdir_archive(f"archive/{league}", {}, protocol=-1)

        self.default_totals = {
            "MLB": 4.671,
            "NBA": 111.667,
            "WNBA": 81.667,
            "NFL": 22.668,
            "NHL": 2.674,
        }

        self._changed_keys = {}

    def _mark_changed(self, league, market):
        self._changed_keys.setdefault(league, set()).add(market)

    def _ensure_loaded(self, league):
        if league not in self._loaded and league in self.archive:
            if isinstance(self.archive[league], cache):
                self.archive[league].load()
            self._loaded.add(league)

    def __getitem__(self, item):
        """Retrieve an item from the archive.

        Args:
            item (str): The key to retrieve from the archive.

        Returns:
            The value associated with the given key in the archive.
        """
        self._ensure_loaded(item)
        return self.archive[item]

    def add(self, o, lines, key):
        """Add data to the archive.

        Args:
            o (dict): The data to add.
            lines (list): The list of lines.
            key (dict): A dictionary for key mapping.

        Returns:
            None
        """
        self._ensure_loaded(o["League"])
        market = o["Market"].replace("H2H ", "")
        market = key.get(market, market)
        cv = stat_cv.get(o["League"], {}).get(market, 1)
        dist = stat_dist.get(o["League"], {}).get(market, "Gamma")
        gate = stat_zi.get(o["League"], {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0
        if o["League"] == "NHL":
            market_swap = {"AST": "assists", "PTS": "points", "BLK": "blocked"}
            market = market_swap.get(market, market)
        if o["League"] == "NBA":
            market = market.replace("underdog", "prizepicks")

        self._mark_changed(o["League"], market)

        if len(lines) < 4:
            lines = [None] * 4

        self.archive.setdefault(o["League"], {}).setdefault(market, {})
        self.archive[o["League"]][market].setdefault(o["Date"], {})
        self.archive[o["League"]][market][o["Date"]].setdefault(o["Player"], {"Lines": []})

        old_evs = self.archive[o["League"]][market][o["Date"]][o["Player"]].get("EV", [None] * 4)
        if len(old_evs) == 0:
            old_evs = [None] * 4

        evs = []
        for i, line in enumerate(lines):
            if line:
                ev = get_ev(
                    float(line["Line"]), float(line["Under"]), cv, dist=dist, gate=gate or None
                )
            else:
                ev = old_evs[i]

            evs = np.append(evs, ev)

        if (
            o["Line"]
            and float(o["Line"])
            not in self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"]
        ):
            self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"].append(
                float(o["Line"])
            )

        self.archive[o["League"]][market][o["Date"]][o["Player"]]["EV"] = evs

    def add_dfs(self, offers, platform, key):
        if not isinstance(offers, list):
            offers = [offers]

        df = pd.DataFrame(offers)
        if "Boost_Over" not in df.columns:
            df["Boost_Over"] = np.nan
        if "Boost" in df.columns:
            df.loc[df["Boost_Over"].isna(), "Boost_Over"] = df.loc[df["Boost_Over"].isna(), "Boost"]
        df["Boost Factor"] = np.abs(df["Boost_Over"] - 1)
        df = df.loc[~df.sort_values("Boost Factor").duplicated(["Player", "Market"])]
        offers = df.to_dict(orient="records")
        for o in offers:
            if not o["Line"]:
                continue
            self._ensure_loaded(o["League"])
            market = o["Market"].replace("H2H ", "")
            market = key.get(market, market)
            cv = stat_cv.get(o["League"], {}).get(market, 1)
            dist = stat_dist.get(o["League"], {}).get(market, "Gamma")
            gate = stat_zi.get(o["League"], {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0
            if o["League"] == "NHL":
                market_swap = {"AST": "assists", "PTS": "points", "BLK": "blocked"}
                market = market_swap.get(market, market)
            if o["League"] == "NBA" or o["League"] == "WNBA":
                market = market.replace("underdog", "prizepicks")

            self._mark_changed(o["League"], market)
            self.archive.setdefault(o["League"], {}).setdefault(market, {}).setdefault(
                o["Date"], {}
            ).setdefault(o["Player"], {"EV": {}, "Lines": []})

            if (
                float(o["Line"])
                not in self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"]
            ):
                self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"].append(
                    float(o["Line"])
                )

            over = o.get("Boost_Over", 0) if o.get("Boost_Over", 0) > 0 else o.get("Boost", 1)
            odds = no_vig_odds(over, o.get("Boost_Under"))
            self.archive[o["League"]][market][o["Date"]][o["Player"]]["EV"][platform] = get_ev(
                o["Line"], odds[1], cv, dist=dist, gate=gate or None
            )

    def get_moneyline(self, league, date, team):
        self._ensure_loaded(league)
        a = []
        w = []
        arr = self.archive.get(league, {}).get("Moneyline", {}).get(date, {}).get(team, {})
        if not arr:
            return 0.5
        elif type(arr) is not dict:
            self.archive.get(league, {}).get("Moneyline", {}).get(date, {}).pop(team)
            return 0.5
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get("Moneyline", {}).get(book, 1))

        return np.average(a, weights=w)

    def get_total(self, league, date, team):
        self._ensure_loaded(league)
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
        self._ensure_loaded(league)
        a = []
        w = []
        arr = (
            self.archive.get(league, {}).get(market, {}).get(date, {}).get(player, {}).get("EV", {})
        )
        if not arr:
            return np.nan
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get(market, {}).get(book, 1))

        return np.average(a, weights=w)

    def get_team_market(self, league, market, date, team):
        self._ensure_loaded(league)
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
        self._ensure_loaded(league)
        arr = (
            self.archive.get(league, {})
            .get(market, {})
            .get(date, {})
            .get(player, {})
            .get("Lines", [np.nan])
        )

        line = np.floor(2 * np.median(arr)) / 2

        return 0 if np.isnan(line) else line

    def to_pandas(self, league, market):
        self._ensure_loaded(league)
        records = {}
        if market not in self.archive[league]:
            return pd.DataFrame()
        for date in list(self.archive[league][market].keys()):
            if (
                market not in ["Moneyline", "Total"]
                and datetime.datetime.strptime(date, "%Y-%m-%d").date()
                < datetime.datetime(2023, 5, 3).date()
            ):
                continue
            for player in list(self.archive[league][market][date].keys()):
                if "EV" in self.archive[league][market][date][player]:
                    line = self.get_line(league, market, date, player)
                    record = self.archive[league][market][date][player]["EV"].copy()
                    record["Line"] = line
                    records[(date, player)] = record
                else:
                    records[(date, player)] = self.archive[league][market][date][player]

        return pd.DataFrame.from_dict(records, orient="index")

    def write(self, all=False):
        """Write the archive data to a file.

        Uses incremental dump (per-market) when possible to avoid
        rewriting unchanged market files.

        Returns:
            None
        """
        if all:
            for league in list(self.archive.keys()):
                if type(self.archive[league]) is not cache:
                    self.archive[league] = hdfdir_archive(
                        f"archive/{league}", self.archive[league], protocol=-1
                    )
                self.archive[league].dump()
        else:
            for league, markets in self._changed_keys.items():
                if type(self.archive[league]) is not cache:
                    # New league from plain dict — must dump everything
                    self.archive[league] = hdfdir_archive(
                        f"archive/{league}", self.archive[league], protocol=-1
                    )
                    self.archive[league].dump()
                else:
                    for market in markets:
                        self.archive[league].dump(market)

        self._changed_keys.clear()

    def clip(self, cutoff_date=None):
        if cutoff_date is None:
            cutoff_date = datetime.datetime.today() - datetime.timedelta(days=7)

        for league in list(self.archive.keys()):
            self._ensure_loaded(league)
            for market in list(self.archive[league].keys()):
                if market not in ["Moneyline", "Totals"]:
                    for date in list(self.archive[league][market].keys()):
                        try:
                            if datetime.datetime.strptime(date, "%Y-%m-%d") < cutoff_date:
                                self.archive[league][market].pop(date)
                                self._mark_changed(league, market)
                        except:
                            self.archive[league][market].pop(date)
                            self._mark_changed(league, market)
                else:
                    for date in list(self.archive[league][market].keys()):
                        try:
                            if datetime.datetime.strptime(date, "%Y-%m-%d") < (
                                datetime.datetime.today() - datetime.timedelta(days=300)
                            ):
                                self.archive[league][market].pop(date)
                                self._mark_changed(league, market)
                        except:
                            self.archive[league][market].pop(date)
                            self._mark_changed(league, market)

    def merge(self, filepath):
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                new_archive = pickle.load(infile)

            if type(new_archive) is dict:
                self.archive = merge_dict(self.archive, new_archive)

    def rename_market(self, league, old_name, new_name):
        """rename_market Rename a market in the archive."""
        self._ensure_loaded(league)
        self._mark_changed(league, new_name)

        if new_name in self.archive[league]:
            self.archive[league][new_name] = merge_dict(
                self.archive[league][new_name], self.archive[league].pop(old_name)
            )
        else:
            self.archive[league][new_name] = self.archive[league].pop(old_name)


_mlb_pitchers_cache = None


def get_mlb_pitchers():
    global _mlb_pitchers_cache
    if _mlb_pitchers_cache is not None:
        return _mlb_pitchers_cache

    mlb_games = mlb.schedule(
        start_date=datetime.date.today(),
        end_date=(datetime.date.today() + datetime.timedelta(days=7)),
    )
    mlb_teams = mlb.get("teams", {"sportId": 1})
    pitchers = {}
    for game in mlb_games:
        if game["status"] in ["Pre-Game", "Scheduled"]:
            awayTeam = [
                team["abbreviation"] for team in mlb_teams["teams"] if team["id"] == game["away_id"]
            ]
            homeTeam = [
                team["abbreviation"] for team in mlb_teams["teams"] if team["id"] == game["home_id"]
            ]
            if len(awayTeam) == 1 and len(homeTeam) == 1:
                awayTeam = awayTeam[0]
                homeTeam = homeTeam[0]
            else:
                continue
            if game["game_num"] == 1:
                if "away_probable_pitcher" in game and awayTeam not in pitchers:
                    pitchers[awayTeam] = remove_accents(game["away_probable_pitcher"])
                if "home_probable_pitcher" in game and homeTeam not in pitchers:
                    pitchers[homeTeam] = remove_accents(game["home_probable_pitcher"])

    pitchers["LA"] = pitchers.get("LAD", "")
    pitchers["ANA"] = pitchers.get("LAA", "")
    pitchers["ARI"] = pitchers.get("AZ", "")
    pitchers["WAS"] = pitchers.get("WSH", "")
    _mlb_pitchers_cache = pitchers
    return pitchers


def fused_loc(
    w,
    ev_a,
    ev_b,
    cv,
    dist,
    *,
    r=None,
    alpha=None,
    sigma=None,
    skew_alpha=None,
    gate_model=None,
    gate_book=None,
):
    """Compute blended distribution parameters for model weight w.

    Blends between model prediction (ev_a) and bookmaker line (ev_b)
    using the logarithmic opinion pool (Genest & Zidek 1986):
    - NegBin: geometric mean of both means and dispersion parameters.
      The model provides per-observation r; the book's r is derived as
      1/cv.  Both μ and r are blended in log-space with the same weight w.
    - Gamma: precision-weighted blend.  The model provides per-observation
      alpha; the book's alpha is derived as 1/cv².  Returns (alpha, beta).
    - SkewNormal: precision-weighted blend of loc/sigma, linear blend of alpha.
      Book uses alpha=0 (symmetric Normal). Returns (ev, sigma, alpha).

    When gate_model and gate_book are supplied (zero-inflated distributions),
    the gate is blended linearly and returned as a final element.  ev_a and
    ev_b should be *base* distribution means (before gate deflation).

    Parameters
    ----------
    w : float
        Weight on model prediction.
    ev_a, ev_b : float or np.ndarray
        Model and bookmaker base distribution means.
    cv : float
        Coefficient of variation for book values.
    dist : str
        Distribution family: "NegBin", "Gamma", or "SkewNormal".
    r : float or np.ndarray, optional
        NegBin per-observation dispersion from model.
    alpha : float or np.ndarray, optional
        Gamma shape parameter from model.
    sigma : float or np.ndarray, optional
        SkewNormal per-observation scale from model.
    skew_alpha : float or np.ndarray, optional
        SkewNormal per-observation skewness from model.
    gate_model : float or np.ndarray, optional
        Model's per-observation zero-inflation gate.
    gate_book : float, optional
        Historical zero-inflation gate for the book side.

    Returns:
    -------
    NegBin    : tuple (r_blend, p, gate_blend)
    Gamma     : tuple (alpha, beta, gate_blend)
    SkewNormal: tuple (blended_ev, blended_sigma, blended_alpha, gate_blend)
    gate_blend is None when no gate parameters are supplied.
    """
    gate_blend = None
    if gate_model is not None and gate_book is not None:
        gate_blend = w * np.asarray(gate_model, dtype=float) + (1 - w) * gate_book
    elif gate_book is not None and gate_book > 0:
        # No model gate (hurdle model) — use book gate directly
        gate_blend = gate_book

    if dist == "NegBin":
        mu = np.exp(
            w * np.log(np.clip(ev_a, 1e-9, None)) + (1 - w) * np.log(np.clip(ev_b, 1e-9, None))
        )
        r_blend = np.exp(w * np.log(np.clip(r, 1e-9, None)) + (1 - w) * np.log(1 / cv))
        p = r_blend / (r_blend + mu)
        return r_blend, p, gate_blend

    elif dist == "SkewNormal":
        ev_a = np.clip(np.asarray(ev_a, dtype=float), 1e-9, None)
        ev_b = np.clip(np.asarray(ev_b, dtype=float), 1e-9, None)
        model_sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
        model_skew = np.asarray(skew_alpha, dtype=float)

        # Book side: symmetric normal (alpha=0), sigma = ev * cv
        book_sigma = np.clip(ev_b * cv, 1e-6, None)

        # Derive loc from EV: loc = EV - sigma * delta * sqrt(2/pi)
        model_delta = model_skew / np.sqrt(1 + model_skew**2)
        model_loc = ev_a - model_sigma * model_delta * np.sqrt(2 / np.pi)
        book_loc = ev_b  # alpha=0 → delta=0 → loc = EV

        # Precision-weighted blend
        prec_m = 1.0 / model_sigma**2
        prec_b = 1.0 / book_sigma**2
        total_prec = w * prec_m + (1 - w) * prec_b
        blended_loc = (w * model_loc * prec_m + (1 - w) * book_loc * prec_b) / total_prec
        blended_sigma = 1.0 / np.sqrt(total_prec)
        blended_skew = w * model_skew  # book alpha=0, so blend reduces to w * model

        # Compute blended EV from blended params
        bl_delta = blended_skew / np.sqrt(1 + blended_skew**2)
        blended_ev = blended_loc + blended_sigma * bl_delta * np.sqrt(2 / np.pi)

        return blended_ev, blended_sigma, blended_skew, gate_blend

    else:  # Gamma – precision-weighted blend
        ev_a = np.clip(np.asarray(ev_a, dtype=float), 1e-9, None)
        ev_b = np.clip(np.asarray(ev_b, dtype=float), 1e-9, None)
        model_alpha = np.clip(np.asarray(alpha, dtype=float), 1e-9, None)
        book_alpha = 1 / cv**2
        inv_var_m = model_alpha / ev_a**2
        inv_var_b = book_alpha / ev_b**2
        total_inv_var = w * inv_var_m + (1 - w) * inv_var_b
        blended_mean = (w * ev_a * inv_var_m + (1 - w) * ev_b * inv_var_b) / total_inv_var
        blended_alpha = blended_mean**2 * total_inv_var
        blended_beta = blended_mean * total_inv_var
        return blended_alpha, blended_beta, gate_blend


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
            total += 1 / i

    if total != 0:
        return count / total
    else:
        return 0


def set_model_start_values(model, dist, X_data, shape_ceiling=None, normalized=False):
    """Set appropriate start values for different distribution types.

    Values are in LightGBMLSS raw space (pre-response-function).
    Response functions per distribution:
      NegBin/ZINB  : total_count → relu,     probs → sigmoid, gate → sigmoid
      Gamma/ZAGamma: concentration → softplus, rate → softplus, gate → sigmoid
      SkewNormal   : loc → identity,  scale → exp,  alpha → identity

    Parameters
    ----------
    model : LightGBMLSS model
    dist : str
        Distribution name ("NegBin", "ZINB", "Gamma", "ZAGamma", "SkewNormal")
    X_data : DataFrame
        Must contain columns "MeanYr" and "STDYr".
    shape_ceiling : float, optional
        Upper bound on shape parameter during training.
    normalized : bool
        If True, targets are normalized to Result/MeanYr ≈ 1.0.
        Start values are set for normalized space.
    """
    from scipy.special import logit

    def _softplus_inv(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 20, x, np.log(np.expm1(np.clip(x, 1e-4, 20))))

    sv = X_data[["MeanYr", "STDYr", "ZeroYr"]].to_numpy()
    n = len(sv)

    mu = np.clip(sv[:, 0], 1e-6, None)
    std = np.clip(sv[:, 1], 1e-6, None)
    hist_gate = np.clip(sv[:, 2], 0, 0.99)

    _r_upper = shape_ceiling if shape_ceiling is not None else 50
    _a_upper = shape_ceiling if shape_ceiling is not None else 100

    if dist == "SkewNormal":
        if normalized:
            # Targets ≈ 1.0 for all players. Use global start values.
            cv_player = np.clip(std / mu, 0.01, 10)
            loc = np.ones(n)
            scale = cv_player  # scale ≈ CV since mean ≈ 1.0
        else:
            loc = mu.copy()
            scale = std.copy()
        alpha_skew = np.zeros(n)  # start symmetric
        # loc: identity → raw = value
        # scale: exp → raw = log(value)
        # alpha: identity → raw = value
        sv = np.column_stack([loc, np.log(np.clip(scale, 1e-6, None)), alpha_skew])

    elif dist in ["NegBin", "ZINB"]:
        # r = mu² / (var - mu); relu response → raw = value (identity for r>0)
        r_init = np.clip(mu**2 / np.clip(std**2 - mu, 1e-6, None), 0.5, _r_upper)
        # PyTorch probs = mu / (mu + r); sigmoid response → raw = logit(probs)
        probs = np.clip(mu / (mu + r_init), 0.01, 0.99)
        if dist == "ZINB":
            nb_zeros = nbinom.pmf(0, r_init, probs)
            hist_gate = np.clip(hist_gate - nb_zeros, 0, 0.99)
            mu = mu / (1 - hist_gate)
            r_init = np.clip(mu**2 / np.clip(std**2 - mu, 1e-6, None), 0.5, _r_upper)
            probs = np.clip(mu / (mu + r_init), 0.01, 0.99)
        sv = np.column_stack([r_init, logit(probs)])

    elif dist in ["Gamma", "ZAGamma"]:
        if dist == "ZAGamma":
            mu = mu / (1 - hist_gate)
        alpha = np.clip((mu / std) ** 2, 0.1, _a_upper)
        beta = np.clip(alpha / np.clip(mu, 1e-6, None), 0.01, 50)
        # softplus response → raw = softplus_inv(value)
        sv = np.column_stack([_softplus_inv(alpha), _softplus_inv(beta)])

    if dist in ["ZINB", "ZAGamma"]:
        gate_val = np.clip(hist_gate, 0.01, 0.99)
        sv = np.column_stack([sv, np.full(n, logit(gate_val))])

    model.start_values = sv
