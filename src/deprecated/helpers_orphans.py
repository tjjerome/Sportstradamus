# ARCHIVED 2026-04-21 from src/sportstradamus/helpers.py
# Reason: Function- and method-level orphans — defined in helpers.py but never
#         called from any reachable entry point. Bundled here so the live
#         helpers package can shrink without losing the implementations.
# Last live SHA: 18253f8 (functions), HEAD (methods, added 2026-04-24)
# Original imports (now unresolved here):
#   import datetime
#   import json
#   import os
#   import pickle
#   import random
#   from time import sleep
#   import numpy as np
#   import pandas as pd
#   import requests
#   from scipy.integrate import dblquad
#   from tqdm.contrib.logging import logging_redirect_tqdm
#   from sportstradamus import creds
#   from sportstradamus.helpers import (
#       get_ev, merge_dict, stat_cv, stat_dist, stat_zi,
#   )
#   from sportstradamus.spiderLogger import logger
#   # `odds_api` was a module-level constant loaded from creds/keys.json.
#
# Method orphans are de-methodized: the first parameter (`archive` or
# `scraper`) stands in for `self`. To re-introduce, copy the body back onto
# the class and restore the `self.` prefix.


def get_active_sports():
    """Returned the subset of NBA/MLB/NHL/NFL currently marked active by the odds API.

    Originally lived at helpers.py:80. Replaced in practice by the per-league
    flags in `data/active_leagues.json` (or equivalent) and never called.
    """
    # Get available sports from the API
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={odds_api}"  # noqa: F821
    res = requests.get(url)  # noqa: F821
    res = res.json()

    # Filter sports
    sports = [
        s["title"]
        for s in res
        if s["title"] in ["NBA", "MLB", "NHL", "NFL"] and s["active"]
    ]

    return sports


def prob_diff(X, Y, line):
    """Probability that X - Y > line, computed as a 2D integral of the joint PDF.

    `X`, `Y` are PDFs (callables). Originally lived at helpers.py:1128. Used
    nowhere; the project's predictive math takes a different route.
    """

    def joint_pdf(x, y):
        return X(x) * Y(y)

    return dblquad(joint_pdf, -np.inf, np.inf, lambda x: x - line, np.inf)  # noqa: F821


def prob_sum(X, Y, line):
    """Probability that X + Y < line, computed as a 2D integral of the joint PDF.

    Mirror of `prob_diff`. Originally helpers.py:1135. Never called.
    """

    def joint_pdf(x, y):
        return X(x) * Y(y)

    return dblquad(joint_pdf, -np.inf, np.inf, -np.inf, lambda x: line - x)  # noqa: F821


def accel_asc(n):
    """Generator yielding all integer partitions of `n` (Kelleher's algorithm).

    Originally helpers.py:1164. Likely intended for parlay-leg combinatorics
    (the same problem opt_parlay.py solved), but never wired in.
    """
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
        l = k + 1  # noqa: E741
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[: k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[: k + 1]


# ---------------------------------------------------------------------------
# Method orphans archived 2026-04-24 (Phase 2b).
# Each was `self`-qualified on the `Scrape` or `Archive` class in
# src/sportstradamus/helpers.py and had zero call sites across the package,
# scripts/, and tests/. De-methodized here: the first parameter stands in
# for `self`.
# ---------------------------------------------------------------------------


def scrape_get_proxy(scraper, url, headers=None):
    """Fetch `url` through the ScrapeOps proxy endpoint.

    Was: Scrape.get_proxy(self, url, headers=None). Never called.
    """
    if headers is None:
        headers = {}
    params = {"api_key": scraper.apikey, "url": url}

    if headers:
        headers = scraper.header | headers
        params["headers"] = json.dumps(headers)  # noqa: F821

    i = 0
    while True:
        i += 1
        response = requests.get("https://scraping.narf.ai/api/v1/", params=params)  # noqa: F821
        if response.status_code != 500 or i > 2:
            break

    if response.status_code == 200:
        try:
            response = response.json()
        except:  # noqa: E722
            return {}

        return response
    else:
        logger.warning("Proxy Failed")  # noqa: F821
        return {}


def scrape_post(scraper, url, max_attempts=3, headers=None, params=None):
    """POST to `url` with ScrapeOps-rotated headers and retry.

    Was: Scrape.post(self, url, ...). Never called. Mirror of Scrape.get.
    """
    if params is None:
        params = {}
    if headers is None:
        headers = {}
    with logging_redirect_tqdm():  # noqa: F821
        for i in range(1, max_attempts + 1):
            if i > 1:
                scraper._new_headers()
                sleep(random.uniform(2, 3))  # noqa: F821
            try:
                response = requests.post(  # noqa: F821
                    url, headers=scraper.header | headers, params=params
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.debug(  # noqa: F821
                        "Attempt " + str(i) + ", Error " + str(response.status_code)
                    )
            except Exception:
                logger.exception("Attempt " + str(i) + ",")  # noqa: F821

        logger.warning("Max Attempts Reached")  # noqa: F821
        return None


def archive_add(archive, o, lines, key):
    """Add a single offer with multi-book lines to the archive.

    Was: Archive.add(self, o, lines, key). Never called. Looks like the
    intended write path for the `confer` pipeline but was never wired in.
    """
    archive._ensure_loaded(o["League"])
    market = o["Market"].replace("H2H ", "")
    market = key.get(market, market)
    cv = stat_cv.get(o["League"], {}).get(market, 1)  # noqa: F821
    dist = stat_dist.get(o["League"], {}).get(market, "Gamma")  # noqa: F821
    gate = (
        stat_zi.get(o["League"], {}).get(market, 0)  # noqa: F821
        if dist in ("ZINB", "ZAGamma")
        else 0
    )
    if o["League"] == "NHL":
        market_swap = {"AST": "assists", "PTS": "points", "BLK": "blocked"}
        market = market_swap.get(market, market)
    if o["League"] == "NBA":
        market = market.replace("underdog", "prizepicks")

    archive._mark_changed(o["League"], market)

    if len(lines) < 4:
        lines = [None] * 4

    archive.archive.setdefault(o["League"], {}).setdefault(market, {})
    archive.archive[o["League"]][market].setdefault(o["Date"], {})
    archive.archive[o["League"]][market][o["Date"]].setdefault(o["Player"], {"Lines": []})

    old_evs = archive.archive[o["League"]][market][o["Date"]][o["Player"]].get(
        "EV", [None] * 4
    )
    if len(old_evs) == 0:
        old_evs = [None] * 4

    evs = []
    for i, line in enumerate(lines):
        if line:
            ev = get_ev(  # noqa: F821
                float(line["Line"]), float(line["Under"]), cv, dist=dist, gate=gate or None
            )
        else:
            ev = old_evs[i]

        evs = np.append(evs, ev)  # noqa: F821

    if (
        o["Line"]
        and float(o["Line"])
        not in archive.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"]
    ):
        archive.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"].append(
            float(o["Line"])
        )

    archive.archive[o["League"]][market][o["Date"]][o["Player"]]["EV"] = evs


def archive_clip(archive, cutoff_date=None):
    """Drop archive entries older than `cutoff_date`.

    Was: Archive.clip(self, cutoff_date=None). Never called. Props older
    than 7 days are dropped; Moneyline/Totals older than 300 days.
    """
    if cutoff_date is None:
        cutoff_date = datetime.datetime.today() - datetime.timedelta(days=7)  # noqa: F821

    for league in list(archive.archive.keys()):
        archive._ensure_loaded(league)
        for market in list(archive.archive[league].keys()):
            if market not in ["Moneyline", "Totals"]:
                for date in list(archive.archive[league][market].keys()):
                    try:
                        if (
                            datetime.datetime.strptime(date, "%Y-%m-%d")  # noqa: F821
                            < cutoff_date
                        ):
                            archive.archive[league][market].pop(date)
                            archive._mark_changed(league, market)
                    except:  # noqa: E722
                        archive.archive[league][market].pop(date)
                        archive._mark_changed(league, market)
            else:
                for date in list(archive.archive[league][market].keys()):
                    try:
                        if datetime.datetime.strptime(date, "%Y-%m-%d") < (  # noqa: F821
                            datetime.datetime.today()  # noqa: F821
                            - datetime.timedelta(days=300)  # noqa: F821
                        ):
                            archive.archive[league][market].pop(date)
                            archive._mark_changed(league, market)
                    except:  # noqa: E722
                        archive.archive[league][market].pop(date)
                        archive._mark_changed(league, market)


def archive_merge(archive, filepath):
    """Merge another pickled archive dict into this one.

    Was: Archive.merge(self, filepath). Never called.
    """
    if os.path.isfile(filepath):  # noqa: F821
        with open(filepath, "rb") as infile:
            new_archive = pickle.load(infile)  # noqa: F821

        if type(new_archive) is dict:
            archive.archive = merge_dict(archive.archive, new_archive)  # noqa: F821


def archive_rename_market(archive, league, old_name, new_name):
    """Rename a market within a league, merging if `new_name` already exists.

    Was: Archive.rename_market(self, league, old_name, new_name). Never
    called — a likely one-off migration helper.
    """
    archive._ensure_loaded(league)
    archive._mark_changed(league, new_name)

    if new_name in archive.archive[league]:
        archive.archive[league][new_name] = merge_dict(  # noqa: F821
            archive.archive[league][new_name], archive.archive[league].pop(old_name)
        )
    else:
        archive.archive[league][new_name] = archive.archive[league].pop(old_name)
