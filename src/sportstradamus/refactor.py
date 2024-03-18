from sportstradamus.helpers import Archive, merge_dict, get_ev, get_odds, odds_to_prob, prob_to_odds, remove_accents
from sportstradamus.stats import StatsNBA
from datetime import datetime
from tqdm import tqdm
from scipy.stats import norm, iqr
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json
import numpy as np
import pickle

with open(pkg_resources.files(data) / "stat_cv.json", "r") as f:
    stat_cv = json.load(f)

with open(pkg_resources.files(data) / "old/stat_cv.json", "r") as f:
    old_stat_cv = json.load(f)

with open(pkg_resources.files(data) / "prop_books.json", "r") as f:
    books = json.load(f)

archive = Archive("All")

book_pos = [5, 6, -1, 9]
leagues = list(archive.archive.keys())
for league in tqdm(leagues, unit="leagues", position=0):
    markets = list(archive[league].keys())
    if "Moneyline" in markets:
        markets.remove("Moneyline")
        markets.remove("Totals")
    for market in tqdm(markets, desc=league, unit="Markets", position=1):
        cv = stat_cv.get(league, {}).get(market, 1)
        for date in tqdm(list(archive[league][market].keys()), desc=market, unit="Gamedays", position=2):
            for player in list(archive[league][market][date].keys()):
                player_name = remove_accents(player)
                archive[league][market][date][player_name] = archive[league][market][date].pop(player)
                ev = np.empty(len(books))*np.nan
                for i, book_ev in enumerate(archive[league][market][date][player_name]["EV"]):
                    if book_ev and book_pos[i]>=0:
                        lines = archive[league][market][date][player_name]["Lines"]
                        if len(lines) > 2:
                            mu = np.median(lines)
                            sig = iqr(lines)
                            lines = [line for line in lines if mu - sig <= line <= mu + sig]

                        line = lines[-1]
                        book_under = get_odds(line, book_ev, old_stat_cv[league].get(market, 1))
                        book_ev = get_ev(line, book_under, stat_cv[league].get(market, 1))
                        ev[book_pos[i]] = book_ev

                archive[league][market][date][player_name]["EV"] = ev


archive.write()
