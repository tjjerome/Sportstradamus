from sportstradamus.helpers import Archive, merge_dict, get_ev, get_odds, odds_to_prob, prob_to_odds, remove_accents, merge_dict
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
df = archive.to_pandas("NBA", "PRA")
book_pos = {0: 'draftkings', 1: 'fanduel', 2: 'pinnacle', 3: 'williamhill_us'}
leagues = list(archive.archive.keys())
for league in tqdm(leagues, unit="leagues", position=0):
    markets = list(archive[league].keys())
    if "Moneyline" in markets:
        markets.remove("Moneyline")
        markets.remove("Totals")
    if "1st 1 innings" in markets:
        markets.remove("1st 1 innings")
    for market in tqdm(markets, desc=league, unit="Markets", position=1):
        cv = stat_cv.get(league, {}).get(market, 1)
        for date in tqdm(list(archive[league][market].keys()), desc=market, unit="Gamedays", position=2):
            players = list(archive[league][market][date].keys())
            for player in players:
                if " + " in player or " vs. " in player:
                    archive[league][market][date].pop(player)
                    continue
                player_name = remove_accents(player)
                if player_name != player:
                    archive[league][market][date][player_name] = merge_dict(archive[league][market][date].get(player_name,{}), archive[league][market][date].pop(player))
                ev = {}
                old_ev = archive[league][market][date][player_name].get("EV", [None]*4)
                if type(old_ev) is np.ndarray:
                    old_ev = list(old_ev)
                lines = archive[league][market][date][player_name]["Lines"]
                if len(lines) > 2:
                    mu = np.median(lines)
                    sig = iqr(lines)
                    lines = [line for line in lines if mu - sig <= line <= mu + sig]

                if type(old_ev) is not list or len(lines) == 0:
                    continue

                line = lines[-1]
                for i, book_ev in enumerate(old_ev):
                    if book_ev:
                        book_under = get_odds(line, book_ev, old_stat_cv[league].get(market, 1))
                        book_ev = get_ev(line, book_under, stat_cv[league].get(market, 1))
                        ev[book_pos[i]] = book_ev

                archive[league][market][date][player_name]["EV"] = ev

                if not archive[league][market][date][player_name]["EV"] and not archive[league][market][date][player_name]["Lines"]:
                    archive[league][market][date].pop(player_name)

            if not archive[league][market][date]:
                archive[league][market].pop(date)

        if not archive[league][market]:
            archive[league].pop(market)


archive.write()
