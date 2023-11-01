from sportstradamus.helpers import Archive, merge_dict, get_ev, odds_to_prob, prob_to_odds
from sportstradamus.stats import StatsNBA
from datetime import datetime
from tqdm import tqdm
from scipy.stats import norm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json
import numpy as np
import pickle

with open(pkg_resources.files(data) / "stat_cv.json", "r") as f:
    stat_cv = json.load(f)

archive = Archive("All")

leagues = ["MLB", "NBA", "NHL", "NFL"]
for league in leagues:
    markets = list(archive[league].keys())
    markets.remove("Moneyline")
    markets.remove("Totals")
    for market in markets:
        # cv = stat_cv[league][market]
        for date in list(archive[league][market].keys()):
            for player in list(archive[league][market][date].keys()):
                lines = list(archive[league][market][date][player].keys())
                if "Closing Lines" in archive[league][market][date][player]:
                    lines.remove("Closing Lines")
                    # EV = []
                    # for i, book in enumerate(archive[league][market][date][player]["Closing Lines"]):
                    #     if book is not None:
                    #         ev = get_ev(float(book["Line"]), odds_to_prob(
                    #             int(book["Under"])), cv)
                    #         EV.append(ev)
                    #         archive[league][market][date][player]["Closing Lines"][i]['EV'] = ev
                    #     else:
                    #         EV.append(None)

                for line in lines:
                    for i, book in enumerate(archive[league][market][date][player][line]):
                        if type(archive[league][market][date][player][line][i]) is dict:
                            archive[league][market][date][player]["Closing Lines"][i] = archive[league][market][date][player][line][i]
                            archive[league][market][date][player][line][i] = None
                        # if EV[i] is not None:
                        #     archive[league][market][date][player][line][i] = norm.sf(
                        #         line, EV[i], cv*EV[i])

archive.write()
