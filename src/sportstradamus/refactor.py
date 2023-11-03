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
                if len(archive[league][market][date][player]['EV']) > 4:
                    archive[league][market][date][player]['EV'] = archive[league][market][date][player]['EV'][:4]
                    archive[league][market][date][player]['Lines'] = archive[league][market][date][player]['Lines'][::-1]

                if np.nanmean(np.array(archive[league][market][date][player]['EV'], dtype=float)) == archive[league][market][date][player]['Lines'][-1]:
                    archive[league][market][date][player]['EV'] = [None]*4
                # lines = list(archive[league][market][date][player].keys())
                # EV = []
                # if "Closing Lines" in archive[league][market][date][player]:
                #     lines.remove("Closing Lines")
                #     for i, book in enumerate(archive[league][market][date][player]["Closing Lines"]):
                #         if book:
                #             line = float(book["Line"])
                #             if line not in lines:
                #                 lines.insert(0, line)
                #             ev = get_ev(line, odds_to_prob(
                #                 int(book["Under"])), cv)
                #             EV.append(ev)
                #         else:
                #             EV.append(None)
                # else:
                #     for line in lines:
                #         books = archive[league][market][date][player][line]
                #         for p in books:
                #             if p:
                #                 ev = get_ev(line, 1-p, cv)
                #                 EV.append(ev)
                #             else:
                #                 EV.append(None)

            #     if len(lines) > 0:
            #         archive[league][market][date][player] = {
            #             "EV": EV,
            #             "Lines": lines
            #         }
            #     else:
            #         archive[league][market][date].pop(player)

            # if len(archive[league][market][date]) == 0:
            #     archive[league][market].pop(date)
        if len(archive[league][market]) == 0:
            archive[league].pop(market)
    if len(archive[league]) == 0:
        archive.archive.pop(league)

archive.write()
