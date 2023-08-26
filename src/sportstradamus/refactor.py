from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsNBA
from datetime import datetime
from tqdm import tqdm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json
import numpy as np

archive = Archive("NBA")
stats = StatsNBA()
stats.load()
stats.update()

for market in [
    "PTS",
    "REB",
    "AST",
    "PRA",
    "PR",
    "RA",
    "PA",
    "FG3M",
    "fantasy score",
    "fantasy points parlay",
    "TOV",
    "BLK",
    "STL",
    "BLST",
    "FG3A",
    "FTM",
    "FGM",
    "FGA",
    "OREB",
    "DREB",
    "PF",
    "MIN",
]:
    results = []
    for date in tqdm(list(archive["NBA"][market].keys()), desc=f"{market}, Pass #1"):
        gameday_df = stats.gamelog.loc[stats.gamelog['gameId'].str[:10] == date.replace(
            "-", "/")]
        if gameday_df.empty:
            continue
        if len(list(archive["NBA"][market][date].keys())) == 0:
            archive["NBA"][market].pop(date)
            continue
        for player in list(archive["NBA"][market][date].keys()):
            if " vs. " not in player:
                continue
            if "Closing Lines" in list(archive["NBA"][market][date][player].keys()):
                continue
            players = player.split(" vs. ")
            res1 = gameday_df.at[(gameday_df['playerName']
                                  == players[0]).idxmax(), market]
            res2 = gameday_df.at[(gameday_df['playerName']
                                  == players[1]).idxmax(), market]
            for line in list(archive["NBA"][market][date][player].keys()):
                r = res1 - res2 + line
                if r > 0:
                    results.append(1)
                else:
                    results.append(0)

    results = np.array(results)
    ratio = np.mean(results) - .5
    sign = ratio/np.abs(ratio)
    num_to_change = int(sign*(np.sum(results) - len(results)/2))
    if num_to_change == 0:
        continue
    flip_count = int(np.sum(
        results)/num_to_change) if sign > 0 else int(np.sum(1-results)/num_to_change)

    count = 0
    num_changed = 0
    for date in tqdm(list(archive["NBA"][market].keys()), desc=f"{market}, Pass #2"):
        gameday_df = stats.gamelog.loc[stats.gamelog['gameId'].str[:10] == date.replace(
            "-", "/")]
        if gameday_df.empty:
            continue
        for player in list(archive["NBA"][market][date].keys()):
            if " vs. " not in player:
                continue
            if "Closing Lines" in list(archive["NBA"][market][date][player].keys()):
                continue
            players = player.split(" vs. ")
            res1 = gameday_df.at[(gameday_df['playerName']
                                  == players[0]).idxmax(), market]
            res2 = gameday_df.at[(gameday_df['playerName']
                                  == players[1]).idxmax(), market]
            for line in list(archive["NBA"][market][date][player].keys()):
                r = res1 - res2 + line
                if sign*r > 0:
                    count += 1
                    if count == flip_count and num_changed < num_to_change:
                        count = 0
                        num_changed += 1
                        flip_player = " vs. ".join(players[::-1])
                        if flip_player not in archive["NBA"][market][date]:
                            archive["NBA"][market][date][flip_player] = {}
                        archive["NBA"][market][date][flip_player][-line] = list(1-np.array(archive["NBA"][market][date][player].pop(
                            line)))

        if len(list(archive["NBA"][market][date][player].keys())) == 0:
            archive["NBA"][market][date].pop(player)

archive.write()

# NBA = StatsNBA()
# NBA.load()
# NBA.update()

# offer = {
#     "Player": "Jeremy Pena + Jose Altuve",
#     "Market": "hits+runs+rbi",
#     "Date": "2023-08-22",
#     "Line": 2.5,
#     "League": "NBA",
#     "Team": "HOU",
#     "Opponent": "BOS"
# }

# NBA.get_stats(offer, date=offer['Date'])
