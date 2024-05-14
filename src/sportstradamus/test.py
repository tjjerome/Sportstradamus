from sportstradamus.stats import StatsNBA, StatsMLB, StatsNFL, StatsNHL
from sportstradamus.helpers import scraper, archive, stat_cv, remove_accents
from urllib.parse import urlencode
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
import json
import re
import os
from scipy.stats import norm, poisson
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import requests

pd.options.mode.chained_assignment = None

NFL = StatsNFL()
NFL.load()
NFL.update()
NFL.update_player_comps()
NFL.load()


gamelog = NFL.gamelog.loc[pd.to_datetime(NFL.gamelog["gameday"]).dt.date > (datetime.today().date() - timedelta(days=300))]

gameRecords = []
gameRecords1 = []
gameRecords2 = []
for i, game in tqdm(gamelog.iterrows(), desc="Checking Player Comps", total=len(gamelog)):
    gameDate = pd.to_datetime(game["gameday"]).date()
    if 9 >= gameDate.month >= 8:
        continue
    dateMask = (pd.to_datetime(NFL.gamelog["gameday"]).dt.date > (gameDate - timedelta(days=300))) & (pd.to_datetime(NFL.gamelog["gameday"]).dt.date < gameDate)

    if not NFL.comps[game["position group"]].get(game["player display name"]):
        continue

    if game["position group"] == "QB":
        markets = ["fantasy points underdog", "passing yards", "qb tds", "sacks taken", "attempts"]
    elif game["position group"] == "RB":
        markets = ["fantasy points underdog", "yards", "tds", "carries"]
    else:
        markets = ["fantasy points underdog", "yards", "tds", "targets"]

    defenseVsLeague = NFL.gamelog.loc[dateMask & (NFL.gamelog["opponent"]==game["opponent"])]
    compVsLeague = NFL.gamelog.loc[dateMask & NFL.gamelog["player display name"].isin(NFL.comps[game["position group"]].get(game["player display name"], []))]
    defenseVsPos = defenseVsLeague.loc[defenseVsLeague["position group"] == game["position group"]]
    defenseVsComp = defenseVsLeague.loc[defenseVsLeague["player display name"].isin(NFL.comps[game["position group"]].get(game["player display name"], []))]
    defenseVsPlayer = defenseVsLeague.loc[defenseVsLeague["player display name"] == game["player display name"]]
    posVsLeague = NFL.gamelog.loc[NFL.gamelog["position group"] == game["position group"]]
        
    defenseResult = (defenseVsComp[markets].mean()/defenseVsLeague[markets].mean().replace(0, np.inf)).fillna(0).to_dict()
    compResult = (defenseVsComp[markets].mean()/compVsLeague[markets].mean().replace(0, np.inf)).fillna(0).to_dict()
    
    gameResult = game[markets].fillna(0).to_dict()
    gameRecords2.append(gameResult|
                        {"comp_"+k:v for k, v in compResult.items()}|
                        {"def_"+k:v for k, v in defenseResult.items()})

# game_df = pd.DataFrame(gameRecords)
# C1 = game_df.corr()
game_df = pd.DataFrame(gameRecords2)
markets = [col for col in game_df.columns if "_" not in col]
C2 = game_df.corr()
C = pd.DataFrame([{market: C2.loc[market, "comp_"+market] for market in markets},
                  {market: C2.loc[market, "def_"+market] for market in markets}],
                  index=['comp', 'def'])
print(C)

# filepath = pkg_resources.files(data) / "banned_combos.json"
# with open(filepath, "r") as infile:
#     banned = json.load(infile)

# for platform in banned.keys():
#     for league in list(banned[platform].keys()):
#         if "modified" in list(banned[platform][league].keys()):
#             for market in list(banned[platform][league]["modified"].keys()):
#                 for submarket in list(banned[platform][league]["modified"][market].keys()):
#                     market2 = market
#                     submarket2 = submarket
#                     if "_OPP_" in submarket:
#                         market2 = "_OPP_"+market2
#                         submarket2 = submarket2.replace("_OPP_", "")
#                     banned[platform][league]["modified"].setdefault(submarket2, {})
#                     banned[platform][league]["modified"][submarket2][market2] = banned[platform][league]["modified"][market][submarket]

# with open(filepath, "w") as outfile:
#     json.dump(banned, outfile, indent=4)

# NFL = StatsNFL()
# NFL.season_start = datetime(2018, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2019, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2020, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2021, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2022, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2023, 9, 1).date()
# NFL.update()

# NBA = StatsNBA()
# NBA.load()
# NBA.update()
# NBA.profile_market("BLST")
# stat=NBA.get_stats({
#     "Player": "Tim Hardaway",
#     "Market": "BLST",
#     "Line": 0.5,
#     "Date": datetime(2024, 3, 19).date(),
#     "Team": "DAL",
#     "Opponent": "SAS",
#     "League": "NBA"
# })
# pass
# NBA.season = "2021-22"
# NBA.season_start = datetime(2021, 10, 1).date()
# NBA.update()
# NBA.season = "2022-23"
# NBA.season_start = datetime(2022, 10, 1).date()
# NBA.update()
# NBA.season = "2023-24"
# NBA.season_start = datetime(2023, 10, 1).date()
# NBA.update()

# NHL = StatsNHL()
# NHL.season_start = datetime(2021, 10, 12).date()
# NHL.update()
# NHL.season_start = datetime(2022, 10, 7).date()
# NHL.update()
# NHL.season_start = datetime(2023, 10, 10).date()
# NHL.update()

# MLB = StatsMLB()
# MLB.load()
# MLB.gamelog = pd.DataFrame()
# MLB.teamlog = pd.DataFrame()
# MLB.season_start = datetime(2021, 3, 1).date()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.season_start = datetime(2022, 3, 1).date()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.season_start = datetime(2023, 3, 30).date()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.update()
