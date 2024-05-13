from sportstradamus.stats import StatsNBA, StatsMLB, StatsNFL, StatsNHL
from sportstradamus.helpers import scraper, archive, stat_cv
from urllib.parse import urlencode
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
import json
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

# players = {}
# NHL.gamelog["season"] = NHL.gamelog.gameId.astype(str).str[:4]
# ppg = NHL.gamelog.groupby(["playerId", "season"])[["goalie fantasy points underdog", "skater fantasy points underdog"]].mean().to_dict('index')
# for year in NHL.players.keys():
#     for player in NHL.players[year].keys():
#         data = NHL.players[year][player]|{"PPG": ppg[(str(player), str(year))]["goalie fantasy points underdog" if NHL.players[year][player]["position"]=="G" else "skater fantasy points underdog"]}
#         players.setdefault(player, {}).update({f"{year}_{k}":v for k, v in data.items()})

# player_df = pd.DataFrame(players).T

# for position in ["C", "W", "D", "G"]:
#     position_df = player_df.loc[player_df["2023_position"] == position].dropna(axis=1, how='all').dropna()
#     position_df.drop(columns=[col for col in position_df.columns if any(substr in col for substr in ["name", "position", "team", "shootsCatches"])], inplace=True)
#     C = position_df.corr()
#     columns = list({col[5:] for col in C.columns})
#     col_stats = {}
#     n = len(NHL.players.keys())
#     for col in columns:
#         yearly = (C.loc[C.index.str.endswith(col), C.columns.str.endswith(col)]*(1-np.tri(3))).sum().sum()/n
#         fantasy = (C.loc[C.index.str.endswith(col), C.columns.str.endswith("PPG")]*np.eye(n)).sum().sum()/n
#         volume = (C.loc[C.index.str.endswith(col), C.columns.str.endswith("timePerGame")]*np.eye(n)).sum().sum()/n

#         col_stats[col] = {
#             "yearly": yearly,
#             "fantasy": fantasy,
#             "volume": volume
#         }    
#     col_df = pd.DataFrame(col_stats).T.sort_values("volume", ascending=False)
#     pass


markets = ["fantasy points prizepicks", "PTS", "REB", "AST", "TOV", "BLK", "STL"]

gamelog = NBA.gamelog.loc[pd.to_datetime(NBA.gamelog["GAME_DATE"]).dt.date > (datetime.today().date() - timedelta(days=300))]

gameRecords = []
gameRecords1 = []
gameRecords2 = []
for i, game in tqdm(gamelog.iterrows(), desc="Checking Player Comps", total=len(gamelog)):
    gameDate = pd.to_datetime(game["GAME_DATE"]).date()
    if 11 >= gameDate.month >= 10:
        continue
    dateMask = (pd.to_datetime(NBA.gamelog["GAME_DATE"]).dt.date > (gameDate - timedelta(days=300))) & (pd.to_datetime(NBA.gamelog["GAME_DATE"]).dt.date < gameDate)

    if not NBA.comps[game["POS"]].get(game["PLAYER_NAME"]):
        continue

    # if game["position"] == "G":
    #     markets = ["goalie fantasy points underdog", "saves", "goalsAgainst"]
    # else:
    #     markets = ["skater fantasy points underdog", "shots", "points", "timeOnIce", "blocked"]

    defenseVsLeague = NBA.gamelog.loc[dateMask & (NBA.gamelog["OPP"]==game["OPP"])]
    compVsLeague = NBA.gamelog.loc[dateMask & NBA.gamelog["PLAYER_NAME"].isin(NBA.comps[game["POS"]].get(game["PLAYER_NAME"], []))]
    defenseVsPos = defenseVsLeague.loc[defenseVsLeague["POS"] == game["POS"]]
    defenseVsComp = defenseVsLeague.loc[defenseVsLeague["PLAYER_NAME"].isin(NBA.comps[game["POS"]].get(game["PLAYER_NAME"], []))]
    defenseVsPlayer = defenseVsLeague.loc[defenseVsLeague["PLAYER_NAME"] == game["PLAYER_NAME"]]
    posVsLeague = NBA.gamelog.loc[NBA.gamelog["POS"] == game["POS"]]

    defenseVsLeague = defenseVsLeague[markets].div(defenseVsLeague["MIN"], axis=0)
    compVsLeague = compVsLeague[markets].div(compVsLeague["MIN"], axis=0)
    defenseVsPos = defenseVsPos[markets].div(defenseVsPos["MIN"], axis=0)
    defenseVsComp = defenseVsComp[markets].div(defenseVsComp["MIN"], axis=0)
    defenseVsPlayer = defenseVsPlayer[markets].div(defenseVsPlayer["MIN"], axis=0)
    posVsLeague = posVsLeague[markets].div(posVsLeague["MIN"], axis=0)
        
    defenseResult = (defenseVsComp.mean()/defenseVsLeague.mean().replace(0, np.inf)).to_dict()
    compResult = (defenseVsComp.mean()/compVsLeague.mean().replace(0, np.inf)).to_dict()
    
    gameResult = (game[markets]/game["MIN"]).to_dict()
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
