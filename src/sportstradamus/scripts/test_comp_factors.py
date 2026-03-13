from sportstradamus.stats import StatsNBA, StatsMLB, StatsNFL, StatsNHL
import nfl_data_py as nfl
from sportstradamus.helpers import remove_accents
import importlib.resources as pkg_resources
from sportstradamus import data
import re
import os
import json
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from itertools import combinations
from math import comb
from tqdm import tqdm

pd.options.mode.chained_assignment = None

NFL = StatsNFL()
NFL.load()

with open(pkg_resources.files(data) / "playerCompStats.json", "r") as infile:
    stats = json.load(infile)

filterStat = {
    "QB": "dropbacks",
    "RB": "attempts",
    "WR": "routes",
    "TE": "routes"
}

players = nfl.import_ids()
players = players.loc[players['position'].isin([
    'QB', 'RB', 'WR', 'TE'])]
players.index = players.name.apply(remove_accents)
players["bmi"] = players["weight"]/players["height"]/players["height"]
players = players[['age', 'height', 'bmi']].dropna()

NFL.profile_market("snap pct")

for position in ["QB", "RB", "WR", "TE"]:
    position_df = pd.DataFrame()
    for year in range(2020, 2024):
        playerProfile = pd.DataFrame()
        ppg = NFL.gamelog.loc[NFL.gamelog.season==year].groupby("player display name")["fantasy points underdog"].mean().to_dict()
        playerFolder = pkg_resources.files(data) / f"player_data/NFL/{year}"
        if os.path.exists(playerFolder):
            for file in os.listdir(playerFolder):
                if file.endswith(".csv"):
                    df = pd.read_csv(playerFolder/file)
                    df.index = df.player_id
                    playerProfile = playerProfile.combine_first(df)

            playerProfile.loc[playerProfile.position=="HB", "position"] = "RB"
            playerProfile.loc[playerProfile.position=="FB", "position"] = "RB"
            playerProfile = playerProfile.loc[playerProfile.position.isin(["QB", "RB", "WR", "TE"])]
            playerProfile.loc[playerProfile.position=="QB", "dropbacks_per_game"] = playerProfile.loc[playerProfile.position=="QB", "dropbacks"] / playerProfile.loc[playerProfile.position=="QB", "player_game_count"]
            playerProfile.loc[playerProfile.position=="QB", "less_grades_pass_diff"] = playerProfile.loc[playerProfile.position=="QB", "less_grades_pass"] - playerProfile.loc[playerProfile.position=="QB", "grades_pass"]
            playerProfile.loc[playerProfile.position=="QB", "blitz_grades_pass_diff"] = playerProfile.loc[playerProfile.position=="QB", "blitz_grades_pass"] - playerProfile.loc[playerProfile.position=="QB", "grades_pass"]
            playerProfile.loc[playerProfile.position=="QB", "scrambles_per_dropback"] = playerProfile.loc[playerProfile.position=="QB", "scrambles"] / playerProfile.loc[playerProfile.position=="QB", "dropbacks"]
            playerProfile.loc[playerProfile.position=="QB", "designed_yards_per_game"] = playerProfile.loc[playerProfile.position=="QB", "designed_yards"] / playerProfile.loc[playerProfile.position=="QB", "player_game_count"]
            playerProfile.loc[playerProfile.position=="RB", "breakaway_yards_per_game"] = playerProfile.loc[playerProfile.position=="RB", "breakaway_yards"] / playerProfile.loc[playerProfile.position=="RB", "player_game_count"]
            playerProfile.loc[playerProfile.position=="RB", "total_touches_per_game"] = playerProfile.loc[playerProfile.position=="RB", "total_touches"] / playerProfile.loc[playerProfile.position=="RB", "player_game_count"]
            playerProfile.loc[playerProfile.position!="QB", "contested_target_rate"] = playerProfile.loc[playerProfile.position!="QB", "contested_targets"] / playerProfile.loc[playerProfile.position!="QB", "targets"]
            playerProfile.loc[playerProfile.position!="QB", "deep_contested_target_rate"] = playerProfile.loc[playerProfile.position!="QB", "deep_contested_targets"] / playerProfile.loc[playerProfile.position!="QB", "targets"]
            playerProfile.loc[playerProfile.position!="QB", "zone_grades_pass_route_diff"] = playerProfile.loc[playerProfile.position!="QB", "zone_grades_pass_route"] - playerProfile.loc[playerProfile.position!="QB", "grades_pass_route"]
            playerProfile.loc[playerProfile.position!="QB", "man_grades_pass_route_diff"] = playerProfile.loc[playerProfile.position!="QB", "man_grades_pass_route"] - playerProfile.loc[playerProfile.position!="QB", "grades_pass_route"]
            playerProfile.index = playerProfile.player.apply(remove_accents)
            playerProfile = playerProfile.join(NFL.playerProfile)
            playerProfile = playerProfile.join(players)

            positionProfile = playerProfile.loc[playerProfile.position==position]
            positionProfile = positionProfile.loc[positionProfile[filterStat[position]] >= positionProfile[filterStat[position]].max()*.1]
            positionProfile.index = positionProfile.player.apply(remove_accents)
            positionProfile = positionProfile[list(stats["NFL"][position].keys())]
            positionProfile["PPG"] = positionProfile.index.map(ppg)
            positionProfile.rename(columns={col: f"{year}_{col}" for col in positionProfile.columns}, inplace=True)
            position_df = position_df.join(positionProfile, how="outer")

    C = position_df.corr(numeric_only=True).replace(np.inf, 0).replace(-np.inf, 0).replace(np.nan, 0).clip(-1,1)
    col_stats = {}
    columns = list({col[5:] for col in C.columns if "_PPG" not in col})
    for col in columns:
        mask = [re.match(rf"\d+_{col}$", row) is not None for row in C.columns]
        n = np.sum(mask)
        if n != C.columns.str.endswith("_PPG").sum():
            continue

        yearly = (C.loc[mask, mask]*(1-np.tri(n))).sum().sum()/n
        fantasy = (C.loc[mask, C.columns.str.endswith("_PPG")]*np.eye(n)).sum().sum()/n

        col_stats[col] = {
            "yearly": yearly,
            "fantasy": np.abs(fantasy),
            "total": 2*np.abs(fantasy) + yearly
        }    
    col_df = pd.DataFrame(col_stats).T.sort_values("total", ascending=False)

    # col_df = col_df.loc[(col_df.fantasy>.3) & (col_df.yearly>.2)]
    columns = list(col_df.index)
    X = np.zeros([len(columns), len(columns)])
    for i, j in tqdm(combinations(range(len(columns)), 2), total=comb(len(columns),2)):
        m1 = [re.match(rf"\d+_{columns[i]}$", row) is not None for row in C.columns]
        m2 = [re.match(rf"\d+_{columns[j]}$", row) is not None for row in C.columns]
        X[i, j] = 1-np.mean(np.abs(C.loc[m1, m2].to_numpy().diagonal()))

    X = np.concatenate([row[i+1:] for i, row in enumerate(X)])
    Z = linkage(X, 'ward')
    col_df["Family"] = fcluster(Z, 8, criterion='maxclust')
    print(position)
    print(col_df.sort_values(["Family", "total"], ascending=False))
