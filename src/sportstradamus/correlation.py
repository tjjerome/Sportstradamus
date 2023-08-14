from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsMLB, StatsNFL
import importlib.resources as pkg_resources
from sportstradamus import data
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import os.path

# MLB
mlb = StatsMLB()
mlb.load()
mlb.update()

if os.path.isfile((pkg_resources.files(data) / "mlb_corr_data.csv")):
    matrix = pd.read_csv((pkg_resources.files(
        data) / "mlb_corr_data.csv"), index_col=0)
else:
    batter_stats = [
        "hits+runs+rbi",
        "total bases",
        "batter strikeouts",
        "hitter fantasy score"
    ]

    pitcher_stats = [
        "pitcher fantasy score",
        "pitcher strikeouts",
        "pitching outs",
        "pitches thrown",
        "hits allowed",
        "runs allowed",
    ]

    games = mlb.gamelog.gameId.unique()
    matrix = []

    for gameId in tqdm(games):
        game_df = mlb.gamelog.loc[mlb.gamelog['gameId'] == gameId]
        gameDate = datetime.strptime(gameId[:10], '%Y/%m/%d')
        if gameDate < datetime(2022, 3, 28):
            continue
        mlb.bucket_stats('hitter fantasy score', date=gameDate)
        homeBatters = game_df.loc[game_df['home'] & game_df['starting batter']].playerName.apply(
            lambda x: mlb.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(5).index.to_list()
        awayBatters = game_df.loc[~game_df['home'] & game_df['starting batter']].playerName.apply(
            lambda x: mlb.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(5).index.to_list()
        homePitcher = game_df.loc[game_df['home']
                                  & game_df['starting pitcher']].index[0]
        awayPitcher = game_df.loc[~game_df['home']
                                  & game_df['starting pitcher']].index[0]
        homeStats = {"P " + k: v for k,
                     v in game_df.loc[homePitcher, pitcher_stats].to_dict().items()}
        for i, row in game_df.loc[homeBatters, batter_stats].reset_index(drop=True).iterrows():
            homeStats.update({"B"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        awayStats = {"P " + k: v for k,
                     v in game_df.loc[awayPitcher, pitcher_stats].to_dict().items()}
        for i, row in game_df.loc[awayBatters, batter_stats].reset_index(drop=True).iterrows():
            awayStats.update({"B"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})

        matrix.append(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})
        matrix.append(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})

    matrix = pd.DataFrame(matrix)
    matrix.to_csv((pkg_resources.files(data) / "mlb_corr_data.csv"))

c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(" ")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(" ")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)
c.loc[c.abs() > .1].to_csv((pkg_resources.files(data) / "mlb_corr.csv"))

# NFL
nfl = StatsNFL()
nfl.load()
nfl.update()

if os.path.isfile((pkg_resources.files(data) / "nfl_corr_data.csv")):
    matrix = pd.read_csv((pkg_resources.files(
        data) / "nfl_corr_data.csv"), index_col=0)
else:
    QB_stats = [
        "passing yards",
        "rushing yards",
        "fantasy points prizepicks",
        "passing tds",
        "rushing tds",
        "completions",
        "interceptions",
        "attempts",
    ]

    WRTE_stats = [
        "yards",
        "fantasy points prizepicks",
        "tds",
        "targets",
        "receptions",
    ]

    RB_stats = [
        "yards",
        "fantasy points prizepicks",
        "tds",
        "targets",
        "receptions",
        "carries",
    ]

    games = nfl.gamelog['game id'].unique()
    matrix = []

    for gameId in tqdm(games):
        game_df = nfl.gamelog.loc[nfl.gamelog['game id'] == gameId]
        gameDate = datetime.strptime(game_df.iat[0, -2], '%Y-%m-%d')
        if gameDate < datetime(2022, 9, 1):
            continue
        nfl.bucket_stats('fantasy points prizepicks', date=gameDate)
        homeWR = game_df.loc[game_df['home'] & (game_df['position group'] == 'WR'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(2).index.to_list()
        awayWR = game_df.loc[~game_df['home'] & (game_df['position group'] == 'WR'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(2).index.to_list()
        homeRB = game_df.loc[game_df['home'] & (game_df['position group'] == 'RB'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(2).index.to_list()
        awayRB = game_df.loc[~game_df['home'] & (game_df['position group'] == 'RB'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(2).index.to_list()
        homeTE = game_df.loc[game_df['home'] & (game_df['position group'] == 'TE'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(1).index.to_list()
        awayTE = game_df.loc[~game_df['home'] & (game_df['position group'] == 'TE'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(1).index.to_list()
        homeQB = game_df.loc[game_df['home'] & (game_df['position group'] == 'QB'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(1).index.to_list()
        awayQB = game_df.loc[~game_df['home'] & (game_df['position group'] == 'QB'), 'player display name'].apply(
            lambda x: nfl.playerStats.get(x, {}).get('avg', 0)).sort_values(ascending=False).head(1).index.to_list()
        homeStats = {"QB " + k: list(v.values())[0] for k,
                     v in game_df.loc[homeQB, QB_stats].to_dict().items()}
        for i, row in game_df.loc[homeWR, WRTE_stats].reset_index(drop=True).iterrows():
            homeStats.update({"WR"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        for i, row in game_df.loc[homeRB, RB_stats].reset_index(drop=True).iterrows():
            homeStats.update({"RB"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        homeStats.update(
            {"TE "+k: (list(v.values())[0] if v else 0) for k, v in game_df.loc[homeTE, WRTE_stats].to_dict().items()})

        awayStats = {"QB " + k: list(v.values())[0] for k,
                     v in game_df.loc[awayQB, QB_stats].to_dict().items()}
        for i, row in game_df.loc[awayWR, WRTE_stats].reset_index(drop=True).iterrows():
            awayStats.update({"WR"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        for i, row in game_df.loc[awayRB, RB_stats].reset_index(drop=True).iterrows():
            awayStats.update({"RB"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        awayStats.update(
            {"TE "+k: (list(v.values())[0] if v else 0) for k, v in game_df.loc[awayTE, WRTE_stats].to_dict().items()})

        matrix.append(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})
        matrix.append(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})

    matrix = pd.DataFrame(matrix)
    matrix.to_csv((pkg_resources.files(data) / "nfl_corr_data.csv"))

c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(" ")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(" ")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)
c.loc[c.abs() > .1].to_csv((pkg_resources.files(data) / "nfl_corr.csv"))
