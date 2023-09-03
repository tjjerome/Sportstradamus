from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsMLB, StatsNFL, StatsNBA, StatsNHL
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
# nfl.update()

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

# NBA
nba = StatsNBA()
nba.load()
nba.update()

if os.path.isfile((pkg_resources.files(data) / "nba_corr_data.csv")):
    matrix = pd.read_csv((pkg_resources.files(
        data) / "nba_corr_data.csv"), index_col=0)
else:
    stats = [
        "PTS",
        "REB",
        "AST",
        "PRA",
        "FG3M",
        "fantasy points prizepicks",
        "TOV",
        "BLK",
        "STL",
        "FTM",
        "PF",
        "MIN",
    ]

    games = nba.gamelog['GAME_ID'].unique()
    matrix = []

    for gameId in tqdm(games):
        game_df = nba.gamelog.loc[nba.gamelog['GAME_ID'] == gameId]
        gameDate = datetime.strptime(
            game_df["GAME_DATE"].max()[:10], '%Y-%m-%d')
        if gameDate < datetime(2022, 9, 1):
            continue
        nba.profile_market('MIN', date=gameDate)

        game_df.index = game_df.PLAYER_NAME
        homeF = game_df.loc[game_df['HOME'] & game_df["POS"].str.contains("Forward"), "PLAYER_NAME"].apply(
            lambda x: nba.playerProfile.loc[x, 'avg'] if x in nba.playerProfile.index else -1).\
            sort_values(ascending=False).head(3).index.to_list()
        homeG = game_df.loc[game_df['HOME'] & game_df["POS"].str.contains("Guard"), "PLAYER_NAME"].apply(
            lambda x: nba.playerProfile.loc[x, 'avg'] if x in nba.playerProfile.index else -1).\
            sort_values(ascending=False).head(3).index.to_list()
        homeC = game_df.loc[game_df['HOME'] & game_df["POS"].str.contains("Center"), "PLAYER_NAME"].apply(
            lambda x: nba.playerProfile.loc[x, 'avg'] if x in nba.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        awayF = game_df.loc[~game_df['HOME'] & game_df["POS"].str.contains("Forward"), "PLAYER_NAME"].apply(
            lambda x: nba.playerProfile.loc[x, 'avg'] if x in nba.playerProfile.index else -1).\
            sort_values(ascending=False).head(3).index.to_list()
        awayG = game_df.loc[~game_df['HOME'] & game_df["POS"].str.contains("Guard"), "PLAYER_NAME"].apply(
            lambda x: nba.playerProfile.loc[x, 'avg'] if x in nba.playerProfile.index else -1).\
            sort_values(ascending=False).head(3).index.to_list()
        awayC = game_df.loc[~game_df['HOME'] & game_df["POS"].str.contains("Center"), "PLAYER_NAME"].apply(
            lambda x: nba.playerProfile.loc[x, 'avg'] if x in nba.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()

        homeStats = {}

        for i, player in enumerate(homeF):
            position = "F"+str(i+1)

            homeStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, stats].to_dict().items()})
        for i, player in enumerate(homeG):
            position = "G"+str(i+1)

            homeStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, stats].to_dict().items()})
        for i, player in enumerate(homeC):
            position = "C"+str(i+1)

            homeStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, stats].to_dict().items()})

        awayStats = {}

        for i, player in enumerate(awayF):
            position = "F"+str(i+1)

            awayStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, stats].to_dict().items()})
        for i, player in enumerate(awayG):
            position = "G"+str(i+1)

            awayStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, stats].to_dict().items()})
        for i, player in enumerate(awayC):
            position = "C"+str(i+1)

            awayStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, stats].to_dict().items()})

        matrix.append(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})
        matrix.append(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})

    matrix = pd.DataFrame(matrix).fillna(0.0)
    matrix.to_csv((pkg_resources.files(data) / "nba_corr_data.csv"))

c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(" ")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(" ")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)
c.loc[c.abs() > .1].to_csv((pkg_resources.files(data) / "nba_corr.csv"))

# NHL
nhl = StatsNHL()
nhl.load()
nhl.update()

if os.path.isfile((pkg_resources.files(data) / "nhl_corr_data.csv")):
    matrix = pd.read_csv((pkg_resources.files(
        data) / "nhl_corr_data.csv"), index_col=0)
else:
    skater_stats = [
        "points",
        "shots",
        "skater fantasy points underdog",
        "blocked",
        "hits",
        "faceOffWins",
        "timeOnIce"
    ]

    goalie_stats = [
        "saves",
        "goalsAgainst",
        "goalie fantasy points underdog",
        "timeOnIce"
    ]

    games = nhl.gamelog.gameId.unique()
    matrix = []

    for gameId in tqdm(games):
        game_df = nhl.gamelog.loc[nhl.gamelog['gameId'] == gameId]
        gameDate = datetime.strptime(game_df.gameDate.max(), '%Y-%m-%d')
        if gameDate < datetime(2022, 9, 1):
            continue
        nhl.profile_market('timeOnIce', date=gameDate)

        game_df.index = game_df.playerName
        homeC = game_df.loc[game_df['home'] & (game_df["position"] == "C"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        homeL = game_df.loc[game_df['home'] & (game_df["position"] == "L"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        homeR = game_df.loc[game_df['home'] & (game_df["position"] == "R"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        homeD = game_df.loc[game_df['home'] & (game_df["position"] == "D"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        awayC = game_df.loc[~game_df['home'] & (game_df["position"] == "C"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        awayL = game_df.loc[~game_df['home'] & (game_df["position"] == "L"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        awayR = game_df.loc[~game_df['home'] & (game_df["position"] == "R"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        awayD = game_df.loc[~game_df['home'] & (game_df["position"] == "D"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        homeGoalie = game_df.loc[game_df['home'] & (game_df["position"] == "G"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()[0]
        awayGoalie = game_df.loc[~game_df['home'] & (game_df["position"] == "G"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()[0]

        homeStats = {"G " + k: v for k,
                     v in game_df.loc[homeGoalie, goalie_stats].to_dict().items()}

        for i, player in enumerate(homeC):
            position = "C"+str(i+1)

            homeStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})
        for i, player in enumerate(homeL):
            position = "L"+str(i+1)

            homeStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})
        for i, player in enumerate(homeR):
            position = "R"+str(i+1)

            homeStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})
        for i, player in enumerate(homeD):
            position = "D"+str(i+1)

            homeStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})

        awayStats = {"G " + k: v for k,
                     v in game_df.loc[awayGoalie, goalie_stats].to_dict().items()}

        for i, player in enumerate(awayC):
            position = "C"+str(i+1)

            awayStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})
        for i, player in enumerate(awayL):
            position = "L"+str(i+1)

            awayStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})
        for i, player in enumerate(awayR):
            position = "R"+str(i+1)

            awayStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})
        for i, player in enumerate(awayD):
            position = "D"+str(i+1)

            awayStats.update({position+" "+k: v for k,
                              v in game_df.loc[player, skater_stats].to_dict().items()})

        matrix.append(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})
        matrix.append(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})

    matrix = pd.DataFrame(matrix)
    matrix.to_csv((pkg_resources.files(data) / "nhl_corr_data.csv"))

c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(" ")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(" ")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)
c.loc[c.abs() > .1].to_csv((pkg_resources.files(data) / "nhl_corr.csv"))
