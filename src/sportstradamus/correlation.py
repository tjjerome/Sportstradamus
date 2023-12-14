from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsMLB, StatsNFL, StatsNBA, StatsNHL
import importlib.resources as pkg_resources
from sportstradamus import data
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import os.path

pd.set_option('mode.chained_assignment', None)

# NFL
nfl = StatsNFL()
nfl.load()
nfl.update()

if os.path.isfile((pkg_resources.files(data) / "nfl_corr_data.csv")):
    matrix = pd.read_csv((pkg_resources.files(
        data) / "nfl_corr_data.csv"), index_col=0)
else:
    stats = {
        "QB": [
            "passing yards",
            "rushing yards",
            "qb yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "passing tds",
            "rushing tds",
            "qb tds",
            "completions",
            "carries",
            "interceptions",
            "attempts",
            "sacks taken",
            "longest completion",
            "longest rush",
            "passing first downs",
            "first downs",
            "fumbles lost",
            "completion percentage"
        ],
        "RB": [
            "rushing yards",
            "receiving yards",
            "yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "tds",
            "rushing tds",
            "receiving tds",
            "carries",
            "receptions",
            "targets",
            "longest rush",
            "longest reception",
            "first downs",
            "fumbles lost"
        ],
        "WRTE": [
            "receiving yards",
            "yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "tds",
            "receiving tds",
            "receptions",
            "targets",
            "longest reception",
            "first downs",
            "fumbles lost"
        ],
    }

    games = nfl.gamelog['game id'].unique()
    matrix = pd.DataFrame()

    for gameId in tqdm(games):
        game_df = nfl.gamelog.loc[nfl.gamelog['game id'] == gameId]
        gameDate = datetime.strptime(game_df.iloc[0]['gameday'], '%Y-%m-%d')
        if gameDate < datetime(2021, 9, 1):
            continue
        nfl.profile_market('snap pct', date=gameDate)
        usage = pd.DataFrame(
            nfl.playerProfile[['snap pct short', 'route participation short']])
        usage.reset_index(inplace=True)
        game_df = game_df.merge(usage)
        ranks = game_df.sort_values('route participation short', ascending=False).groupby(
            ["recent team", "position group"]).rank(ascending=False, method='first')["snap pct short"].astype(int)
        game_df['position group'] = game_df['position group'] + \
            ranks.astype(str)
        game_df.index = game_df['position group']

        homeStats = game_df.loc[game_df['home'] & game_df['position group'].str.contains(
            "QB"), stats["QB"]].to_dict('index')
        homeStats.update(game_df.loc[game_df['home'] & game_df['position group'].str.contains(
            "WR"), stats["WRTE"]].to_dict('index'))
        homeStats.update(game_df.loc[game_df['home'] & game_df['position group'].str.contains(
            "TE"), stats["WRTE"]].to_dict('index'))
        homeStats.update(game_df.loc[game_df['home'] & game_df['position group'].str.contains(
            "RB"), stats["RB"]].to_dict('index'))

        awayStats = game_df.loc[~game_df['home'] & game_df['position group'].str.contains(
            "QB"), stats["QB"]].to_dict('index')
        awayStats.update(game_df.loc[~game_df['home'] & game_df['position group'].str.contains(
            "WR"), stats["WRTE"]].to_dict('index'))
        awayStats.update(game_df.loc[~game_df['home'] & game_df['position group'].str.contains(
            "TE"), stats["WRTE"]].to_dict('index'))
        awayStats.update(game_df.loc[~game_df['home'] & game_df['position group'].str.contains(
            "RB"), stats["RB"]].to_dict('index'))

        matrix = pd.concat([matrix, pd.json_normalize(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})], ignore_index=True)
        matrix = pd.concat([matrix, pd.json_normalize(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})], ignore_index=True)

    matrix.to_csv((pkg_resources.files(data) / "nfl_corr_data.csv"))

matrix = matrix.dropna(axis=1, thresh=int(len(matrix)/2))
matrix = matrix.reindex(sorted(matrix.columns), axis=1)
c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(".")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(".")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)

c.to_csv((pkg_resources.files(data) / "NFL_corr.csv"))

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
        "PR",
        "RA",
        "PA",
        "FG3M",
        "fantasy points prizepicks",
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
        "MIN"
    ]

    games = nba.gamelog['GAME_ID'].unique()
    matrix = pd.DataFrame()

    for gameId in tqdm(games):
        game_df = nba.gamelog.loc[nba.gamelog['GAME_ID'] == gameId]
        gameDate = datetime.strptime(
            game_df["GAME_DATE"].max()[:10], '%Y-%m-%d')
        if gameDate < datetime(2021, 10, 26):
            continue
        nba.profile_market('MIN', date=gameDate)
        usage = pd.DataFrame(nba.playerProfile[['MIN short', 'USG_PCT short']])
        usage.reset_index(inplace=True)
        game_df = game_df.merge(usage)
        ranks = game_df.sort_values('USG_PCT short', ascending=False).groupby(
            ["TEAM_ABBREVIATION", "POS"]).rank(ascending=False, method='first')["MIN short"].astype(int)
        game_df['POS'] = game_df['POS'] + \
            ranks.astype(str)
        game_df.index = game_df['POS']

        homeStats = game_df.loc[game_df['HOME'], stats].to_dict('index')
        awayStats = game_df.loc[~game_df['HOME'].astype(
            bool), stats].to_dict('index')

        matrix = pd.concat([matrix, pd.json_normalize(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})], ignore_index=True)
        matrix = pd.concat([matrix, pd.json_normalize(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})], ignore_index=True)

    matrix.to_csv((pkg_resources.files(data) / "nba_corr_data.csv"))

matrix = matrix.dropna(axis=1, thresh=int(len(matrix)/2))
matrix = matrix.reindex(sorted(matrix.columns), axis=1)
c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(".")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(".")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)

c.to_csv((pkg_resources.files(data) / "NBA_corr.csv"))


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
        "sogBS",
        "fantasy points prizepicks",
        "skater fantasy points underdog",
        "blocked",
        "hits",
        "goals",
        "assists",
        "faceOffWins",
        "timeOnIce",
    ]

    goalie_stats = [
        "saves",
        "goalsAgainst",
        "goalie fantasy points underdog"
    ]

    games = nhl.gamelog.gameId.unique()
    matrix = pd.DataFrame()

    for gameId in tqdm(games):
        game_df = nhl.gamelog.loc[nhl.gamelog['gameId'] == gameId]
        gameDate = datetime.strptime(game_df.gameDate.max(), '%Y-%m-%d')
        if gameDate < datetime(2021, 10, 19):
            continue
        nhl.profile_market('TimeShare', date=gameDate)
        usage = pd.DataFrame(
            nhl.playerProfile[['TimeShare short', 'Fenwick short']])
        usage.reset_index(inplace=True)
        game_df = game_df.merge(usage)
        ranks = game_df.sort_values('Fenwick short', ascending=False).groupby(["team", "position"]).rank(
            ascending=False, method='first')["TimeShare short"].astype(int)
        game_df['position'] = game_df['position'] + \
            ranks.astype(str)
        game_df.index = game_df['position']

        homeStats = game_df.loc[game_df['home'] &
                                (game_df['position'] == "G"), goalie_stats].to_dict('index')
        homeStats.update(
            game_df.loc[game_df['home'] & (game_df['position'] != "G"), skater_stats].to_dict('index'))

        awayStats = game_df.loc[~game_df['home'] &
                                (game_df['position'] == "G"), goalie_stats].to_dict('index')
        awayStats.update(
            game_df.loc[~game_df['home'] & (game_df['position'] != "G"), skater_stats].to_dict('index'))

        matrix = pd.concat([matrix, pd.json_normalize(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})], ignore_index=True)
        matrix = pd.concat([matrix, pd.json_normalize(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})], ignore_index=True)

    matrix = pd.DataFrame(matrix)
    matrix.to_csv((pkg_resources.files(data) / "nhl_corr_data.csv"))

matrix = matrix.dropna(axis=1, thresh=int(len(matrix)/2))
matrix = matrix.reindex(sorted(matrix.columns), axis=1)
c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(".")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(".")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)

c.to_csv((pkg_resources.files(data) / "NHL_corr.csv"))


# MLB
mlb = StatsMLB()
mlb.load()
mlb.update()

if os.path.isfile((pkg_resources.files(data) / "mlb_corr_data.csv")):
    matrix = pd.read_csv((pkg_resources.files(
        data) / "mlb_corr_data.csv"), index_col=0)
else:
    stats = {
        "pitcher": [
            "pitcher strikeouts",
            "pitching outs",
            "pitches thrown",
            "hits allowed",
            "runs allowed",
            "1st inning runs allowed",
            "1st inning hits allowed",
            "pitcher fantasy score",
            "pitcher fantasy points underdog",
            "walks allowed"
        ],
        "batter": [
            "hitter fantasy score",
            "hitter fantasy points underdog",
            "hits+runs+rbi",
            "total bases",
            "walks",
            "stolen bases",
            "hits",
            "runs",
            "rbi",
            "batter strikeouts",
            "singles"
        ]
    }

    games = mlb.gamelog.gameId.unique()
    matrix = pd.DataFrame()

    for gameId in tqdm(games):
        game_df = mlb.gamelog.loc[mlb.gamelog['gameId'] == gameId]
        bat_df = game_df.loc[game_df['starting batter']]
        bat_df.position = "B" + bat_df.battingOrder.astype(str)
        bat_df.index = bat_df.position
        pitch_df = game_df.loc[game_df['starting pitcher']]
        pitch_df.position = "P"
        pitch_df.index = pitch_df.position
        gameDate = datetime.strptime(game_df.gameDate.values[0], '%Y-%m-%d')

        homeStats = pitch_df.loc[pitch_df['home'],
                                 stats['pitcher']].to_dict("index")
        homeStats.update(
            bat_df.loc[bat_df['home'], stats['batter']].to_dict("index"))

        awayStats = pitch_df.loc[~pitch_df['home'],
                                 stats['pitcher']].to_dict("index")
        awayStats.update(
            bat_df.loc[~bat_df['home'], stats['batter']].to_dict("index"))

        matrix = pd.concat([matrix, pd.json_normalize(
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})], ignore_index=True)
        matrix = pd.concat([matrix, pd.json_normalize(
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})], ignore_index=True)

    matrix.to_csv((pkg_resources.files(data) / "mlb_corr_data.csv"))

matrix = matrix.dropna(axis=1, thresh=int(len(matrix)/2))
matrix = matrix.reindex(sorted(matrix.columns), axis=1)
c = matrix.corr().unstack()
c = c.iloc[:int(len(c)/2)]
l1 = [i.split(".")[0] for i in c.index.get_level_values(0).to_list()]
l2 = [i.split(".")[0] for i in c.index.get_level_values(1).to_list()]
c = c.loc[[x != y and "OPP" not in x for x, y in zip(l1, l2)]]
c = c.reindex(c.abs().sort_values(ascending=False).index)

c = c.loc[~(c.index.get_level_values(0).str.startswith("P") &
            c.index.get_level_values(1).str.startswith("_OPP_B"))]
c = c.loc[~(c.index.get_level_values(0).str.startswith("B") &
            c.index.get_level_values(1).str.startswith("_OPP_P"))]

c.to_csv((pkg_resources.files(data) / "MLB_corr.csv"))