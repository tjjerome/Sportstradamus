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
    matrix = []

    for gameId in tqdm(games):
        game_df = mlb.gamelog.loc[mlb.gamelog['gameId'] == gameId]
        gameDate = datetime.strptime(game_df.gameDate.values[0], '%Y-%m-%d')
        homeBatters = game_df.loc[game_df['home'] & game_df['starting batter']].sort_values(
            'battingOrder').index.to_list()
        awayBatters = game_df.loc[~game_df['home'] & game_df['starting batter']].sort_values(
            'battingOrder').index.to_list()
        homePitcher = game_df.loc[game_df['home']
                                  & game_df['starting pitcher'], 'playerName']
        awayPitcher = game_df.loc[~game_df['home']
                                  & game_df['starting pitcher'], 'playerName']
        if homePitcher.empty or awayPitcher.empty:
            continue

        homePitcher = homePitcher.index[0]
        awayPitcher = awayPitcher.index[0]
        homeStats = {"P " + k: v for k,
                     v in game_df.loc[homePitcher, stats['pitcher']].to_dict().items()}
        for i, row in game_df.loc[homeBatters, stats['batter']].iterrows():
            homeStats.update({"B"+str(game_df.loc[homeBatters, 'battingOrder'].at[i])+" "+k: v for k,
                              v in row.to_dict().items()})
        awayStats = {"P " + k: v for k,
                     v in game_df.loc[awayPitcher, stats['pitcher']].to_dict().items()}
        for i, row in game_df.loc[awayBatters, stats['batter']].iterrows():
            awayStats.update({"B"+str(game_df.loc[awayBatters, 'battingOrder'].at[i])+" "+k: v for k,
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
c = c.loc[c.abs() > .1]

c = c.loc[~(c.index.get_level_values(0).str.startswith("P") &
            c.index.get_level_values(1).str.startswith("_OPP_B"))]
c = c.loc[~(c.index.get_level_values(0).str.startswith("B") &
            c.index.get_level_values(1).str.startswith("_OPP_P"))]

# Same team
banned_combos = [
    ("fantasy", "fantasy"),
    ("fantasy", "hits+runs+rbi"),
    ("fantasy", "rbi"),
    ("fantasy", "runs"),
    ("hits+runs+rbi", "hits+runs+rbi"),
    ("hits+runs+rbi", "rbi"),
    ("hits+runs+rbi", "runs"),
    ("rbi", "hits+runs+rbi"),
    ("rbi", "runs"),
    ("runs", "hits+runs+rbi"),
    ("runs", "rbi"),
    ("runs", "runs"),
]

for m1, m2 in banned_combos:
    c = c.loc[~(c.index.get_level_values(0).str.contains(m1) & c.index.get_level_values(
        1).str.contains(m2) & ~c.index.get_level_values(1).str.startswith("_OPP_"))]
    c = c.loc[~(c.index.get_level_values(0).str.contains(m2) & c.index.get_level_values(
        1).str.contains(m1) & ~c.index.get_level_values(1).str.startswith("_OPP_"))]

# Other team
# banned_combos = [

# ]

# for m1, m2 in banned_combos:
#     c = c.loc[~(c.index.get_level_values(0).str.contains(m1) & c.index.get_level_values(
#         1).str.contains(m2) & c.index.get_level_values(1).str.startswith("_OPP_"))]
#     c = c.loc[~(c.index.get_level_values(0).str.contains(m2) & c.index.get_level_values(
#         1).str.contains(m1) & c.index.get_level_values(1).str.startswith("_OPP_"))]

c.to_csv((pkg_resources.files(data) / "MLB_corr.csv"))

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
            "attempts"
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
            "targets"
        ],
        "WRTE": [
            "receiving yards",
            "yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "tds",
            "receiving tds",
            "receptions",
            "targets"
        ],
    }

    games = nfl.gamelog['game id'].unique()
    matrix = []

    for gameId in tqdm(games):
        game_df = nfl.gamelog.loc[nfl.gamelog['game id'] == gameId]
        gameDate = datetime.strptime(game_df.iloc[0]['gameday'], '%Y-%m-%d')
        if gameDate < datetime(2020, 9, 17):
            continue
        nfl.profile_market('snap pct', date=gameDate)

        homeWR = game_df.loc[game_df['home'] & (game_df['position group'] == 'WR'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(3).index.to_list()
        awayWR = game_df.loc[~game_df['home'] & (game_df['position group'] == 'WR'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(3).index.to_list()
        homeRB = game_df.loc[game_df['home'] & (game_df['position group'] == 'RB'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        awayRB = game_df.loc[~game_df['home'] & (game_df['position group'] == 'RB'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(2).index.to_list()
        homeTE = game_df.loc[game_df['home'] & (game_df['position group'] == 'TE'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()
        awayTE = game_df.loc[~game_df['home'] & (game_df['position group'] == 'TE'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()
        homeQB = game_df.loc[game_df['home'] & (game_df['position group'] == 'QB'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()
        awayQB = game_df.loc[~game_df['home'] & (game_df['position group'] == 'QB'), 'player display name'].apply(
            lambda x: nfl.playerProfile.loc[x, 'avg'] if x in nfl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()

        if len(homeWR) == 0 or len(homeRB) == 0 or len(homeTE) == 0 or len(homeQB) == 0 or len(awayWR) == 0 or len(awayRB) == 0 or len(awayTE) == 0 or len(awayQB) == 0:
            continue

        homeStats = {"QB " + k: list(v.values())[0] for k,
                     v in game_df.loc[homeQB, stats['QB']].to_dict().items()}
        for i, row in game_df.loc[homeWR, stats['WRTE']].reset_index(drop=True).iterrows():
            homeStats.update({"WR"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        for i, row in game_df.loc[homeRB, stats['RB']].reset_index(drop=True).iterrows():
            homeStats.update({"RB"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        homeStats.update(
            {"TE "+k: (list(v.values())[0] if v else 0) for k, v in game_df.loc[homeTE, stats['WRTE']].to_dict().items()})

        awayStats = {"QB " + k: list(v.values())[0] for k,
                     v in game_df.loc[awayQB, stats['QB']].to_dict().items()}
        for i, row in game_df.loc[awayWR, stats['WRTE']].reset_index(drop=True).iterrows():
            awayStats.update({"WR"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        for i, row in game_df.loc[awayRB, stats['RB']].reset_index(drop=True).iterrows():
            awayStats.update({"RB"+str(i+1)+" "+k: v for k,
                              v in row.to_dict().items()})
        awayStats.update(
            {"TE "+k: (list(v.values())[0] if v else 0) for k, v in game_df.loc[awayTE, stats['WRTE']].to_dict().items()})

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
c = c.loc[c.abs() > .1]

banned_combos = [
    ("passing tds", "tds"),
    ("passing tds", "receiving tds"),
    ("completions", "receptions"),
    ("attempts", "carries")
]

for m1, m2 in banned_combos:
    c = c.loc[~(c.index.get_level_values(0).str.contains(m1) & c.index.get_level_values(
        1).str.contains(m2) & ~c.index.get_level_values(1).str.startswith("_OPP_"))]
    c = c.loc[~(c.index.get_level_values(0).str.contains(m2) & c.index.get_level_values(
        1).str.contains(m1) & ~c.index.get_level_values(1).str.startswith("_OPP_"))]

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
    matrix = []

    for gameId in tqdm(games):
        game_df = nba.gamelog.loc[nba.gamelog['GAME_ID'] == gameId]
        gameDate = datetime.strptime(
            game_df["GAME_DATE"].max()[:10], '%Y-%m-%d')
        if gameDate < datetime(2021, 10, 26):
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
c.loc[c.abs() > .1].to_csv((pkg_resources.files(data) / "NBA_corr.csv"))

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
    matrix = []

    for gameId in tqdm(games):
        game_df = nhl.gamelog.loc[nhl.gamelog['gameId'] == gameId]
        gameDate = datetime.strptime(game_df.gameDate.max(), '%Y-%m-%d')
        if gameDate < datetime(2021, 10, 19):
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
            sort_values(ascending=False).head(4).index.to_list()
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
            sort_values(ascending=False).head(4).index.to_list()
        homeGoalie = game_df.loc[game_df['home'] & (game_df["position"] == "G"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()
        awayGoalie = game_df.loc[~game_df['home'] & (game_df["position"] == "G"), "playerName"].apply(
            lambda x: nhl.playerProfile.loc[x, 'avg'] if x in nhl.playerProfile.index else -1).\
            sort_values(ascending=False).head(1).index.to_list()

        if len(homeGoalie) == 0 or len(awayGoalie) == 0:
            continue

        homeGoalie = homeGoalie[0]
        awayGoalie = awayGoalie[0]

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
c = c.loc[c.abs() > .1]

# Same team
banned_combos = [
    ("fantasy", "fantasy"),
    ("points", "points"),
    ("points", "goals"),
    ("points", "assists"),
    ("goals", "assists"),
]

for m1, m2 in banned_combos:
    c = c.loc[~(c.index.get_level_values(0).str.contains(m1) & c.index.get_level_values(
        1).str.contains(m2) & ~c.index.get_level_values(1).str.startswith("_OPP_"))]
    c = c.loc[~(c.index.get_level_values(0).str.contains(m2) & c.index.get_level_values(
        1).str.contains(m1) & ~c.index.get_level_values(1).str.startswith("_OPP_"))]

# Other team
banned_combos = [
    ("goalsAgainst", "points"),
    ("goalsAgainst", "goals"),
    ("goalsAgainst", "assists"),
    ("goalsAgainst", "fantasy"),
    ("saves", "points"),
    ("saves", "goals"),
    ("saves", "assists"),
    ("saves", "fantasy"),
    ("saves", "shots"),
    ("fantasy", "fantasy"),
]

for m1, m2 in banned_combos:
    c = c.loc[~(c.index.get_level_values(0).str.contains(m1) & c.index.get_level_values(
        1).str.contains(m2) & c.index.get_level_values(1).str.startswith("_OPP_"))]
    c = c.loc[~(c.index.get_level_values(0).str.contains(m2) & c.index.get_level_values(
        1).str.contains(m1) & c.index.get_level_values(1).str.startswith("_OPP_"))]

c.to_csv((pkg_resources.files(data) / "NHL_corr.csv"))
