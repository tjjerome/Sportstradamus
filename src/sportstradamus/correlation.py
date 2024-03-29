from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsMLB, StatsNFL, StatsNBA, StatsNHL
import importlib.resources as pkg_resources
from sportstradamus import data
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os.path

pd.set_option('mode.chained_assignment', None)

nfl = StatsNFL()
nfl.load()
nfl.update()

nba = StatsNBA()
nba.load()
nba.update()

nhl = StatsNHL()
nhl.load()
nhl.update()

mlb = StatsMLB()
mlb.load()
mlb.update()


tracked_stats = { 
    "NFL": {
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
        "WR": [
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
        "TE": [
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
    },
    "NHL": {
        "G": [
            "saves",
            "goalsAgainst",
            "goalie fantasy points underdog"
        ],
        "C": [
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
        ],
        "L": [
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
        ],
        "R": [
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
        ],
        "D": [
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
    },
    "NBA": {
        "C": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
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
        ],
        "P": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
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
        ],
        "B": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
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
        ],
        "F": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
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
        ],
        "W": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
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
    },
    "MLB": {
        "P": [
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
        "B1": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B2": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B3": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B4": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B5": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B6": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B7": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B8": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
        "B9": [
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
            "singles",
            "doubles",
            "triples",
            "home runs"
        ],
    }
}

log_strings = {
    "NFL": {
        "game": "game id",
        "date": "gameday",
        "usage": "snap pct",
        "usage_sec": "route participation",
        "position": "position group",
        "team": "recent team",
        "home": "home"
    },
    "NBA": {
        "game": "GAME_ID",
        "date": "GAME_DATE",
        "usage": "MIN",
        "usage_sec": "USG_PCT",
        "position": "POS",
        "team": "TEAM_ABBREVIATION",
        "home": "HOME"
    },
    "NHL": {
        "game": "gameId",
        "date": "gameDate",
        "usage": "TimeShare",
        "usage_sec": "Fenwick",
        "position": "position",
        "team": "team",
        "home": "home"
    },
    "MLB": {
        "game": "gameId",
        "date": "gameDate",
        "position": "position",
        "team": "team",
        "home": "home"
    },
}
gamelogs = {
    "NFL": nfl,
    "NBA": nba,
    "NHL": nhl,
    "MLB": mlb
}

for league in ["NHL", "NBA", "MLB", "NFL"]:

    stats = tracked_stats[league]
    log = gamelogs[league]
    log_str = log_strings[league]

    filepath = pkg_resources.files(data) / f"{league}_corr_data.csv"
    if os.path.isfile(filepath) and datetime.fromtimestamp(os.path.getmtime(filepath)) > datetime.today() - timedelta(days=27):
        matrix = pd.read_csv(filepath, index_col=0)
    else:

        games = log.gamelog[log_str["game"]].unique()
        matrix = []

        for gameId in tqdm(games):
            game_df = log.gamelog.loc[log.gamelog[log_str["game"]] == gameId]
            gameDate = datetime.fromisoformat(game_df.iloc[0][log_str["date"]])
            if gameDate < datetime.today()-timedelta(days=400):
                continue
            home_team = game_df.loc[game_df[log_str["home"]], log_str["team"]].iloc[0]
            away_team = game_df.loc[~game_df[log_str["home"]].astype(bool), log_str["team"]].iloc[0]

            if league == "MLB":
                bat_df = game_df.loc[game_df['starting batter']]
                bat_df.position = "B" + bat_df.battingOrder.astype(str)
                bat_df.index = bat_df.position
                pitch_df = game_df.loc[game_df['starting pitcher']]
                pitch_df.position = "P"
                pitch_df.index = pitch_df.position
                game_df = pd.concat([bat_df, pitch_df])
            else:
                log.profile_market(log_str["usage"], date=gameDate)
                usage = pd.DataFrame(
                    log.playerProfile[[f"{log_str.get('usage')} short", f"{log_str.get('usage_sec')} short"]])
                usage.reset_index(inplace=True)
                game_df = game_df.merge(usage, how="left").fillna(0)
                ranks = game_df.sort_values(f"{log_str.get('usage_sec')} short", ascending=False).groupby(
                    [log_str["team"], log_str["position"]]).rank(ascending=False, method='first')[f"{log_str.get('usage')} short"].astype(int)
                game_df[log_str["position"]] = game_df[log_str["position"]] + \
                    ranks.astype(str)
                game_df.index = game_df[log_str["position"]]

            homeStats = {}
            awayStats = {}
            for position in stats.keys():
                homeStats.update(game_df.loc[game_df[log_str["home"]] & game_df[log_str["position"]].str.contains(
                    position), stats[position]].to_dict('index'))
                awayStats.update(game_df.loc[~game_df[log_str["home"]] & game_df[log_str["position"]].str.contains(
                    position), stats[position]].to_dict('index'))

            matrix.append({"TEAM": home_team} |
                homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})
            matrix.append({"TEAM": away_team} |
                awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})

        matrix = pd.json_normalize(matrix)
        matrix.to_csv(filepath)

    big_c = {}
    matrix.fillna(0, inplace=True)
    for team in matrix.TEAM.unique():
        team_matrix = matrix.loc[matrix.TEAM == team].drop(columns="TEAM")
        team_matrix = team_matrix.loc[:,((team_matrix==0).mean() < .5)]
        team_matrix = team_matrix.reindex(sorted(team_matrix.columns), axis=1)
        c = team_matrix.corr(min_periods=int(len(team_matrix)*.75)).unstack()
        # c = c.iloc[:int(len(c)/2)]
        # l1 = [i.split(".")[0] for i in c.index.get_level_values(0).to_list()]
        # l2 = [i.split(".")[0] for i in c.index.get_level_values(1).to_list()]
        # c = c.loc[[x != y for x, y in zip(l1, l2)]]
        c = c.reindex(c.abs().sort_values(ascending=False).index).dropna()
        c = c.loc[c>0.001]
        big_c.update({team: c})

    pd.concat(big_c).to_csv((pkg_resources.files(data) / f"{league}_corr.csv"))