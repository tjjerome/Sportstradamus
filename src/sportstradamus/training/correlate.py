"""Per-league player-stat correlation matrix builder."""

import importlib.resources as pkg_resources
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data

_TRACKED_STATS: dict[str, dict] = {
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
            "completion percentage",
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
            "fumbles lost",
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
            "fumbles lost",
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
            "fumbles lost",
        ],
    },
    "NHL": {
        "G": ["saves", "goalsAgainst", "goalie fantasy points underdog"],
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
        "W": [
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
        ],
    },
    "NBA": {
        "C": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
        "P": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
        "B": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
        "F": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
        "W": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
    },
    "WNBA": {
        "G": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
        "F": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
        "C": [
            "PTS", "REB", "AST", "PRA", "PR", "RA", "PA", "FG3M",
            "fantasy points prizepicks", "fantasy points underdog",
            "TOV", "BLK", "STL", "BLST", "FG3A", "FTM", "FGM", "FGA",
            "OREB", "DREB", "PF", "MIN",
        ],
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
            "walks allowed",
        ],
        "B": [
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
            "home runs",
        ],
    },
}


def correlate(league: str, stat_data, force: bool = False) -> None:
    """Calculate feature correlations with outcomes for feature engineering."""
    print(f"Correlating {league}...")
    stats = _TRACKED_STATS[league]
    log = stat_data
    log_str = log.log_strings

    filepath = pkg_resources.files(data) / f"training_data/{league}_corr.csv"
    if filepath.is_file() and not force:
        matrix = pd.read_csv(filepath, index_col=0)
        matrix.DATE = pd.to_datetime(matrix.DATE, format="mixed")
        latest_date = matrix.DATE.max()
        matrix = matrix.loc[datetime.today() - timedelta(days=300) <= matrix.DATE]
    else:
        matrix = pd.DataFrame()
        latest_date = datetime.today() - timedelta(days=300)

    games = log.gamelog[log_str["game"]].unique()
    game_data = []

    for gameId in tqdm(games):
        game_df = log.gamelog.loc[log.gamelog[log_str["game"]] == gameId]
        gameDate = datetime.fromisoformat(game_df.iloc[0][log_str["date"]])
        if gameDate < latest_date or len(game_df[log_str["team"]].unique()) != 2:
            continue
        [home_team, away_team] = tuple(
            game_df.sort_values(log_str["home"], ascending=False)[log_str["team"]].unique()
        )

        if league == "MLB":
            bat_df = game_df.loc[game_df["starting batter"]]
            bat_df.position = "B" + bat_df.battingOrder.astype(str)
            bat_df.index = bat_df.position
            pitch_df = game_df.loc[game_df["starting pitcher"]]
            pitch_df.position = "P"
            pitch_df.index = pitch_df.position
            game_df = pd.concat([bat_df, pitch_df])
        else:
            log.profile_market(log_str["usage"], date=gameDate)
            usage = pd.DataFrame(
                log.playerProfile[
                    [f"{log_str.get('usage')} short", f"{log_str.get('usage_sec')} short"]
                ]
            )
            usage.reset_index(inplace=True)
            game_df = game_df.merge(usage, how="left")
            game_df = game_df.loc[game_df[log_str["position"]].apply(lambda x: isinstance(x, str))]
            game_df = game_df.fillna(0).infer_objects(copy=False)
            ranks = (
                game_df.sort_values(f"{log_str.get('usage_sec')} short", ascending=False)
                .groupby([log_str["team"], log_str["position"]])
                .rank(ascending=False, method="first")[f"{log_str.get('usage')} short"]
                .astype(int)
            )
            game_df[log_str["position"]] = game_df[log_str["position"]] + ranks.astype(str)
            game_df.index = game_df[log_str["position"]]

        homeStats = {}
        awayStats = {}
        for position in stats:
            homeStats.update(
                game_df.loc[
                    (game_df[log_str["team"]] == home_team)
                    & game_df[log_str["position"]].str.contains(position),
                    stats[position],
                ].to_dict("index")
            )
            awayStats.update(
                game_df.loc[
                    (game_df[log_str["team"]] == away_team)
                    & game_df[log_str["position"]].str.contains(position),
                    stats[position],
                ].to_dict("index")
            )

        game_data.append(
            {"TEAM": home_team}
            | {"DATE": gameDate.date()}
            | homeStats
            | {"_OPP_" + k: v for k, v in awayStats.items()}
        )
        game_data.append(
            {"TEAM": away_team}
            | {"DATE": gameDate.date()}
            | awayStats
            | {"_OPP_" + k: v for k, v in homeStats.items()}
        )

    matrix = pd.concat([matrix, pd.json_normalize(game_data)], ignore_index=True)
    matrix.to_csv(filepath)

    big_c = {}
    matrix.drop(columns="DATE", inplace=True)
    matrix.fillna(0, inplace=True)
    for team in matrix.TEAM.unique():
        team_matrix = matrix.loc[team == matrix.TEAM].drop(columns="TEAM")
        team_matrix = team_matrix.loc[:, ((team_matrix == 0).mean() < 0.5)]
        team_matrix = team_matrix.reindex(sorted(team_matrix.columns), axis=1)
        c_spearman = team_matrix.corr(
            method="spearman", min_periods=int(len(team_matrix) * 0.75)
        ).unstack()
        c = 2 * np.sin(np.pi / 6 * c_spearman)
        c = c.reindex(c.abs().sort_values(ascending=False).index).dropna()
        c = c.loc[c.abs() > 0.05]
        big_c.update({team: c})

    pd.concat(big_c).to_csv(pkg_resources.files(data) / f"{league}_corr.csv")
