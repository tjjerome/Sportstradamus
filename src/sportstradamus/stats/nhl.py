"""StatsNHL: NHL player stats loading, feature engineering, and prediction."""

import importlib.resources as pkg_resources
import json
import os.path
import pickle
import warnings
from datetime import datetime, timedelta
from io import StringIO
from time import sleep

import line_profiler
import numpy as np
import pandas as pd
import requests
from scipy.stats import iqr, norm, poisson
from sklearn.neighbors import BallTree
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import (
    Archive,
    Scrape,
    abbreviations,
    combo_props,
    feature_filter,
    get_ev,
    get_mlb_pitchers,
    get_odds,
    remove_accents,
    set_model_start_values,
    stat_cv,
    stat_dist,
)
from sportstradamus.spiderLogger import logger
from sportstradamus.stats.base import Stats, archive, clean_data, scraper


class StatsNHL(Stats):
    """A class for handling and analyzing NHL statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        skater_data (list): A list containing skater statistics.
        goalie_data (list): A list containing goalie statistics.
        season_start (datetime.datetime): The start date of the season.

    Additional Methods:
        None
    """

    def __init__(self):
        super().__init__()
        self.season_start = datetime(2024, 10, 4).date()
        self.skater_stats = [
            "GOE",
            "Fenwick",
            "TimeShare",
            "ShotShare",
            "Shot60",
            "Blk60",
            "Hit60",
            "Ast60",
        ]
        self.stat_types = {
            "skater": [
                "GOE",
                "Fenwick",
                "TimeShare",
                "ShotShare",
                "Shot60",
                "Blk60",
                "Hit60",
                "Ast60",
            ],
            "goalie": ["SV", "SOE", "goalsAgainst", "Freeze", "Rebound", "RG"],
        }
        self.team_stat_types = [
            "Corsi",
            "Fenwick",
            "Hits",
            "Takeaways",
            "PIM",
            "Corsi_Pct",
            "Fenwick_Pct",
            "Hits_Pct",
            "Takeaways_Pct",
            "PIM_Pct",
            "Block_Pct",
            "xGoals",
            "xGoalsAgainst",
            "goalsAgainst",
            "GOE",
            "SV",
            "SOE",
            "Freeze",
            "Rebound",
            "RG",
        ]
        self.volume_stats = ["timeOnIce", "shotsAgainst"]
        self.default_total = 2.674
        self.positions = ["C", "W", "D", "G"]
        self.league = "NHL"
        self.log_strings = {
            "game": "gameId",
            "date": "gameDate",
            "player": "playerName",
            "usage": "TimeShare",
            "usage_sec": "Fenwick",
            "position": "position",
            "team": "team",
            "opponent": "opponent",
            "home": "home",
            "win": "WL",
            "score": "goals",
        }
        self.usage_stat = "TimeShare"
        self.tiebreaker_stat = "Fenwick short"
        self._volume_model_cache = None

    def load(self):
        """Loads NHL skater and goalie data from files.

        Args:
            None

        Returns:
            None
        """
        filepath = pkg_resources.files(data) / "nhl_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                nhl_data = pickle.load(infile)
                self.players = nhl_data.get("players", {})
                self.gamelog = nhl_data.get("gamelog", {})
                self.teamlog = nhl_data.get("teamlog", {})

    def build_comp_profile(self, playerDict=None):
        """Build NHL player comp profile from loaded player data.

        Args:
            playerDict: Optional flat dict of {player_id: stats_dict}.
                If None, uses all seasons from self.players.

        Returns:
            (playerProfile, all_players, id_to_name) where playerProfile is
            a DataFrame indexed by player IDs, all_players is the flat dict,
            and id_to_name maps integer IDs to string player names.
        """
        if playerDict is None:
            playerDict = {}
            for season_key in self.players:
                playerDict.update(self.players[season_key])

        if not playerDict:
            return pd.DataFrame(), {}, {}

        playerProfile = pd.DataFrame(playerDict).T
        id_to_name = {pid: v.get("playerName", pid) for pid, v in playerDict.items()}

        return playerProfile, playerDict, id_to_name

    def update_player_comps(self, year=None):
        if year is None:
            year = self.season_start.year
        with open(pkg_resources.files(data) / "playerCompStats.json") as infile:
            stats = json.load(infile)

        players = self.players.get(self.season_start.year - 1, {})
        players.update(self.players.get(self.season_start.year, {}))
        playerProfile, all_players, id_to_name = self.build_comp_profile(players)

        comps = {}
        for position in ["C", "W", "D", "G"]:
            pos_players = [
                p
                for p, v in all_players.items()
                if v.get("position") == position and p in playerProfile.index
            ]
            positionProfile = playerProfile.loc[
                pos_players, list(stats["NHL"][position].keys())
            ].replace([np.nan, np.inf, -np.inf], 0)
            positionProfile.index = positionProfile.index.map(lambda x: id_to_name.get(x, x))
            positionProfile = positionProfile[~positionProfile.index.duplicated(keep="first")]
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(np.sqrt(list(stats["NHL"][position].values())))
            knn = BallTree(positionProfile)
            min_k = 4 if position == "G" else 5
            comps[position] = self._build_comps(knn, positionProfile, min_comps=min_k, max_comps=20)

        filepath = pkg_resources.files(data) / "nhl_comps.json"
        with open(filepath, "w") as outfile:
            json.dump(comps, outfile, indent=4)

    def _compute_comps(self):
        """Build comps from loaded data at runtime (no JSON I/O)."""
        with open(pkg_resources.files(data) / "playerCompStats.json") as f:
            stats = json.load(f)

        playerProfile, all_players, id_to_name = self.build_comp_profile()
        if playerProfile.empty:
            return

        comps = {}
        for position in ["C", "W", "D", "G"]:
            pos_players = [
                p
                for p, v in all_players.items()
                if v.get("position") == position and p in playerProfile.index
            ]
            if len(pos_players) < 7:
                continue
            positionProfile = playerProfile.loc[
                pos_players, list(stats["NHL"][position].keys())
            ].replace([np.nan, np.inf, -np.inf], 0)
            positionProfile.index = positionProfile.index.map(lambda x: id_to_name.get(x, x))
            positionProfile = positionProfile[~positionProfile.index.duplicated(keep="first")]
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(np.sqrt(list(stats["NHL"][position].values())))
            knn = BallTree(positionProfile)
            min_k = 4 if position == "G" else 5
            comps[position] = self._build_comps(knn, positionProfile, min_comps=min_k, max_comps=20)

        self.comps = comps

    def parse_game(self, gameId, gameDate):
        gamelog = []
        teamlog = []
        game = scraper.get(f"https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore")
        season = game["season"]
        res = requests.get(
            f"https://moneypuck.com/moneypuck/playerData/games/{season}/{gameId}.csv"
        )
        if res.status_code != 200:
            return gamelog, teamlog
        game_df = pd.read_csv(StringIO(res.text))
        pp_df = game_df.loc[game_df.situation == "5on4"]
        game_df = game_df.loc[game_df.situation == "all"]
        if game and not game_df.empty:
            team_map = {"SJS": "SJ", "LAK": "LA", "NJD": "NJ", "TBL": "TB", "WSH": "WAS"}
            awayTeam = game["awayTeam"]["abbrev"]
            homeTeam = game["homeTeam"]["abbrev"]

            awayTeam = team_map.get(awayTeam, awayTeam)
            homeTeam = team_map.get(homeTeam, homeTeam)
            game_df.team = game_df.team.apply(lambda x: team_map.get(x, x))
            game_df["position"] = game_df["position"].replace("L", "W")
            game_df["position"] = game_df["position"].replace("R", "W")

            for _i, player in game_df.iterrows():
                team = player["team"]
                team = team_map.get(team, team)
                home = team == homeTeam
                opponent = awayTeam if home else homeTeam
                win = (game["homeTeam"]["score"] > game["awayTeam"]["score"]) == home

                if player["position"] == "Team Level":
                    n = {
                        "gameId": gameId,
                        "gameDate": gameDate,
                        "team": team,
                        "opponent": opponent,
                        "home": home,
                    }
                    stats = {
                        "Corsi": float(player["OffIce_F_shotAttempts"]),
                        "Fenwick": float(player["OffIce_F_unblockedShotAttempts"]),
                        "Hits": float(player["OffIce_F_hits"]),
                        "Takeaways": float(player["OffIce_F_takeaways"]),
                        "PIM": float(player["OffIce_F_penalityMinutes"]),
                        "Corsi_Pct": float(player["OffIce_shotAttempts_For_Percentage"]),
                        "Fenwick_Pct": float(player["OffIce_unblockedShotAttempts_For_Percentage"]),
                        "Hits_Pct": float(player["OffIce_hits_For_Percentage"]),
                        "Takeaways_Pct": float(player["OffIce_takeaways_For_Percentage"]),
                        "PIM_Pct": float(player["OffIce_penalityMinutes_For_Percentage"]),
                        "Block_Pct": float(player["OffIce_A_blockedShotAttempts"])
                        / float(player["OffIce_A_shotAttempts"]),
                        "xGoals": float(player["OffIce_F_flurryScoreVenueAdjustedxGoals"]),
                        "xGoalsAgainst": float(player["OffIce_A_flurryScoreVenueAdjustedxGoals"]),
                        "goalsAgainst": float(player["OffIce_A_goals"]),
                        "goals": float(player["OffIce_F_goals"]),
                    }
                    shotsAgainst = float(player["OffIce_A_shotsOnGoal"])
                    stats.update(
                        {
                            "WL": "W" if stats["goals"] > stats["goalsAgainst"] else "L",
                            "GOE": (float(player["OffIce_F_goals"]) - stats["xGoals"])
                            / float(player["OffIce_F_shotAttempts"]),
                            "SV": (float(player["OffIce_A_savedShotsOnGoal"]) / shotsAgainst)
                            if shotsAgainst
                            else 0,
                            "SOE": (
                                (
                                    float(player["OffIce_A_flurryScoreVenueAdjustedxGoals"])
                                    - float(player["OffIce_A_goals"])
                                )
                                / shotsAgainst
                            )
                            if shotsAgainst
                            else 0,
                            "Freeze": (
                                (
                                    float(player["OffIce_A_freeze"])
                                    - float(player["OffIce_A_xFreeze"])
                                )
                                / shotsAgainst
                            )
                            if shotsAgainst
                            else 0,
                            "Rebound": (
                                (
                                    float(player["OffIce_A_rebounds"])
                                    - float(player["OffIce_A_xRebounds"])
                                )
                                / shotsAgainst
                            )
                            if shotsAgainst
                            else 0,
                            "RG": (
                                (
                                    float(player["OffIce_A_reboundGoals"])
                                    - float(player["OffIce_A_reboundxGoals"])
                                )
                                / float(player["OffIce_A_rebounds"])
                            )
                            if float(player["OffIce_A_rebounds"])
                            else 0,
                        }
                    )
                    teamlog.append(n | stats)
                else:
                    n = {
                        "gameId": gameId,
                        "gameDate": gameDate,
                        "team": team,
                        "opponent": opponent,
                        "opponent goalie": remove_accents(
                            game_df.loc[
                                (game_df.position == "G") & (game_df.team != team), "playerName"
                            ].iat[0]
                        ),
                        "home": home,
                        "playerId": player["playerId"],
                        "playerName": remove_accents(player["playerName"]),
                        "position": player["position"],
                    }
                    stats = {
                        "points": float(player["I_F_points"]),
                        "shots": float(player["I_F_shotsOnGoal"]),
                        "blocked": float(player["I_A_blockedShotAttempts"]),
                        "sogBS": float(player["I_F_shotsOnGoal"])
                        + float(player["I_A_blockedShotAttempts"]),
                        "goals": float(player["I_F_goals"]),
                        "assists": float(player["I_F_primaryAssists"])
                        + float(player["I_F_secondaryAssists"]),
                        "hits": float(player["I_F_hits"]),
                        "faceOffWins": float(player["I_F_faceOffsWon"]),
                        "timeOnIce": float(player["I_F_iceTime"]) / 60,
                        "saves": float(player["OnIce_A_savedShotsOnGoal"]),
                        "shotsAgainst": float(player["OnIce_A_shotsOnGoal"]),
                        "goalsAgainst": float(player["OnIce_A_goals"]),
                    }
                    if player["playerName"] in pp_df["playerName"].to_list():
                        stats["powerPlayPoints"] = float(
                            pp_df.loc[pp_df["playerName"] == player["playerName"]][
                                "I_F_points"
                            ].iat[0]
                        )
                    else:
                        stats["powerPlayPoints"] = 0
                    stats.update(
                        {
                            "fantasy points prizepicks": stats.get("goals", 0) * 8
                            + stats.get("assists", 0) * 5
                            + stats.get("sogBS", 0) * 1.5,
                            "goalie fantasy points underdog": int(win) * 6
                            + stats.get("saves", 0) * 0.6
                            - stats.get("goalsAgainst", 0) * 3,
                            "skater fantasy points underdog": stats.get("goals", 0) * 6
                            + stats.get("assists", 0) * 4
                            + stats.get("sogBS", 0)
                            + stats.get("hits", 0) * 0.5
                            + stats.get("powerPlayPoints", 0) * 0.5,
                            "goalie fantasy points parlay": stats.get("saves", 0) * 0.25
                            - stats.get("goalsAgainst", 0),
                            "skater fantasy points parlay": stats.get("goals", 0) * 3
                            + stats.get("assists", 0) * 2
                            + stats.get("shots", 0) * 0.5
                            + stats.get("hits", 0)
                            + stats.get("blocked", 0),
                        }
                    )
                    team = {v: k for k, v in team_map.items()}.get(team, team)
                    shots = float(player["I_F_shotAttempts"])
                    shotsAgainst = float(player["OnIce_A_shotsOnGoal"])
                    stats.update(
                        {
                            "GOE": (
                                (
                                    stats["goals"]
                                    - float(player["I_F_flurryScoreVenueAdjustedxGoals"])
                                )
                                / shots
                            )
                            if shots
                            else 0,
                            "Fenwick": float(player["OnIce_unblockedShotAttempts_For_Percentage"]),
                            "TimeShare": stats["timeOnIce"]
                            / (
                                float(
                                    game_df.loc[
                                        game_df["playerName"] == team, "OffIce_F_iceTime"
                                    ].iat[0]
                                )
                                / 60
                            ),
                            "ShotShare": stats["shots"]
                            / float(
                                game_df.loc[
                                    game_df["playerName"] == team, "OffIce_F_shotsOnGoal"
                                ].iat[0]
                            ),
                            "Shot60": stats["shots"] * 60 / stats["timeOnIce"],
                            "Blk60": stats["blocked"] * 60 / stats["timeOnIce"],
                            "Hit60": stats["hits"] * 60 / stats["timeOnIce"],
                            "Ast60": stats["assists"] * 60 / stats["timeOnIce"],
                            "SV": (float(player["OnIce_A_savedShotsOnGoal"]) / shotsAgainst)
                            if shotsAgainst
                            else 0,
                            "SOE": (
                                (
                                    float(player["OnIce_A_flurryScoreVenueAdjustedxGoals"])
                                    - stats["goalsAgainst"]
                                )
                                / shotsAgainst
                            )
                            if shotsAgainst
                            else 0,
                            "Freeze": (
                                (float(player["OnIce_A_freeze"]) - float(player["OnIce_A_xFreeze"]))
                                / shotsAgainst
                            )
                            if shotsAgainst
                            else 0,
                            "Rebound": (
                                (
                                    float(player["OnIce_A_rebounds"])
                                    - float(player["OnIce_A_xRebounds"])
                                )
                                / shotsAgainst
                            )
                            if shotsAgainst
                            else 0,
                            "RG": (
                                (
                                    float(player["OnIce_A_reboundGoals"])
                                    - float(player["OnIce_A_reboundxGoals"])
                                )
                                / float(player["OnIce_A_rebounds"])
                            )
                            if float(player["OnIce_A_rebounds"])
                            else 0,
                        }
                    )
                    gamelog.append(n | stats)

        return gamelog, teamlog

    def update(self):
        """Updates the NHL skater and goalie data.

        Args:
            None

        Returns:
            None
        """
        # Get game ids
        latest_date = self.season_start
        if not self.gamelog.empty:
            latest_date = pd.to_datetime(self.gamelog["gameDate"]).max().date() + timedelta(days=1)
        today = datetime.today().date()
        ids = []
        while latest_date <= today:
            start_date = latest_date.strftime("%Y-%m-%d")
            res = scraper.get(f"https://api-web.nhle.com/v1/schedule/{start_date}")
            latest_date = datetime.strptime(
                res.get("nextStartDate", (today + timedelta(days=1)).strftime("%Y-%m-%d")),
                "%Y-%m-%d",
            ).date()

            if len(res.get("gameWeek", [])) > 0:
                for day in res.get("gameWeek"):
                    ids.extend([(game["id"], day["date"]) for game in day["games"]])

            else:
                break

        # Parse the game stats
        nhl_gamelog = []
        nhl_teamlog = []
        for gameId, date in tqdm(ids, desc="Getting NHL Stats"):
            if datetime.strptime(date, "%Y-%m-%d").date() < today:
                gamelog, teamlog = self.parse_game(gameId, date)
                if type(gamelog) is list:
                    nhl_gamelog.extend(gamelog)
                if type(teamlog) is list:
                    nhl_teamlog.extend(teamlog)

        nhl_df = pd.DataFrame(nhl_gamelog).fillna(0).infer_objects(copy=False)
        if not nhl_df.empty:
            nhl_df.drop_duplicates(subset=["gameId", "playerName"], keep="last", inplace=True)
            if not self.gamelog.empty:
                existing = set(
                    self.gamelog[["gameId", "playerName"]].itertuples(index=False, name=None)
                )
                nhl_df = nhl_df[
                    ~nhl_df.apply(lambda x: (x["gameId"], x["playerName"]) in existing, axis=1)
                ]
            nhl_df.loc[:, "moneyline"] = nhl_df.apply(
                lambda x: archive.get_moneyline(self.league, x["gameDate"], x["team"]), axis=1
            )
            nhl_df.loc[:, "totals"] = nhl_df.apply(
                lambda x: archive.get_total(self.league, x["gameDate"], x["team"]), axis=1
            )
        nhl_teamlog_df = pd.DataFrame(nhl_teamlog).fillna(0).infer_objects(copy=False)
        if not nhl_teamlog_df.empty:
            nhl_teamlog_df.drop_duplicates(subset=["gameId", "team"], keep="last", inplace=True)
        self.gamelog = (
            pd.concat([nhl_df, self.gamelog]).sort_values("gameDate").reset_index(drop=True)
        )
        self.teamlog = (
            pd.concat([nhl_teamlog_df, self.teamlog]).sort_values("gameDate").reset_index(drop=True)
        )

        res = scraper.get(
            "https://core.api.dobbersports.com/v1/weekly-schedule/weekly-games?week=0"
        )
        self.upcoming_games = {}
        for team in res.get("data", {}).get("content", {}).get("weeklyGames", []):
            for game in team.get("games", []):
                abbr = abbreviations["NHL"].get(
                    remove_accents(team["teamName"]), remove_accents(team["teamName"])
                )
                if abbr in self.upcoming_games:
                    continue
                idx = game["gameId"]
                details = scraper.get(f"https://core.api.dobbersports.com/v1/game/{idx}")
                if (
                    datetime.strptime(
                        details.get("data", {}).get("gameDate"), "%Y-%m-%dT%H:%M:%S%z"
                    )
                    .astimezone()
                    .date()
                    < today
                ):
                    continue
                opp = abbreviations["NHL"].get(
                    remove_accents(game["opponentTeam"]["name"]),
                    remove_accents(game["opponentTeam"]["name"]),
                )
                home = game["teamType"] == "HOME"
                if home:
                    goalie = details.get("data", {}).get("predictedGoalies", {}).get("HOME", [])
                else:
                    goalie = details.get("data", {}).get("predictedGoalies", {}).get("AWAY", [])
                goalie = remove_accents(goalie[0]["goalie"]["fullName"]) if goalie else ""
                self.upcoming_games[abbr] = {"Opponent": opp, "Home": home, "Goalie": goalie}

        res = requests.get(
            "https://moneypuck.com/moneypuck/playerData/playerBios/allPlayersLookup.csv"
        )
        player_df = pd.read_csv(StringIO(res.text))
        player_df.rename(columns={"name": "playerName"}, inplace=True)
        player_df.height = (
            player_df.height.str[:-1]
            .str.split("' ")
            .apply(lambda x: 12 * int(x[0]) + int(x[1]) if type(x) is list else 0)
        )
        player_df["bmi"] = player_df["weight"] / player_df["height"] / player_df["height"]
        player_df["age"] = (
            datetime.now() - pd.to_datetime(player_df["birthDate"])
        ).dt.days / 365.25
        player_df.playerName = player_df.playerName.apply(remove_accents)
        player_df["position"] = player_df["position"].replace("R", "W")
        player_df["position"] = player_df["position"].replace("L", "W")

        res = requests.get(
            f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{self.season_start.year}/regular/skaters.csv"
        )
        if res.status_code == 200:
            skater_df = pd.read_csv(StringIO(res.text))
            skater_df.rename(columns={"name": "playerName"}, inplace=True)
            skater_df = skater_df.loc[skater_df["situation"] == "all"]
            skater_df["Fenwick"] = (
                skater_df["onIce_fenwickPercentage"] - skater_df["offIce_fenwickPercentage"]
            )
            skater_df["timePerGame"] = skater_df["icetime"] / skater_df["games_played"] / 60
            skater_df["timePerShift"] = skater_df["icetime"] / skater_df["I_F_shifts"]
            skater_df["xGoals"] = (
                skater_df["I_F_flurryScoreVenueAdjustedxGoals"] / skater_df["I_F_shotAttempts"]
            )
            skater_df["shotsOnGoal"] = skater_df["I_F_shotsOnGoal"] / skater_df["I_F_shotAttempts"]
            skater_df["goals"] = (
                skater_df["I_F_goals"] - skater_df["I_F_flurryScoreVenueAdjustedxGoals"]
            ) / skater_df["I_F_shotAttempts"]
            skater_df["rebounds"] = skater_df["I_F_rebounds"] / skater_df["I_F_shotAttempts"]
            skater_df["freeze"] = skater_df["I_F_freeze"] / skater_df["I_F_shotAttempts"]
            skater_df["oZoneStarts"] = skater_df["I_F_oZoneShiftStarts"] / (
                skater_df["I_F_oZoneShiftStarts"] + skater_df["I_F_dZoneShiftStarts"]
            )
            skater_df["flyStarts"] = skater_df["I_F_flyShiftStarts"] / skater_df["I_F_shifts"]
            skater_df["shotAttempts"] = (
                skater_df["I_F_shotAttempts"] / skater_df["icetime"] * 60 * 60
            )
            skater_df["hits"] = skater_df["I_F_hits"] / skater_df["icetime"] * 60 * 60
            skater_df["takeaways"] = skater_df["I_F_takeaways"] / skater_df["icetime"] * 60 * 60
            skater_df["giveaways"] = skater_df["I_F_giveaways"] / skater_df["icetime"] * 60 * 60
            skater_df["assists"] = (
                (skater_df["I_F_primaryAssists"] + skater_df["I_F_secondaryAssists"])
                / skater_df["icetime"]
                * 60
                * 60
            )
            skater_df["penaltyMinutes"] = (
                skater_df["penalityMinutes"] / skater_df["icetime"] * 60 * 60
            )
            skater_df["penaltyMinutesDrawn"] = (
                skater_df["penalityMinutesDrawn"] / skater_df["icetime"] * 60 * 60
            )
            skater_df["blockedShots"] = (
                skater_df["shotsBlockedByPlayer"] / skater_df["icetime"] * 60 * 60
            )
            skater_df = skater_df[
                [
                    "playerId",
                    "playerName",
                    "team",
                    "position",
                    "Fenwick",
                    "timePerGame",
                    "timePerShift",
                    "shotAttempts",
                    "xGoals",
                    "shotsOnGoal",
                    "goals",
                    "rebounds",
                    "freeze",
                    "oZoneStarts",
                    "flyStarts",
                    "hits",
                    "takeaways",
                    "giveaways",
                    "assists",
                    "penaltyMinutes",
                    "penaltyMinutesDrawn",
                    "blockedShots",
                ]
            ]

            skater_df = player_df.merge(
                skater_df, how="right", on="playerId", suffixes=[None, "_y"]
            )
            skater_df.dropna(inplace=True)
            skater_df.index = skater_df.playerId
            skater_df.drop(
                columns=[
                    "playerId",
                    "birthDate",
                    "nationality",
                    "primaryNumber",
                    "primaryPosition",
                    "playerName_y",
                    "team_y",
                    "position_y",
                ],
                inplace=True,
            )

        else:
            skater_df = pd.DataFrame()

        # res = requests.get(f"https://moneypuck.com/moneypuck/playerData/shots/shots_{self.season_start.year}.csv")
        # shot_df = pd.read_csv(StringIO(res.text))

        res = requests.get(
            f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{self.season_start.year}/regular/goalies.csv"
        )
        if res.status_code == 200:
            goalie_df = pd.read_csv(StringIO(res.text))
            goalie_df.rename(columns={"name": "playerName"}, inplace=True)
            goalie_df = goalie_df.loc[goalie_df["situation"] == "all"]
            goalie_df["timePerGame"] = goalie_df["icetime"] / goalie_df["games_played"] / 60
            goalie_df["saves"] = goalie_df["ongoal"] - goalie_df["goals"]
            goalie_df["savePct"] = goalie_df["saves"] / goalie_df["ongoal"]
            goalie_df["freezeAgainst"] = (goalie_df["freeze"] - goalie_df["xFreeze"]) / goalie_df[
                "saves"
            ]
            goalie_df["reboundsAgainst"] = (
                goalie_df["rebounds"] - goalie_df["xRebounds"]
            ) / goalie_df["saves"]
            goalie_df["goalsAgainst"] = (
                goalie_df["goals"] - goalie_df["flurryAdjustedxGoals"]
            ) / goalie_df["ongoal"]
            goalie_df = goalie_df[
                [
                    "playerId",
                    "playerName",
                    "team",
                    "position",
                    "timePerGame",
                    "savePct",
                    "freezeAgainst",
                    "reboundsAgainst",
                    "goalsAgainst",
                ]
            ]

            goalie_df = player_df.merge(
                goalie_df, how="right", on="playerId", suffixes=[None, "_y"]
            )
            goalie_df.dropna(inplace=True)
            goalie_df.index = goalie_df.playerId
            goalie_df.drop(
                columns=[
                    "playerId",
                    "birthDate",
                    "nationality",
                    "primaryNumber",
                    "primaryPosition",
                    "playerName_y",
                    "team_y",
                    "position_y",
                ],
                inplace=True,
            )

        else:
            goalie_df = pd.DataFrame()

        self.players[self.season_start.year] = skater_df.to_dict("index") | goalie_df.to_dict(
            "index"
        )

        # Remove old games to prevent file bloat
        four_years_ago = today - timedelta(days=1431)
        self.gamelog = self.gamelog[
            pd.to_datetime(self.gamelog["gameDate"]).dt.date >= four_years_ago
        ]
        self.gamelog.drop_duplicates(subset=["gameId", "playerName"], keep="last", inplace=True)
        self.teamlog = self.teamlog[
            pd.to_datetime(self.teamlog["gameDate"]).dt.date >= four_years_ago
        ]
        self.teamlog.drop_duplicates(subset=["gameId", "team"], keep="last", inplace=True)

        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog["playerName"] = self.gamelog["playerName"].apply(remove_accents)
            self.gamelog.loc[:, "moneyline"] = self.gamelog.apply(
                lambda x: archive.get_moneyline(self.league, x["gameDate"], x["team"]), axis=1
            )
            self.gamelog.loc[:, "totals"] = self.gamelog.apply(
                lambda x: archive.get_total(self.league, x["gameDate"], x["team"]), axis=1
            )

        # Write to file
        with open((pkg_resources.files(data) / "nhl_data.dat"), "wb") as outfile:
            pickle.dump(
                {"players": self.players, "gamelog": self.gamelog, "teamlog": self.teamlog},
                outfile,
                -1,
            )

    def dump_goalie_list(self):
        filepath = pkg_resources.files(data) / "goalies.json"
        with open(filepath, "w") as outfile:
            json.dump(
                list(self.gamelog.loc[self.gamelog.position == "G", "playerName"].unique()), outfile
            )

    def get_volume_stats(self, offers, date=datetime.today().date(), pitcher=False):
        flat_offers = {}
        if isinstance(offers, dict):
            for players in offers.values():
                flat_offers.update(players)
        else:
            flat_offers = offers

        market = "shotsAgainst" if pitcher else "timeOnIce"

        if isinstance(offers, dict):
            flat_offers.update(offers.get(market, {}))
        self.profile_market(market, date)
        self.get_depth(flat_offers, date)
        playerStats = self.get_stats(market, flat_offers, date)

        filename = "_".join([self.league, market]).replace(" ", "-")
        filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
        if self._volume_model_cache is None:
            self._volume_model_cache = {}
        if filename not in self._volume_model_cache:
            if os.path.isfile(filepath):
                with open(filepath, "rb") as infile:
                    self._volume_model_cache[filename] = pickle.load(infile)
            else:
                logger.warning(f"{filename} missing")
                return

        if filename in self._volume_model_cache:
            filedict = self._volume_model_cache[filename]
            # Slice to the trained schema embedded in the pickle.
            playerStats = playerStats[filedict["expected_columns"]]
            model = filedict["model"]
            dist = filedict["distribution"]

            categories = ["Home", "Player position"]
            if "Player position" not in playerStats.columns:
                categories.remove("Player position")
            for c in categories:
                playerStats[c] = playerStats[c].astype("category")

            set_model_start_values(model, dist, playerStats)

            prob_params = pd.DataFrame()
            preds = model.predict(playerStats, pred_type="parameters")
            preds.index = playerStats.index
            prob_params = pd.concat([prob_params, preds])

            prob_params.sort_index(inplace=True)
            playerStats.sort_index(inplace=True)

        else:
            logger.warning(f"{filename} missing")
            return

        # Drop gate column for ZI distributions — not needed for budget normalization
        prob_params.drop(columns=["gate"], inplace=True, errors="ignore")

        # SkewNormal: rename loc/scale/alpha to proj {market} loc/scale/alpha
        rename_map = {
            "loc": f"proj {market} loc",
            "scale": f"proj {market} scale",
            "alpha": f"proj {market} alpha",
        }

        self.playerProfile = self.playerProfile.join(
            prob_params.rename(columns=rename_map), lsuffix="_obs"
        )
        self.playerProfile.drop(
            columns=[col for col in self.playerProfile.columns if "_obs" in col], inplace=True
        )

        if not pitcher:
            # ------------------------------------------------------------------
            # Budget parameters — derived by analyzing historical NHL gamelogs.
            #
            # Methodology:
            #   typical_rotation : median skaters (non-G) logging >3 min per team-game
            #   ot_rate          : fraction of team-games where total skater TOI > 300
            #   ot_extra         : mean extra team TOI above 300 when OT occurs (9.6 min)
            #   avg_unmodeled_min: mean TOI for ranked 8-18 skaters (rank > 7 tier)
            #   per_player_floor : 5th-percentile TOI for top-7 tier skaters
            #   per_player_cap   : 99th-percentile TOI for top-7 tier skaters
            # ------------------------------------------------------------------
            reg_minutes = 300  # 5 skaters × 60 min regulation
            ot_rate = 0.189  # measured: 18.9% of team-games go to OT
            ot_extra = 9.6  # measured: mean extra team TOI when OT occurs
            typical_rotation = 18  # measured: median active skaters per team-game
            avg_unmodeled_min = 14  # measured: mean TOI for skaters ranked 8-18
            per_player_floor = 17  # measured: 5th-pct TOI for top-7 tier skaters
            per_player_cap = 29  # measured: 99th-pct TOI for top-7 tier skaters

            # Expected total team TOI including OT
            budget_mean = reg_minutes + ot_rate * ot_extra

            teams = self.playerProfile.loc[self.playerProfile["team"] != 0].groupby("team")
            for _team, team_df in teams:
                # SkewNormal: E[X] = loc + scale * delta * sqrt(2/pi)
                loc = team_df[f"proj {market} loc"].copy()
                scale = team_df[f"proj {market} scale"].copy()
                sn_alpha = (
                    team_df[f"proj {market} alpha"].copy()
                    if f"proj {market} alpha" in team_df.columns
                    else pd.Series(0, index=loc.index)
                )
                N = len(team_df)

                delta = sn_alpha / np.sqrt(1 + sn_alpha**2)
                true_means = loc + scale * delta * np.sqrt(2 / np.pi)
                true_vars = scale**2

                total = true_means.sum()
                if total <= 0:
                    continue

                # Reserve TOI for the many unmodeled skaters (rank 8-18)
                unmodeled_count = max(0, typical_rotation - N)
                unmodeled_reserve = unmodeled_count * avg_unmodeled_min

                upper_target = budget_mean - unmodeled_reserve
                lower_target = N * per_player_floor
                target = max(upper_target, lower_target)

                # Precision-weighted deficit distribution
                deficit = target - total
                total_var = true_vars.sum()
                if total_var > 0:
                    adjustments = true_vars / total_var * deficit
                else:
                    adjustments = true_means / total * deficit

                new_means = (true_means + adjustments).clip(lower=0, upper=per_player_cap)

                # Scale both loc and scale to preserve coefficient of variation
                ratio = new_means / true_means.replace(0, np.nan)
                ratio = ratio.fillna(1.0).clip(lower=0.1, upper=10.0)
                new_scale = scale * ratio
                new_loc = new_means - new_scale * delta * np.sqrt(2 / np.pi)

                self.playerProfile.loc[loc.index, f"proj {market} loc"] = new_loc
                self.playerProfile.loc[scale.index, f"proj {market} scale"] = new_scale
                self.playerProfile.loc[loc.index, f"proj {market} mean"] = new_means
                self.playerProfile.loc[scale.index, f"proj {market} std"] = new_scale

        # Drop SkewNormal parameters (keep only mean/std for downstream use)
        self.playerProfile.drop(
            columns=[f"proj {market} loc", f"proj {market} scale", f"proj {market} alpha"],
            inplace=True,
            errors="ignore",
        )
        self.playerProfile.fillna(0, inplace=True)

    def check_combo_markets(self, market, player, date=datetime.today().date()):
        player_games = self.short_gamelog.loc[
            self.short_gamelog[self.log_strings["player"]] == player
        ]
        cv = stat_cv.get(self.league, {}).get(market, 1)
        dist = stat_dist.get(self.league, {}).get(market, "Gamma")
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()

        if date < datetime.today().date():
            todays_games = self.gamelog.loc[
                pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date == date
            ]
            player_game = todays_games.loc[todays_games[self.log_strings["player"]] == player]
            if player_game.empty:
                return 0

            team = player_game[self.log_strings["team"]].iloc[0]
            opponent = player_game[self.log_strings["opponent"]].iloc[0]

        else:
            team = player_games[self.log_strings["team"]].iloc[-1]
            opponent = self.upcoming_games[team][self.log_strings["opponent"]]

        date = date.strftime("%Y-%m-%d")
        ev = 0
        if market in combo_props:
            for submarket in combo_props.get(market, []):
                sub_cv = stat_cv["NHL"].get(submarket, 1)
                sub_dist = stat_dist.get("NHL", {}).get(submarket, "Gamma")
                v = archive.get_ev("NHL", submarket, date, player)
                subline = archive.get_line("NHL", submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist)
                if np.isnan(v) or v == 0:
                    ev = 0
                    break

                else:
                    ev += v

        elif market == "goalsAgainst":
            ev = archive.get_total("NHL", date, opponent)

        elif "fantasy" in market:
            ev = 0
            book_odds = False
            if "prizepicks" in market:
                fantasy_props = [("goals", 8), ("assists", 5), ("shots", 1.5), ("blocked", 1.5)]
            elif ("underdog" in market) and ("skater" in market):
                fantasy_props = [
                    ("goals", 6),
                    ("assists", 4),
                    ("shots", 1),
                    ("blocked", 1),
                    ("hits", 0.5),
                    ("powerPlayPoints", 0.5),
                ]
            else:
                fantasy_props = [("saves", 0.6), ("goalsAgainst", -3), ("Moneyline", 6)]
            for submarket, weight in fantasy_props:
                sub_cv = stat_cv["NHL"].get(submarket, 1)
                sub_dist = stat_dist.get("NHL", {}).get(submarket, "Gamma")
                v = archive.get_ev("NHL", submarket, date, player)
                subline = archive.get_line("NHL", submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist)
                if np.isnan(v) or v == 0:
                    if submarket == "Moneyline":
                        p = archive.get_moneyline("NHL", date, team)
                        ev += p * weight
                    elif submarket == "goalsAgainst":
                        v = archive.get_total("NHL", date, opponent)
                        subline = np.floor(v) + 0.5
                        v = get_ev(
                            subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                        )
                        ev += v * weight
                    else:
                        if subline == 0 and not player_games.empty:
                            subline = np.floor(player_games.iloc[-10:][submarket].median()) + 0.5

                        if subline != 0:
                            under = (player_games[submarket] < subline).mean()
                            ev += get_ev(subline, under, sub_cv, dist=sub_dist) * weight
                else:
                    book_odds = True
                    ev += v * weight

            if not book_odds:
                ev = 0

        return 0 if np.isnan(ev) else ev

