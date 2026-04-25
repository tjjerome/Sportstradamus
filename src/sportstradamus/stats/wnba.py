"""StatsWNBA: WNBA player stats, inherits from StatsNBA."""

import importlib.resources as pkg_resources
import json
import os.path
import pickle
import warnings
from datetime import datetime, timedelta
from io import StringIO
from time import sleep

import line_profiler
import nba_api.stats.endpoints as nba
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
from sportstradamus.stats.nba import StatsNBA


class StatsWNBA(StatsNBA):
    def __init__(self):
        super().__init__()
        self.league = "WNBA"
        self.positions = ["G", "F", "C"]
        self.season_start = datetime(2026, 5, 15).date()
        self.default_total = 81.667

        self.gamelog.columns = [stat.replace("_48", "_40") for stat in self.gamelog.columns]
        self.stat_types = [stat.replace("_48", "_40") for stat in self.stat_types]

        self.season = self.season_start.year

    def load(self):
        """Load data from files."""
        filepath = pkg_resources.files(data) / "wnba_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                wnba_data = pickle.load(infile)
                self.players = wnba_data["players"]
                self.gamelog = wnba_data["gamelog"]
                self.teamlog = wnba_data["teamlog"]

    def update(self):
        team_abbr_map = {
            "CONN": "CON",
            "NY": "NYL",
            "LA": "LAS",
            "LV": "LVA",
            "PHO": "PHX",
            "WSH": "WAS",
        }

        pos_map = {"G-F": "G", "F-G": "F", "F-C": "C"}

        i = 0
        while i < 10:
            try:
                player_df = nba.playerindex.PlayerIndex(
                    season=self.season_start.year, league_id="10", historical_nullable=1
                ).get_normalized_dict()["PlayerIndex"]
                player_df = pd.DataFrame(player_df).rename(
                    columns={"POSITION": "POS", "Position": "POS", "PERSON_ID": "PLAYER_ID"}
                )
                player_df.TEAM_ABBREVIATION = player_df.TEAM_ABBREVIATION.apply(
                    lambda x: team_abbr_map.get(x, x)
                )
                player_df.POS = player_df.POS.apply(lambda x: pos_map.get(x, x))
                break
            except:
                player_df = pd.DataFrame(
                    columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POS"]
                )

            sleep(0.1)
            i = i + 1

        i = 0
        while i < 10:
            try:
                playerBios = nba.leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                    season=self.season_start.year, league_id="10"
                ).get_normalized_dict()["LeagueDashPlayerBioStats"]

                shotData = nba.leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                    **{
                        "season": self.season_start.year,
                        "league_id_nullable": "10",
                        "season_type_all_star": "Regular Season",
                        "distance_range": "By Zone",
                        "per_mode_detailed": "Per40",
                    }
                ).get_dict()["resultSets"]
                break
            except:
                playerBios = {
                    "PLAYER_NAME": [],
                    "PLAYER_ID": [],
                    "PLAYER_HEIGHT_INCHES": [],
                    "PLAYER_WEIGHT": [],
                    "TEAM_ABBREVIATION": [],
                }
                shotData = {"rowSet": []}
                sleep(0.1)
                i = i + 1

        playerBios = pd.DataFrame(playerBios)
        if not playerBios.empty:
            playerBios.PLAYER_NAME = playerBios.PLAYER_NAME.apply(remove_accents)
            shotMap = []
            for row in shotData["rowSet"]:
                fga = np.nansum(np.array(row[7:-6:3], dtype=float))
                if fga > 0:
                    record = {
                        "PLAYER_NAME": remove_accents(row[1]),
                        "TEAM_ABBREVIATION": row[3],
                        "RA_PCT": (row[7] if row[7] else 0) / fga,
                        "ITP_PCT": (row[10] if row[10] else 0) / fga,
                        "MR_PCT": (row[13] if row[13] else 0) / fga,
                        "C3_PCT": (row[28] if row[28] else 0) / fga,
                        "B3_PCT": (row[22] if row[22] else 0) / fga,
                    }
                    shotMap.append(record)

            player_df = player_df.merge(playerBios, on="PLAYER_ID", suffixes=(None, "_y")).merge(
                pd.DataFrame(shotMap),
                on=["PLAYER_NAME", "TEAM_ABBREVIATION"],
                suffixes=(None, "_y"),
            )
            # list(player_df.loc[player_df.isna().any(axis=1)].index.unique()) TODO handle these names
            player_df.PLAYER_WEIGHT = player_df.PLAYER_WEIGHT.astype(float)
            player_df.POS = player_df.POS.str[0]
            player_df.index = player_df.PLAYER_NAME
            player_df["PLAYER_BMI"] = (
                player_df["PLAYER_WEIGHT"]
                / player_df["PLAYER_HEIGHT_INCHES"]
                / player_df["PLAYER_HEIGHT_INCHES"]
            )
            wnba_numeric_cols = [
                "AGE",
                "PLAYER_HEIGHT_INCHES",
                "PLAYER_BMI",
                "USG_PCT",
                "TS_PCT",
                "RA_PCT",
                "ITP_PCT",
                "MR_PCT",
                "C3_PCT",
                "B3_PCT",
            ]
            player_dict = player_df.groupby("TEAM_ABBREVIATION")[["POS", *wnba_numeric_cols]].apply(
                lambda x: x
            )
            player_dict[wnba_numeric_cols] = player_dict[wnba_numeric_cols].replace(
                [np.nan, np.inf, -np.inf], 0
            )

            player_dict = {
                level: player_dict.xs(level).T.to_dict() for level in player_dict.index.levels[0]
            }
            if self.season_start.year in self.players:
                self.players[self.season_start.year] = {
                    team: players | player_dict.get(team, {})
                    for team, players in self.players[self.season_start.year].items()
                }
            else:
                self.players[self.season_start.year] = player_dict

        self.upcoming_games = {}
        today = datetime.today().date()

        # Drop records with incomplete advanced stats so they can be re-fetched
        if "OFF_RATING" in self.gamelog.columns:
            self.gamelog = self.gamelog.dropna(subset=["OFF_RATING"])

        latest_date = self.season_start
        if not self.gamelog.empty:
            latest_date = pd.to_datetime(self.gamelog[self.log_strings["date"]]).max().date()
            latest_date = max(latest_date, self.season_start)

        try:
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=10&Season={self.season_start.year}&RegionID=1&Date={today.strftime('%m/%d/%Y')}&EST=Y"

            ug_res = scraper.get(ug_url)["resultSets"][1]["CompleteGameList"]

            next_day = today + timedelta(days=1)
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=10&Season={self.season_start.year}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

            ug_res.extend(scraper.get(ug_url)["resultSets"][1]["CompleteGameList"])

            next_day = next_day + timedelta(days=1)
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=10&Season={self.season_start.year}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

            ug_res.extend(scraper.get(ug_url)["resultSets"][1]["CompleteGameList"])

            for game in ug_res:
                if game["vtAbbreviation"] not in self.upcoming_games:
                    self.upcoming_games[game["vtAbbreviation"]] = {
                        "Opponent": game["htAbbreviation"],
                        "Home": False,
                    }
                if game["htAbbreviation"] not in self.upcoming_games:
                    self.upcoming_games[game["htAbbreviation"]] = {
                        "Opponent": game["vtAbbreviation"],
                        "Home": True,
                    }

        except:
            pass

        params = {
            "season_nullable": self.season_start.year,
            "league_id_nullable": "10",
            "date_from_nullable": latest_date.strftime("%m/%d/%Y"),
            "date_to_nullable": today.strftime("%m/%d/%Y"),
        }

        i = 0

        while i < 10:
            try:
                nba_gamelog = nba.playergamelogs.PlayerGameLogs(**params).get_normalized_dict()[
                    "PlayerGameLogs"
                ]
                adv_gamelog = nba.playergamelogs.PlayerGameLogs(
                    **(params | {"measure_type_player_game_logs_nullable": "Advanced"})
                ).get_normalized_dict()["PlayerGameLogs"]
                usg_gamelog = nba.playergamelogs.PlayerGameLogs(
                    **(params | {"measure_type_player_game_logs_nullable": "Usage"})
                ).get_normalized_dict()["PlayerGameLogs"]
                teamlog = nba.teamgamelogs.TeamGameLogs(**(params)).get_normalized_dict()[
                    "TeamGameLogs"
                ]
                sco_teamlog = nba.teamgamelogs.TeamGameLogs(
                    **(params | {"measure_type_player_game_logs_nullable": "Scoring"})
                ).get_normalized_dict()["TeamGameLogs"]
                adv_teamlog = nba.teamgamelogs.TeamGameLogs(
                    **(params | {"measure_type_player_game_logs_nullable": "Advanced"})
                ).get_normalized_dict()["TeamGameLogs"]

                # Fetch playoffs game logs
                if (today.month >= 9) or (today - latest_date).days > 150:
                    params.update({"season_type_nullable": "Playoffs"})
                    nba_gamelog.extend(
                        nba.playergamelogs.PlayerGameLogs(**params).get_normalized_dict()[
                            "PlayerGameLogs"
                        ]
                    )
                    adv_gamelog.extend(
                        nba.playergamelogs.PlayerGameLogs(
                            **(params | {"measure_type_player_game_logs_nullable": "Advanced"})
                        ).get_normalized_dict()["PlayerGameLogs"]
                    )
                    usg_gamelog.extend(
                        nba.playergamelogs.PlayerGameLogs(
                            **(params | {"measure_type_player_game_logs_nullable": "Usage"})
                        ).get_normalized_dict()["PlayerGameLogs"]
                    )
                    teamlog.extend(
                        nba.teamgamelogs.TeamGameLogs(**(params)).get_normalized_dict()[
                            "TeamGameLogs"
                        ]
                    )
                    sco_teamlog.extend(
                        nba.teamgamelogs.TeamGameLogs(
                            **(params | {"measure_type_player_game_logs_nullable": "Scoring"})
                        ).get_normalized_dict()["TeamGameLogs"]
                    )
                    adv_teamlog.extend(
                        nba.teamgamelogs.TeamGameLogs(
                            **(params | {"measure_type_player_game_logs_nullable": "Advanced"})
                        ).get_normalized_dict()["TeamGameLogs"]
                    )

                break
            except:
                sleep(0.1)
                i += 1

        if nba_gamelog and adv_gamelog and usg_gamelog:
            nba_gamelog.sort(key=lambda x: (x["GAME_ID"], x["PLAYER_ID"]))
            adv_gamelog.sort(key=lambda x: (x["GAME_ID"], x["PLAYER_ID"]))
            usg_gamelog.sort(key=lambda x: (x["GAME_ID"], x["PLAYER_ID"]))

            stat_df = (
                pd.DataFrame(nba_gamelog)
                .merge(
                    pd.DataFrame(adv_gamelog), on=["PLAYER_ID", "GAME_ID"], suffixes=[None, "_y"]
                )
                .merge(
                    pd.DataFrame(usg_gamelog), on=["PLAYER_ID", "GAME_ID"], suffixes=[None, "_y"]
                )
            )
            stat_df = stat_df[[col for col in stat_df.columns if "_y" not in col]]
            stat_df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last", inplace=True)
            stat_df.PLAYER_NAME = stat_df.PLAYER_NAME.apply(remove_accents)
            stat_df = stat_df.loc[stat_df["MIN"] > 1]
            stat_df = stat_df.loc[~stat_df.TEAM_ABBREVIATION.isna()]
            # Filter out games already in the gamelog
            if not self.gamelog.empty:
                existing = set(
                    self.gamelog[["PLAYER_ID", "GAME_ID"]].itertuples(index=False, name=None)
                )
                stat_df = stat_df[
                    ~stat_df.apply(lambda x: (x["PLAYER_ID"], x["GAME_ID"]) in existing, axis=1)
                ]

            stat_df["HOME"] = stat_df.MATCHUP.str.contains(" vs. ")
            stat_df = stat_df.merge(player_df[["PLAYER_ID", "POS"]], on="PLAYER_ID")
            stat_df["OPP"] = stat_df.MATCHUP.apply(lambda x: x[x.rfind(" ") :].strip())

            stat_df["PRA"] = stat_df["PTS"] + stat_df["REB"] + stat_df["AST"]
            stat_df["PR"] = stat_df["PTS"] + stat_df["REB"]
            stat_df["PA"] = stat_df["PTS"] + stat_df["AST"]
            stat_df["RA"] = stat_df["REB"] + stat_df["AST"]
            stat_df["BLST"] = stat_df["BLK"] + stat_df["STL"]
            stat_df["fantasy points prizepicks"] = (
                stat_df["PTS"]
                + stat_df["REB"] * 1.2
                + stat_df["AST"] * 1.5
                + stat_df["BLST"] * 3
                - stat_df["TOV"]
            )
            stat_df["fantasy points underdog"] = (
                stat_df["PTS"]
                + stat_df["REB"] * 1.2
                + stat_df["AST"] * 1.5
                + stat_df["BLST"] * 3
                - stat_df["TOV"]
            )
            stat_df["fantasy points parlay"] = stat_df["PRA"] + stat_df["BLST"] * 2 - stat_df["TOV"]
            stat_df["FTR"] = stat_df["FTM"] / stat_df["FGA"]
            stat_df["FG3_RATIO"] = stat_df["FG3A"] / stat_df["FGA"]
            stat_df["BLK_PCT"] = (
                (stat_df["BLK"] / stat_df["BLKA"]).fillna(0).infer_objects(copy=False)
            )
            stat_df["FGA_40"] = stat_df["FGA"] / stat_df["MIN"] * 40
            stat_df["FG3A_40"] = stat_df["FG3A"] / stat_df["MIN"] * 40
            stat_df["REB_40"] = stat_df["REB"] / stat_df["MIN"] * 40
            stat_df["OREB_40"] = stat_df["OREB"] / stat_df["MIN"] * 40
            stat_df["DREB_40"] = stat_df["DREB"] / stat_df["MIN"] * 40
            stat_df["AST_40"] = stat_df["AST"] / stat_df["MIN"] * 40
            stat_df["TOV_40"] = stat_df["TOV"] / stat_df["MIN"] * 40
            stat_df["BLKA_40"] = stat_df["BLKA"] / stat_df["MIN"] * 40
            stat_df["STL_40"] = stat_df["STL"] / stat_df["MIN"] * 40
            stat_df.fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)
            stat_df.TEAM_ABBREVIATION = stat_df.TEAM_ABBREVIATION.apply(
                lambda x: team_abbr_map.get(x, x)
            )
            stat_df.OPP = stat_df.OPP.apply(lambda x: team_abbr_map.get(x, x))

            stat_df["GAME_DATE"] = pd.to_datetime(stat_df["GAME_DATE"]).astype(str)

            if not stat_df.empty:
                stat_df.loc[:, "moneyline"] = stat_df.apply(
                    lambda x: archive.get_moneyline(
                        self.league, x[self.log_strings["date"]], x["TEAM_ABBREVIATION"]
                    ),
                    axis=1,
                )
                stat_df.loc[:, "totals"] = stat_df.apply(
                    lambda x: archive.get_total(
                        self.league, x[self.log_strings["date"]], x["TEAM_ABBREVIATION"]
                    ),
                    axis=1,
                )
                self.gamelog = (
                    pd.concat([stat_df[self.gamelog.columns], self.gamelog])
                    .sort_values(self.log_strings["date"])
                    .reset_index(drop=True)
                )

        if teamlog and sco_teamlog and adv_teamlog:
            teamlog.sort(key=lambda x: (x["GAME_ID"], x["TEAM_ID"]))
            sco_teamlog.sort(key=lambda x: (x["GAME_ID"], x["TEAM_ID"]))
            adv_teamlog.sort(key=lambda x: (x["GAME_ID"], x["TEAM_ID"]))

            team_df = (
                pd.DataFrame(teamlog)
                .merge(pd.DataFrame(adv_teamlog), on=["TEAM_ID", "GAME_ID"], suffixes=[None, "_y"])
                .merge(pd.DataFrame(sco_teamlog), on=["TEAM_ID", "GAME_ID"], suffixes=[None, "_y"])
            )
            team_df = team_df[[col for col in team_df.columns if "_y" not in col]]
            team_df = team_df.loc[~team_df.TEAM_ABBREVIATION.isna()]

            team_df["HOME"] = team_df.MATCHUP.str.contains(" vs. ")
            team_df["OPP"] = team_df.MATCHUP.apply(lambda x: x[x.rfind(" ") :].strip())
            team_df["FTR"] = team_df["FTM"] / team_df["FGA"]
            team_df["BLK_RATIO"] = team_df["BLK"] / team_df["BLKA"]
            team_df.fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)
            team_df.TEAM_ABBREVIATION = team_df.TEAM_ABBREVIATION.apply(
                lambda x: team_abbr_map.get(x, x)
            )
            team_df.OPP = team_df.OPP.apply(lambda x: team_abbr_map.get(x, x))

            stats = [stat for stat in self.teamlog.columns if "OPP_" in stat]
            home_teams = team_df.loc[team_df.HOME]
            home_teams.index = home_teams.GAME_ID
            away_teams = team_df.loc[~team_df.HOME]
            away_teams.index = away_teams.GAME_ID
            home_teams = home_teams.join(away_teams.add_prefix("OPP_")[stats])
            away_teams = away_teams.join(home_teams.add_prefix("OPP_")[stats])
            team_df = pd.concat([home_teams, away_teams], ignore_index=True)

            team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"]).astype(str)

            if not team_df.empty:
                self.teamlog = (
                    pd.concat([team_df[self.teamlog.columns], self.teamlog])
                    .sort_values(self.log_strings["date"])
                    .reset_index(drop=True)
                )

        self.gamelog.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last", inplace=True)
        self.teamlog.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)

        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog.loc[:, "moneyline"] = self.gamelog.apply(
                lambda x: archive.get_moneyline(
                    self.league, x[self.log_strings["date"]][:10], x["TEAM_ABBREVIATION"]
                ),
                axis=1,
            )
            self.gamelog.loc[:, "totals"] = self.gamelog.apply(
                lambda x: archive.get_total(
                    self.league, x[self.log_strings["date"]][:10], x["TEAM_ABBREVIATION"]
                ),
                axis=1,
            )
            self.gamelog["GAME_DATE"] = self.gamelog["GAME_DATE"].astype(str)
            self.teamlog["GAME_DATE"] = self.teamlog["GAME_DATE"].astype(str)

        # Save the updated player data
        with open(pkg_resources.files(data) / "wnba_data.dat", "wb") as outfile:
            pickle.dump(
                {"players": self.players, "gamelog": self.gamelog, "teamlog": self.teamlog}, outfile
            )

    def build_comp_profile(self, playerList=None):
        """Build merged player profile DataFrame for comp computation.

        Args:
            playerList: Optional dict of {team: {player_name: stats_dict}}.
                If None, uses all seasons from self.players.

        Returns:
            (playerProfile, playerDict) where playerProfile is indexed by
            PLAYER_NAME, and playerDict maps player_name to stats_dict.
        """
        if self.playerProfile.empty:
            self.profile_market("MIN")

        if playerList is None:
            playerList = {}
            for season_key in self.players:
                playerList.update(self.players[season_key])

        players = []
        for team in playerList:
            players.extend(
                [
                    v | {"PLAYER_NAME": k, "TEAM_ABBREVIATION": team}
                    for k, v in playerList[team].items()
                ]
            )

        playerProfile = self.playerProfile.merge(
            pd.DataFrame(players).drop_duplicates(subset="PLAYER_NAME"),
            on="PLAYER_NAME",
            how="outer",
            suffixes=("_x", None),
        ).set_index("PLAYER_NAME")

        # Coalesce _x columns (gamelog values shadowed by roster values during merge)
        _x_cols = [c for c in playerProfile.columns if c.endswith("_x")]
        for col in _x_cols:
            base = col[:-2]
            if base in playerProfile.columns:
                playerProfile[base] = playerProfile[base].fillna(playerProfile[col])
        playerProfile.drop(columns=_x_cols, inplace=True, errors="ignore")
        playerProfile.fillna(0, inplace=True)

        playerDict = {}
        for team in playerList.values():
            playerDict.update(team)

        return playerProfile, playerDict

    def update_player_comps(self, year=None):
        if year is None:
            year = self.season_start.year
        with open(pkg_resources.files(data) / "playerCompStats.json") as infile:
            stats = json.load(infile)

        playerList = self.players.get(self.season_start.year - 1, {})
        playerList.update(self.players.get(self.season_start.year, {}))
        playerProfile, playerDict = self.build_comp_profile(playerList)
        all_features = set()
        for pos_weights in stats["WNBA"].values():
            all_features.update(pos_weights.keys())
        all_features = list(all_features)
        playerProfile = playerProfile[
            [f for f in all_features if f in playerProfile.columns]
        ].replace([np.nan, np.inf, -np.inf], 0)

        comps = {}
        for position in self.positions:
            pos_weights = stats["WNBA"][position]
            pos_features = list(pos_weights.keys())
            pos_players = [
                p
                for p, v in playerDict.items()
                if v["POS"] == position and p in playerProfile.index
            ]
            positionProfile = playerProfile.loc[pos_players, pos_features]
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(np.sqrt(list(pos_weights.values())))
            knn = BallTree(positionProfile)
            comps[position] = self._build_comps(knn, positionProfile, min_comps=5, max_comps=20)

        self.comps = comps
        filepath = pkg_resources.files(data) / "wnba_comps.json"
        with open(filepath, "w") as outfile:
            json.dump(comps, outfile, indent=4)

    def _compute_comps(self):
        """Build comps from loaded data at runtime (no JSON I/O)."""
        with open(pkg_resources.files(data) / "playerCompStats.json") as f:
            stats = json.load(f)

        all_features = set()
        for pos_weights in stats["WNBA"].values():
            all_features.update(pos_weights.keys())
        all_features = list(all_features)

        playerProfile, playerDict = self.build_comp_profile()
        playerProfile = playerProfile[
            [f for f in all_features if f in playerProfile.columns]
        ].replace([np.nan, np.inf, -np.inf], 0)

        comps = {}
        for position in self.positions:
            pos_weights = stats["WNBA"][position]
            pos_features = list(pos_weights.keys())
            pos_players = [
                p
                for p, v in playerDict.items()
                if v["POS"] == position and p in playerProfile.index
            ]
            if len(pos_players) < 7:
                continue
            positionProfile = playerProfile.loc[pos_players, pos_features]
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(np.sqrt(list(pos_weights.values())))
            knn = BallTree(positionProfile)
            comps[position] = self._build_comps(knn, positionProfile, min_comps=5, max_comps=20)

        self.comps = comps

