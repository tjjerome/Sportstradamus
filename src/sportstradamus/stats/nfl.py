"""StatsNFL: NFL player stats loading, feature engineering, and prediction."""

import importlib.resources as pkg_resources
import json
import os.path
import pickle
import warnings
from datetime import datetime, timedelta
from io import StringIO
from time import sleep

import line_profiler
import nfl_data_py as nfl
import nflreadpy as nflr
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


class StatsNFL(Stats):
    """A class for handling and analyzing NFL statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        None

    Additional Methods:
        None
    """

    def __init__(self):
        """Initialize the StatsNFL class."""
        super().__init__()
        self.season_start = datetime(2026, 9, 10).date()
        cols = [
            "player id",
            "player display name",
            "position group",
            "team",
            "season",
            "week",
            "season type",
            "snap pct",
            "completions",
            "attempts",
            "passing yards",
            "passing tds",
            "interceptions",
            "sacks",
            "sack fumbles",
            "sack fumbles lost",
            "passing 2pt conversions",
            "carries",
            "rushing yards",
            "rushing tds",
            "rushing fumbles",
            "rushing fumbles lost",
            "rushing 2pt conversions",
            "receptions",
            "targets",
            "receiving yards",
            "receiving tds",
            "receiving fumbles",
            "receiving fumbles lost",
            "receiving 2pt conversions",
            "fumbles",
            "fumbles lost",
            "yards",
            "tds",
            "qb yards",
            "qb tds",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "fantasy points parlayplay",
            "home",
            "opponent",
            "gameday",
            "game id",
            "target share",
            "air yards share",
            "wopr",
            "yards per target",
            "yards per carry",
            "completion percentage over expected",
            "completion percentage",
            "passer rating",
            "passer adot",
            "passer adot differential",
            "time to throw",
            "aggressiveness",
            "pass yards per attempt",
            "rushing yards over expected",
            "rushing success rate",
            "yac over expected",
            "separation created",
            "targets per route run",
            "first read targets per route run",
            "route participation",
            "midfield target rate",
            "midfield tprr",
            "yards per route run",
            "average depth of target",
            "receiver cp over expected",
            "first read target share",
            "redzone target share",
            "redzone carry share",
            "carry share",
            "longest completion",
            "longest rush",
            "longest reception",
            "sacks taken",
            "passing first downs",
            "first downs",
        ]
        self.gamelog = pd.DataFrame(columns=cols)
        team_cols = [
            "season",
            "week",
            "team",
            "gameday",
            "points",
            "WL",
            "pass_rate",
            "pass_rate_over_expected",
            "pass_rate_over_expected_110",
            "pass_rate_against",
            "pass_rate_over_expected_against",
            "rush_success_rate",
            "pass_success_rate",
            "redzone_success_rate",
            "first_read_success_rate",
            "midfield_success_rate",
            "rush_success_rate_allowed",
            "pass_success_rate_allowed",
            "redzone_success_rate_allowed",
            "first_read_success_rate_allowed",
            "midfield_success_rate_allowed",
            "epa_per_rush",
            "epa_per_pass",
            "redzone_epa",
            "first_read_epa",
            "midfield_epa",
            "yards_per_rush",
            "yards_per_pass",
            "epa_allowed_per_rush",
            "epa_allowed_per_pass",
            "redzone_epa_allowed",
            "first_read_epa_allowed",
            "midfield_epa_allowed",
            "yards_allowed_per_rush",
            "yards_allowed_per_pass",
            "completion_percentage_allowed",
            "cpoe_allowed",
            "pressure_per_pass",
            "stuffs_per_rush",
            "pressure_allowed_per_pass",
            "stuffs_allowed_per_rush",
            "expected_yards_per_rush",
            "blitz_rate",
            "epa_per_blitz",
            "epa_allowed_per_blitz",
            "exp_per_rush",
            "exp_per_pass",
            "exp_allowed_per_rush",
            "exp_allowed_per_pass",
            "plays_per_game",
            "time_of_possession",
            "time_per_play",
        ]

        self.teamlog = pd.DataFrame(columns=team_cols)
        self.stat_types = {
            "passing": [
                "completion percentage over expected",
                "completion percentage",
                "passer rating",
                "passer adot",
                "passer adot differential",
                "time to throw",
                "aggressiveness",
                "pass yards per attempt",
                "receiver drops",
                "midfield target rate",
                "longest completion",
            ],
            "receiving": [
                "target share",
                "air yards share",
                "wopr",
                "yards per target",
                "yac over expected",
                "separation created",
                "targets per route run",
                "first read targets per route run",
                "route participation",
                "yards per route run",
                "midfield tprr",
                "average depth of target",
                "receiver cp over expected",
                "first read target share",
                "redzone target share",
                "drop rate",
                "longest reception",
            ],
            "rushing": [
                "snap pct",
                "rushing yards over expected",
                "rushing success rate",
                "redzone carry share",
                "carry share",
                "yards per carry",
                "breakaway yards",
                "broken tackles",
                "longest rush",
            ],
            "offense": [
                "pass_rate",
                "pass_rate_over_expected",
                "pass_rate_over_expected_110",
                "rush_success_rate",
                "pass_success_rate",
                "redzone_success_rate",
                "first_read_success_rate",
                "midfield_success_rate",
                "epa_per_rush",
                "epa_per_pass",
                "redzone_epa",
                "first_read_epa",
                "midfield_epa",
                "exp_per_rush",
                "exp_per_pass",
                "yards_per_rush",
                "yards_per_pass",
                "pressure_allowed_per_pass",
                "stuffs_allowed_per_rush",
                "expected_yards_per_rush",
                "epa_per_blitz",
                "plays_per_game",
                "time_of_possession",
                "time_per_play",
            ],
            "defense": [
                "pass_rate_against",
                "pass_rate_over_expected_against",
                "rush_success_rate_allowed",
                "pass_success_rate_allowed",
                "redzone_success_rate_allowed",
                "first_read_success_rate_allowed",
                "midfield_success_rate_allowed",
                "epa_allowed_per_rush",
                "epa_allowed_per_pass",
                "redzone_epa_allowed",
                "first_read_epa_allowed",
                "midfield_epa_allowed",
                "exp_allowed_per_rush",
                "exp_allowed_per_pass",
                "yards_allowed_per_rush",
                "yards_allowed_per_pass",
                "completion_percentage_allowed",
                "cpoe_allowed",
                "pressure_per_pass",
                "stuffs_per_rush",
                "blitz_rate",
                "epa_allowed_per_blitz",
                "plays_per_game",
                "time_of_possession",
                "time_per_play",
            ],
        }
        self.volume_stats = ["attempts", "carries", "targets"]
        self.need_pbp = True
        self.default_total = 22.668
        self.positions = ["QB", "WR", "RB", "TE"]
        self.league = "NFL"
        self.log_strings = {
            "game": "game id",
            "date": "gameday",
            "player": "player display name",
            "usage": "snap pct",
            "usage_sec": "route participation",
            "position": "position",
            "team": "team",
            "opponent": "opponent",
            "home": "home",
            "win": "WL",
            "score": "points",
        }
        self.usage_stat = "snap pct"
        self.tiebreaker_stat = "route participation short"
        self._volume_model_cache = None

    def load(self):
        """Load data from files."""
        filepath = pkg_resources.files(data) / "nfl_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                pdata = pickle.load(infile)
                if type(pdata) is dict:
                    self.gamelog = pdata["gamelog"]
                    self.teamlog = pdata["teamlog"]
                    if "players" in pdata:
                        self.players = pdata["players"]
                else:
                    self.gamelog = pdata

    def update(self):
        """Update data from the web API."""
        # Fetch game logs
        self.need_pbp = True
        cols = [
            "player_id",
            "player_display_name",
            "position_group",
            "team",
            "season",
            "week",
            "season_type",
            "completions",
            "attempts",
            "passing_yards",
            "passing_tds",
            "interceptions",
            "sacks",
            "sack_fumbles",
            "sack_fumbles_lost",
            "passing_2pt_conversions",
            "carries",
            "rushing_yards",
            "rushing_tds",
            "rushing_fumbles",
            "rushing_fumbles_lost",
            "rushing_2pt_conversions",
            "receptions",
            "targets",
            "receiving_yards",
            "receiving_tds",
            "receiving_fumbles",
            "receiving_fumbles_lost",
            "receiving_2pt_conversions",
            "target_share",
            "air_yards_share",
            "wopr",
        ]
        try:
            nfl_data = nflr.load_player_stats().to_pandas()
            nfl_data["interceptions"] = (
                nfl_data["passing_interceptions"].fillna(0).infer_objects(copy=False)
            )
            nfl_data["sacks"] = nfl_data["sacks_suffered"].fillna(0).infer_objects(copy=False)
            nfl_data = nfl_data[cols]
        except:
            nfl_data = pd.DataFrame(columns=cols)

        try:
            snaps = nflr.load_snap_counts().to_pandas()
        except:
            snaps = pd.DataFrame(
                columns=[
                    "game_id",
                    "pfr_game_id",
                    "season",
                    "game_type",
                    "week",
                    "player",
                    "pfr_player_id",
                    "position",
                    "team",
                    "opponent",
                    "offense_snaps",
                    "offense_pct",
                    "defense_snaps",
                    "defense_pct",
                    "st_snaps",
                    "st_pct",
                ]
            )

        try:
            sched = nfl.import_schedules([self.season_start.year])
            sched.loc[sched["away_team"] == "LA", "away_team"] = "LAR"
            sched.loc[sched["home_team"] == "LA", "home_team"] = "LAR"
            sched.loc[sched["away_team"] == "OAK", "away_team"] = "LV"
            sched.loc[sched["home_team"] == "OAK", "home_team"] = "LV"
            sched.loc[sched["away_team"] == "WSH", "away_team"] = "WAS"
            sched.loc[sched["home_team"] == "WSH", "home_team"] = "WAS"
        except Exception:
            logger.warning(
                "Failed to fetch NFL schedule data; upcoming games and game metadata will not be updated this run"
            )
            sched = pd.DataFrame(
                columns=[
                    "away_team",
                    "home_team",
                    "gameday",
                    "weekday",
                    "gametime",
                    "week",
                    "game_id",
                ]
            )
        upcoming_games = sched.loc[
            pd.to_datetime(sched["gameday"]).dt.date >= datetime.today().date(),
            ["gameday", "away_team", "home_team", "weekday", "gametime"],
        ]
        if not upcoming_games.empty:
            upcoming_games["gametime"] = (
                upcoming_games["weekday"].str[:-3] + " " + upcoming_games["gametime"]
            )
            df1 = upcoming_games.rename(columns={"home_team": "Team", "away_team": "Opponent"})
            df2 = upcoming_games.rename(columns={"away_team": "Team", "home_team": "Opponent"})
            df1["Home"] = True
            df2["Home"] = False
            upcoming_games = pd.concat([df1, df2]).sort_values("gameday")
            self.upcoming_games = (
                upcoming_games.groupby("Team")
                .apply(lambda x: x.head(1))
                .droplevel(1)[["Opponent", "Home", "gameday", "gametime"]]
                .to_dict(orient="index")
            )

        nfl_data = nfl_data.merge(
            pd.concat(
                [
                    sched.rename(columns={"home_team": "recent_team"}),
                    sched.rename(columns={"away_team": "recent_team"}),
                ]
            )[["recent_team", "week", "gameday"]],
            how="left",
        )
        nfl_data = nfl_data.loc[nfl_data["position_group"].isin(["QB", "WR", "RB", "TE"])]
        snaps = snaps.loc[snaps["position"].isin(["QB", "WR", "RB", "TE"])]
        snaps["player_display_name"] = snaps["player"].map(remove_accents)
        snaps["snap_pct"] = snaps["offense_pct"]
        snaps = snaps[["player_display_name", "season", "week", "snap_pct"]]

        nfl_data["player_display_name"] = nfl_data["player_display_name"].map(remove_accents)

        nfl_data = nfl_data.merge(snaps, on=["player_display_name", "season", "week"])

        nfl_data["fumbles"] = (
            nfl_data["sack_fumbles"] + nfl_data["rushing_fumbles"] + nfl_data["receiving_fumbles"]
        )
        nfl_data["fumbles_lost"] = (
            nfl_data["sack_fumbles_lost"]
            + nfl_data["rushing_fumbles_lost"]
            + nfl_data["receiving_fumbles_lost"]
        )
        nfl_data["yards"] = nfl_data["receiving_yards"] + nfl_data["rushing_yards"]
        nfl_data["qb_yards"] = nfl_data["passing_yards"] + nfl_data["rushing_yards"]
        nfl_data["tds"] = nfl_data["rushing_tds"] + nfl_data["receiving_tds"]
        nfl_data["qb_tds"] = nfl_data["rushing_tds"] + nfl_data["passing_tds"]

        nfl_data["fantasy_points_prizepicks"] = (
            nfl_data["passing_yards"] / 25
            + nfl_data["passing_tds"] * 4
            - nfl_data["interceptions"]
            + nfl_data["yards"] / 10
            + nfl_data["tds"] * 6
            + nfl_data["receptions"]
            - nfl_data["fumbles_lost"]
            + nfl_data["passing_2pt_conversions"] * 2
            + nfl_data["rushing_2pt_conversions"] * 2
            + nfl_data["receiving_2pt_conversions"] * 2
        )

        nfl_data["fantasy_points_underdog"] = (
            nfl_data["passing_yards"] / 25
            + nfl_data["passing_tds"] * 4
            - nfl_data["interceptions"]
            + nfl_data["yards"] / 10
            + nfl_data["tds"] * 6
            + nfl_data["receptions"] / 2
            - nfl_data["fumbles_lost"] * 2
            + nfl_data["passing_2pt_conversions"] * 2
            + nfl_data["rushing_2pt_conversions"] * 2
            + nfl_data["receiving_2pt_conversions"] * 2
        )

        nfl_data["fantasy_points_parlayplay"] = (
            nfl_data["passing_yards"] / 20
            + nfl_data["passing_tds"] * 5
            - nfl_data["interceptions"] * 5
            + nfl_data["yards"] / 5
            + nfl_data["tds"] * 5
            - nfl_data["fumbles_lost"] * 3
        )

        nfl_data["yards_per_target"] = nfl_data["receiving_yards"] / nfl_data["targets"]

        nfl_data.rename(columns=lambda x: x.replace("_", " "), inplace=True)
        nfl_data.rename(columns={"position group": "position"}, inplace=True)
        nfl_data.drop(columns=["recent team"], inplace=True)
        nfl_data[["target share", "air yards share", "wopr", "yards per target"]] = (
            nfl_data[["target share", "air yards share", "wopr", "yards per target"]]
            .fillna(0)
            .infer_objects(copy=False)
        )

        nfl_data.loc[nfl_data["team"] == "LA", "team"] = "LAR"
        nfl_data.loc[nfl_data["team"] == "WSH", "team"] = "WAS"
        nfl_data.loc[nfl_data["team"] == "OAK", "team"] = "LV"

        if not nfl_data.empty:
            nfl_data.loc[:, "moneyline"] = nfl_data.apply(
                lambda x: archive.get_moneyline(self.league, x["gameday"], x["team"]), axis=1
            )
            nfl_data.loc[:, "totals"] = nfl_data.apply(
                lambda x: archive.get_total(self.league, x["gameday"], x["team"]), axis=1
            )

        self.gamelog = (
            pd.concat([self.gamelog, nfl_data], ignore_index=True)
            .drop_duplicates(["season", "week", "player id"], ignore_index=True)
            .reset_index(drop=True)
        )
        self.gamelog["player display name"] = self.gamelog["player display name"].apply(
            remove_accents
        )

        self.players = nfl.import_ids()
        self.players = self.players.loc[
            self.players["position"].isin(["QB", "RB", "WR", "TE"])
            & (self.players["team"] != "FA")
            & (self.players["team"] != "FA*")
        ]
        self.players.name = self.players.name.apply(remove_accents)
        self.players.team = self.players.team.map(
            {
                "NOS": "NO",
                "GBP": "GB",
                "TBB": "TB",
                "SFO": "SF",
                "KCC": "KC",
                "LVR": "LV",
                "JAC": "JAX",
            }
        )
        ids = self.players[["name", "gsis_id"]].dropna()
        ids.index = ids.name
        self.ids = ids.gsis_id.to_dict()
        self.players = self.players.drop_duplicates("name")
        self.players.index = self.players["name"]
        self.players = self.players[["age", "height", "weight", "team", "position"]]

        teamDataList = []
        for i, row in tqdm(
            self.gamelog.loc[self.gamelog.isna().any(axis=1)].iterrows(),
            desc="Updating NFL data",
            unit="game",
            total=len(self.gamelog.loc[self.gamelog.isna().any(axis=1)]),
        ):
            if row["opponent"] != row["opponent"]:
                if row["team"] in sched.loc[sched["week"] == row["week"], "home_team"].unique():
                    self.gamelog.at[i, "home"] = True
                    self.gamelog.at[i, "opponent"] = sched.loc[
                        (sched["week"] == row["week"]) & (sched["home_team"] == row["team"]),
                        "away_team",
                    ].values[0]
                    self.gamelog.at[i, "gameday"] = sched.loc[
                        (sched["week"] == row["week"]) & (sched["home_team"] == row["team"]),
                        "gameday",
                    ].values[0]
                    self.gamelog.at[i, "game id"] = sched.loc[
                        (sched["week"] == row["week"]) & (sched["home_team"] == row["team"]),
                        "game_id",
                    ].values[0]
                else:
                    self.gamelog.at[i, "home"] = False
                    self.gamelog.at[i, "opponent"] = sched.loc[
                        (sched["week"] == row["week"]) & (sched["away_team"] == row["team"]),
                        "home_team",
                    ].values[0]
                    self.gamelog.at[i, "gameday"] = sched.loc[
                        (sched["week"] == row["week"]) & (sched["away_team"] == row["team"]),
                        "gameday",
                    ].values[0]
                    self.gamelog.at[i, "game id"] = sched.loc[
                        (sched["week"] == row["week"]) & (sched["away_team"] == row["team"]),
                        "game_id",
                    ].values[0]
            if row.isna().any():
                if self.season_start.year != row["season"]:
                    self.season_start = datetime(row["season"], 9, 1).date()
                    self.need_pbp = True

                playerData = self.parse_pbp(
                    row["week"], row["team"], row["season"], row["player display name"]
                )
                if type(playerData) is not int:
                    for k, v in playerData.items():
                        self.gamelog.at[i, k.replace("_", " ")] = np.nan_to_num(v)

            if row["team"] not in self.teamlog.loc[
                (self.teamlog.season == row.season) & (self.teamlog.week == row.week), "team"
            ].to_list() and (row["week"], row["team"]) not in [
                (t["week"], t["team"]) for t in teamDataList
            ]:
                teamData = {
                    "season": row.season,
                    "week": row.week,
                    "team": row["team"],
                    "gameday": self.gamelog.at[i, "gameday"],
                }
                team_pbp = self.parse_pbp(row["week"], row["team"], row["season"])

                if type(team_pbp) is not int:
                    teamData.update(team_pbp)
                teamDataList.append(teamData)

        self.teamlog = pd.concat(
            [self.teamlog, pd.DataFrame.from_records(teamDataList)], ignore_index=True
        )
        self.teamlog = self.teamlog.sort_values("gameday").fillna(0).infer_objects(copy=False)
        self.gamelog = self.gamelog.sort_values("gameday")

        # Remove old games to prevent file bloat
        six_years_ago = datetime.today().date() - timedelta(days=2191)
        self.gamelog = self.gamelog[
            self.gamelog["gameday"].apply(
                lambda x: six_years_ago
                <= datetime.strptime(x, "%Y-%m-%d").date()
                <= datetime.today().date()
            )
        ]
        self.gamelog = self.gamelog[~self.gamelog["opponent"].isin(["AFC", "NFC"])]
        self.teamlog = self.teamlog[
            self.teamlog["gameday"].apply(
                lambda x: six_years_ago
                <= datetime.strptime(x, "%Y-%m-%d").date()
                <= datetime.today().date()
            )
        ]
        self.gamelog.drop_duplicates(inplace=True)
        self.teamlog.drop_duplicates(inplace=True)

        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog["player display name"] = self.gamelog["player display name"].apply(
                remove_accents
            )
            self.gamelog.loc[:, "moneyline"] = self.gamelog.apply(
                lambda x: archive.get_moneyline(self.league, x["gameday"], x["team"]), axis=1
            )
            self.gamelog.loc[:, "totals"] = self.gamelog.apply(
                lambda x: archive.get_total(self.league, x["gameday"], x["team"]), axis=1
            )

        # Save the updated player data
        filepath = pkg_resources.files(data) / "nfl_data.dat"
        with open(filepath, "wb") as outfile:
            pickle.dump(
                {"players": self.players, "gamelog": self.gamelog, "teamlog": self.teamlog},
                outfile,
                -1,
            )

    def parse_pbp(self, week, team, year, playerName=""):
        if self.need_pbp:
            self.pbp = nflr.load_pbp(year).to_pandas()
            self.pbp["play_time"] = (
                self.pbp["game_seconds_remaining"].diff(-1).fillna(0).infer_objects(copy=False)
            )
            self.pbp = self.pbp.loc[
                self.pbp["play_type"].isin(["run", "pass"]) | (self.pbp["desc"] == "END GAME")
            ]
            if self.season_start.year > 2021:
                ftn = nfl.import_ftn_data([self.season_start.year])
                ftn["game_id"] = ftn["nflverse_game_id"]
                ftn["play_id"] = ftn["nflverse_play_id"]
                ftn.drop(columns=["week", "season", "nflverse_game_id"], inplace=True)
                self.pbp = self.pbp.merge(ftn, on=["game_id", "play_id"], how="left")
            else:
                self.pbp["is_qb_out_of_pocket"] = False
                self.pbp["is_throw_away"] = False
                self.pbp["read_thrown"] = 0
                self.pbp["n_blitzers"] = 0

            self.pbp["pass"] = self.pbp["pass"].astype(bool)
            self.pbp["rush"] = self.pbp["rush"].astype(bool)
            self.pbp["qb_hit"] = self.pbp["qb_hit"].astype(bool)
            self.pbp["sack"] = self.pbp["sack"].astype(bool)
            self.pbp["qb_dropback"] = self.pbp["qb_dropback"].astype(bool)
            self.pbp["pass_attempt"] = self.pbp["pass_attempt"].astype(bool)
            self.pbp["redzone"] = (self.pbp["yardline_100"] <= 20).astype(bool)
            self.pbp.loc[self.pbp["home_team"] == "LA", "home_team"] = "LAR"
            self.pbp.loc[self.pbp["away_team"] == "LA", "away_team"] = "LAR"
            self.pbp.loc[self.pbp["posteam"] == "LA", "posteam"] = "LAR"
            self.pbp.loc[self.pbp["home_team"] == "WSH", "home_team"] = "WAS"
            self.pbp.loc[self.pbp["away_team"] == "WSH", "away_team"] = "WAS"
            self.pbp.loc[self.pbp["posteam"] == "WSH", "posteam"] = "WAS"
            self.pbp.loc[self.pbp["home_team"] == "OAK", "home_team"] = "LV"
            self.pbp.loc[self.pbp["away_team"] == "OAK", "away_team"] = "LV"
            self.pbp.loc[self.pbp["posteam"] == "OAK", "posteam"] = "LV"
            self.ngs = nfl.import_ngs_data("passing", [self.season_start.year])
            self.ngs = self.ngs.merge(
                nfl.import_ngs_data("receiving", [self.season_start.year]), how="outer"
            )
            self.ngs = self.ngs.merge(
                nfl.import_ngs_data("rushing", [self.season_start.year]), how="outer"
            )
            self.ngs["player_display_name"] = self.ngs["player_display_name"].apply(remove_accents)
            self.pfr = nfl.import_weekly_pfr("pass", [self.season_start.year])
            self.pfr = self.pfr.merge(
                nfl.import_weekly_pfr("rush", [self.season_start.year]), how="outer"
            )
            self.pfr = self.pfr.merge(
                nfl.import_weekly_pfr("rec", [self.season_start.year]), how="outer"
            )
            self.pfr["pfr_player_name"] = self.pfr["pfr_player_name"].apply(remove_accents)
            self.need_pbp = False
        pbp = self.pbp.loc[
            (self.pbp.week == week) & ((self.pbp.home_team == team) | (self.pbp.away_team == team))
        ]
        if pbp.empty:
            return 0
        pbp_off = pbp.loc[pbp.posteam == team]
        pbp_def = pbp.loc[pbp.posteam != team]
        home = pbp.iloc[0].home_team == team
        if playerName == "":
            pr = pbp_off["pass"].mean()
            proe = pbp_off["pass"].mean() - pbp_off["xpass"].mean()
            proe110 = (
                pbp_off.loc[(pbp_off["down"] == 1) & (pbp_off["ydstogo"] == 10), "pass"].mean()
                - pbp_off.loc[(pbp_off["down"] == 1) & (pbp_off["ydstogo"] == 10), "xpass"].mean()
            )
            pr_against = pbp_def["pass"].mean()
            proe_against = pbp_def["pass"].mean() - pbp_def["xpass"].mean()
            off_rush_sr = (pbp_off.loc[pbp_off["rush"], "epa"] > 0).mean()
            off_pass_sr = (pbp_off.loc[pbp_off["pass"], "epa"] > 0).mean()
            off_rz_sr = (pbp_off.loc[pbp_off["redzone"], "epa"] > 0).mean()
            off_fr_sr = (
                pbp_off.loc[(pbp_off["pass"]) & (pbp_off["read_thrown"] == "1"), "epa"] > 0
            ).mean()
            off_mid_sr = (
                pbp_off.loc[(pbp_off["pass"]) & (pbp_off["pass_location"] == "middle"), "epa"] > 0
            ).mean()
            def_rush_sr = (pbp_def.loc[pbp_def["rush"], "epa"] > 0).mean()
            def_pass_sr = (pbp_def.loc[pbp_def["pass"], "epa"] > 0).mean()
            def_rz_sr = (pbp_def.loc[pbp_def["redzone"], "epa"] > 0).mean()
            def_fr_sr = (
                pbp_def.loc[(pbp_def["pass"]) & (pbp_def["read_thrown"] == "1"), "epa"] > 0
            ).mean()
            def_mid_sr = (
                pbp_def.loc[(pbp_def["pass"]) & (pbp_def["pass_location"] == "middle"), "epa"] > 0
            ).mean()
            off_rush_epa = pbp_off.loc[pbp_off["rush"], "epa"].mean()
            off_pass_epa = pbp_off.loc[pbp_off["pass"], "epa"].mean()
            off_rz_epa = pbp_off.loc[pbp_off["redzone"], "epa"].mean()
            off_fr_epa = pbp_off.loc[
                (pbp_off["pass"]) & (pbp_off["read_thrown"] == "1"), "epa"
            ].mean()
            off_mid_epa = pbp_off.loc[
                (pbp_off["pass"]) & (pbp_off["pass_location"] == "middle"), "epa"
            ].mean()
            def_rush_epa = pbp_def.loc[pbp_def["rush"], "epa"].mean()
            def_pass_epa = pbp_def.loc[pbp_def["pass"], "epa"].mean()
            def_rz_epa = pbp_def.loc[pbp_def["redzone"], "epa"].mean()
            def_fr_epa = pbp_def.loc[
                (pbp_def["pass"]) & (pbp_def["read_thrown"] == "1"), "epa"
            ].mean()
            def_mid_epa = pbp_def.loc[
                (pbp_def["pass"]) & (pbp_def["pass_location"] == "middle"), "epa"
            ].mean()
            off_rush_ypa = pbp_off.loc[pbp_off["rush"], "yards_gained"].mean()
            off_pass_ypa = pbp_off.loc[pbp_off["pass"], "yards_gained"].mean()
            def_rush_ypa = pbp_def.loc[pbp_def["rush"], "yards_gained"].mean()
            def_pass_ypa = pbp_def.loc[pbp_def["pass"], "yards_gained"].mean()
            off_rush_exp = (pbp_off.loc[pbp_off["rush"], "yards_gained"] > 15).mean()
            off_pass_exp = (pbp_off.loc[pbp_off["pass"], "yards_gained"] > 15).mean()
            def_rush_exp = (pbp_def.loc[pbp_def["rush"], "yards_gained"] > 15).mean()
            def_pass_exp = (pbp_def.loc[pbp_def["pass"], "yards_gained"] > 15).mean()
            def_cpoe = pbp_def.loc[pbp_def["pass"], "cpoe"].mean() / 100
            def_cp = pbp_def.loc[pbp_def["pass"], "complete_pass"].mean()
            def_press = self.pfr.loc[
                (self.pfr["week"] == pbp.week.max()) & (self.pfr["opponent"] == team),
                "times_pressured",
            ].sum() / len(pbp_def.loc[pbp_def["qb_dropback"]])
            def_stuff = (pbp_def.loc[pbp_def["rush"], "yards_gained"] <= 0).mean()
            off_press = self.pfr.loc[
                (self.pfr["week"] == pbp.week.max()) & (self.pfr["team"] == team), "times_pressured"
            ].sum() / len(pbp_off.loc[pbp_off["qb_dropback"]])
            off_stuff = (pbp_off.loc[pbp_off["rush"], "yards_gained"] <= 0).mean()
            rush_ngs = self.ngs.loc[
                (self.ngs["player_position"] == "RB")
                & (self.ngs["team_abbr"] == team)
                & (self.ngs["week"] == pbp.week.max()),
                ["expected_rush_yards", "rush_attempts"],
            ].sum()
            off_rush_xya = rush_ngs.iloc[0] / rush_ngs.iloc[1]
            blitz_rate = pbp_def.loc[
                pbp_def["pass"] & (pbp_def["n_blitzers"] > 0), "n_blitzers"
            ].count() / len(pbp_def.loc[pbp_def["qb_dropback"]])
            off_blitz_epa = pbp_off.loc[
                pbp_off["qb_dropback"] & (pbp_off["n_blitzers"] > 0), "epa"
            ].mean()
            def_blitz_epa = pbp_def.loc[
                pbp_def["qb_dropback"] & (pbp_def["n_blitzers"] > 0), "epa"
            ].mean()
            plays = len(pbp_off)
            time_of_possession = pbp_off["play_time"].sum() / pbp["play_time"].sum()
            time_per_play = pbp_off["play_time"].mean()
            points = pbp["home_score"].iloc[-1] if home else pbp["away_score"].iloc[-1]
            pointsAgainst = pbp["away_score"].iloc[-1] if home else pbp["home_score"].iloc[-1]
            win = "W" if points > pointsAgainst else "L"

            return {
                "pass_rate": pr,
                "pass_rate_over_expected": proe,
                "pass_rate_over_expected_110": proe110,
                "pass_rate_against": pr_against,
                "pass_rate_over_expected_against": proe_against,
                "rush_success_rate": off_rush_sr,
                "pass_success_rate": off_pass_sr,
                "redzone_success_rate": off_rz_sr,
                "first_read_success_rate": off_fr_sr,
                "midfield_success_rate": off_mid_sr,
                "rush_success_rate_allowed": def_rush_sr,
                "pass_success_rate_allowed": def_pass_sr,
                "redzone_success_rate_allowed": def_rz_sr,
                "first_read_success_rate_allowed": def_fr_sr,
                "midfield_success_rate_allowed": def_mid_sr,
                "epa_per_rush": off_rush_epa,
                "epa_per_pass": off_pass_epa,
                "redzone_epa": off_rz_epa,
                "first_read_epa": off_fr_epa,
                "midfield_epa": off_mid_epa,
                "epa_allowed_per_rush": def_rush_epa,
                "epa_allowed_per_pass": def_pass_epa,
                "redzone_epa_allowed": def_rz_epa,
                "first_read_epa_allowed": def_fr_epa,
                "midfield_epa_allowed": def_mid_epa,
                "exp_per_rush": off_rush_exp,
                "exp_per_pass": off_pass_exp,
                "exp_allowed_per_rush": def_rush_exp,
                "exp_allowed_per_pass": def_pass_exp,
                "yards_per_rush": off_rush_ypa,
                "yards_per_pass": off_pass_ypa,
                "yards_allowed_per_rush": def_rush_ypa,
                "yards_allowed_per_pass": def_pass_ypa,
                "completion_percentage_allowed": def_cp,
                "cpoe_allowed": def_cpoe,
                "pressure_per_pass": def_press,
                "stuffs_per_rush": def_stuff,
                "pressure_allowed_per_pass": off_press,
                "stuffs_allowed_per_rush": off_stuff,
                "expected_yards_per_rush": off_rush_xya,
                "blitz_rate": blitz_rate,
                "epa_per_blitz": off_blitz_epa,
                "epa_allowed_per_blitz": def_blitz_epa,
                "plays_per_game": plays,
                "time_of_possession": time_of_possession,
                "time_per_play": time_per_play,
                "points": points,
                "WL": win,
            }

        else:
            if self.ids.get(playerName) is None:
                return {
                    "completion_percentage_over_expected": 0,
                    "completion_percentage": 0,
                    "passer_rating": 0,
                    "passer_adot": 0,
                    "passer_adot_differential": 0,
                    "time_to_throw": 0,
                    "aggressiveness": 0,
                    "pass_yards_per_attempt": 0,
                    "receiver_drops": 0,
                    "midfield_target_rate": 0,
                    "rushing_yards_over_expected": 0,
                    "rushing_success_rate": 0,
                    "breakaway_yards": 0,
                    "broken_tackles": 0,
                    "drop_rate": 0,
                    "yac_over_expected": 0,
                    "separation_created": 0,
                    "targets_per_route_run": 0,
                    "first_read_targets_per_route_run": 0,
                    "route_participation": 0,
                    "yards_per_route_run": 0,
                    "yards_per_carry": 0,
                    "midfield_tprr": 0,
                    "average_depth_of_target": 0,
                    "receiver_cp_over_expected": 0,
                    "first_read_target_share": 0,
                    "redzone_target_share": 0,
                    "redzone_carry_share": 0,
                    "carry_share": 0,
                    "longest_completion": 0,
                    "longest_rush": 0,
                    "longest_reception": 0,
                    "sacks_taken": 0,
                    "passing_first_downs": 0,
                    "first_downs": 0,
                }

            cpoe = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "completion_percentage_above_expectation",
            ].mean()
            cp = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "completion_percentage",
            ].mean()
            qbr = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "passer_rating",
            ].mean()
            pass_adot = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "avg_intended_air_yards",
            ].mean()
            pass_adot_diff = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "avg_air_yards_differential",
            ].mean()
            time_to_throw = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "avg_time_to_throw",
            ].mean()
            aggressiveness = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "aggressiveness",
            ].mean()
            ryoe = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "rush_yards_over_expected",
            ].mean()
            rush_sr = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "rush_pct_over_expected",
            ].mean()
            yacoe = self.ngs.loc[
                (self.ngs["player_display_name"] == playerName)
                & (self.ngs["week"] == pbp.week.max()),
                "avg_yac_above_expectation",
            ].mean()
            sep = (
                self.ngs.loc[
                    (self.ngs["player_display_name"] == playerName)
                    & (self.ngs["week"] == pbp.week.max()),
                    "avg_separation",
                ].mean()
                - self.ngs.loc[
                    (self.ngs["player_display_name"] == playerName)
                    & (self.ngs["week"] == pbp.week.max()),
                    "avg_cushion",
                ].mean()
            )
            broken_tackles = (
                self.pfr.loc[
                    (self.pfr["pfr_player_name"] == playerName)
                    & (self.pfr["week"] == pbp.week.max()),
                    ["rushing_broken_tackles", "receiving_broken_tackles"],
                ]
                .sum(axis=1)
                .mean()
            )
            breakaway_yards = self.pfr.loc[
                (self.pfr["pfr_player_name"] == playerName) & (self.pfr["week"] == pbp.week.max()),
                "rushing_yards_after_contact_avg",
            ].mean()
            drop_pct = self.pfr.loc[
                (self.pfr["pfr_player_name"] == playerName) & (self.pfr["week"] == pbp.week.max()),
                "receiving_drop_pct",
            ].mean()
            pass_drop_pct = self.pfr.loc[
                (self.pfr["pfr_player_name"] == playerName) & (self.pfr["week"] == pbp.week.max()),
                "passing_drop_pct",
            ].mean()
            pass_ypa = (
                pbp_off.loc[pbp_off["passer_player_id"] == self.ids.get(playerName), "yards_gained"]
                .fillna(0)
                .infer_objects(copy=False)
                .mean()
            )
            ypc = (
                pbp_off.loc[pbp_off["rusher_player_id"] == self.ids.get(playerName), "yards_gained"]
                .fillna(0)
                .infer_objects(copy=False)
                .mean()
            )
            routes_run = int(
                pbp_off["pass"].sum()
                * self.gamelog.loc[
                    (self.gamelog.season == year)
                    & (self.gamelog.week == week)
                    & (self.gamelog["player display name"] == playerName),
                    "snap pct",
                ].mean()
            )
            targets = len(pbp_off.loc[pbp_off["receiver_player_id"] == self.ids.get(playerName)])
            fr_targets = len(
                pbp_off.loc[
                    (pbp_off["receiver_player_id"] == self.ids.get(playerName))
                    & (pbp_off["read_thrown"] == "1")
                ]
            )
            pass_attempts = len(pbp_off.loc[pbp_off["pass"]])
            fr_pass_attempts = len(pbp_off.loc[(pbp_off["pass"]) & (pbp_off["read_thrown"] == "1")])
            mid_target_rate = (
                (
                    len(
                        pbp_off.loc[
                            (pbp_off["passer_player_id"] == self.ids.get(playerName))
                            & (pbp_off["pass_location"] == "middle")
                        ]
                    )
                    / pass_attempts
                )
                if pass_attempts > 0
                else np.nan
            )
            mid_targets = (
                (
                    len(
                        pbp_off.loc[
                            (pbp_off["receiver_player_id"] == self.ids.get(playerName))
                            & (pbp_off["pass_location"] == "middle")
                        ]
                    )
                    / routes_run
                )
                if routes_run > 0
                else np.nan
            )
            tprr = (targets / routes_run) if routes_run > 0 else np.nan
            frtprr = (fr_targets / routes_run) if routes_run > 0 else np.nan
            frt_pct = (fr_targets / fr_pass_attempts) if fr_pass_attempts > 0 else np.nan
            route_participation = (routes_run / pass_attempts) if pass_attempts > 0 else np.nan
            yprr = (
                (
                    pbp_off.loc[
                        pbp_off["receiver_player_id"] == self.ids.get(playerName), "yards_gained"
                    ].sum()
                    / routes_run
                )
                if routes_run > 0
                else np.nan
            )
            adot = pbp_off.loc[
                pbp_off["receiver_player_id"] == self.ids.get(playerName), "air_yards"
            ].mean()
            rec_cpoe = pbp_off.loc[
                pbp_off["receiver_player_id"] == self.ids.get(playerName), "cpoe"
            ].mean()
            rz_passes = len(pbp_off.loc[pbp_off["pass_attempt"] & pbp_off["redzone"]])
            rz_target_pct = (
                (
                    len(
                        pbp_off.loc[
                            (pbp_off["receiver_player_id"] == self.ids.get(playerName))
                            & pbp_off["redzone"]
                        ]
                    )
                    / rz_passes
                )
                if rz_passes > 0
                else np.nan
            )
            rz_rushes = len(pbp_off.loc[pbp_off["rush"] & pbp_off["redzone"]])
            rz_attempt_pct = (
                (
                    len(
                        pbp_off.loc[
                            (pbp_off["rusher_player_id"] == self.ids.get(playerName))
                            & pbp_off["redzone"]
                        ]
                    )
                    / rz_rushes
                )
                if rz_rushes > 0
                else np.nan
            )
            rushes = len(pbp_off.loc[pbp_off["rush"]])
            attempt_pct = (
                (len(pbp_off.loc[pbp_off["rusher_player_id"] == self.ids.get(playerName)]) / rushes)
                if rushes > 0
                else np.nan
            )

            sacks_taken = pbp_off.loc[
                pbp_off["passer_player_id"] == self.ids.get(playerName), "sack"
            ].sum()
            longest_completion = pbp_off.loc[
                pbp_off["passer_player_id"] == self.ids.get(playerName), "passing_yards"
            ].max()
            longest_rush = pbp_off.loc[
                pbp_off["rusher_player_id"] == self.ids.get(playerName), "rushing_yards"
            ].max()
            longest_reception = pbp_off.loc[
                pbp_off["receiver_player_id"] == self.ids.get(playerName), "receiving_yards"
            ].max()
            passing_first_downs = len(
                pbp_off.loc[
                    (pbp_off["passer_player_id"] == self.ids.get(playerName))
                    & (pbp_off["yards_gained"] > pbp_off["ydstogo"])
                ]
            )
            first_downs = len(
                pbp_off.loc[
                    (
                        (pbp_off["rusher_player_id"] == self.ids.get(playerName))
                        | (pbp_off["receiver_player_id"] == self.ids.get(playerName))
                    )
                    & (pbp_off["yards_gained"] > pbp_off["ydstogo"])
                ]
            )

            return {
                "completion_percentage_over_expected": cpoe,
                "completion_percentage": cp,
                "passer_rating": qbr,
                "passer_adot": pass_adot,
                "passer_adot_differential": pass_adot_diff,
                "time_to_throw": time_to_throw,
                "aggressiveness": aggressiveness,
                "pass_yards_per_attempt": pass_ypa,
                "receiver_drops": pass_drop_pct,
                "midfield_target_rate": mid_target_rate,
                "rushing_yards_over_expected": ryoe,
                "rushing_success_rate": rush_sr,
                "breakaway_yards": breakaway_yards,
                "broken_tackles": broken_tackles,
                "drop_rate": drop_pct,
                "yac_over_expected": yacoe,
                "separation_created": sep,
                "targets_per_route_run": tprr,
                "first_read_targets_per_route_run": frtprr,
                "route_participation": route_participation,
                "yards_per_route_run": yprr,
                "midfield_tprr": mid_targets,
                "average_depth_of_target": adot,
                "receiver_cp_over_expected": rec_cpoe,
                "first_read_target_share": frt_pct,
                "redzone_target_share": rz_target_pct,
                "redzone_carry_share": rz_attempt_pct,
                "carry_share": attempt_pct,
                "yards_per_carry": ypc,
                "longest_completion": longest_completion,
                "longest_rush": longest_rush,
                "longest_reception": longest_reception,
                "sacks_taken": sacks_taken,
                "passing_first_downs": passing_first_downs,
                "first_downs": first_downs,
            }

    def build_comp_profile(self):
        """Build the full NFL player comp profile with all derived features.

        Loads PFF CSV data, derives rate/differential features, and joins
        with nfl.import_ids() for age/height/bmi.

        Returns:
            playerProfile DataFrame indexed by player name, or empty DataFrame on failure.
            Call load() before this method.
        """
        if self.playerProfile.empty:
            self.profile_market("snap pct")

        try:
            nfl_players = nfl.import_ids()
            nfl_players = nfl_players.loc[nfl_players["position"].isin(["QB", "RB", "WR", "TE"])]
            nfl_players.index = nfl_players.name.apply(remove_accents)
            nfl_players["bmi"] = (
                nfl_players["weight"] / nfl_players["height"] / nfl_players["height"]
            )
            nfl_players = nfl_players[["age", "height", "bmi"]].dropna()
        except Exception:
            nfl_players = pd.DataFrame()

        year = self.season_start.year
        playerProfile = pd.DataFrame()
        for y in reversed(range(year - 3, year + 1)):
            playerFolder = pkg_resources.files(data) / f"player_data/NFL/{y}"
            if os.path.exists(playerFolder):
                for file in os.listdir(playerFolder):
                    if file.endswith(".csv"):
                        df = pd.read_csv(playerFolder / file)
                        df.index = df.player_id
                        playerProfile = playerProfile.combine_first(df)

        if playerProfile.empty:
            return playerProfile

        playerProfile.loc[playerProfile.position == "HB", "position"] = "RB"
        playerProfile.loc[playerProfile.position == "FB", "position"] = "RB"
        playerProfile = playerProfile.loc[playerProfile.position.isin(["QB", "RB", "WR", "TE"])]
        playerProfile.loc[playerProfile.position == "QB", "dropbacks_per_game"] = (
            playerProfile.loc[playerProfile.position == "QB", "dropbacks"]
            / playerProfile.loc[playerProfile.position == "QB", "player_game_count"]
        )
        playerProfile.loc[playerProfile.position == "QB", "blitz_grades_pass_diff"] = (
            playerProfile.loc[playerProfile.position == "QB", "blitz_grades_pass"]
            - playerProfile.loc[playerProfile.position == "QB", "grades_pass"]
        )
        playerProfile.loc[playerProfile.position == "QB", "pa_grades_pass_diff"] = (
            playerProfile.loc[playerProfile.position == "QB", "pa_grades_pass"]
            - playerProfile.loc[playerProfile.position == "QB", "grades_pass"]
        )
        playerProfile.loc[playerProfile.position == "QB", "screen_grades_pass_diff"] = (
            playerProfile.loc[playerProfile.position == "QB", "screen_grades_pass"]
            - playerProfile.loc[playerProfile.position == "QB", "grades_pass"]
        )
        playerProfile.loc[playerProfile.position == "QB", "deep_grades_pass_diff"] = (
            playerProfile.loc[playerProfile.position == "QB", "deep_grades_pass"]
            - playerProfile.loc[playerProfile.position == "QB", "grades_pass"]
        )
        playerProfile.loc[playerProfile.position == "QB", "cm_grades_pass_diff"] = (
            playerProfile.loc[playerProfile.position == "QB", "center_medium_grades_pass"]
            - playerProfile.loc[playerProfile.position == "QB", "grades_pass"]
        )
        playerProfile.loc[playerProfile.position == "QB", "scrambles_per_dropback"] = (
            playerProfile.loc[playerProfile.position == "QB", "scrambles"]
            / playerProfile.loc[playerProfile.position == "QB", "dropbacks"]
        )
        playerProfile.loc[playerProfile.position == "QB", "designed_yards_per_game"] = (
            playerProfile.loc[playerProfile.position == "QB", "designed_yards"]
            / playerProfile.loc[playerProfile.position == "QB", "player_game_count"]
        )
        playerProfile.loc[playerProfile.position != "QB", "man_grades_pass_route_diff"] = (
            playerProfile.loc[playerProfile.position != "QB", "man_grades_pass_route"]
            - playerProfile.loc[playerProfile.position != "QB", "grades_pass_route"]
        )
        playerProfile.loc[playerProfile.position == "RB", "breakaway_yards_per_game"] = (
            playerProfile.loc[playerProfile.position == "RB", "breakaway_yards"]
            / playerProfile.loc[playerProfile.position == "RB", "player_game_count"]
        )
        playerProfile.loc[playerProfile.position == "RB", "total_touches_per_game"] = (
            playerProfile.loc[playerProfile.position == "RB", "total_touches"]
            / playerProfile.loc[playerProfile.position == "RB", "player_game_count"]
        )
        playerProfile.loc[playerProfile.position != "QB", "contested_target_rate"] = (
            playerProfile.loc[playerProfile.position != "QB", "contested_targets"]
            / playerProfile.loc[playerProfile.position != "QB", "targets"]
        )
        playerProfile.loc[playerProfile.position != "QB", "deep_contested_target_rate"] = (
            playerProfile.loc[playerProfile.position != "QB", "deep_contested_targets"]
            / playerProfile.loc[playerProfile.position != "QB", "targets"]
        )
        playerProfile.loc[playerProfile.position != "QB", "zone_grades_pass_route_diff"] = (
            playerProfile.loc[playerProfile.position != "QB", "zone_grades_pass_route"]
            - playerProfile.loc[playerProfile.position != "QB", "grades_pass_route"]
        )
        playerProfile.loc[playerProfile.position != "QB", "man_grades_pass_route_diff"] = (
            playerProfile.loc[playerProfile.position != "QB", "man_grades_pass_route"]
            - playerProfile.loc[playerProfile.position != "QB", "grades_pass_route"]
        )
        playerProfile.index = playerProfile.player.apply(remove_accents)
        playerProfile = playerProfile.join(self.playerProfile[self.playerProfile.columns[9:]])
        if not nfl_players.empty:
            playerProfile = playerProfile.join(nfl_players)

        return playerProfile

    def update_player_comps(self, year=None):
        if year is None:
            year = self.season_start.year
        with open(pkg_resources.files(data) / "playerCompStats.json") as infile:
            stats = json.load(infile)

        filterStat = {"QB": "dropbacks", "RB": "attempts", "WR": "routes", "TE": "routes"}

        playerProfile = self.build_comp_profile()
        if playerProfile.empty:
            return

        comps = {}
        for position in ["QB", "RB", "WR", "TE"]:
            positionProfile = playerProfile.loc[playerProfile.position == position]
            positionProfile[filterStat[position]] = (
                positionProfile[filterStat[position]] / positionProfile["player_game_count"]
            )
            positionProfile = positionProfile.loc[
                positionProfile[filterStat[position]]
                >= positionProfile[filterStat[position]].quantile(0.25)
            ]
            positionProfile = positionProfile[list(stats["NFL"][position].keys())].replace(
                [np.nan, np.inf, -np.inf], 0
            )
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(np.sqrt(list(stats["NFL"][position].values())))
            knn = BallTree(positionProfile)
            comps[position] = self._build_comps(knn, positionProfile, min_comps=5, max_comps=15)

        filepath = pkg_resources.files(data) / "nfl_comps.json"
        with open(filepath, "w") as outfile:
            json.dump(comps, outfile, indent=4)

    def _compute_comps(self):
        """Build comps from loaded data at runtime (no JSON I/O)."""
        with open(pkg_resources.files(data) / "playerCompStats.json") as f:
            stats = json.load(f)

        filterStat = {"QB": "dropbacks", "RB": "attempts", "WR": "routes", "TE": "routes"}

        playerProfile = self.build_comp_profile()
        if playerProfile.empty:
            return

        comps = {}
        for position in ["QB", "RB", "WR", "TE"]:
            positionProfile = playerProfile.loc[playerProfile.position == position]
            positionProfile[filterStat[position]] = (
                positionProfile[filterStat[position]] / positionProfile["player_game_count"]
            )
            positionProfile = positionProfile.loc[
                positionProfile[filterStat[position]]
                >= positionProfile[filterStat[position]].quantile(0.25)
            ]
            positionProfile = positionProfile[list(stats["NFL"][position].keys())].replace(
                [np.nan, np.inf, -np.inf], 0
            )
            if len(positionProfile) < 7:
                continue
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(np.sqrt(list(stats["NFL"][position].values())))
            knn = BallTree(positionProfile)
            comps[position] = self._build_comps(knn, positionProfile, min_comps=5, max_comps=15)

        self.comps = comps


    def check_combo_markets(self, market, player, date=datetime.today().date()):
        return 0  # TODO reimplement this
        player_games = self.short_gamelog.loc[
            self.short_gamelog[self.log_strings["player"]] == player
        ]
        cv = stat_cv.get(self.league, {}).get(market, 1)
        dist = stat_dist.get(self.league, {}).get(market, "Gamma")
        if not isinstance(date, str):
            date = date.strftime("%Y-%m-%d")
        if market in combo_props:
            ev = 0
            for submarket in combo_props.get(market, []):
                sub_cv = stat_cv[self.league].get(submarket, 1)
                sub_dist = stat_dist.get(self.league, {}).get(submarket, "Gamma")
                v = archive.get_ev(self.league, submarket, date, player)
                subline = archive.get_line(self.league, submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist)
                if np.isnan(v) or v == 0:
                    ev = 0
                    break
                else:
                    ev += v

        elif market in ["rushing tds", "receiving tds"]:
            ev = (
                (
                    archive.get_ev(self.league, "tds", date, player)
                    * player_games[market].sum()
                    / player_games["tds"].sum()
                )
                if player_games["tds"].sum()
                else 0
            )

        elif "fantasy" in market:
            ev = 0
            book_odds = False
            if "prizepicks" in market:
                fantasy_props = [
                    ("passing yards", 1 / 25),
                    ("passing tds", 4),
                    ("interceptions", -1),
                    ("rushing yards", 0.1),
                    ("receiving yards", 0.1),
                    ("tds", 6),
                    ("receptions", 1),
                ]
            else:
                fantasy_props = [
                    ("passing yards", 1 / 25),
                    ("passing tds", 4),
                    ("interceptions", -1),
                    ("rushing yards", 0.1),
                    ("receiving yards", 0.1),
                    ("tds", 6),
                    ("receptions", 0.5),
                ]
            for submarket, weight in fantasy_props:
                sub_cv = stat_cv[self.league].get(submarket, 1)
                sub_dist = stat_dist.get(self.league, {}).get(submarket, "Gamma")
                v = archive.get_ev(self.league, submarket, date, player)
                subline = archive.get_line(self.league, submarket, date, player)
                if sub_dist != dist and not np.isnan(v):
                    v = get_ev(subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist)
                if np.isnan(v) or v == 0:
                    if subline == 0 and not player_games.empty:
                        subline = np.floor(player_games.iloc[-10:][submarket].median()) + 0.5

                    if not subline <= 0:
                        under = (player_games[submarket] < subline).mean()
                        ev += get_ev(subline, under, sub_cv, dist=sub_dist)
                else:
                    book_odds = True
                    ev += v * weight

            if not book_odds:
                ev = 0
        else:
            ev = 0

        return 0 if np.isnan(ev) else ev

    def get_volume_stats(self, offers, date=datetime.today().date()):
        flat_offers = {}
        if isinstance(offers, dict):
            for players in offers.values():
                flat_offers.update(players)
        else:
            flat_offers = offers

        position_map = {"attempts": [1], "carries": [1, 3], "targets": [2, 3, 4]}

        for market in self.volume_stats:
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
                # Slice to the trained schema embedded in the pickle. This is the
                # source of truth — never re-derive from feature_filter.json here,
                # which can drift between when the volume model was trained and now.
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
                prob_params = prob_params.loc[
                    playerStats["Player position"].isin(position_map[market])
                ]

            else:
                logger.warning(f"{filename} missing")
                return

            # ---------------------------------------------------------------
            # Store distribution parameters in playerProfile.
            #
            # attempts  → SkewNormal: loc, scale, alpha
            # carries / targets → NegBin/ZINB: probs (p), total_count (n)
            #   NegBin mean     μ = n(1−p)/p
            #   NegBin variance σ² = n(1−p)/p² = μ/p
            # ---------------------------------------------------------------
            # Drop gate column for ZI distributions — not needed for budget normalization
            prob_params.drop(columns=["gate"], inplace=True, errors="ignore")
            # All NFL volume stats (attempts, carries, targets) now use SkewNormal
            rename_map = {
                "loc": f"proj {market} loc",
                "scale": f"proj {market} scale",
                "alpha": f"proj {market} alpha",
            }

            self.playerProfile = self.playerProfile.join(
                prob_params.rename(columns=rename_map), lsuffix="_obs"
            )
            self.playerProfile.drop(
                columns=[col for col in self.playerProfile.columns if "_obs" in col],
                inplace=True,
            )

            # ---------------------------------------------------------------
            # Volume normalization — precision-weighted, always-adjust.
            #
            # Two hard constraints for the rush/receiving markets:
            #   (B) total carries ≈ plays_per_game − proj_attempts − unmodeled_carry_reserve
            #   (C) total targets ≈ proj_attempts − unmodeled_target_reserve
            #
            # proj_attempts is derived from SkewNormal:
            #   E[X] = loc + scale * delta * sqrt(2/π) where delta = alpha/sqrt(1+alpha²)
            # ---------------------------------------------------------------
            if market == "attempts":
                # SkewNormal: E[X] = loc + scale * delta * sqrt(2/pi)
                loc = self.playerProfile[f"proj {market} loc"].fillna(0)
                scale = self.playerProfile[f"proj {market} scale"].fillna(0)
                sn_alpha = (
                    self.playerProfile[f"proj {market} alpha"].fillna(0)
                    if f"proj {market} alpha" in self.playerProfile.columns
                    else 0
                )

                delta = sn_alpha / np.sqrt(1 + sn_alpha**2)
                true_means = loc + scale * delta * np.sqrt(2 / np.pi)
                true_stds = scale

                self.playerProfile.loc[loc.index, f"proj {market} mean"] = true_means
                self.playerProfile.loc[scale.index, f"proj {market} std"] = true_stds
                self.playerProfile.drop(
                    columns=[f"proj {market} loc", f"proj {market} scale"],
                    inplace=True,
                    errors="ignore",
                )
                self.playerProfile.drop(
                    columns=[f"proj {market} alpha"], inplace=True, errors="ignore"
                )

                self.playerProfile.fillna(0, inplace=True)
                continue

            unmodeled_carry_reserve = 6  # measured: mean carries for rank 3+ rushers
            unmodeled_target_reserve = 8  # measured: mean targets for rank 5+ receivers
            carry_cap = 40  # absolute single-player carry ceiling
            target_cap = 20  # absolute single-player target ceiling

            teams = self.playerProfile.loc[self.playerProfile["team"] != 0].groupby("team")
            for team, team_df in teams:
                # SkewNormal: E[X] = loc + scale * delta * sqrt(2/pi)
                loc = team_df[f"proj {market} loc"].copy()
                scale = team_df[f"proj {market} scale"].copy()
                sn_alpha = (
                    team_df[f"proj {market} alpha"].copy()
                    if f"proj {market} alpha" in team_df.columns
                    else 0
                )

                delta = sn_alpha / np.sqrt(1 + sn_alpha**2)
                true_means = loc + scale * delta * np.sqrt(2 / np.pi)
                true_vars = scale**2
                total = true_means.sum()

                if total <= 0:
                    continue

                plays = self.teamProfile.loc[team, "plays_per_game"]

                # proj_attempts for this team: from attempts market
                if "proj attempts mean" in self.playerProfile.columns:
                    proj_attempts = self.playerProfile.loc[
                        self.playerProfile["team"] == team, "proj attempts mean"
                    ].sum()
                else:
                    pass_rate = self.teamProfile.loc[team, "pass_rate"]
                    proj_attempts = plays * pass_rate

                if market == "carries":
                    cap = carry_cap
                    rush_budget = plays - proj_attempts
                    target = max(rush_budget - unmodeled_carry_reserve, len(team_df))
                else:  # targets
                    cap = target_cap
                    target = max(proj_attempts - unmodeled_target_reserve, len(team_df))

                # Precision-weighted deficit distribution
                deficit = target - total
                total_var = true_vars.sum()
                if total_var > 0:
                    adjustments = true_vars / total_var * deficit
                else:
                    adjustments = true_means / total * deficit

                new_means = (true_means + adjustments).clip(lower=0, upper=cap)

                # SkewNormal: scale both loc and scale to preserve CV
                ratio = new_means / true_means.replace(0, np.nan)
                ratio = ratio.fillna(1.0).clip(lower=0.1, upper=10.0)
                new_scale = scale * ratio
                new_loc = new_means - new_scale * delta * np.sqrt(2 / np.pi)

                self.playerProfile.loc[loc.index, f"proj {market} loc"] = new_loc
                self.playerProfile.loc[scale.index, f"proj {market} scale"] = new_scale
                self.playerProfile.loc[loc.index, f"proj {market} mean"] = new_means
                self.playerProfile.loc[scale.index, f"proj {market} std"] = new_scale
            self.playerProfile.fillna(0, inplace=True)

        # Defragment after 3-market loop's sequential .join / .loc column inserts.
        self.playerProfile = self.playerProfile.copy()

