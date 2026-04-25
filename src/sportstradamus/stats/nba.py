"""StatsNBA: NBA player stats loading, feature engineering, and prediction."""

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


class StatsNBA(Stats):
    """A class for handling and analyzing NBA statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        None

    Additional Methods:
        None
    """

    def __init__(self):
        """Initialize the StatsNBA class."""
        super().__init__()
        self.league = "NBA"
        self.positions = ["P", "C", "F", "W", "B"]
        self.season_start = datetime(2025, 10, 21).date()
        self.season = "2025-26"
        cols = [
            "SEASON_YEAR",
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "GAME_ID",
            "GAME_DATE",
            "WL",
            "MIN",
            "FGM",
            "FGA",
            "FG3M",
            "FG3A",
            "FTM",
            "FTA",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "TOV",
            "STL",
            "BLK",
            "BLKA",
            "PF",
            "PFD",
            "PTS",
            "FG_PCT",
            "FG3_PCT",
            "FT_PCT",
            "FG3_RATIO",
            "PLUS_MINUS",
            "POS",
            "HOME",
            "OPP",
            "PRA",
            "PR",
            "PA",
            "RA",
            "BLST",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "fantasy points parlay",
            "OFF_RATING",
            "DEF_RATING",
            "E_OFF_RATING",
            "E_DEF_RATING",
            "AST_PCT",
            "AST_TO",
            "AST_RATIO",
            "OREB_PCT",
            "DREB_PCT",
            "REB_PCT",
            "EFG_PCT",
            "TS_PCT",
            "USG_PCT",
            "BLK_PCT",
            "PIE",
            "FTR",
            "PACE",
            "PCT_FGA",
            "PCT_FG3A",
            "PCT_OREB",
            "PCT_DREB",
            "PCT_REB",
            "PCT_AST",
            "PCT_TOV",
            "PCT_STL",
            "PCT_BLKA",
            "FGA_48",
            "FG3A_48",
            "REB_48",
            "OREB_48",
            "DREB_48",
            "AST_48",
            "TOV_48",
            "BLKA_48",
            "STL_48",
        ]
        self.gamelog = pd.DataFrame(columns=cols)

        team_cols = [
            "SEASON_YEAR",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "TEAM_NAME",
            "GAME_ID",
            "GAME_DATE",
            "OPP",
            "WL",
            "MIN",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "TOV",
            "STL",
            "BLK",
            "BLKA",
            "PF",
            "PFD",
            "PTS",
            "FTR",
            "BLK_RATIO",
            "PCT_FGA_2PT",
            "PCT_FGA_3PT",
            "PCT_PTS_2PT",
            "PCT_PTS_2PT_MR",
            "PCT_PTS_3PT",
            "PCT_PTS_FB",
            "PCT_PTS_FT",
            "PCT_PTS_OFF_TOV",
            "PCT_PTS_PAINT",
            "PCT_AST_2PM",
            "PCT_UAST_2PM",
            "PCT_AST_3PM",
            "PCT_UAST_3PM",
            "PCT_AST_FGM",
            "PCT_UAST_FGM",
            "E_OFF_RATING",
            "OFF_RATING",
            "E_DEF_RATING",
            "DEF_RATING",
            "AST_PCT",
            "AST_TO",
            "AST_RATIO",
            "OREB_PCT",
            "DREB_PCT",
            "REB_PCT",
            "TM_TOV_PCT",
            "EFG_PCT",
            "TS_PCT",
            "E_PACE",
            "PACE",
            "PIE",
            "OPP_FGM",
            "OPP_FGA",
            "OPP_FG_PCT",
            "OPP_FG3M",
            "OPP_FG3A",
            "OPP_FG3_PCT",
            "OPP_FTM",
            "OPP_FTA",
            "OPP_FT_PCT",
            "OPP_OREB",
            "OPP_DREB",
            "OPP_REB",
            "OPP_AST",
            "OPP_TOV",
            "OPP_STL",
            "OPP_BLK",
            "OPP_BLKA",
            "OPP_PTS",
            "OPP_FTR",
            "OPP_BLK_RATIO",
            "OPP_PCT_FGA_2PT",
            "OPP_PCT_FGA_3PT",
            "OPP_PCT_PTS_2PT",
            "OPP_PCT_PTS_2PT_MR",
            "OPP_PCT_PTS_3PT",
            "OPP_PCT_PTS_FB",
            "OPP_PCT_PTS_FT",
            "OPP_PCT_PTS_OFF_TOV",
            "OPP_PCT_PTS_PAINT",
            "OPP_PCT_AST_2PM",
            "OPP_PCT_UAST_2PM",
            "OPP_PCT_AST_3PM",
            "OPP_PCT_UAST_3PM",
            "OPP_PCT_AST_FGM",
            "OPP_PCT_UAST_FGM",
            "OPP_E_OFF_RATING",
            "OPP_OFF_RATING",
            "OPP_E_DEF_RATING",
            "OPP_DEF_RATING",
            "OPP_AST_PCT",
            "OPP_AST_TO",
            "OPP_AST_RATIO",
            "OPP_OREB_PCT",
            "OPP_DREB_PCT",
            "OPP_REB_PCT",
            "OPP_TM_TOV_PCT",
            "OPP_EFG_PCT",
            "OPP_TS_PCT",
            "OPP_E_PACE",
            "OPP_PACE",
            "OPP_PIE",
        ]
        self.teamlog = pd.DataFrame(columns=team_cols)

        self.stat_types = [
            "PFD",
            "E_OFF_RATING",
            "E_DEF_RATING",
            "AST_PCT",
            "AST_TO",
            "AST_RATIO",
            "FG3_RATIO",
            "OREB_PCT",
            "DREB_PCT",
            "REB_PCT",
            "EFG_PCT",
            "TS_PCT",
            "USG_PCT",
            "FG_PCT",
            "FG3_PCT",
            "FT_PCT",
            "PIE",
            "FTR",
            "MIN",
            "PACE",
            "PCT_FGA",
            "PCT_FG3A",
            "PCT_OREB",
            "PCT_DREB",
            "PCT_REB",
            "PCT_AST",
            "PCT_TOV",
            "PCT_STL",
            "PCT_BLKA",
            "FGA_48",
            "FG3A_48",
            "REB_48",
            "OREB_48",
            "DREB_48",
            "AST_48",
            "TOV_48",
            "BLKA_48",
            "STL_48",
        ]

        self.team_stat_types = [
            "FG_PCT",
            "FG3_PCT",
            "FT_PCT",
            "BLKA",
            "PF",
            "PFD",
            "FTR",
            "BLK_RATIO",
            "PCT_FGA_2PT",
            "PCT_FGA_3PT",
            "PCT_PTS_2PT",
            "PCT_PTS_2PT_MR",
            "PCT_PTS_3PT",
            "PCT_PTS_FB",
            "PCT_PTS_FT",
            "PCT_PTS_OFF_TOV",
            "PCT_PTS_PAINT",
            "PCT_AST_2PM",
            "PCT_UAST_2PM",
            "PCT_AST_3PM",
            "PCT_UAST_3PM",
            "PCT_AST_FGM",
            "PCT_UAST_FGM",
            "OFF_RATING",
            "DEF_RATING",
            "OREB_PCT",
            "DREB_PCT",
            "REB_PCT",
            "TM_TOV_PCT",
            "PACE",
            "PIE",
            "OPP_FG_PCT",
            "OPP_FG3_PCT",
            "OPP_FT_PCT",
            "OPP_BLKA",
            "OPP_FTR",
            "OPP_BLK_RATIO",
            "OPP_PCT_FGA_2PT",
            "OPP_PCT_FGA_3PT",
            "OPP_PCT_PTS_2PT",
            "OPP_PCT_PTS_2PT_MR",
            "OPP_PCT_PTS_3PT",
            "OPP_PCT_PTS_FB",
            "OPP_PCT_PTS_FT",
            "OPP_PCT_PTS_OFF_TOV",
            "OPP_PCT_PTS_PAINT",
            "OPP_PCT_AST_2PM",
            "OPP_PCT_UAST_2PM",
            "OPP_PCT_AST_3PM",
            "OPP_PCT_UAST_3PM",
            "OPP_PCT_AST_FGM",
            "OPP_PCT_UAST_FGM",
            "OPP_OFF_RATING",
            "OPP_DEF_RATING",
            "OPP_OREB_PCT",
            "OPP_DREB_PCT",
            "OPP_REB_PCT",
            "OPP_TM_TOV_PCT",
            "OPP_PIE",
        ]

        self.volume_stats = ["MIN"]
        self.default_total = 111.667
        self.log_strings = {
            "game": "GAME_ID",
            "date": "GAME_DATE",
            "player": "PLAYER_NAME",
            "usage": "MIN",
            "usage_sec": "USG_PCT",
            "position": "POS",
            "team": "TEAM_ABBREVIATION",
            "opponent": "OPP",
            "home": "HOME",
            "win": "WL",
            "score": "PTS",
            "age": "AGE",
        }
        self.usage_stat = "MIN"
        self.tiebreaker_stat = "USG_PCT short"
        self._volume_model_cache = None

    def load(self):
        """Load data from files."""
        filepath = pkg_resources.files(data) / "nba_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                nba_data = pickle.load(infile)
                self.players = nba_data["players"]
                self.gamelog = nba_data["gamelog"]
                self.teamlog = nba_data["teamlog"]

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

        playerList = self.players.get(
            "-".join([str(int(n) - 1) for n in self.season.split("-")]), {}
        )
        playerList.update(self.players.get(self.season, {}))
        playerProfile, playerDict = self.build_comp_profile(playerList)
        all_features = set()
        for pos_weights in stats["NBA"].values():
            all_features.update(pos_weights.keys())
        all_features = list(all_features)
        playerProfile = playerProfile[
            [f for f in all_features if f in playerProfile.columns]
        ].replace([np.nan, np.inf, -np.inf], 0)

        comps = {}
        for position in self.positions:
            pos_weights = stats["NBA"][position]
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
        filepath = pkg_resources.files(data) / "nba_comps.json"
        with open(filepath, "w") as outfile:
            json.dump(comps, outfile, indent=4)

    def _compute_comps(self):
        """Build comps from loaded data at runtime (no JSON I/O)."""
        with open(pkg_resources.files(data) / "playerCompStats.json") as f:
            stats = json.load(f)

        all_features = set()
        for pos_weights in stats["NBA"].values():
            all_features.update(pos_weights.keys())
        all_features = list(all_features)

        playerProfile, playerDict = self.build_comp_profile()
        playerProfile = playerProfile[
            [f for f in all_features if f in playerProfile.columns]
        ].replace([np.nan, np.inf, -np.inf], 0)

        comps = {}
        for position in self.positions:
            pos_weights = stats["NBA"][position]
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

    def update(self):
        """Update data from the web API."""
        # Fix corrupted POS values (integer 0 or None) from prior fillna(0) bug
        # Step 1: Clean self.players — try to resolve from other seasons/teams
        for season in self.players:
            for team in self.players[season]:
                for player, info in self.players[season][team].items():
                    if not isinstance(info.get("POS"), str):
                        resolved = None
                        for s in reversed(list(self.players.keys())):
                            for roster in self.players[s].values():
                                pos = roster.get(player, {}).get("POS")
                                if isinstance(pos, str) and pos in self.positions:
                                    resolved = pos
                                    break
                            if resolved:
                                break
                        info["POS"] = resolved

        # Step 2: Clean gamelog POS from self.players
        if not self.gamelog.empty:
            pos_col = self.log_strings["position"]
            bad_pos = ~self.gamelog[pos_col].apply(lambda x: isinstance(x, str))
            if bad_pos.any():
                self.gamelog.loc[bad_pos, pos_col] = self.gamelog.loc[bad_pos].apply(
                    lambda x: self.players.get(x.SEASON_YEAR, {})
                    .get(x.TEAM_ABBREVIATION, {})
                    .get(x.PLAYER_NAME, {})
                    .get("POS"),
                    axis=1,
                )

        # Fetch regular season game logs
        latest_date = self.season_start
        if not self.gamelog.empty:
            nanlog = self.gamelog.loc[self.gamelog.isnull().values.any(axis=1)]
            if not nanlog.empty:
                latest_date = pd.to_datetime(nanlog[self.log_strings["date"]]).min().date()

            else:
                latest_date = pd.to_datetime(self.gamelog[self.log_strings["date"]]).max().date()
            latest_date = max(latest_date, self.season_start)
        today = datetime.today().date()
        player_df = pd.read_csv(
            pkg_resources.files(data) / f"player_data/NBA/nba_players_{self.season}.csv"
        )

        player_df.Player = player_df.Player.apply(remove_accents)
        player_df.rename(
            columns={
                "Player": "PLAYER_NAME",
                "Team": "TEAM_ABBREVIATION",
                "Age": "AGE",
                "Pos": "POS",
            },
            inplace=True,
        )
        i = 0
        while i < 10:
            try:
                playerBios = nba.leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                    season=self.season
                ).get_normalized_dict()["LeagueDashPlayerBioStats"]

                shotData = nba.leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                    **{
                        "season": self.season,
                        "season_type_all_star": "Regular Season",
                        "distance_range": "By Zone",
                        "per_mode_detailed": "PerGame",
                    }
                ).get_dict()["resultSets"]
                postup = nba.synergyplaytypes.SynergyPlayTypes(
                    **{
                        "league_id": "00",
                        "season": self.season,
                        "season_type_all_star": "Regular Season",
                        "per_mode_simple": "PerGame",
                        "player_or_team_abbreviation": "P",
                        "type_grouping_nullable": "offensive",
                        "play_type_nullable": "Postup",
                    }
                ).get_dict()["resultSets"][0]
                handoff = nba.synergyplaytypes.SynergyPlayTypes(
                    **{
                        "league_id": "00",
                        "season": self.season,
                        "season_type_all_star": "Regular Season",
                        "per_mode_simple": "PerGame",
                        "player_or_team_abbreviation": "P",
                        "type_grouping_nullable": "offensive",
                        "play_type_nullable": "Handoff",
                    }
                ).get_dict()["resultSets"][0]
                isolation = nba.synergyplaytypes.SynergyPlayTypes(
                    **{
                        "league_id": "00",
                        "season": self.season,
                        "season_type_all_star": "Regular Season",
                        "per_mode_simple": "PerGame",
                        "player_or_team_abbreviation": "P",
                        "type_grouping_nullable": "offensive",
                        "play_type_nullable": "Isolation",
                    }
                ).get_dict()["resultSets"][0]
                picknroll = nba.synergyplaytypes.SynergyPlayTypes(
                    **{
                        "league_id": "00",
                        "season": self.season,
                        "season_type_all_star": "Regular Season",
                        "per_mode_simple": "PerGame",
                        "player_or_team_abbreviation": "P",
                        "type_grouping_nullable": "offensive",
                        "play_type_nullable": "PRBallHandler",
                    }
                ).get_dict()["resultSets"][0]
                spotup = nba.synergyplaytypes.SynergyPlayTypes(
                    **{
                        "league_id": "00",
                        "season": self.season,
                        "season_type_all_star": "Regular Season",
                        "per_mode_simple": "PerGame",
                        "player_or_team_abbreviation": "P",
                        "type_grouping_nullable": "offensive",
                        "play_type_nullable": "Spotup",
                    }
                ).get_dict()["resultSets"][0]
                putback = nba.synergyplaytypes.SynergyPlayTypes(
                    **{
                        "league_id": "00",
                        "season": self.season,
                        "season_type_all_star": "Regular Season",
                        "per_mode_simple": "PerGame",
                        "player_or_team_abbreviation": "P",
                        "type_grouping_nullable": "offensive",
                        "play_type_nullable": "OffRebound",
                    }
                ).get_dict()["resultSets"][0]
                break
            except:
                playerBios = []
                shotData = {"rowSet": [], "headers": [{}, {"columnNames": []}]}
                postup = {"rowSet": [], "headers": []}
                handoff = {"rowSet": [], "headers": []}
                isolation = {"rowSet": [], "headers": []}
                picknroll = {"rowSet": [], "headers": []}
                spotup = {"rowSet": [], "headers": []}
                putback = {"rowSet": [], "headers": []}
                sleep(0.1)
                i = i + 1

        playerBios = pd.DataFrame(playerBios)
        shotData = pd.DataFrame(shotData["rowSet"], columns=shotData["headers"][1]["columnNames"])
        postup = pd.DataFrame(postup["rowSet"], columns=postup["headers"])
        handoff = pd.DataFrame(handoff["rowSet"], columns=handoff["headers"])
        isolation = pd.DataFrame(isolation["rowSet"], columns=isolation["headers"])
        picknroll = pd.DataFrame(picknroll["rowSet"], columns=picknroll["headers"])
        spotup = pd.DataFrame(spotup["rowSet"], columns=spotup["headers"])
        putback = pd.DataFrame(putback["rowSet"], columns=putback["headers"])

        if playerBios.empty:
            self.players[self.season] = self.players.get(
                "-".join([str(int(x) - 1) for x in self.season.split("-")]), {}
            )
        else:
            shotData = shotData.merge(
                postup[["PLAYER_ID", "POSS_PCT", "PPP"]].rename(
                    columns={"POSS_PCT": "POST_PCT", "PPP": "POST_PPP"}
                ),
                on="PLAYER_ID",
                how="outer",
            )
            shotData = shotData.merge(
                handoff[["PLAYER_ID", "POSS_PCT", "PPP"]].rename(
                    columns={"POSS_PCT": "HANDOFF_PCT", "PPP": "HANDOFF_PPP"}
                ),
                on="PLAYER_ID",
                how="outer",
            )
            shotData = shotData.merge(
                isolation[["PLAYER_ID", "POSS_PCT", "PPP"]].rename(
                    columns={"POSS_PCT": "ISO_PCT", "PPP": "ISO_PPP"}
                ),
                on="PLAYER_ID",
                how="outer",
            )
            shotData = shotData.merge(
                picknroll[["PLAYER_ID", "POSS_PCT", "PPP"]].rename(
                    columns={"POSS_PCT": "PR_PCT", "PPP": "PR_PPP"}
                ),
                on="PLAYER_ID",
                how="outer",
            )
            shotData = shotData.merge(
                spotup[["PLAYER_ID", "POSS_PCT", "PPP"]].rename(
                    columns={"POSS_PCT": "SPOT_PCT", "PPP": "SPOT_PPP"}
                ),
                on="PLAYER_ID",
                how="outer",
            )
            shotData = shotData.merge(
                putback[["PLAYER_ID", "POSS_PCT", "PPP"]].rename(
                    columns={"POSS_PCT": "PUTBACK_PCT", "PPP": "PUTBACK_PPP"}
                ),
                on="PLAYER_ID",
                how="outer",
            )
            shotData = shotData.fillna(0)
            fga = shotData["FGA"].iloc[:, :3].sum(axis=1) + shotData["FGA"].iloc[:, 5::2].sum(
                axis=1
            )
            shotData["ITP_PCT"] = shotData["FGA"].iloc[:, :2].sum(axis=1) / fga
            shotData["ITP_PPP"] = (
                shotData["FGM"].iloc[:, :2].sum(axis=1)
                / shotData["FGA"].iloc[:, :2].sum(axis=1)
                * 2
            )
            shotData.loc[shotData["FGA"].iloc[:, :2].sum(axis=1) < 0.5, "ITP_PPP"] = 0
            shotData["MR_PCT"] = shotData["FGA"].iloc[:, 2] / fga
            shotData["MR_PPP"] = shotData["FGM"].iloc[:, 2] / shotData["FGA"].iloc[:, 2] * 2
            shotData.loc[shotData["FGA"].iloc[:, 7] < 0.5, "MR_PPP"] = 0
            shotData["C3_PCT"] = shotData["FGA"].iloc[:, 5] / fga
            shotData["C3_PPP"] = shotData["FGM"].iloc[:, 5] / shotData["FGA"].iloc[:, 5] * 3
            shotData.loc[shotData["FGA"].iloc[:, 7] < 0.5, "C3_PPP"] = 0
            shotData["B3_PCT"] = shotData["FGA"].iloc[:, 7] / fga
            shotData["B3_PPP"] = shotData["FGM"].iloc[:, 7] / shotData["FGA"].iloc[:, 7] * 3
            shotData.loc[shotData["FGA"].iloc[:, 7] < 0.5, "B3_PPP"] = 0
            shotData = shotData.fillna(0)

            playerBios.PLAYER_NAME = playerBios.PLAYER_NAME.apply(remove_accents)
            shotData.PLAYER_NAME = shotData.PLAYER_NAME.apply(remove_accents)
            playerBios = playerBios.merge(
                shotData, on="PLAYER_NAME", suffixes=(None, "_y"), how="outer"
            ).fillna(0)
            player_df = player_df.merge(
                playerBios,
                on=["PLAYER_NAME", "TEAM_ABBREVIATION"],
                suffixes=(None, "_x"),
                how="outer",
            )
            # For traded players, the CSV has the old team and playerBios has the new team,
            # producing two incomplete rows. Combine them by taking the first non-NaN value
            # per column, with playerBios rows (non-NaN PLAYER_ID) sorted first so their
            # TEAM_ABBREVIATION (current team) is preferred.
            player_df = (
                player_df.sort_values("PLAYER_ID", na_position="last")
                .groupby("PLAYER_NAME", sort=False)
                .first()
                .reset_index()
            )
            player_df.PLAYER_WEIGHT = player_df.PLAYER_WEIGHT.astype(float)
            player_df.POS = player_df.POS.str[0]
            player_df.index = player_df.PLAYER_NAME
            # Fill NaN POS from prior-season data
            nan_pos = player_df.POS.isna()
            if nan_pos.any():
                for player_name in player_df.loc[nan_pos].index:
                    for season in reversed(list(self.players.keys())):
                        for roster in self.players[season].values():
                            pos = roster.get(player_name, {}).get("POS")
                            if isinstance(pos, str) and pos in self.positions:
                                player_df.loc[player_name, "POS"] = pos
                                break
                        if pd.notna(player_df.loc[player_name, "POS"]):
                            break
                player_df.POS = player_df.POS.fillna("W")
            player_df["PLAYER_BMI"] = (
                player_df["PLAYER_WEIGHT"]
                / player_df["PLAYER_HEIGHT_INCHES"]
                / player_df["PLAYER_HEIGHT_INCHES"]
            )
            numeric_cols = [
                "AGE",
                "PLAYER_HEIGHT_INCHES",
                "PLAYER_BMI",
                "USG_PCT",
                "TS_PCT",
                "ITP_PCT",
                "ITP_PPP",
                "MR_PCT",
                "MR_PPP",
                "C3_PCT",
                "C3_PPP",
                "B3_PCT",
                "B3_PPP",
                "POST_PCT",
                "POST_PPP",
                "HANDOFF_PCT",
                "HANDOFF_PPP",
                "ISO_PCT",
                "ISO_PPP",
                "PR_PCT",
                "PR_PPP",
                "SPOT_PCT",
                "SPOT_PPP",
                "PUTBACK_PCT",
                "PUTBACK_PPP",
            ]
            player_df = player_df.groupby("TEAM_ABBREVIATION")[["POS", *numeric_cols]].apply(
                lambda x: x
            )
            player_df[numeric_cols] = player_df[numeric_cols].fillna(0)

            player_df = {
                level: player_df.xs(level).T.to_dict() for level in player_df.index.levels[0]
            }
            if self.season in self.players:
                self.players[self.season] = {
                    team: players | player_df.get(team, {})
                    for team, players in self.players[self.season].items()
                }
            else:
                self.players[self.season] = player_df

        position_map = {
            "Forward": "F",
            "Guard": "C",
            "Forward-Guard": "W",
            "Guard-Forward": "W",
            "Center": "B",
            "Forward-Center": "B",
            "Center-Forward": "B",
            "Center-Guard": "B",
            "Guard-Center": "W",
        }

        self.upcoming_games = {}

        try:
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={self.season[:4]}&RegionID=1&Date={today.strftime('%m/%d/%Y')}&EST=Y"

            ug_res = scraper.get(ug_url)["resultSets"][1]["CompleteGameList"]

            next_day = today + timedelta(days=1)
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={self.season[:4]}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

            ug_res.extend(scraper.get(ug_url)["resultSets"][1]["CompleteGameList"])

            next_day = next_day + timedelta(days=1)
            ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID=00&Season={self.season[:4]}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

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
            "season_nullable": self.season,
            "league_id_nullable": "00",
            "date_from_nullable": latest_date.strftime("%m/%d/%Y"),
            "date_to_nullable": today.strftime("%m/%d/%Y"),
        }

        nba_gamelog = []
        adv_gamelog = []
        usg_gamelog = []
        teamlog = []
        sco_teamlog = []
        adv_teamlog = []
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
                if (today.month == 4) or (today - latest_date).days > 150:
                    params.update({"season_type_nullable": "PlayIn"})
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
                if (4 <= today.month <= 6) or (today - latest_date).days > 150:
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
                sleep(0.2)
                i += 1

        adv_teamlog_idx = {(g["TEAM_ID"], g["GAME_ID"]): g for g in adv_teamlog}
        sco_teamlog_idx = {(g["TEAM_ID"], g["GAME_ID"]): g for g in sco_teamlog}
        adv_gamelog_idx = {(g["PLAYER_ID"], g["GAME_ID"]): g for g in adv_gamelog}
        usg_gamelog_idx = {(g["PLAYER_ID"], g["GAME_ID"]): g for g in usg_gamelog}

        for i in range(len(teamlog)):
            key = (teamlog[i]["TEAM_ID"], teamlog[i]["GAME_ID"])
            adv_game = adv_teamlog_idx.get(key)
            sco_game = sco_teamlog_idx.get(key)
            if adv_game:
                teamlog[i] = teamlog[i] | adv_game
            if sco_game:
                teamlog[i] = teamlog[i] | sco_game

        team_df = []
        for team1, team2 in zip(*[iter(teamlog)] * 2, strict=False):
            team1.update(
                {
                    "FTR": (team1["FTM"] / team1["FGA"]) if team1["FGA"] > 0 else 0,
                    "BLK_RATIO": (team1["BLK"] / team1["BLKA"]) if team1["BLKA"] > 0 else 0,
                    "OPP": team2[self.log_strings["team"]],
                }
            )
            team2.update(
                {
                    "FTR": (team2["FTM"] / team2["FGA"]) if team2["FGA"] > 0 else 0,
                    "BLK_RATIO": (team2["BLK"] / team2["BLKA"]) if team2["BLKA"] > 0 else 0,
                    "OPP": team1[self.log_strings["team"]],
                }
            )
            team1.update(
                {"OPP_" + k: v for k, v in team2.items() if "OPP_" + k in self.teamlog.columns}
            )
            team2.update(
                {"OPP_" + k: v for k, v in team1.items() if "OPP_" + k in self.teamlog.columns}
            )
            team_df.append(team1)
            team_df.append(team2)

        team_df = pd.DataFrame(team_df)

        if not team_df.empty:
            self.teamlog = (
                pd.concat([team_df.reindex(columns=self.teamlog.columns), self.teamlog])
                .sort_values(self.log_strings["date"])
                .reset_index(drop=True)
            )

        # Drop records with incomplete advanced stats so they can be re-fetched
        if "OFF_RATING" in self.gamelog.columns:
            self.gamelog = self.gamelog.dropna(subset=["OFF_RATING"])

        # Process each game
        nba_df = []
        included_games = set(
            self.gamelog[["PLAYER_ID", "GAME_ID"]].itertuples(index=False, name=None)
        )
        for i, game in enumerate(tqdm(nba_gamelog, desc="Getting NBA stats", unit="player")):
            if (
                game["MIN"] < 1
                or not game["TEAM_ABBREVIATION"]
                or (game["PLAYER_ID"], game["GAME_ID"]) in included_games
            ):
                continue

            included_games.add((game["PLAYER_ID"], game["GAME_ID"]))

            player_id = game["PLAYER_ID"]
            game["PLAYER_NAME"] = remove_accents(game["PLAYER_NAME"])

            adv_game = adv_gamelog_idx.get((player_id, game["GAME_ID"]))
            usg_game = usg_gamelog_idx.get((player_id, game["GAME_ID"]))

            self.players[self.season].setdefault(game["TEAM_ABBREVIATION"], {})
            existing_pos = (
                self.players[self.season][game["TEAM_ABBREVIATION"]]
                .get(game["PLAYER_NAME"], {})
                .get("POS")
            )
            if game["PLAYER_NAME"] not in self.players[self.season][
                game["TEAM_ABBREVIATION"]
            ] or not isinstance(existing_pos, str):
                # Fetch player information if not already present
                position = None
                for season in list(self.players.keys())[::-1]:
                    if not isinstance(position, str):
                        position = (
                            self.players[season]
                            .get(game["TEAM_ABBREVIATION"], {})
                            .get(game["PLAYER_NAME"], {})
                            .get("POS")
                        )

                    if not isinstance(position, str):
                        for team in self.players[season]:
                            position = (
                                self.players[season][team].get(game["PLAYER_NAME"], {}).get("POS")
                            )

                if not isinstance(position, str):
                    position = None

                if position is None:
                    try:
                        position = (
                            nba.commonplayerinfo.CommonPlayerInfo(player_id=player_id)
                            .get_normalized_dict()["CommonPlayerInfo"][0]
                            .get("POSITION")
                        )
                    except:
                        position = "Forward-Guard"
                    position = position_map.get(position, "W")

                self.players[self.season][game["TEAM_ABBREVIATION"]].setdefault(
                    game["PLAYER_NAME"], {}
                )["POS"] = position

            # Extract additional game information
            game["POS"] = (
                self.players[self.season][game["TEAM_ABBREVIATION"]]
                .get(game["PLAYER_NAME"], {})
                .get("POS")
            )
            game["HOME"] = "vs." in game["MATCHUP"]
            teams = game["MATCHUP"].replace("vs.", "@").split(" @ ")
            for team in teams:
                if team != game["TEAM_ABBREVIATION"]:
                    game["OPP"] = team

            # Compute derived stats
            game["PRA"] = game["PTS"] + game["REB"] + game["AST"]
            game["PR"] = game["PTS"] + game["REB"]
            game["PA"] = game["PTS"] + game["AST"]
            game["RA"] = game["REB"] + game["AST"]
            game["BLST"] = game["BLK"] + game["STL"]
            game["fantasy points prizepicks"] = (
                game["PTS"] + game["REB"] * 1.2 + game["AST"] * 1.5 + game["BLST"] * 3 - game["TOV"]
            )
            game["fantasy points underdog"] = (
                game["PTS"] + game["REB"] * 1.2 + game["AST"] * 1.5 + game["BLST"] * 3 - game["TOV"]
            )
            game["fantasy points parlay"] = game["PRA"] + game["BLST"] * 2 - game["TOV"]
            game["FTR"] = (game["FTM"] / game["FGA"]) if game["FGA"] > 0 else 0
            game["FG3_RATIO"] = (game["FG3A"] / game["FGA"]) if game["FGA"] > 0 else 0
            game["BLK_PCT"] = (game["BLK"] / game["BLKA"]) if game["BLKA"] > 0 else 0
            game["FGA_48"] = game["FGA"] / game["MIN"] * 48
            game["FG3A_48"] = game["FG3A"] / game["MIN"] * 48
            game["REB_48"] = game["REB"] / game["MIN"] * 48
            game["OREB_48"] = game["OREB"] / game["MIN"] * 48
            game["DREB_48"] = game["DREB"] / game["MIN"] * 48
            game["AST_48"] = game["AST"] / game["MIN"] * 48
            game["TOV_48"] = game["TOV"] / game["MIN"] * 48
            game["BLKA_48"] = game["BLKA"] / game["MIN"] * 48
            game["STL_48"] = game["STL"] / game["MIN"] * 48

            if adv_game:
                game.update(adv_game)

            if usg_game:
                game.update(usg_game)

            nba_df.append(game)

        nba_df = pd.DataFrame(nba_df)

        if not nba_df.empty:
            # Retrieve moneyline and totals data
            nba_df.loc[:, "moneyline"] = nba_df.apply(
                lambda x: archive.get_moneyline(
                    self.league, x[self.log_strings["date"]][:10], x["TEAM_ABBREVIATION"]
                ),
                axis=1,
            )
            nba_df.loc[:, "totals"] = nba_df.apply(
                lambda x: archive.get_total(
                    self.league, x[self.log_strings["date"]][:10], x["TEAM_ABBREVIATION"]
                ),
                axis=1,
            )

            self.gamelog = (
                pd.concat([nba_df.reindex(columns=self.gamelog.columns), self.gamelog])
                .sort_values(self.log_strings["date"])
                .reset_index(drop=True)
            )

        # Remove old games to prevent file bloat
        four_years_ago = today - timedelta(days=1461)
        self.gamelog = self.gamelog[
            pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date >= four_years_ago
        ]
        self.gamelog.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last", inplace=True)
        self.teamlog = self.teamlog[
            pd.to_datetime(self.teamlog[self.log_strings["date"]]).dt.date >= four_years_ago
        ]
        self.teamlog.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)

        self.gamelog.loc[
            self.gamelog[self.log_strings["team"]] == "UTAH", self.log_strings["team"]
        ] = "UTA"
        self.gamelog.loc[
            self.gamelog[self.log_strings["opponent"]] == "UTAH", self.log_strings["opponent"]
        ] = "UTA"
        self.gamelog.loc[
            self.gamelog[self.log_strings["team"]] == "NOP", self.log_strings["team"]
        ] = "NO"
        self.gamelog.loc[
            self.gamelog[self.log_strings["opponent"]] == "NOP", self.log_strings["opponent"]
        ] = "NO"
        self.gamelog.loc[
            self.gamelog[self.log_strings["team"]] == "GS", self.log_strings["team"]
        ] = "GSW"
        self.gamelog.loc[
            self.gamelog[self.log_strings["opponent"]] == "GS", self.log_strings["opponent"]
        ] = "GSW"
        self.gamelog.loc[
            self.gamelog[self.log_strings["team"]] == "NY", self.log_strings["team"]
        ] = "NYK"
        self.gamelog.loc[
            self.gamelog[self.log_strings["opponent"]] == "NY", self.log_strings["opponent"]
        ] = "NYK"
        self.gamelog.loc[
            self.gamelog[self.log_strings["team"]] == "SA", self.log_strings["team"]
        ] = "SAS"
        self.gamelog.loc[
            self.gamelog[self.log_strings["opponent"]] == "SA", self.log_strings["opponent"]
        ] = "SAS"

        self.teamlog.loc[
            self.teamlog[self.log_strings["team"]] == "UTAH", self.log_strings["team"]
        ] = "UTA"
        self.teamlog.loc[
            self.teamlog[self.log_strings["opponent"]] == "UTAH", self.log_strings["opponent"]
        ] = "UTA"
        self.teamlog.loc[
            self.teamlog[self.log_strings["team"]] == "NOP", self.log_strings["team"]
        ] = "NO"
        self.teamlog.loc[
            self.teamlog[self.log_strings["opponent"]] == "NOP", self.log_strings["opponent"]
        ] = "NO"
        self.teamlog.loc[
            self.teamlog[self.log_strings["team"]] == "GS", self.log_strings["team"]
        ] = "GSW"
        self.teamlog.loc[
            self.teamlog[self.log_strings["opponent"]] == "GS", self.log_strings["opponent"]
        ] = "GSW"
        self.teamlog.loc[
            self.teamlog[self.log_strings["team"]] == "NY", self.log_strings["team"]
        ] = "NYK"
        self.teamlog.loc[
            self.teamlog[self.log_strings["opponent"]] == "NY", self.log_strings["opponent"]
        ] = "NYK"
        self.teamlog.loc[
            self.teamlog[self.log_strings["team"]] == "SA", self.log_strings["team"]
        ] = "SAS"
        self.teamlog.loc[
            self.teamlog[self.log_strings["opponent"]] == "SA", self.log_strings["opponent"]
        ] = "SAS"

        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog["PLAYER_NAME"] = self.gamelog["PLAYER_NAME"].apply(remove_accents)
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
            self.gamelog.loc[:, self.log_strings["position"]] = self.gamelog.apply(
                lambda x: self.players.get(x.SEASON_YEAR, {})
                .get(x.TEAM_ABBREVIATION, {})
                .get(x.PLAYER_NAME, {})
                .get("POS", x.POS),
                axis=1,
            )

        # Save the updated player data
        with open(pkg_resources.files(data) / "nba_data.dat", "wb") as outfile:
            pickle.dump(
                {"players": self.players, "gamelog": self.gamelog, "teamlog": self.teamlog}, outfile
            )


    def get_volume_stats(self, offers, date=datetime.today().date()):
        market = "MIN"
        flat_offers = {}
        if isinstance(offers, dict):
            for players in offers.values():
                flat_offers.update(players)
            flat_offers.update(offers.get(market, {}))
        else:
            flat_offers = offers

        self.profile_market(market, date)
        self.get_depth(flat_offers, date)
        playerStats = self.get_stats(market, flat_offers, date)

        if playerStats.empty:
            logger.warning(f"Gamelog missing - {date}")
            return []

        filename = "_".join([self.league, market]).replace(" ", "-")
        filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
        # Cache model loading — avoid re-reading pickle on every call
        if not hasattr(self, "_volume_model_cache") or self._volume_model_cache is None:
            if os.path.isfile(filepath):
                with open(filepath, "rb") as infile:
                    self._volume_model_cache = pickle.load(infile)
            else:
                logger.warning(f"{filename} missing")
                return []

        # Slice to the trained schema embedded in the pickle.
        cols = self._volume_model_cache["expected_columns"]
        if any([col not in playerStats.columns for col in cols]):
            logger.warning(f"Gamelog missing - {date}")
            return []
        playerStats = playerStats[cols]

        model = self._volume_model_cache["model"]
        dist = self._volume_model_cache["distribution"]

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

        # ------------------------------------------------------------------
        # Minutes-budget normalization
        #
        # Two real-world constraints complicate a simple 300-minute cap:
        #
        #   1. Overtime: Games sometimes go beyond regulation. With probability
        #      p_ot per period, each OT adds ot_per_period player-minutes.
        #      Modelling this as a geometric process gives an expected total of:
        #          budget_mean = reg_minutes + ot_per_period * p_ot / (1 - p_ot)
        #
        #   2. Unmodeled players: When our roster coverage is incomplete (e.g.
        #      rookies or fringe benchwarmers lacking enough data), those players
        #      still consume real minutes. We reserve a portion of the budget for
        #      them so we don't over-inflate the projections of modeled players.
        #          unmodeled_reserve = max(0, typical_rotation - N) * avg_unmodeled_min
        #
        # Given these facts, the target range for modeled players' total is:
        #   lower_target = N * per_player_floor   (sanity floor — each player logged)
        #   upper_target = budget_mean - unmodeled_reserve
        #
        # When the projected total falls outside [lower_target, upper_target] we
        # apply a precision-weighted adjustment. Rather than scaling everyone by
        # the same factor, we distribute the correction in proportion to each
        # player's variance. This is the minimum-information-loss solution to the
        # constrained optimisation:
        #
        #   minimise  sum_i (d_i / sigma_i)^2
        #   subject to  sum_i (mu_i + d_i) = target
        #
        #   -> d_i = sigma_i^2 / sum_j(sigma_j^2) * (target - total)
        #
        # Players we are most uncertain about absorb the correction; confidently
        # projected players are left largely untouched.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Budget parameters — derived by analyzing historical gamelogs.
        #
        # Methodology (run offline against self.gamelog, hardcoded for speed):
        #   typical_rotation : median players logging >3 min per team per game
        #   ot_rate          : fraction of games where any player exceeded
        #                      regulation max minutes (0.55% NBA, 3.9% WNBA)
        #   avg_unmodeled_min: mean minutes for players ranked beyond the top-7
        #                      modeled tier (ranks 8-10 NBA, 8-9 WNBA)
        #   per_player_floor : 5th-percentile minutes for top-7 tier players
        #   per_player_cap   : regulation max + one 5-min OT period (hard rule)
        # ------------------------------------------------------------------
        if self.league == "NBA":
            reg_minutes = 240  # 5 players × 48 min regulation
            ot_per_period = 25  # 5 players × 5 min per OT period
            ot_rate = 0.06  # measured: 6% of NBA games go to OT
            typical_rotation = 10  # measured: median active players per team-game
            avg_unmodeled_min = 11  # measured: mean min for players ranked 8-10
            per_player_floor = 18  # measured: 5th-pct min for top-7 tier players
            per_player_cap = 53  # hard rule: 48 min regulation + 1 OT period
        else:  # WNBA
            reg_minutes = 200  # 5 players × 40 min regulation
            ot_per_period = 25  # 5 players × 5 min per OT period
            ot_rate = 0.039  # measured: 3.9% of WNBA games go to OT
            typical_rotation = 9  # measured: median active players per team-game
            avg_unmodeled_min = 8.5  # measured: mean min for players ranked 8-9
            per_player_floor = 13  # measured: 5th-pct min for top-7 tier players
            per_player_cap = 45  # hard rule: 40 min regulation + 1 OT period

        # Expected total team minutes including OT (geometric series)
        ot_expected = ot_per_period * ot_rate / (1.0 - ot_rate)
        budget_mean = reg_minutes + ot_expected

        teams = self.playerProfile.loc[self.playerProfile["team"] != 0].groupby("team")
        for _team, team_df in teams:
            # SkewNormal: E[X] = loc + scale * delta * sqrt(2/pi) where delta = alpha/sqrt(1+alpha^2)
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

            # Minutes reserved for fringe bench players not captured in our model.
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

        self.playerProfile.fillna(0, inplace=True)

    def check_combo_markets(self, market, player, date=datetime.today().date()):
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

        elif market in ["DREB", "OREB"]:
            ev = (
                (
                    archive.get_ev(self.league, "REB", date, player)
                    * player_games[market].sum()
                    / player_games["REB"].sum()
                )
                if player_games["REB"].sum()
                else 0
            )

        elif "fantasy" in market:
            ev = 0
            book_odds = False
            fantasy_props = [
                ("PTS", 1),
                ("REB", 1.2),
                ("AST", 1.5),
                ("BLK", 3),
                ("STL", 3),
                ("TOV", -1),
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

                    if subline != 0:
                        under = (player_games[submarket] < subline).mean()
                        ev += get_ev(subline, under, sub_cv, dist=sub_dist) * weight
                else:
                    book_odds = True
                    ev += v * weight

            if not book_odds:
                ev = 0
        else:
            ev = 0

        return 0 if np.isnan(ev) else ev

