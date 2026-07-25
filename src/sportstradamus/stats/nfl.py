"""StatsNFL: NFL player stats loading, feature engineering, and prediction."""

import importlib.resources as pkg_resources
import json
import os.path
import pickle
from datetime import date, datetime, timedelta

import nfl_data_py as nfl
import nflreadpy as nflr
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import (
    remove_accents,
    set_model_start_values,
)
from sportstradamus.helpers.io import read_gamelog, write_gamelog
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import (
    nfl_fp_loader,
    nfl_fp_team_weekly,
    nfl_fp_team_weekly_aggregate,
    nfl_fp_weekly_aggregate,
)
from sportstradamus.stats.base import (
    _VOLUME_SCALE_RATIO_CAP,
    _VOLUME_SCALE_RATIO_FLOOR,
    Stats,
    clean_data,
)

# Phase-1.5 lookback rule:
#   - Post-week-4 (target_week >= 5) of the target's season: comp pool is
#     restricted to current-season weeks 1..target_week-1 (no prior season).
#     ~4 in-season games are more predictive of the next game than last
#     year's full-season number for the same player.
#   - Pre-week-5 (target_week 1..4): not enough current-season sample to
#     stand alone, so the prior season's BACK HALF (weeks 11..18 -- last
#     8 weeks of the regular season) is blended with any current-season
#     games already played. The back-half-only choice trims early-season
#     noise from the prior year on the assumption that late-season form
#     is more representative of the player's current talent level.
# These thresholds encode the operator-confirmed rule; the named constants
# document the policy rather than burying it as a magic literal inside
# build_comp_profile / _lookback_windows.
COMP_LOOKBACK_CURRENT_SEASON_ONLY_WEEK = 5
COMP_LOOKBACK_PRIOR_SEASON_FROM_WEEK = 11

# Default NFL regular-season length in weeks; the season-CSV path used 17
# (pre-2021) / 18 (2021+). Phase-1.5 always reads the per-game weekly
# snapshots, so 18 is the right ceiling for the comp lookback window.
NFL_REGULAR_SEASON_WEEKS = 18

# Number of prior NFL seasons retained in the legacy (None-date) fallback
# path. Only used when ``build_comp_profile`` is called without a
# ``target_game_date`` -- the post-Phase-1.5 production path uses the
# back-half rule above instead.
COMP_PRIOR_SEASON_DEPTH = 3

# Positions that actually accrue each market stat. Rows for other positions are
# all-zero (or wrong-population) noise that depresses MeanYr and inflates the zero
# fraction (Stage A1.6; passing-yards global mean read 38 vs 216 QB-only). Only the
# fantasy-points composites are absent (genuinely all-position — every player has a
# fantasy score). "qb yards"/"qb tds" are QB total-offense markets; "yards"/"tds"
# are skill scrimmage markets.
NFL_MARKET_POSITIONS = {
    # Passing + QB total offense — QB only
    "passing yards": {"QB"},
    "attempts": {"QB"},  # passing attempts
    "completions": {"QB"},
    "passing tds": {"QB"},
    "passing first downs": {"QB"},
    "interceptions": {"QB"},
    "sacks taken": {"QB"},
    "qb yards": {"QB"},  # pass+rush yards — a QB market
    "qb tds": {"QB"},  # pass+rush TDs — a QB market
    # Rushing — QB + RB (QBs scramble; excluding them skews scramble games)
    "rushing yards": {"QB", "RB"},
    "carries": {"QB", "RB"},
    "rushing tds": {"QB", "RB"},
    # Receiving + skill scrimmage — WR + RB + TE (no QB)
    "receiving yards": {"WR", "RB", "TE"},
    "targets": {"WR", "RB", "TE"},
    "receptions": {"WR", "RB", "TE"},
    "receiving tds": {"WR", "RB", "TE"},
    "yards": {"WR", "RB", "TE"},  # rush+rec scrimmage yards
    "tds": {"WR", "RB", "TE"},  # rush+rec TDs
}

# nfl-data-py delivers the advanced-stat columns (yards per carry, passer
# rating, the target-share metrics, ...) as object-dtype numeric strings. Every
# gamelog column EXCEPT these identifier / categorical / boolean ones is coerced
# to numeric on load and update (see StatsNFL._coerce_numeric_gamelog); leaving
# them as strings crashes base_profile's tail(5).mean() short-window aggregate.
_NON_NUMERIC_GAMELOG_COLS = frozenset(
    {
        "player id",
        "player display name",
        "position",
        "position group",
        "team",
        "season type",
        "opponent",
        "gameday",
        "game id",
        "home",
    }
)

# Volume-normalization budget constants used in get_volume_stats.
# Values measured from historical gamelog distributions (mean carries /
# targets for depth-chart players not individually modeled).
_UNMODELED_CARRY_RESERVE: int = 6  # mean carries for rank-3+ rushers per game
_UNMODELED_TARGET_RESERVE: int = 8  # mean targets for rank-5+ receivers per game
_CARRY_CAP: int = 40  # absolute single-player carry ceiling per game
_TARGET_CAP: int = 20  # absolute single-player target ceiling per game

# Gamelog lookback window for data hygiene: drop rows older than this many days
# (~6 seasons). Keeping more than 6 seasons of NFL data inflates storage and
# includes pre-analytics-era play-by-play that is noisier than modern data.
_GAMELOG_RETENTION_DAYS: int = 2191  # 6 * 365 + 1 leap day

# KNN comp-pool filtering thresholds. The percentile gate removes very
# low-usage players who would pollute the nearest-neighbor set with
# small-sample noise. The min-players guard avoids fitting a BallTree
# with too few points to produce meaningful comps.
_COMP_POOL_MIN_USAGE_PERCENTILE: float = 0.25  # bottom quartile filtered out
_COMP_POOL_MIN_PLAYERS: int = 7  # skip position if fewer candidates survive filter

# BallTree KNN comp count bounds. min_comps prevents degenerate 1-comp
# matches; max_comps caps the neighborhood to avoid averaging over
# players who are barely similar.
_COMP_KNN_MIN: int = 5
_COMP_KNN_MAX: int = 15

# Per-market position-number filters for load_volume_model_params.
# Numbers correspond to self.positions = ["QB","WR","RB","TE"] (index+1):
#   QB=1, WR=2, RB=3, TE=4. Must mirror NFL_MARKET_POSITIONS (name→number).
_VOLUME_MARKET_POSITION_FILTER: dict[str, list[int]] = {
    "attempts": [1],  # QB only
    "carries": [1, 3],  # QB + RB (QBs scramble)
    "targets": [2, 3, 4],  # WR + RB + TE
}


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

    def _market_position_filter(self, gamelog: pd.DataFrame, market: str) -> pd.DataFrame:
        """Restrict the training gamelog to positions that accrue ``market``'s stat.

        Overrides the base no-op hook. Markets absent from ``NFL_MARKET_POSITIONS``
        (the fantasy-points composites) keep all four positions. See Stage A1.6 in
        docs/gbdt_mean_regression_plan.md.
        """
        eligible = NFL_MARKET_POSITIONS.get(market)
        if eligible is None:
            return gamelog
        return gamelog.loc[gamelog[self.log_strings["position"]].isin(eligible)]

    def _coerce_numeric_gamelog(self):
        """Coerce numeric-as-string gamelog columns to numeric dtype in place.

        The nfl-data-py advanced-stat columns arrive as object-dtype numeric
        strings. Uncoerced, they survive base_profile's numeric_only=True
        aggregate by being silently dropped, but crash the short-window
        tail(5).mean() aggregate (no numeric_only argument available). Coerce
        every column except the known identifier/categorical/boolean columns;
        unparseable values become NaN (downstream code fills them with 0).
        """
        numeric_cols = [c for c in self.gamelog.columns if c not in _NON_NUMERIC_GAMELOG_COLS]
        self.gamelog[numeric_cols] = self.gamelog[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

    def load(self):
        """Read the NFL gamelog bundle and apply NFL-specific cleanup.

        Differs from the base :meth:`Stats.load` in two ways:

        * The ``players`` payload is only assigned when non-empty, so a
          stale or corrupt read does not blow away a previously populated
          roster snapshot.
        * The gamelog object-dtype numeric columns are coerced to real
          numeric dtypes before downstream feature engineering.
        """
        nfl_data = read_gamelog("nfl")
        self.gamelog = nfl_data["gamelog"]
        self.teamlog = nfl_data["teamlog"]
        players = nfl_data["players"]
        if (isinstance(players, pd.DataFrame) and not players.empty) or (
            isinstance(players, dict) and players
        ):
            self.players = players
        self._coerce_numeric_gamelog()

    def _update(self):
        """Update data from the web API."""
        self.need_pbp = True
        # QW-4 gametime / own rest / opp rest were reverted 2026-06-20 (the
        # PrimeTime/RestDiff features they fed carried zero importance). Shed them
        # from any pre-revert cache so the ``isna().any`` backfill selector below
        # doesn't flag every historical row -- they were NaN across all of it.
        self.gamelog = self.gamelog.drop(
            columns=["gametime", "own rest", "opp rest"], errors="ignore"
        )
        nfl_data = self._fetch_player_stats()
        snaps = self._fetch_snap_counts()
        sched = self._fetch_schedule()
        self._compute_upcoming_games(sched)
        nfl_data = self._assemble_gamelog_frame(nfl_data, sched, snaps)
        if not nfl_data.empty:
            self._enrich_new_rows(nfl_data)
        self.gamelog = (
            pd.concat([self.gamelog, nfl_data], ignore_index=True)
            .drop_duplicates(["season", "week", "player id"], ignore_index=True)
            .reset_index(drop=True)
        )
        self.gamelog["player display name"] = self.gamelog["player display name"].apply(
            remove_accents
        )
        self._load_player_ids()
        teamDataList = self._backfill_missing_rows(sched)
        self._finalize_logs(teamDataList)
        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog["player display name"] = self.gamelog["player display name"].apply(
                remove_accents
            )
            self._enrich_team_markets(self.gamelog)

        self._coerce_numeric_gamelog()

        write_gamelog("nfl", self.gamelog, self.teamlog, self.players)

    def _fetch_player_stats(self):
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
        except Exception as exc:
            logger.warning("load_player_stats() failed: %s", exc)
            nfl_data = pd.DataFrame(columns=cols)
        return nfl_data

    def _fetch_snap_counts(self):
        try:
            snaps = nflr.load_snap_counts().to_pandas()
        except Exception as exc:
            logger.warning("load_snap_counts() failed: %s", exc)
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
        return snaps

    def _fetch_schedule(self):
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
            sched = pd.DataFrame(columns=["away_team", "home_team", "gameday", "week", "game_id"])
        return sched

    def _compute_upcoming_games(self, sched):
        upcoming_games = sched.loc[
            pd.to_datetime(sched["gameday"]).dt.date >= datetime.today().date(),
            ["gameday", "away_team", "home_team", "week"],
        ]
        if not upcoming_games.empty:
            # Playoff games run past the regular-season schedule (week 19+).
            upcoming_games["playoff"] = upcoming_games["week"] > NFL_REGULAR_SEASON_WEEKS
            df1 = upcoming_games.rename(columns={"home_team": "Team", "away_team": "Opponent"})
            df2 = upcoming_games.rename(columns={"away_team": "Team", "home_team": "Opponent"})
            df1["Home"] = True
            df2["Home"] = False
            upcoming_games = pd.concat([df1, df2]).sort_values("gameday")
            self.upcoming_games = (
                upcoming_games.groupby("Team")
                .apply(lambda x: x.head(1))
                .droplevel(1)[["Opponent", "Home", "gameday", "playoff"]]
                .to_dict(orient="index")
            )

    def _schedule_context(self, stats, date, teams):
        """QW-4: emit the ``Weekday`` game-script feature (game date day-of-week).

        Historical reads the realized gamelog row for ``date``; upcoming reads
        ``upcoming_games``. Both derive Weekday from the game date, so they agree.
        """
        if date < datetime.today().date():
            todays = (
                self.gamelog.loc[pd.to_datetime(self.gamelog["gameday"]).dt.date == date]
                .drop_duplicates("player display name")
                .set_index("player display name")
            )
            gameday = pd.to_datetime(todays["gameday"]).reindex(stats.index)
        else:
            ug = self.upcoming_games
            rows = {p: ug.get(teams.get(p), {}) for p in stats.index}
            gameday = pd.to_datetime(pd.Series({p: r.get("gameday") for p, r in rows.items()}))

        stats["Weekday"] = gameday.dt.dayofweek

    def _playoff_flag(self, stats, date, teams):
        """Flag postseason games from the gamelog ``season type`` (historical) or the
        schedule-derived ``playoff`` bool carried on ``upcoming_games`` (upcoming).
        """
        if date < datetime.today().date():
            todays = (
                self.gamelog.loc[pd.to_datetime(self.gamelog["gameday"]).dt.date == date]
                .drop_duplicates("player display name")
                .set_index("player display name")
            )
            playoff = todays["season type"].reindex(stats.index) == "POST"
        else:
            ug = self.upcoming_games
            playoff = pd.Series(
                {p: bool(ug.get(teams.get(p), {}).get("playoff")) for p in stats.index}
            )
        stats["Playoff"] = playoff.astype(int)

    def _series_context(self, stats, date, teams, opponents):
        """NFL playoffs are single-elimination: every postseason game is win-or-go-home
        for both sides and a win clinches advancement, so there is no multi-game series
        record. FacingElimination and CanClinch both collapse onto the Playoff flag.
        """
        stats["SeriesWins"] = 0
        stats["SeriesLosses"] = 0
        stats["FacingElimination"] = stats["Playoff"]
        stats["CanClinch"] = stats["Playoff"]

    def _assemble_gamelog_frame(self, nfl_data, sched, snaps):
        # Attach each row's gameday from the schedule -- home and away side both map
        # their team to recent_team so the merge reaches every player row.
        sched_long = pd.concat(
            [
                sched.rename(columns={"home_team": "recent_team"}),
                sched.rename(columns={"away_team": "recent_team"}),
            ]
        )[["recent_team", "week", "gameday"]]
        nfl_data = nfl_data.merge(sched_long, how="left")
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
        return nfl_data

    def _enrich_new_rows(self, nfl_data):
        # Only enrich rows that will survive the post-concat
        # drop_duplicates below. Without this filter the bulk path
        # still wins, but most of the work is thrown away when the
        # cached gamelog already covers the (season, week, player) tuple.
        cached_keys: set[tuple] = set()
        if {"season", "week", "player id"}.issubset(self.gamelog.columns):
            cached_keys = set(
                zip(
                    self.gamelog["season"],
                    self.gamelog["week"],
                    self.gamelog["player id"],
                    strict=False,
                )
            )
        if cached_keys:
            new_mask = np.fromiter(
                (
                    key not in cached_keys
                    for key in zip(
                        nfl_data["season"],
                        nfl_data["week"],
                        nfl_data["player id"],
                        strict=False,
                    )
                ),
                dtype=bool,
                count=len(nfl_data),
            )
        else:
            new_mask = None
        self._enrich_team_markets(nfl_data, mask=new_mask)

    def _load_player_ids(self):
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

    def _backfill_missing_rows(self, sched):
        teamDataList = []
        for i, row in tqdm(
            self.gamelog.loc[self.gamelog.isna().any(axis=1)].iterrows(),
            desc="Updating NFL data",
            unit="game",
            total=len(self.gamelog.loc[self.gamelog.isna().any(axis=1)]),
        ):
            self._backfill_schedule_fields(i, row, sched)
            self._backfill_player_pbp(i, row)
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
        return teamDataList

    def _backfill_schedule_fields(self, i, row, sched):
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

    def _backfill_player_pbp(self, i, row):
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

    def _finalize_logs(self, teamDataList):
        self.teamlog = pd.concat(
            [self.teamlog, pd.DataFrame.from_records(teamDataList)], ignore_index=True
        )
        self.teamlog = self.teamlog.sort_values("gameday").fillna(0).infer_objects(copy=False)
        self.gamelog = self.gamelog.sort_values("gameday")

        six_years_ago = datetime.today().date() - timedelta(days=_GAMELOG_RETENTION_DAYS)
        self.gamelog = self.gamelog[
            self.gamelog["gameday"].apply(
                lambda x: (
                    six_years_ago
                    <= datetime.strptime(x, "%Y-%m-%d").date()
                    <= datetime.today().date()
                )
            )
        ]
        self.gamelog = self.gamelog[~self.gamelog["opponent"].isin(["AFC", "NFC"])]
        self.teamlog = self.teamlog[
            self.teamlog["gameday"].apply(
                lambda x: (
                    six_years_ago
                    <= datetime.strptime(x, "%Y-%m-%d").date()
                    <= datetime.today().date()
                )
            )
        ]
        self.gamelog.drop_duplicates(inplace=True)
        self.teamlog.drop_duplicates(inplace=True)

    def parse_pbp(self, week, team, year, playerName=""):
        self._ensure_pbp_loaded(year)
        pbp = self.pbp.loc[
            (self.pbp.week == week) & ((self.pbp.home_team == team) | (self.pbp.away_team == team))
        ]
        if pbp.empty:
            return 0
        pbp_off = pbp.loc[pbp.posteam == team]
        pbp_def = pbp.loc[pbp.posteam != team]
        home = pbp.iloc[0].home_team == team
        if playerName == "":
            return self._team_pbp_stats(pbp, pbp_off, pbp_def, home, team)
        if self.ids.get(playerName) is None:
            return self._empty_player_pbp_stats()
        return self._player_pbp_stats(pbp, pbp_off, week, year, playerName)

    def _ensure_pbp_loaded(self, year):
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

    def _ratio(self, num, den):
        return num / den if den > 0 else np.nan

    def _ngs_week(self, playerName, wk, col):
        return self.ngs.loc[
            (self.ngs["player_display_name"] == playerName) & (self.ngs["week"] == wk), col
        ].mean()

    def _pfr_week(self, playerName, wk, col):
        return self.pfr.loc[
            (self.pfr["pfr_player_name"] == playerName) & (self.pfr["week"] == wk), col
        ].mean()

    def _team_pbp_stats(self, pbp, pbp_off, pbp_def, home, team):
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
        off_fr_epa = pbp_off.loc[(pbp_off["pass"]) & (pbp_off["read_thrown"] == "1"), "epa"].mean()
        off_mid_epa = pbp_off.loc[
            (pbp_off["pass"]) & (pbp_off["pass_location"] == "middle"), "epa"
        ].mean()
        def_rush_epa = pbp_def.loc[pbp_def["rush"], "epa"].mean()
        def_pass_epa = pbp_def.loc[pbp_def["pass"], "epa"].mean()
        def_rz_epa = pbp_def.loc[pbp_def["redzone"], "epa"].mean()
        def_fr_epa = pbp_def.loc[(pbp_def["pass"]) & (pbp_def["read_thrown"] == "1"), "epa"].mean()
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

    def _empty_player_pbp_stats(self):
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

    def _player_pbp_stats(self, pbp, pbp_off, week, year, playerName):
        wk = pbp.week.max()
        cpoe = self._ngs_week(playerName, wk, "completion_percentage_above_expectation")
        cp = self._ngs_week(playerName, wk, "completion_percentage")
        qbr = self._ngs_week(playerName, wk, "passer_rating")
        pass_adot = self._ngs_week(playerName, wk, "avg_intended_air_yards")
        pass_adot_diff = self._ngs_week(playerName, wk, "avg_air_yards_differential")
        time_to_throw = self._ngs_week(playerName, wk, "avg_time_to_throw")
        aggressiveness = self._ngs_week(playerName, wk, "aggressiveness")
        ryoe = self._ngs_week(playerName, wk, "rush_yards_over_expected")
        rush_sr = self._ngs_week(playerName, wk, "rush_pct_over_expected")
        yacoe = self._ngs_week(playerName, wk, "avg_yac_above_expectation")
        sep = self._ngs_week(playerName, wk, "avg_separation") - self._ngs_week(
            playerName, wk, "avg_cushion"
        )
        broken_tackles = (
            self.pfr.loc[
                (self.pfr["pfr_player_name"] == playerName) & (self.pfr["week"] == wk),
                ["rushing_broken_tackles", "receiving_broken_tackles"],
            ]
            .sum(axis=1)
            .mean()
        )
        breakaway_yards = self._pfr_week(playerName, wk, "rushing_yards_after_contact_avg")
        drop_pct = self._pfr_week(playerName, wk, "receiving_drop_pct")
        pass_drop_pct = self._pfr_week(playerName, wk, "passing_drop_pct")
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
        mid_target_rate = self._ratio(
            len(
                pbp_off.loc[
                    (pbp_off["passer_player_id"] == self.ids.get(playerName))
                    & (pbp_off["pass_location"] == "middle")
                ]
            ),
            pass_attempts,
        )
        mid_targets = self._ratio(
            len(
                pbp_off.loc[
                    (pbp_off["receiver_player_id"] == self.ids.get(playerName))
                    & (pbp_off["pass_location"] == "middle")
                ]
            ),
            routes_run,
        )
        tprr = self._ratio(targets, routes_run)
        frtprr = self._ratio(fr_targets, routes_run)
        frt_pct = self._ratio(fr_targets, fr_pass_attempts)
        route_participation = self._ratio(routes_run, pass_attempts)
        yprr = self._ratio(
            pbp_off.loc[
                pbp_off["receiver_player_id"] == self.ids.get(playerName), "yards_gained"
            ].sum(),
            routes_run,
        )
        adot = pbp_off.loc[
            pbp_off["receiver_player_id"] == self.ids.get(playerName), "air_yards"
        ].mean()
        rec_cpoe = pbp_off.loc[
            pbp_off["receiver_player_id"] == self.ids.get(playerName), "cpoe"
        ].mean()
        rz_passes = len(pbp_off.loc[pbp_off["pass_attempt"] & pbp_off["redzone"]])
        rz_target_pct = self._ratio(
            len(
                pbp_off.loc[
                    (pbp_off["receiver_player_id"] == self.ids.get(playerName)) & pbp_off["redzone"]
                ]
            ),
            rz_passes,
        )
        rz_rushes = len(pbp_off.loc[pbp_off["rush"] & pbp_off["redzone"]])
        rz_attempt_pct = self._ratio(
            len(
                pbp_off.loc[
                    (pbp_off["rusher_player_id"] == self.ids.get(playerName)) & pbp_off["redzone"]
                ]
            ),
            rz_rushes,
        )
        rushes = len(pbp_off.loc[pbp_off["rush"]])
        attempt_pct = self._ratio(
            len(pbp_off.loc[pbp_off["rusher_player_id"] == self.ids.get(playerName)]), rushes
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

    def build_comp_profile(self, target_game_date: date | None = None) -> pd.DataFrame:
        """Build the NFL player comp profile from per-game FP snapshots + PBP aggregates.

        Phase-1.5 of the PFF -> FantasyPoints migration. Inputs come from
        the per-game weekly parquets via
        :func:`sportstradamus.stats.nfl_fp_weekly_aggregate.load_through_one_year`,
        which season-to-dates per-game rows using the Pattern A / B / C
        helpers in :mod:`sportstradamus.stats.nfl_fp_aggregation`. PBP/NGS
        aggregates are joined the same way the legacy season-CSV path
        did. The Phase-1G rolling transform smooths moderate-stability
        features across the lookback window, then the most-recent row per
        player is selected for the downstream KNN join.

        The lookback window enforces the operator's Phase-1.5 v2 rule:
        after week 4 of ``target_game_date``'s season the comp pool is
        current-season-only (no prior seasons blended in). Pre-week-5
        the prior season's BACK HALF (weeks
        ``COMP_LOOKBACK_PRIOR_SEASON_FROM_WEEK..18``) is pooled with
        any current-season games already played so the KNN has enough
        sample. When ``target_game_date`` is None the function falls
        back to a full-season aggregate across the prior three seasons +
        the current season's complete window -- preserving the legacy
        contract for callers that haven't been migrated to per-game-date
        comp profiles yet.

        Results are cached per ``(season, week)`` cell on ``self`` so
        repeated calls within a training loop don't re-aggregate the same
        snapshots.

        Args:
            target_game_date: Date of the game the comp profile is being
                built for. When provided, the lookback window cuts off
                at ``target_week - 1`` of the corresponding season (post-
                week-4: current season only; pre-week-5: prior-season
                back half + current partial). When ``None``, full-season
                aggregates land for the prior three seasons + current
                season (legacy path).

        Returns:
            DataFrame indexed by accent-stripped player name with
            ``position``, ``player_game_count``, the comp-filter aliases
            (``dropbacks`` / ``attempts`` / ``routes``), every FP-prefixed
            metric, every ``pbp_*`` aggregate, and (when available)
            ``age`` / ``height`` / ``bmi``. Empty DataFrame if no FP
            snapshots are found in the lookback window.
        """
        if self.playerProfile.empty:
            self.profile_market("snap pct")

        cache_key = self._resolve_comp_cache_key(target_game_date)
        cache = self._comp_profile_cache()
        if cache_key not in cache:
            cache[cache_key] = self._compute_comp_profile(cache_key)
        playerProfile = cache[cache_key]
        if playerProfile.empty:
            return playerProfile

        return playerProfile.join(self.playerProfile[self.playerProfile.columns[9:]])

    def _resolve_comp_cache_key(self, target_game_date: date | None) -> tuple[int, int | None]:
        """Translate the public ``target_game_date`` arg into a cache key + lookback spec.

        Returns a ``(target_season, target_week)`` tuple. ``target_week``
        carries a sentinel ``None`` for the "no date given" mode --
        triggers the full-season fallback inside
        :meth:`_compute_comp_profile`. ``target_season`` defaults to
        ``self.season_start.year`` so callers without a date still see
        the right season for the current cron run.
        """
        if target_game_date is None:
            return self.season_start.year, None
        target_season, target_week = self._lookup_season_week(target_game_date)
        return target_season, target_week

    def _comp_target_key(self, date: "date | None") -> "tuple[int, int | None]":
        """NFL override: cache the per-week comp pool on ``(season, week)``.

        The base implementation buckets on the most-recent Wednesday's week
        index. NFL has authoritative schedule weeks that already align with
        the Wed→Wed window (games Thu/Sun/Mon; Tue/Wed = retrain idle), so
        reusing :meth:`_resolve_comp_cache_key` keeps the comp cache aligned
        with both :meth:`build_comp_profile`'s ``_comp_profile_cache`` and
        the operator's confirmed retrain cadence.
        """
        return self._resolve_comp_cache_key(date)

    def _lookup_season_week(self, target_game_date: datetime | date) -> tuple[int, int]:
        """Resolve a calendar date to ``(season, week)`` via nfl_data_py's schedule.

        Builds + caches a date → (season, week) index across the
        seasons that overlap the requested date. For dates in the
        regular-season window the schedule lookup is exact; for
        bye-week / off-season dates the function returns the next
        scheduled game's week so the comp pool covers everything the
        target player could plausibly have played in already.
        """
        if isinstance(target_game_date, datetime):
            target_game_date = target_game_date.date()

        sched_index = self._schedule_index()
        if sched_index.empty:
            return self.season_start.year, NFL_REGULAR_SEASON_WEEKS

        match = sched_index.loc[sched_index["gameday"] >= target_game_date].head(1)
        if match.empty:
            # All schedule rows are in the past -> use the season's last week.
            last = sched_index.iloc[-1]
            return int(last["season"]), int(last["week"])
        row = match.iloc[0]
        return int(row["season"]), int(row["week"])

    def _schedule_index(self) -> pd.DataFrame:
        """Cached ``(gameday, season, week)`` index spanning the comp lookback."""
        if getattr(self, "_sched_index", None) is None:
            if self.snapshot_only_rebuild:
                sched = self.gamelog[["gameday", "season", "week"]].copy()
            else:
                years = list(
                    range(
                        self.season_start.year - COMP_PRIOR_SEASON_DEPTH,
                        self.season_start.year + 2,
                    )
                )
                try:
                    sched = nfl.import_schedules(years)
                except Exception as exc:
                    logger.warning("import_schedules(%s) failed: %s", years, exc)
                    self._sched_index = pd.DataFrame(columns=["gameday", "season", "week"])
                    return self._sched_index
            sched = sched.dropna(subset=["gameday", "season", "week"]).copy()
            sched["gameday"] = pd.to_datetime(sched["gameday"]).dt.date
            self._sched_index = (
                sched[["gameday", "season", "week"]]
                .drop_duplicates()
                .sort_values("gameday", kind="stable")
                .reset_index(drop=True)
            )
        return self._sched_index

    def _comp_profile_cache(self) -> dict[tuple[int, int | None], pd.DataFrame]:
        """Lazily-initialised per-(season, week) comp profile cache."""
        if getattr(self, "_comp_cache", None) is None:
            self._comp_cache: dict[tuple[int, int | None], pd.DataFrame] = {}
        return self._comp_cache

    def _join_fp_team_features(
        self, date: datetime | date
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Phase-2 NFL hook: aggregate FP team-grain snapshots into team / defense feature frames.

        Resolves ``date`` to ``(season, target_week)`` via
        ``_lookup_season_week``, applies the Phase-1.5 lookback rule via
        ``_lookback_windows`` (post-week-4 = current-season only;
        pre-week-5 = prior season weeks 11..18 BLENDED with current
        partial), and hands the windows + the target snapshot to the
        bridge. The bridge handles Pattern A (rate stats pooled across
        windows) and Pattern B (``line_matchups`` single-snapshot at
        target week).

        Returns ``(None, None)`` when the team-data fetcher hasn't
        backfilled any snapshots for the target season AND the prior
        season -- the existing nfl_data_py-derived teamlog aggregates
        carry the run alone, so production stays live on mixed-fill data.
        """
        season, target_week = self._lookup_season_week(date)
        if getattr(self, "_fp_team_feature_cache", None) is None:
            self._fp_team_feature_cache: dict[
                tuple[int, int], tuple[pd.DataFrame | None, pd.DataFrame | None]
            ] = {}
        cache_key = (season, target_week)
        if cache_key in self._fp_team_feature_cache:
            return self._fp_team_feature_cache[cache_key]
        pattern_a_windows = self._lookback_windows(season, target_week)
        has_data = any(
            nfl_fp_team_weekly.available_snapshots(window_season)
            for window_season, _, _ in pattern_a_windows
        ) or bool(nfl_fp_team_weekly.available_snapshots(season))
        if not has_data:
            self._fp_team_feature_cache[cache_key] = (None, None)
            return None, None
        team_features, defense_features = (
            nfl_fp_team_weekly_aggregate.load_team_and_defense_features(
                pattern_a_windows=pattern_a_windows,
                pattern_b_snapshot=(season, target_week),
            )
        )
        self._fp_team_feature_cache[cache_key] = (team_features, defense_features)
        return self._fp_team_feature_cache[cache_key]

    def _join_fp_player_features(self, date: datetime | date) -> pd.DataFrame | None:
        """NFL hook: project FP per-player season-to-date aggregates to ``{col}_asof``.

        Resolves ``date`` to ``(season, target_week)`` via
        ``_lookup_season_week``, then loads the player-grain season-to-date
        profile through ``load_through_one_year(season, target_week=week-1)``
        — the "as of last week" snapshot, leakage-clean for the target row's
        date. Each non-meta column is renamed to raw ``{col}_asof`` (no
        ``Player`` prefix) so the downstream ``add_prefix("Player ")`` step
        prepends the namespace once, producing ``Player {col}_asof`` keys
        that do not collide with the legacy gamelog ``Player {col}`` columns.

        Pre-week-5 falls back to the prior season's late weeks. Returns
        ``None`` when no FP weekly snapshot exists for the resolved window —
        the model falls back to gamelog-derived features alone.
        """
        season, target_week = self._lookup_season_week(date)
        if getattr(self, "_fp_asof_cache", None) is None:
            self._fp_asof_cache: dict[tuple[int, int], pd.DataFrame | None] = {}
        key = (season, target_week)
        if key in self._fp_asof_cache:
            return self._fp_asof_cache[key]
        frame = self._compute_fp_asof_features(season, target_week)
        self._fp_asof_cache[key] = frame
        return frame

    def _compute_fp_asof_features(self, season: int, week: int) -> pd.DataFrame | None:
        """Load FP season-to-date profile for ``(season, week-1)`` and rename to raw ``{col}_asof``.

        Pre-week-5 routes to the prior season's late weeks via
        ``COMP_LOOKBACK_PRIOR_SEASON_FROM_WEEK..NFL_REGULAR_SEASON_WEEKS``;
        this mirrors the comp-profile blend semantics so the early-season
        asof view sees real signal instead of an empty frame.
        """
        lookback_week = max(week - 1, 0)
        if lookback_week >= 1:
            frame = nfl_fp_weekly_aggregate.load_through_one_year(
                season,
                target_week=lookback_week,
                snapshot_only=self.snapshot_only_rebuild,
            )
        else:
            # Early-season: fall back to prior season's late weeks.
            frame = nfl_fp_weekly_aggregate.load_through_one_year(
                season - 1,
                target_week=NFL_REGULAR_SEASON_WEEKS,
                snapshot_only=self.snapshot_only_rebuild,
            )
        if frame is None or frame.empty or "Name" not in frame.columns:
            return None
        frame = frame.copy()
        # Drop rows with missing player names (rare aggregation artifacts).
        frame = frame.loc[frame["Name"].notna()]
        if frame.empty:
            return None
        # Join key matches the gamelog ``player display name`` after the
        # ``remove_accents`` pass that ``base_profile`` applies to
        # short_gamelog. Both sides must strip accents for the merge to land.
        frame.index = frame["Name"].astype(str).apply(remove_accents)
        frame.index.name = self.log_strings["player"]
        meta = {"Name", "Team", "position"}
        feat_cols = [c for c in frame.columns if c not in meta]
        if not feat_cols:
            return None
        out = frame[feat_cols].copy()
        # Drop duplicate-index rows (player traded mid-season -> two Team rows).
        # Keep the row with the most non-NaN cells so we don't lose signal to
        # the smaller-sample team's snapshot.
        if out.index.has_duplicates:
            non_na_rank = out.notna().sum(axis=1)
            out = out.assign(_rank=non_na_rank).sort_values("_rank", ascending=False)
            out = out[~out.index.duplicated(keep="first")].drop(columns="_rank")
        # Emit raw ``{col}_asof`` (no Player prefix). Downstream
        # ``add_prefix("Player ")`` in get_stats / get_feature_columns
        # prepends the namespace once, producing ``Player {col}_asof``.
        # Prefixing here would double-prefix to ``Player Player {col}_asof``.
        out.columns = [f"{c}_asof" for c in out.columns]
        return out

    def _compute_comp_profile(self, cache_key: tuple[int, int | None]) -> pd.DataFrame:
        """Aggregate the per-game snapshots into a comp-profile frame for one cell.

        Applies the Phase-1.5 lookback rule:

        * ``target_week is None`` -> legacy fallback. Load full-season
          aggregates for the prior 3 seasons + current season,
          season-by-season, then rolling-transform + latest-row dedupe.
          Per-season independent aggregation matches the pre-Phase-1.5
          contract for callers (e.g. ``optimize_comp_weights``) that
          haven't been migrated to per-game-date comp profiles yet.
        * ``target_week < 5`` -> pre-cutoff. Prior season's back half
          (weeks ``COMP_LOOKBACK_PRIOR_SEASON_FROM_WEEK..18``) BLENDED
          with the current season's partial weeks through
          ``target_week - 1``. "Blended" here means a player's per-game
          rows from BOTH seasons land in one aggregate row -- the
          aggregator pools raw per-game data across windows before the
          per-player groupby, so a player with 8 prior-back-half games
          + 2 current-season games gets one 10-game aggregate (not
          most-recent-season-wins).
        * ``target_week >= 5`` -> post-cutoff. Current season only,
          weeks ``1..target_week - 1``. Prior seasons drop out because
          ~4 in-season games are more predictive of the next game than
          last year's full-season number for the same player (the comp
          pool also restricts to current-season players who have a
          meaningful in-season sample).
        """
        target_season, target_week = cache_key
        windows = self._lookback_windows(target_season, target_week)
        if target_week is None:
            return self._compute_comp_profile_legacy(windows)

        playerProfile = nfl_fp_weekly_aggregate.load_multi_window_one_year(
            windows,
            snapshot_only=self.snapshot_only_rebuild,
        )
        if playerProfile.empty:
            return playerProfile
        playerProfile = playerProfile.copy()
        playerProfile["season"] = target_season

        if "Name" in playerProfile.columns:
            playerProfile = playerProfile.assign(player=playerProfile["Name"])
        return playerProfile

    def _compute_comp_profile_legacy(
        self,
        windows: list[tuple[int, int, int]],
    ) -> pd.DataFrame:
        """Per-season aggregation + rolling-transform path used by the None-date fallback.

        Preserves the pre-Phase-1.5 contract: aggregate each season
        independently, stack, run ``apply_rolling_transforms`` for
        cross-season smoothing, then keep the most-recent row per
        player. Callers that pass a ``target_game_date`` route through
        the pooled aggregator instead.
        """
        per_year_frames: list[pd.DataFrame] = []
        for season, start_week, end_week in windows:
            per_year = nfl_fp_weekly_aggregate.load_window_one_year(
                season,
                start_week,
                end_week,
                snapshot_only=self.snapshot_only_rebuild,
            )
            if per_year.empty:
                continue
            per_year = per_year.copy()
            per_year["season"] = season
            per_year_frames.append(per_year)

        if not per_year_frames:
            return pd.DataFrame()

        stacked = pd.concat(per_year_frames)
        stacked = nfl_fp_loader.apply_rolling_transforms(stacked)
        stacked = stacked.sort_values("season", kind="stable")
        playerProfile = stacked.loc[~stacked.index.duplicated(keep="last")]

        if "Name" in playerProfile.columns:
            # The concat-then-sort can leave a NaN ``Name`` cell for any
            # row whose player only appeared in one of the lookback's
            # seasons; remove_accents chokes on NaN. Drop those before
            # re-indexing rather than letting the dispatch fail.
            playerProfile = playerProfile.dropna(subset=["Name"])
            if playerProfile.empty:
                return playerProfile
            playerProfile = playerProfile.assign(player=playerProfile["Name"])
            playerProfile.index = playerProfile["Name"].apply(remove_accents)
        return playerProfile

    @staticmethod
    def _lookback_windows(
        target_season: int, target_week: int | None
    ) -> list[tuple[int, int, int]]:
        """Return the ``[(season, start_week, end_week), ...]`` list for the comp lookback.

        Encodes the Phase-1.5 rule. Documented case-by-case so future
        edits to the cutoff land here in one place rather than scattered
        across the aggregation path.

        **Postseason-clean contract** (pinned by
        ``tests/golden/test_nfl_lookback_windows.py``):

        * Regular-season targets (``target_week`` in 1..18) MUST NOT pull
          any window past ``NFL_REGULAR_SEASON_WEEKS = 18``. The
          prior-season window caps at 18 explicitly; the current-season
          window caps at ``target_week - 1 <= 17``. Postseason rows are a
          measurably different distribution at team grain (2022 W11-18
          vs W19-22 measured PROE +8.4pp, ``def_two_high_pct`` +7.7pp,
          dropbacks/game +9.2%); blending them into a reg-season prior
          biases the feature in the wrong direction. Full diagnostic +
          cross-sport precedent: Phase 2.1 section of the team-features
          plan doc.
        * Postseason targets (``target_week`` in 19..22) pass through the
          post-cutoff branch and pull ``(target_season, 1, target_week - 1)``,
          which DOES include this season's earlier postseason rows. That
          is the symmetric rule's INCLUDE branch -- a postseason target
          should see postseason priors. The branch is unreachable today
          (no postseason props are priced), but the test pins it so a
          future tightening does not kill it before the operator decides
          to price one.

        Pattern B (``line_matchups``) is NOT routed through this method;
        the single-snapshot lookup at ``(target_season, target_week)``
        applies whether the target is reg-season or postseason.
        """
        if target_week is None:
            # Legacy fallback -- full-season aggregates for prior 3 + current.
            # Cap at NFL_REGULAR_SEASON_WEEKS so optimize_comp_weights' default
            # path inherits the postseason-clean contract above.
            return [
                (s, 1, NFL_REGULAR_SEASON_WEEKS)
                for s in range(
                    target_season - COMP_PRIOR_SEASON_DEPTH,
                    target_season + 1,
                )
            ]
        if target_week >= COMP_LOOKBACK_CURRENT_SEASON_ONLY_WEEK:
            # Post-cutoff -- current season only. For target_week in 5..18 the
            # ``target_week - 1`` end is <= 17 (reg-season only); for
            # target_week in 19..22 (postseason target) the end >= 18 and
            # postseason rows are intentionally included per the symmetric rule.
            return [(target_season, 1, target_week - 1)]
        # Pre-week-5 -- prior season's back half (clipped at 18 -- the cap is
        # load-bearing, not cosmetic) + current partial (if any).
        windows = [
            (
                target_season - 1,
                COMP_LOOKBACK_PRIOR_SEASON_FROM_WEEK,
                NFL_REGULAR_SEASON_WEEKS,
            )
        ]
        if target_week > 1:
            windows.append((target_season, 1, target_week - 1))
        return windows

    def update_player_comps(self, year: int | None = None) -> None:
        """Rebuild comp clusters for each NFL position and write to ``comps.json``.

        Applies the Phase-1.5 lookback rule via ``build_comp_profile``: after
        week 4 of the current season, the comp pool is restricted to
        current-season weeks 1..W-1; pre-week-5, the prior season's back half
        (weeks 11..18) is blended with any current-season games. Writes the
        result to ``data/leagues/nfl/comps.json`` for reuse by the prediction
        pipeline.

        Args:
            year: Override the season year. Defaults to ``self.season_start.year``.
        """
        if year is None:
            year = self.season_start.year
        with open(pkg_resources.files(data) / "config" / "playerCompStats.json") as infile:
            stats = json.load(infile)

        filterStat = {"QB": "dropbacks", "RB": "attempts", "WR": "routes", "TE": "routes"}

        # Pass today's date so the Phase-1.5 lookback rule applies: after
        # week 4 of the current season the comp pool is restricted to
        # current-season weeks 1..W-1 (`_lookback_windows`), which is the
        # operator-confirmed cron behaviour. Pre-week-5 (or off-season),
        # the prior season's back half (weeks 11..18) is blended with
        # any current-season games already played.
        playerProfile = self.build_comp_profile(target_game_date=datetime.today().date())
        if playerProfile.empty:
            return

        comps = {}
        for position in ["QB", "RB", "WR", "TE"]:
            positionProfile = playerProfile.loc[playerProfile.position == position].copy()
            positionProfile[filterStat[position]] = (
                positionProfile[filterStat[position]] / positionProfile["player_game_count"]
            )
            positionProfile = positionProfile.loc[
                positionProfile[filterStat[position]]
                >= positionProfile[filterStat[position]].quantile(_COMP_POOL_MIN_USAGE_PERCENTILE)
            ]
            positionProfile = positionProfile.reindex(
                columns=list(stats["NFL"][position].keys())
            ).replace([np.nan, np.inf, -np.inf], 0)
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(
                np.sqrt(np.abs(np.asarray(list(stats["NFL"][position].values()))))
            )
            knn = BallTree(positionProfile)
            comps[position] = self._build_comps(
                knn, positionProfile, min_comps=_COMP_KNN_MIN, max_comps=_COMP_KNN_MAX
            )

        filepath = pkg_resources.files(data) / "leagues" / "nfl" / "comps.json"
        with open(filepath, "w") as outfile:
            json.dump(comps, outfile, indent=4)

    def _compute_comps(self, target_game_date: date | None = None) -> None:
        """Build comps from loaded data at runtime (no JSON I/O).

        Args:
            target_game_date: Date the comps should reflect "as of". The
                training-matrix path passes the per-gameday date so the comp
                pool is bounded to games strictly before it (eliminates the
                look-ahead leakage where a 2023 row's comps would otherwise
                see the 2025 player population). When ``None`` (inference /
                cron path), today's date is used and the Phase-1.5 lookback
                rule routes through ``_lookback_windows`` to restrict comps
                to current-season-only candidates after week 4.
        """
        with open(pkg_resources.files(data) / "config" / "playerCompStats.json") as f:
            stats = json.load(f)

        filterStat = {"QB": "dropbacks", "RB": "attempts", "WR": "routes", "TE": "routes"}

        if target_game_date is None:
            target_game_date = datetime.today().date()
        playerProfile = self.build_comp_profile(target_game_date=target_game_date)
        if playerProfile.empty:
            return

        comps = {}
        for position in ["QB", "RB", "WR", "TE"]:
            positionProfile = playerProfile.loc[playerProfile.position == position].copy()
            positionProfile[filterStat[position]] = (
                positionProfile[filterStat[position]] / positionProfile["player_game_count"]
            )
            positionProfile = positionProfile.loc[
                positionProfile[filterStat[position]]
                >= positionProfile[filterStat[position]].quantile(_COMP_POOL_MIN_USAGE_PERCENTILE)
            ]
            positionProfile = positionProfile.reindex(
                columns=list(stats["NFL"][position].keys())
            ).replace([np.nan, np.inf, -np.inf], 0)
            if len(positionProfile) < _COMP_POOL_MIN_PLAYERS:
                continue
            positionProfile = positionProfile.apply(
                lambda x: (x - x.mean()) / x.std(), axis=0
            ).fillna(0)
            positionProfile = positionProfile.mul(
                np.sqrt(np.abs(np.asarray(list(stats["NFL"][position].values()))))
            )
            knn = BallTree(positionProfile)
            comps[position] = self._build_comps(
                knn, positionProfile, min_comps=_COMP_KNN_MIN, max_comps=_COMP_KNN_MAX
            )

        self.comps = comps

    def check_combo_markets(
        self, market: str, player: str, date: date = datetime.today().date()
    ) -> int:
        return 0  # combo-market EV pending reimplementation

    def get_volume_stats(self, offers: dict, date: date = datetime.today().date()) -> None:
        for market in self.volume_stats:
            if not self.load_volume_model_params(
                offers,
                market,
                date,
                position_filter=_VOLUME_MARKET_POSITION_FILTER[market],
            ):
                return

            if market == "attempts":
                self._finalize_attempts_volume(market)
                continue

            teams = self.playerProfile.loc[self.playerProfile["team"] != 0].groupby("team")
            for team, team_df in teams:
                self._rescale_team_volume(market, team, team_df)
            self.playerProfile.fillna(0, inplace=True)

        # Defragment after 3-market loop's sequential .join / .loc column inserts.
        self.playerProfile = self.playerProfile.copy()

    def _finalize_attempts_volume(self, market):
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
        self.playerProfile.drop(columns=[f"proj {market} alpha"], inplace=True, errors="ignore")

        self.playerProfile.fillna(0, inplace=True)

    def _rescale_team_volume(self, market, team, team_df):
        # Precision-weighted, always-adjust normalization.
        # Budget constraints:
        #   carries:  total ≈ plays_per_game − proj_attempts − _UNMODELED_CARRY_RESERVE
        #   targets:  total ≈ proj_attempts − _UNMODELED_TARGET_RESERVE
        # proj_attempts from SkewNormal: E[X] = loc + scale*delta*sqrt(2/π),
        # delta = alpha/sqrt(1+alpha²)
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
            return

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
            cap = _CARRY_CAP
            rush_budget = plays - proj_attempts
            target = max(rush_budget - _UNMODELED_CARRY_RESERVE, len(team_df))
        else:  # targets
            cap = _TARGET_CAP
            target = max(proj_attempts - _UNMODELED_TARGET_RESERVE, len(team_df))

        deficit = target - total
        total_var = true_vars.sum()
        if total_var > 0:
            adjustments = true_vars / total_var * deficit
        else:
            adjustments = true_means / total * deficit

        new_means = (true_means + adjustments).clip(lower=0, upper=cap)

        # SkewNormal: scale both loc and scale to preserve CV
        ratio = new_means / true_means.replace(0, np.nan)
        ratio = ratio.fillna(1.0).clip(
            lower=_VOLUME_SCALE_RATIO_FLOOR, upper=_VOLUME_SCALE_RATIO_CAP
        )
        new_scale = scale * ratio
        new_loc = new_means - new_scale * delta * np.sqrt(2 / np.pi)

        self.playerProfile.loc[loc.index, f"proj {market} loc"] = new_loc
        self.playerProfile.loc[scale.index, f"proj {market} scale"] = new_scale
        self.playerProfile.loc[loc.index, f"proj {market} mean"] = new_means
        self.playerProfile.loc[scale.index, f"proj {market} std"] = new_scale
