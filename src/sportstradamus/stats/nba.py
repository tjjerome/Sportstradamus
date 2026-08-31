"""StatsNBA: NBA player stats loading, feature engineering, and prediction."""

import importlib.resources as pkg_resources
import json
import warnings
from datetime import date, datetime, timedelta
from io import StringIO

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
    abbreviations,
    combo_props,
    feature_filter,
    get_ev,
    get_mlb_pitchers,
    get_odds,
    remove_accents,
    stat_cv,
    stat_dist,
)
from sportstradamus.helpers.io import read_gamelog, write_gamelog
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import nba_client
from sportstradamus.stats.base import (
    ComboSpec,
    Stats,
    archive,
    clean_data,
    fetch_upcoming_games,
    scale_team_volume_to_budget,
)
from sportstradamus.stats.nba_client import NBAStatsError

# Days since last fetch beyond which play-in and playoff logs are re-pulled (off-season catch-up)
_CATCHUP_THRESHOLD_DAYS: int = 150
# Retention window: keep 4 seasons of logs to bound file growth
_GAMELOG_RETENTION_DAYS: int = 1461

# Cleaning the Glass (NBA) dated-snapshot roots + join schema. The key/meta lists
# are pinned from a real authenticated CTG capture (see docs/cleaningtheglass.md);
# they stay empty until the endpoint catalog + a snapshot exist, which makes the
# feature hooks below return None and NBA features stay gamelog-only.
_CTG_PLAYER_DIR = "player_data/NBA/cleaningtheglass"
_CTG_TEAM_DIR = "team_data/NBA/cleaningtheglass"
_CTG_PLAYER_KEY_COL = ""
_CTG_PLAYER_META_COLS: frozenset[str] = frozenset()
_CTG_TEAM_KEY_COL = ""
_CTG_TEAM_META_COLS: frozenset[str] = frozenset()

# DFS scoring weights shared by PrizePicks and Underdog NBA fantasy (settled gamelog formula in
# _compute_derived_player_stats; its BLST*3 term is carried as separate BLK/STL components
# because they are distinct book markets with their own correlation cross terms).
NBA_FANTASY_WEIGHTS = (("PTS", 1), ("REB", 1.2), ("AST", 1.5), ("BLK", 3), ("STL", 3), ("TOV", -1))

# The three components sportsbooks price thinly or not at all, measured over the
# 2025-26 archive: NBA `TOV` reaches a real book on 15.7% of its rows (325 against
# PTS's 20712) and WNBA quotes none of the three — `BLK` has no archived row at
# any book, `STL` and `TOV` only pick'em platforms. Under all-or-nothing admission
# that one gap denies the whole fantasy board, so they fall back to the player's
# trailing rate (`Stats._assumed_component`). Shared with WNBA by inheritance: the
# fallback only fires where no sportsbook priced the component, which on NBA's
# densely quoted `BLK`/`STL` is the rare row rather than the rule.
NBA_ASSUMABLE_FANTASY_COMPONENTS = frozenset({"TOV", "BLK", "STL"})


class StatsNBA(Stats):
    """NBA player statistics: game log loading, feature engineering, and prediction."""

    def __init__(self):
        super().__init__()
        self.league = "NBA"
        self.positions = ["P", "C", "F", "W", "B"]
        self._set_season_start(datetime(2025, 10, 21).date())
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

    def load(self) -> None:
        """Hydrate players, gamelog, and teamlog from the on-disk cache written by update()."""
        nba_data = read_gamelog("nba")
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

    def _comp_season_pair(self) -> tuple[str, str]:
        """Return the ``(prior, current)`` ``self.players`` keys for comp assembly.

        NBA keys are ``"YYYY-YY"`` strings; the WNBA subclass overrides this to
        return integer-year keys.
        """
        prior = "-".join([str(int(n) - 1) for n in self.season.split("-")])
        return prior, self.season

    def _current_season_key(self, target_game_date: date) -> str:
        """Return the ``self.players`` key for the season containing ``target_game_date``.

        NBA seasons span Oct–Jun and ``self.players`` is keyed by the
        ``"YYYY-YY"`` notation. Dates in Oct–Dec belong to the season that
        started that calendar year; Jan–Sep dates belong to the season that
        started the prior year.
        """
        year = target_game_date.year
        if target_game_date.month >= 10:
            return f"{year}-{(year + 1) % 100:02d}"
        return f"{year - 1}-{year % 100:02d}"

    def _ctg_season(self, date) -> int:
        """CTG's integer season label (starting year) for a game ``date``."""
        return int(self._current_season_key(date)[:4])

    def _join_fp_player_features(self, date):
        """NBA hook: Cleaning the Glass per-player season-to-date features as of ``date``."""
        return self._collector_asof_features(
            _CTG_PLAYER_DIR,
            _CTG_PLAYER_KEY_COL,
            _CTG_PLAYER_META_COLS,
            self._ctg_season(date),
            date,
        )

    def _join_fp_team_features(self, date):
        """NBA hook: CTG team-grain features as of ``date`` (defense split deferred)."""
        team = self._collector_asof_features(
            _CTG_TEAM_DIR, _CTG_TEAM_KEY_COL, _CTG_TEAM_META_COLS, self._ctg_season(date), date
        )
        return team, None

    def _player_seasons_through(self, target_game_date: date) -> dict:
        """Merge ``self.players`` seasons whose key is ``<=`` ``target_game_date``'s season.

        Returns a ``{team: {player_name: stats_dict}}`` mapping ready for
        :meth:`build_comp_profile`. Iterates oldest → newest so the current
        season's roster wins on dict-update conflicts.
        """
        cutoff = self._current_season_key(target_game_date)
        merged: dict = {}
        for season_key in sorted(self.players.keys()):
            if season_key > cutoff:
                continue
            for team, roster in self.players[season_key].items():
                merged.setdefault(team, {}).update(roster)
        return merged

    def _compute_comps(self, target_game_date: date | None = None) -> None:
        """Build comps from loaded data at runtime (no JSON I/O).

        When ``target_game_date`` is provided, the player pool is bounded to
        seasons whose key (NBA "YYYY-YY") is ``<=`` the target date's season,
        so a 2022 training row's comps no longer pool with 2024+ rookies.
        ``self.playerProfile`` is already date-bounded by
        :meth:`base_profile` upstream (it filters ``short_gamelog`` to the
        300-day window ending at ``date``), so this fix only needs to gate
        the per-season roster set that feeds :meth:`build_comp_profile`.
        Today() default preserves inference / cron behavior.
        """
        with open(pkg_resources.files(data) / "config" / "playerCompStats.json") as f:
            stats = json.load(f)

        league_weights = stats[self.league]
        all_features = set()
        for pos_weights in league_weights.values():
            all_features.update(pos_weights.keys())
        all_features = list(all_features)

        if target_game_date is None:
            target_game_date = datetime.today().date()
        playerList = self._player_seasons_through(target_game_date)
        playerProfile, playerDict = self.build_comp_profile(playerList=playerList)
        playerProfile = playerProfile[
            [f for f in all_features if f in playerProfile.columns]
        ].replace([np.nan, np.inf, -np.inf], 0)

        comps = {}
        for position in self.positions:
            pos_weights = league_weights[position]
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

    def _set_season_start(self, day):
        self.season_start = day
        self.season = f"{day.year}-{(day.year + 1) % 100:02d}"

    def _update(self) -> None:
        """Update data from the web API."""
        self._clean_player_positions()

        latest_date = self._latest_gamelog_date()
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

        playerBios, shotData, synergy = self._fetch_league_endpoints()
        self._build_player_profiles(player_df, playerBios, shotData, synergy)

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
        self.upcoming_games = fetch_upcoming_games("00", self.season[:4], today)

        params = {
            "season_nullable": self.season,
            "league_id_nullable": "00",
            "date_from_nullable": latest_date.strftime("%m/%d/%Y"),
            "date_to_nullable": today.strftime("%m/%d/%Y"),
        }
        include_playin = (today.month == 4) or (today - latest_date).days > _CATCHUP_THRESHOLD_DAYS
        include_playoffs = (4 <= today.month <= 6) or (
            today - latest_date
        ).days > _CATCHUP_THRESHOLD_DAYS
        nba_gamelog, adv_gamelog, usg_gamelog, teamlog, sco_teamlog, adv_teamlog = (
            self._fetch_game_logs(params, include_playin, include_playoffs)
        )

        adv_teamlog_idx = {(g["TEAM_ID"], g["GAME_ID"]): g for g in adv_teamlog}
        sco_teamlog_idx = {(g["TEAM_ID"], g["GAME_ID"]): g for g in sco_teamlog}
        adv_gamelog_idx = {(g["PLAYER_ID"], g["GAME_ID"]): g for g in adv_gamelog}
        usg_gamelog_idx = {(g["PLAYER_ID"], g["GAME_ID"]): g for g in usg_gamelog}

        self._merge_team_logs(teamlog, adv_teamlog_idx, sco_teamlog_idx)
        team_df = pd.DataFrame(self._pair_team_rows(teamlog))
        if not team_df.empty:
            self.teamlog = (
                pd.concat([team_df.reindex(columns=self.teamlog.columns), self.teamlog])
                .sort_values(self.log_strings["date"])
                .reset_index(drop=True)
            )

        # Drop records with incomplete advanced stats so they can be re-fetched
        if "OFF_RATING" in self.gamelog.columns:
            self.gamelog = self.gamelog.dropna(subset=["OFF_RATING"])

        nba_df = pd.DataFrame(
            self._assemble_player_rows(nba_gamelog, adv_gamelog_idx, usg_gamelog_idx, position_map)
        )
        if not nba_df.empty:
            self._enrich_team_markets(
                nba_df, date_col=self.log_strings["date"], team_col="TEAM_ABBREVIATION"
            )
            self.gamelog = (
                pd.concat([nba_df.reindex(columns=self.gamelog.columns), self.gamelog])
                .sort_values(self.log_strings["date"])
                .reset_index(drop=True)
            )

        # Remove old games to prevent file bloat
        four_years_ago = today - timedelta(days=_GAMELOG_RETENTION_DAYS)
        self.gamelog = self.gamelog[
            pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date >= four_years_ago
        ]
        self.gamelog.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last", inplace=True)
        self.teamlog = self.teamlog[
            pd.to_datetime(self.teamlog[self.log_strings["date"]]).dt.date >= four_years_ago
        ]
        self.teamlog.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)

        self._canonicalize_team_abbrevs(self.gamelog)
        self._canonicalize_team_abbrevs(self.teamlog)

        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog["PLAYER_NAME"] = self.gamelog["PLAYER_NAME"].apply(remove_accents)
            self._enrich_team_markets(
                self.gamelog, date_col=self.log_strings["date"], team_col="TEAM_ABBREVIATION"
            )
            self.gamelog.loc[:, self.log_strings["position"]] = self.gamelog.apply(
                lambda x: (
                    self.players.get(x.SEASON_YEAR, {})
                    .get(x.TEAM_ABBREVIATION, {})
                    .get(x.PLAYER_NAME, {})
                    .get("POS", x.POS)
                ),
                axis=1,
            )

        write_gamelog("nba", self.gamelog, self.teamlog, self.players)

    def _resolve_pos_from_history(self, player):
        """First valid POS for ``player`` scanning seasons newest-first, else None."""
        for season in reversed(list(self.players.keys())):
            for roster in self.players[season].values():
                pos = roster.get(player, {}).get("POS")
                if isinstance(pos, str) and pos in self.positions:
                    return pos
        return None

    def _clean_player_positions(self):
        """Repair non-string POS values left by a prior fillna(0) bug."""
        for season in self.players:
            for team in self.players[season]:
                for player, info in self.players[season][team].items():
                    if not isinstance(info.get("POS"), str):
                        info["POS"] = self._resolve_pos_from_history(player)

        if not self.gamelog.empty:
            pos_col = self.log_strings["position"]
            bad_pos = ~self.gamelog[pos_col].apply(lambda x: isinstance(x, str))
            if bad_pos.any():
                self.gamelog.loc[bad_pos, pos_col] = self.gamelog.loc[bad_pos].apply(
                    lambda x: (
                        self.players.get(x.SEASON_YEAR, {})
                        .get(x.TEAM_ABBREVIATION, {})
                        .get(x.PLAYER_NAME, {})
                        .get("POS")
                    ),
                    axis=1,
                )

    def _latest_gamelog_date(self):
        """Earliest NaN-row date (refetch point) or the latest complete date."""
        if self.gamelog.empty:
            return self.season_start
        nanlog = self.gamelog.loc[self.gamelog.isnull().values.any(axis=1)]
        if not nanlog.empty:
            latest = pd.to_datetime(nanlog[self.log_strings["date"]]).min().date()
        else:
            latest = pd.to_datetime(self.gamelog[self.log_strings["date"]]).max().date()
        return max(latest, self.season_start)

    def _fetch_league_endpoints(self):
        """Pull the season-level bio / shot-location / synergy endpoints.

        Returns ``(playerBios, shotData, synergy)`` as DataFrames / a dict of
        play-type DataFrames; on an API error every result is its empty shape.
        """

        def _synergy(play_type):
            return nba_client.fetch(
                nba.synergyplaytypes.SynergyPlayTypes,
                league_id="00",
                season=self.season,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame",
                player_or_team_abbreviation="P",
                type_grouping_nullable="offensive",
                play_type_nullable=play_type,
            ).get_dict()["resultSets"][0]

        synergy_types = [
            ("POST", "Postup"),
            ("HANDOFF", "Handoff"),
            ("ISO", "Isolation"),
            ("PR", "PRBallHandler"),
            ("SPOT", "Spotup"),
            ("PUTBACK", "OffRebound"),
        ]
        try:
            with tqdm(total=8, desc="NBA league endpoints", unit="endpoint", leave=False) as pbar:
                pbar.set_postfix_str("PlayerBioStats")
                playerBios = nba_client.fetch(
                    nba.leaguedashplayerbiostats.LeagueDashPlayerBioStats, season=self.season
                ).get_normalized_dict()["LeagueDashPlayerBioStats"]
                pbar.update(1)
                pbar.set_postfix_str("ShotLocations")
                shotData = nba_client.fetch(
                    nba.leaguedashplayershotlocations.LeagueDashPlayerShotLocations,
                    season=self.season,
                    season_type_all_star="Regular Season",
                    distance_range="By Zone",
                    per_mode_detailed="PerGame",
                ).get_dict()["resultSets"]
                pbar.update(1)
                synergy_raw = {}
                for name, play_type in synergy_types:
                    pbar.set_postfix_str(f"Synergy:{play_type}")
                    synergy_raw[name] = _synergy(play_type)
                    pbar.update(1)
        except NBAStatsError:
            playerBios = []
            shotData = {"rowSet": [], "headers": [{}, {"columnNames": []}]}
            synergy_raw = {name: {"rowSet": [], "headers": []} for name, _ in synergy_types}

        playerBios = pd.DataFrame(playerBios)
        shotData = pd.DataFrame(shotData["rowSet"], columns=shotData["headers"][1]["columnNames"])
        synergy = {
            name: pd.DataFrame(raw["rowSet"], columns=raw["headers"])
            for name, raw in synergy_raw.items()
        }
        return playerBios, shotData, synergy

    @staticmethod
    def _enrich_shot_data(shotData, synergy):
        """Merge synergy play-type rates onto shot data and derive zone PCT/PPP."""
        for name, df in synergy.items():
            shotData = shotData.merge(
                df[["PLAYER_ID", "POSS_PCT", "PPP"]].rename(
                    columns={"POSS_PCT": f"{name}_PCT", "PPP": f"{name}_PPP"}
                ),
                on="PLAYER_ID",
                how="outer",
            )
        shotData = shotData.fillna(0)
        fga = shotData["FGA"].iloc[:, :3].sum(axis=1) + shotData["FGA"].iloc[:, 5::2].sum(axis=1)
        shotData["ITP_PCT"] = shotData["FGA"].iloc[:, :2].sum(axis=1) / fga
        shotData["ITP_PPP"] = (
            shotData["FGM"].iloc[:, :2].sum(axis=1) / shotData["FGA"].iloc[:, :2].sum(axis=1) * 2
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
        return shotData.fillna(0)

    def _build_player_profiles(self, player_df, playerBios, shotData, synergy):
        """Merge bios + shot/synergy data into per-team player profiles on self.players."""
        if playerBios.empty:
            self.players[self.season] = self.players.get(
                "-".join([str(int(x) - 1) for x in self.season.split("-")]), {}
            )
            return

        shotData = self._enrich_shot_data(shotData, synergy)
        playerBios.PLAYER_NAME = playerBios.PLAYER_NAME.apply(remove_accents)
        shotData.PLAYER_NAME = shotData.PLAYER_NAME.apply(remove_accents)
        playerBios = playerBios.merge(
            shotData, on="PLAYER_NAME", suffixes=(None, "_y"), how="outer"
        ).fillna(0)
        player_df = player_df.merge(
            playerBios, on=["PLAYER_NAME", "TEAM_ABBREVIATION"], suffixes=(None, "_x"), how="outer"
        )
        # For traded players the CSV has the old team and bios the new team, producing two
        # incomplete rows; combine by first non-NaN per column, bios rows (non-NaN PLAYER_ID)
        # sorted first so their current TEAM_ABBREVIATION wins.
        player_df = (
            player_df.sort_values("PLAYER_ID", na_position="last")
            .groupby("PLAYER_NAME", sort=False)
            .first()
            .reset_index()
        )
        player_df.PLAYER_WEIGHT = player_df.PLAYER_WEIGHT.astype(float)
        player_df.POS = player_df.POS.str[0]
        player_df.index = player_df.PLAYER_NAME
        nan_pos = player_df.POS.isna()
        if nan_pos.any():
            for player_name in player_df.loc[nan_pos].index:
                player_df.loc[player_name, "POS"] = self._resolve_pos_from_history(player_name)
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
        player_df = {level: player_df.xs(level).T.to_dict() for level in player_df.index.levels[0]}
        if self.season in self.players:
            self.players[self.season] = {
                team: players | player_df.get(team, {})
                for team, players in self.players[self.season].items()
            }
        else:
            self.players[self.season] = player_df

    def _fetch_game_logs(self, params, include_playin, include_playoffs):
        """Pull player/team game logs (base/advanced/usage/scoring) for the window.

        Returns ``(nba_gamelog, adv_gamelog, usg_gamelog, teamlog, sco_teamlog,
        adv_teamlog)``; all empty on API error (a partial pull would mix
        populated and empty logs).
        """

        def _player_logs(measure=None):
            extra = {} if measure is None else {"measure_type_player_game_logs_nullable": measure}
            return nba_client.fetch(
                nba.playergamelogs.PlayerGameLogs, **(params | extra)
            ).get_normalized_dict()["PlayerGameLogs"]

        def _team_logs(measure=None):
            extra = {} if measure is None else {"measure_type_player_game_logs_nullable": measure}
            return nba_client.fetch(
                nba.teamgamelogs.TeamGameLogs, **(params | extra)
            ).get_normalized_dict()["TeamGameLogs"]

        nba_gamelog, adv_gamelog, usg_gamelog = [], [], []
        teamlog, sco_teamlog, adv_teamlog = [], [], []
        # (postfix label, target list, log fn, measure)
        base_calls = [
            ("PlayerGameLogs:Base", nba_gamelog, _player_logs, None),
            ("PlayerGameLogs:Advanced", adv_gamelog, _player_logs, "Advanced"),
            ("PlayerGameLogs:Usage", usg_gamelog, _player_logs, "Usage"),
            ("TeamGameLogs:Base", teamlog, _team_logs, None),
            ("TeamGameLogs:Scoring", sco_teamlog, _team_logs, "Scoring"),
            ("TeamGameLogs:Advanced", adv_teamlog, _team_logs, "Advanced"),
        ]
        season_types = [None]
        if include_playin:
            season_types.append("PlayIn")
        if include_playoffs:
            season_types.append("Playoffs")
        try:
            with tqdm(
                total=len(season_types) * len(base_calls),
                desc="NBA player/team gamelogs",
                unit="endpoint",
                leave=False,
            ) as pbar:
                for season_type in season_types:
                    if season_type is not None:
                        params.update({"season_type_nullable": season_type})
                    prefix = "" if season_type is None else f"{season_type}:"
                    for label, target, fn, measure in base_calls:
                        pbar.set_postfix_str(f"{prefix}{label}")
                        target.extend(fn(measure))
                        pbar.update(1)
        except NBAStatsError:
            return [], [], [], [], [], []
        return nba_gamelog, adv_gamelog, usg_gamelog, teamlog, sco_teamlog, adv_teamlog

    @staticmethod
    def _merge_team_logs(teamlog, adv_idx, sco_idx):
        """Fold advanced + scoring team-log rows into the base rows in place."""
        for i, row in enumerate(teamlog):
            key = (row["TEAM_ID"], row["GAME_ID"])
            if adv := adv_idx.get(key):
                row = row | adv
            if sco := sco_idx.get(key):
                row = row | sco
            teamlog[i] = row

    def _pair_team_rows(self, teamlog):
        """Pair consecutive team rows into opponent-annotated records."""
        team_cols = self.teamlog.columns
        team_key = self.log_strings["team"]
        team_df = []
        for team1, team2 in zip(*[iter(teamlog)] * 2, strict=False):
            team1.update(
                {
                    "FTR": (team1["FTM"] / team1["FGA"]) if team1["FGA"] > 0 else 0,
                    "BLK_RATIO": (team1["BLK"] / team1["BLKA"]) if team1["BLKA"] > 0 else 0,
                    "OPP": team2[team_key],
                }
            )
            team2.update(
                {
                    "FTR": (team2["FTM"] / team2["FGA"]) if team2["FGA"] > 0 else 0,
                    "BLK_RATIO": (team2["BLK"] / team2["BLKA"]) if team2["BLKA"] > 0 else 0,
                    "OPP": team1[team_key],
                }
            )
            team1.update({"OPP_" + k: v for k, v in team2.items() if "OPP_" + k in team_cols})
            team2.update({"OPP_" + k: v for k, v in team1.items() if "OPP_" + k in team_cols})
            team_df.append(team1)
            team_df.append(team2)
        return team_df

    def _resolve_player_position(self, game, position_map):
        """Resolve a player's POS from history, falling back to a live API call."""
        name = game["PLAYER_NAME"]
        team_abbr = game["TEAM_ABBREVIATION"]
        position = None
        for season in reversed(list(self.players.keys())):
            if not isinstance(position, str):
                position = self.players[season].get(team_abbr, {}).get(name, {}).get("POS")
            if not isinstance(position, str):
                for team in self.players[season]:
                    position = self.players[season][team].get(name, {}).get("POS")
        if not isinstance(position, str):
            position = None
        if position is None:
            try:
                position = (
                    nba_client.fetch(
                        nba.commonplayerinfo.CommonPlayerInfo, player_id=game["PLAYER_ID"]
                    )
                    .get_normalized_dict()["CommonPlayerInfo"][0]
                    .get("POSITION")
                )
            except NBAStatsError:
                position = "Forward-Guard"
            position = position_map.get(position, "W")
        return position

    @staticmethod
    def _skip_game(game, included_games):
        return (
            game["MIN"] < 1
            or not game["TEAM_ABBREVIATION"]
            or (game["PLAYER_ID"], game["GAME_ID"]) in included_games
        )

    @staticmethod
    def _compute_derived_player_stats(game):
        """Add PRA/PR/PA/RA/BLST, fantasy scores, ratios, and per-48 rates to ``game``."""
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

    def _assemble_player_rows(self, nba_gamelog, adv_idx, usg_idx, position_map):
        """Build per-player gamelog rows, resolving POS and derived stats per game."""
        nba_df = []
        included_games = set(
            self.gamelog[["PLAYER_ID", "GAME_ID"]].itertuples(index=False, name=None)
        )
        for game in tqdm(nba_gamelog, desc="Getting NBA stats", unit="player"):
            if self._skip_game(game, included_games):
                continue
            included_games.add((game["PLAYER_ID"], game["GAME_ID"]))
            game["PLAYER_NAME"] = remove_accents(game["PLAYER_NAME"])
            team_abbr = game["TEAM_ABBREVIATION"]
            adv_game = adv_idx.get((game["PLAYER_ID"], game["GAME_ID"]))
            usg_game = usg_idx.get((game["PLAYER_ID"], game["GAME_ID"]))

            self.players[self.season].setdefault(team_abbr, {})
            existing_pos = (
                self.players[self.season][team_abbr].get(game["PLAYER_NAME"], {}).get("POS")
            )
            if game["PLAYER_NAME"] not in self.players[self.season][team_abbr] or not isinstance(
                existing_pos, str
            ):
                self.players[self.season][team_abbr].setdefault(game["PLAYER_NAME"], {})["POS"] = (
                    self._resolve_player_position(game, position_map)
                )

            game["POS"] = (
                self.players[self.season][team_abbr].get(game["PLAYER_NAME"], {}).get("POS")
            )
            game["HOME"] = "vs." in game["MATCHUP"]
            teams = game["MATCHUP"].replace("vs.", "@").split(" @ ")
            for team in teams:
                if team != team_abbr:
                    game["OPP"] = team

            self._compute_derived_player_stats(game)
            if adv_game:
                game.update(adv_game)
            if usg_game:
                game.update(usg_game)
            nba_df.append(game)
        return nba_df

    def _canonicalize_team_abbrevs(self, df):
        """Map alternate team codes (UTAH/NOP/GS/NY/SA) to their canonical forms."""
        fixups = {"UTAH": "UTA", "NOP": "NO", "GS": "GSW", "NY": "NYK", "SA": "SAS"}
        team, opp = self.log_strings["team"], self.log_strings["opponent"]
        df[team] = df[team].replace(fixups)
        df[opp] = df[opp].replace(fixups)

    def get_volume_stats(self, offers, date=datetime.today().date()):
        """Predict minutes and normalize projections against the team-game minute budget.

        Scores each player in ``offers`` with the trained MIN model, then rescales
        the per-team projections so they sum to the expected team-game minute budget
        (regulation + OT premium). Writes proj columns back into self.playerProfile
        in place.
        """
        market = "MIN"
        if not self.load_volume_model_params(offers, market, date):
            return []

        # Budget parameters derived from historical gamelogs (methodology in
        # scale_team_volume_to_budget docstring):
        #   typical_rotation : median players logging >3 min per team per game
        #   ot_rate          : fraction of games going to OT (6% NBA, 3.9% WNBA)
        #   avg_unmodeled_min: mean min for players ranked beyond the modeled tier
        #   per_player_floor : 5th-pct min for top-7 tier players
        #   per_player_cap   : regulation max + one 5-min OT period (hard rule)
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

        # The budget kwargs are computed above from league-specific physics
        # (NBA/WNBA basketball minutes with an OT geometric series; NHL uses a
        # different ice-time model), so only the call shape repeats across
        # leagues. Absorbing it into a base method would be a pure forwarder,
        # which the no-wrapper rule bans -- the parallel call stays, justified.
        # pylint: disable=duplicate-code
        scale_team_volume_to_budget(
            self.playerProfile,
            market,
            budget_mean=budget_mean,
            typical_rotation=typical_rotation,
            avg_unmodeled_min=avg_unmodeled_min,
            per_player_floor=per_player_floor,
            per_player_cap=per_player_cap,
        )
        # pylint: enable=duplicate-code

        self.playerProfile.fillna(0, inplace=True)
        return None

    def _fantasy_combo_spec(self, market, player):
        if market in ("fantasy points prizepicks", "fantasy points underdog"):
            return ComboSpec(
                marginals=NBA_FANTASY_WEIGHTS,
                assumable=NBA_ASSUMABLE_FANTASY_COMPONENTS,
            )
        return None

    def check_combo_markets(self, market, player, date=datetime.today().date()):
        """Return an EV estimate for derived markets (combo, OREB/DREB split, fantasy) from archive."""
        player_games = self.short_gamelog.loc[
            self.short_gamelog[self.log_strings["player"]] == player
        ]
        cv = stat_cv.get(self.league, {}).get(market, 1)
        dist = stat_dist.get(self.league, {}).get(market, "Gamma")
        if not isinstance(date, str):
            date = date.strftime("%Y-%m-%d")
        if market in combo_props:
            ev = self._combo_market_ev(market, date, player, dist, cv)
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
                v, subline, sub_cv, sub_dist = self._submarket_ev(submarket, date, player, dist, cv)
                contribution, from_book = self._fantasy_default_contribution(
                    submarket, weight, v, subline, sub_cv, sub_dist, player_games
                )
                ev += contribution
                book_odds |= from_book
            if not book_odds:
                ev = 0
        else:
            ev = 0
        return 0 if np.isnan(ev) else ev
