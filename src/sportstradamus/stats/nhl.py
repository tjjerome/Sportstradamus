"""StatsNHL: NHL player stats loading, feature engineering, and prediction."""

import importlib.resources as pkg_resources
import json
import os.path
import pickle
import warnings
from datetime import date, datetime, timedelta
from time import sleep

import line_profiler
import numpy as np
import pandas as pd
from scipy.stats import iqr, norm, poisson
from sklearn.neighbors import BallTree
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import (
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
from sportstradamus.helpers.io import write_gamelog
from sportstradamus.spiderLogger import logger
from sportstradamus.stats.base import (
    Stats,
    archive,
    clean_data,
    scale_team_volume_to_budget,
    scraper,
)

# Days of gamelog history to retain (~4 NHL seasons); older rows pruned on each update.
_GAMELOG_RETENTION_DAYS = 1431
# Days since season start before triggering a full gamelog re-enrichment pass.
_STALE_SEASON_DAYS = 300


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
        self.season_start = datetime(2025, 10, 7).date()
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
        with open(pkg_resources.files(data) / "config" / "playerCompStats.json") as infile:
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

        filepath = pkg_resources.files(data) / "leagues" / "nhl" / "comps.json"
        with open(filepath, "w") as outfile:
            json.dump(comps, outfile, indent=4)

    def _current_season_key(self, target_game_date: date) -> int:
        """Return the ``self.players`` key for the season containing ``target_game_date``.

        NHL seasons run Oct–Jun and ``self.players`` is keyed by the season
        *start* year (int). Oct–Dec dates belong to that year's season;
        Jan–Sep dates belong to the prior year's season.
        """
        if target_game_date.month >= 10:
            return target_game_date.year
        return target_game_date.year - 1

    def _player_seasons_through(self, target_game_date: date) -> dict:
        """Merge ``self.players`` seasons ``<=`` ``target_game_date``'s season.

        Returns a flat ``{player_id: stats_dict}`` mapping ready for
        :meth:`build_comp_profile`. Iterates oldest → newest so the current
        season's roster wins on dict-update conflicts (matches the
        :meth:`update_player_comps` ``prior.update(current)`` order).
        """
        cutoff = self._current_season_key(target_game_date)
        merged: dict = {}
        for season_key in sorted(self.players.keys()):
            if season_key > cutoff:
                continue
            merged.update(self.players[season_key])
        return merged

    def _compute_comps(self, target_game_date: date | None = None) -> None:
        """Build comps from loaded data at runtime (no JSON I/O).

        When ``target_game_date`` is provided, the player pool is bounded to
        ``self.players`` seasons that started on or before the target date's
        season-start year, so a 2022 training row's comps no longer pool
        with 2024+ rookies. ``self.playerProfile`` is already date-bounded
        by :meth:`base_profile` upstream (it filters ``short_gamelog`` to
        the 300-day window ending at ``date``), so the gate here is on the
        per-season roster set that feeds :meth:`build_comp_profile`.
        Today() default preserves inference / cron behavior.
        """
        with open(pkg_resources.files(data) / "config" / "playerCompStats.json") as f:
            stats = json.load(f)

        if target_game_date is None:
            target_game_date = datetime.today().date()
        playerDict = self._player_seasons_through(target_game_date)
        playerProfile, all_players, id_to_name = self.build_comp_profile(playerDict=playerDict)
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
        game_df = scraper.get_csv(
            f"https://moneypuck.com/moneypuck/playerData/games/{season}/{gameId}.csv"
        )
        if game_df.empty:
            return gamelog, teamlog
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
                    teamlog.append(
                        self._team_level_row(player, gameId, gameDate, team, opponent, home)
                    )
                else:
                    gamelog.append(
                        self._skater_row(
                            player,
                            gameId,
                            gameDate,
                            team,
                            opponent,
                            home,
                            win,
                            game_df,
                            pp_df,
                            team_map,
                        )
                    )

        return gamelog, teamlog

    def _team_level_row(self, player, gameId, gameDate, team, opponent, home):
        """Build one Team Level (on-ice aggregate) teamlog row from a moneypuck row."""
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
                    (float(player["OffIce_A_freeze"]) - float(player["OffIce_A_xFreeze"]))
                    / shotsAgainst
                )
                if shotsAgainst
                else 0,
                "Rebound": (
                    (float(player["OffIce_A_rebounds"]) - float(player["OffIce_A_xRebounds"]))
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
        return n | stats

    def _skater_row(
        self, player, gameId, gameDate, team, opponent, home, win, game_df, pp_df, team_map
    ):
        """Build one skater/goalie gamelog row (box score + on-ice rates + fantasy points)."""
        n = {
            "gameId": gameId,
            "gameDate": gameDate,
            "team": team,
            "opponent": opponent,
            "opponent goalie": remove_accents(
                game_df.loc[(game_df.position == "G") & (game_df.team != team), "playerName"].iat[0]
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
            "sogBS": float(player["I_F_shotsOnGoal"]) + float(player["I_A_blockedShotAttempts"]),
            "goals": float(player["I_F_goals"]),
            "assists": float(player["I_F_primaryAssists"]) + float(player["I_F_secondaryAssists"]),
            "hits": float(player["I_F_hits"]),
            "faceOffWins": float(player["I_F_faceOffsWon"]),
            "timeOnIce": float(player["I_F_iceTime"]) / 60,
            "saves": float(player["OnIce_A_savedShotsOnGoal"]),
            "shotsAgainst": float(player["OnIce_A_shotsOnGoal"]),
            "goalsAgainst": float(player["OnIce_A_goals"]),
        }
        if player["playerName"] in pp_df["playerName"].to_list():
            stats["powerPlayPoints"] = float(
                pp_df.loc[pp_df["playerName"] == player["playerName"]]["I_F_points"].iat[0]
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
                    (stats["goals"] - float(player["I_F_flurryScoreVenueAdjustedxGoals"])) / shots
                )
                if shots
                else 0,
                "Fenwick": float(player["OnIce_unblockedShotAttempts_For_Percentage"]),
                "TimeShare": stats["timeOnIce"]
                / (
                    float(game_df.loc[game_df["playerName"] == team, "OffIce_F_iceTime"].iat[0])
                    / 60
                ),
                "ShotShare": stats["shots"]
                / float(game_df.loc[game_df["playerName"] == team, "OffIce_F_shotsOnGoal"].iat[0]),
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
                    (float(player["OnIce_A_rebounds"]) - float(player["OnIce_A_xRebounds"]))
                    / shotsAgainst
                )
                if shotsAgainst
                else 0,
                "RG": (
                    (float(player["OnIce_A_reboundGoals"]) - float(player["OnIce_A_reboundxGoals"]))
                    / float(player["OnIce_A_rebounds"])
                )
                if float(player["OnIce_A_rebounds"])
                else 0,
            }
        )
        return n | stats

    def _update(self):
        """Updates the NHL skater and goalie data."""
        latest_date = self.season_start
        if not self.gamelog.empty:
            latest_date = pd.to_datetime(self.gamelog["gameDate"]).max().date() + timedelta(days=1)
        today = datetime.today().date()

        ids = self._collect_game_ids(latest_date, today)
        nhl_gamelog, nhl_teamlog = self._parse_new_games(ids, today)
        self._merge_new_gamelogs(nhl_gamelog, nhl_teamlog)
        self._fetch_upcoming_games_nhl(today)

        player_df = self._fetch_player_bios_nhl()
        skater_df = self._fetch_skater_summary(player_df)
        goalie_df = self._fetch_goalie_summary(player_df)
        self.players[self.season_start.year] = skater_df.to_dict("index") | goalie_df.to_dict(
            "index"
        )

        four_years_ago = today - timedelta(days=_GAMELOG_RETENTION_DAYS)
        self.gamelog = self.gamelog[
            pd.to_datetime(self.gamelog["gameDate"]).dt.date >= four_years_ago
        ]
        self.gamelog.drop_duplicates(subset=["gameId", "playerName"], keep="last", inplace=True)
        self.teamlog = self.teamlog[
            pd.to_datetime(self.teamlog["gameDate"]).dt.date >= four_years_ago
        ]
        self.teamlog.drop_duplicates(subset=["gameId", "team"], keep="last", inplace=True)

        if (
            self.season_start < datetime.today().date() - timedelta(days=_STALE_SEASON_DAYS)
            or clean_data
        ):
            self.gamelog["playerName"] = self.gamelog["playerName"].apply(remove_accents)
            self._enrich_team_markets(self.gamelog, date_col="gameDate", team_col="team")

        write_gamelog("nhl", self.gamelog, self.teamlog, self.players)

    def _collect_game_ids(self, latest_date, today):
        """Walk the NHL schedule day-by-day collecting (gameId, date) since latest_date."""
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
        return ids

    def _parse_new_games(self, ids, today):
        """Parse each completed game id into (gamelog, teamlog) row lists."""
        nhl_gamelog = []
        nhl_teamlog = []
        for gameId, game_date_str in tqdm(ids, desc="Getting NHL Stats"):
            if datetime.strptime(game_date_str, "%Y-%m-%d").date() < today:
                gamelog, teamlog = self.parse_game(gameId, game_date_str)
                if type(gamelog) is list:
                    nhl_gamelog.extend(gamelog)
                if type(teamlog) is list:
                    nhl_teamlog.extend(teamlog)
        return nhl_gamelog, nhl_teamlog

    def _merge_new_gamelogs(self, nhl_gamelog, nhl_teamlog):
        """Dedup new player/team rows, enrich markets, and prepend to the stored logs."""
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
            self._enrich_team_markets(nhl_df, date_col="gameDate", team_col="team")
        nhl_teamlog_df = pd.DataFrame(nhl_teamlog).fillna(0).infer_objects(copy=False)
        if not nhl_teamlog_df.empty:
            nhl_teamlog_df.drop_duplicates(subset=["gameId", "team"], keep="last", inplace=True)
        self.gamelog = (
            pd.concat([nhl_df, self.gamelog]).sort_values("gameDate").reset_index(drop=True)
        )
        self.teamlog = (
            pd.concat([nhl_teamlog_df, self.teamlog]).sort_values("gameDate").reset_index(drop=True)
        )

    def _fetch_upcoming_games_nhl(self, today):
        """Populate self.upcoming_games with opponent + predicted goalie from dobbersports."""
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

    def _fetch_player_bios_nhl(self):
        """All-players bio lookup (height/weight/bmi/age/position) from moneypuck."""
        player_df = scraper.get_csv(
            "https://moneypuck.com/moneypuck/playerData/playerBios/allPlayersLookup.csv"
        )
        if player_df.empty:
            return player_df
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
        return player_df

    def _fetch_skater_summary(self, player_df):
        """Per-60 / per-attempt skater rates from the moneypuck season summary."""
        skater_df = scraper.get_csv(
            f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{self.season_start.year}/regular/skaters.csv"
        )
        if skater_df.empty:
            return pd.DataFrame()
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
        skater_df["shotAttempts"] = skater_df["I_F_shotAttempts"] / skater_df["icetime"] * 60 * 60
        skater_df["hits"] = skater_df["I_F_hits"] / skater_df["icetime"] * 60 * 60
        skater_df["takeaways"] = skater_df["I_F_takeaways"] / skater_df["icetime"] * 60 * 60
        skater_df["giveaways"] = skater_df["I_F_giveaways"] / skater_df["icetime"] * 60 * 60
        skater_df["assists"] = (
            (skater_df["I_F_primaryAssists"] + skater_df["I_F_secondaryAssists"])
            / skater_df["icetime"]
            * 60
            * 60
        )
        skater_df["penaltyMinutes"] = skater_df["penalityMinutes"] / skater_df["icetime"] * 60 * 60
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

        return self._merge_player_bio(player_df, skater_df)

    def _fetch_goalie_summary(self, player_df):
        """Per-save / per-shot goalie rates from the moneypuck season summary."""
        goalie_df = scraper.get_csv(
            f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{self.season_start.year}/regular/goalies.csv"
        )
        if goalie_df.empty:
            return pd.DataFrame()
        goalie_df.rename(columns={"name": "playerName"}, inplace=True)
        goalie_df = goalie_df.loc[goalie_df["situation"] == "all"]
        goalie_df["timePerGame"] = goalie_df["icetime"] / goalie_df["games_played"] / 60
        goalie_df["saves"] = goalie_df["ongoal"] - goalie_df["goals"]
        goalie_df["savePct"] = goalie_df["saves"] / goalie_df["ongoal"]
        goalie_df["freezeAgainst"] = (goalie_df["freeze"] - goalie_df["xFreeze"]) / goalie_df[
            "saves"
        ]
        goalie_df["reboundsAgainst"] = (goalie_df["rebounds"] - goalie_df["xRebounds"]) / goalie_df[
            "saves"
        ]
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

        return self._merge_player_bio(player_df, goalie_df)

    def _merge_player_bio(self, player_df, summary_df):
        """Right-join a moneypuck season summary onto player bios; drop merge cruft."""
        merged = player_df.merge(summary_df, how="right", on="playerId", suffixes=[None, "_y"])
        merged.dropna(inplace=True)
        merged.index = merged.playerId
        merged.drop(
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
        return merged

    def dump_goalie_list(self):
        filepath = pkg_resources.files(data) / "config" / "goalies.json"
        with open(filepath, "w") as outfile:
            json.dump(
                list(self.gamelog.loc[self.gamelog.position == "G", "playerName"].unique()), outfile
            )

    def get_volume_stats(self, offers, date=datetime.today().date(), pitcher=False):
        market = "shotsAgainst" if pitcher else "timeOnIce"
        if not self.load_volume_model_params(
            offers,
            market,
            date,
        ):
            return

        if not pitcher:
            # Budget parameters derived from historical NHL gamelogs:
            #   typical_rotation : median skaters (non-G) logging >3 min per team-game
            #   ot_rate          : fraction of team-games where total skater TOI > 300
            #   ot_extra         : mean extra team TOI above 300 when OT occurs (9.6 min)
            #   avg_unmodeled_min: mean TOI for ranked 8-18 skaters (rank > 7 tier)
            #   per_player_floor : 5th-pct TOI for top-7 tier skaters
            #   per_player_cap   : 99th-pct TOI for top-7 tier skaters
            reg_minutes = 300  # 5 skaters × 60 min regulation
            ot_rate = 0.189  # measured: 18.9% of team-games go to OT
            ot_extra = 9.6  # measured: mean extra team TOI when OT occurs
            typical_rotation = 18  # measured: median active skaters per team-game
            avg_unmodeled_min = 14  # measured: mean TOI for skaters ranked 8-18
            per_player_floor = 17  # measured: 5th-pct TOI for top-7 tier skaters
            per_player_cap = 29  # measured: 99th-pct TOI for top-7 tier skaters

            # Expected total team TOI including OT
            budget_mean = reg_minutes + ot_rate * ot_extra

            scale_team_volume_to_budget(
                self.playerProfile,
                market,
                budget_mean=budget_mean,
                typical_rotation=typical_rotation,
                avg_unmodeled_min=avg_unmodeled_min,
                per_player_floor=per_player_floor,
                per_player_cap=per_player_cap,
            )

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
            opponent = self.upcoming_games[team]["Opponent"]

        date = date.strftime("%Y-%m-%d")
        ev = 0
        if market in combo_props:
            ev = self._combo_market_ev(market, date, player, dist, cv)
        elif market == "goalsAgainst":
            ev = archive.get_total("NHL", date, opponent)
        elif "fantasy" in market:
            ev = self._check_nhl_fantasy(
                market, date, player, team, opponent, dist, cv, player_games
            )
        return 0 if np.isnan(ev) else ev

    def _check_nhl_fantasy(self, market, date, player, team, opponent, dist, cv, player_games):
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

        ev = 0
        book_odds = False
        for submarket, weight in fantasy_props:
            v, subline, sub_cv, sub_dist = self._submarket_ev(submarket, date, player, dist, cv)
            if not (np.isnan(v) or v == 0):
                book_odds = True
                ev += v * weight
            elif submarket == "Moneyline":
                ev += archive.get_moneyline("NHL", date, team) * weight
            elif submarket == "goalsAgainst":
                v = archive.get_total("NHL", date, opponent)
                subline = np.floor(v) + 0.5
                v = get_ev(subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist)
                ev += v * weight
            else:
                contribution, _ = self._fantasy_default_contribution(
                    submarket, weight, v, subline, sub_cv, sub_dist, player_games
                )
                ev += contribution
        return ev if book_odds else 0
