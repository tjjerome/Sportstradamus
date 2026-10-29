"""StatsNHL: NHL player stats loading, feature engineering, and prediction."""

import importlib.resources as pkg_resources
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import (
    abbreviations,
    combo_props,
    get_ev,
    get_odds,
    remove_accents,
    stat_cv,
    stat_dist,
)
from sportstradamus.helpers.io import VOLUME_STATS, write_gamelog
from sportstradamus.stats.base import (
    _SEASON_ROLLOVER_MONTH,
    GAMELOG_RETENTION_DAYS,
    ComboSpec,
    Stats,
    archive,
    scale_team_volume_to_budget,
    scraper,
)
from sportstradamus.stats.nhl_position_policy import allowed_nhl_positions

# DFS scoring weights (settled gamelog formulas in _skater_row; the sogBS terms there
# are carried as separate shots/blocked components because those are distinct book
# markets with their own correlation cross terms).
NHL_SKATER_UNDERDOG_WEIGHTS = (
    ("goals", 6),
    ("assists", 4),
    ("shots", 1),
    ("blocked", 1),
    ("hits", 0.5),
    ("powerPlayPoints", 0.5),
)
NHL_SKATER_PRIZEPICKS_WEIGHTS = (("goals", 8), ("assists", 5), ("shots", 1.5), ("blocked", 1.5))
# `hits` is a pick'em-only market: 3200 archived player-gamedays, none of them
# from a sportsbook. It is the sole component of the underdog skater spec that
# no book prices, so without a fallback that spec never quotes at all.
NHL_ASSUMABLE_SKATER_COMPONENTS = frozenset({"hits"})
# The research brief's 0.6*shotsAgainst - 3.6*goalsAgainst re-parameterization is
# unusable here: shotsAgainst has zero book quotes in the archive (model-side volume
# stat), so the goalie spec stays on the quoted pair; the same-player corr matrix
# carries the residual saves x goalsAgainst co-movement (weakly negative for
# starters — save% luck at fixed shot volume — despite saves + GA == SA on raw counts).
NHL_GOALIE_UNDERDOG_WEIGHTS = (("saves", 0.6), ("goalsAgainst", -3))
NHL_GOALIE_UNDERDOG_BERNOULLI = (("win", 6),)
# Keeps the goalie win probability off the exact 0/1 poles so it always clears
# combo_quote's strict open-interval admission check.
_GOALIE_WIN_P_CLIP: float = 0.01
# The per-game sums behind the moneypuck season-summary rates (_fetch_skater_summary,
# _fetch_goalie_summary), kept per gamelog row so a comp profile can stop at a game date.
SEASON_SUM_COLUMNS = {
    "shifts": "I_F_shifts",
    "shotAttempts": "I_F_shotAttempts",
    "xGoals": "I_F_flurryScoreVenueAdjustedxGoals",
    "shotRebounds": "I_F_rebounds",
    "shotFreezes": "I_F_freeze",
    "oZoneStarts": "I_F_oZoneShiftStarts",
    "dZoneStarts": "I_F_dZoneShiftStarts",
    "flyStarts": "I_F_flyShiftStarts",
    "takeaways": "I_F_takeaways",
    "giveaways": "I_F_giveaways",
    "penaltyMinutes": "I_F_penalityMinutes",
    "penaltyMinutesDrawn": "I_A_penalityMinutes",
    "onIceFenwickFor": "OnIce_F_unblockedShotAttempts",
    "onIceFenwickAgainst": "OnIce_A_unblockedShotAttempts",
    "offIceFenwickFor": "OffIce_F_unblockedShotAttempts",
    "offIceFenwickAgainst": "OffIce_A_unblockedShotAttempts",
    "xGoalsAgainst": "OnIce_A_flurryAdjustedxGoals",
    "freezesAgainst": "OnIce_A_freeze",
    "xFreezesAgainst": "OnIce_A_xFreeze",
    "reboundsAgainst": "OnIce_A_rebounds",
    "xReboundsAgainst": "OnIce_A_xRebounds",
}
# Characters 5-6 of an NHL game id (2025020001 -> 02): the season summary a comp profile
# mirrors counts regular-season games only.
_REGULAR_SEASON_GAME_TYPE: int = 2
# The fields of a stored players.json row that describe the player, not his season. A comp
# pool reads only these from the running season's row: its stats are season-to-date as of
# the last roster fetch, so for a training row they look ahead.
PLAYER_BIO_FIELDS = (
    "age",
    "bmi",
    "height",
    "weight",
    "shootsCatches",
    "playerName",
    "position",
    "team",
    "birthDate",
)


def _game_player_key(frame: pd.DataFrame) -> pd.MultiIndex:
    """(gameId, playerId) row key; the stored playerId column is object-typed, so both sides cast."""
    return pd.MultiIndex.from_arrays(
        [frame["gameId"].astype(int), pd.to_numeric(frame["playerId"], errors="coerce")]
    )


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
        self.volume_stats = VOLUME_STATS["NHL"]
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
        self._combo_player_games_cache_key = None
        self._combo_player_games_cache = {}
        self._combo_archive_cache_key = None
        self._combo_archive_cache = {}

    def _market_position_filter(self, gamelog: pd.DataFrame, market: str) -> pd.DataFrame:
        """Keep only players who can accrue the requested NHL market."""
        allowed = allowed_nhl_positions(market)
        position = gamelog[self.log_strings["position"]].astype("string")
        return gamelog.loc[position.isin(allowed)].copy()

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

    def _current_season_key(self, target_game_date: date) -> int:
        """Return the ``self.players`` key (season start year) holding ``target_game_date``.

        The key rolls over in August with the depth charts (``_SEASON_ROLLOVER_MONTH``), so an
        offseason date belongs to the coming season: 2026-09-30 -> 2026, 2026-07-31 -> 2025.
        """
        if target_game_date.month >= _SEASON_ROLLOVER_MONTH:
            return target_game_date.year
        return target_game_date.year - 1

    def _season_to_date_profile(self, season: int, target_game_date: date) -> pd.DataFrame:
        """Rebuild moneypuck's season-summary rates from the games before ``target_game_date``.

        One row per player id over the regular-season games of ``season``, with the formulas
        of :meth:`_fetch_skater_summary` and :meth:`_fetch_goalie_summary` applied once to the
        gamelog's per-game sums. Skater rows carry the skater rates, goalie rows the goalie rates.
        Counting stats are per 60 minutes of ice time; ``timePerGame`` is in minutes and
        ``timePerShift`` in seconds.

        Raises:
            RuntimeError: a row of the slice lacks the ``SEASON_SUM_COLUMNS``.
        """
        game_id = self.gamelog["gameId"]
        rows = self.gamelog.loc[
            (game_id // 1_000_000 == season)
            & (game_id // 10_000 % 100 == _REGULAR_SEASON_GAME_TYPE)
            & (self._gamelog_dates() < pd.Timestamp(target_game_date))
        ]
        if rows.reindex(columns=list(SEASON_SUM_COLUMNS)).isna().to_numpy().any():
            raise RuntimeError(
                f"NHL {season} gamelog rows before {target_game_date} lack the season sums; "
                "run `sportstradamus admin backfill-nhl-gamelog` first"
            )
        by_player = rows.groupby(rows["playerId"].astype(int))
        sums = by_player[
            [
                *SEASON_SUM_COLUMNS,
                "timeOnIce",
                "shots",
                "goals",
                "assists",
                "hits",
                "blocked",
                "shotsAgainst",
                "goalsAgainst",
            ]
        ].sum()
        icetime = sums["timeOnIce"] * 60
        per_hour = 3600 / icetime
        skaters = pd.DataFrame(
            {
                "Fenwick": sums["onIceFenwickFor"]
                / (sums["onIceFenwickFor"] + sums["onIceFenwickAgainst"])
                - sums["offIceFenwickFor"]
                / (sums["offIceFenwickFor"] + sums["offIceFenwickAgainst"]),
                "timePerGame": sums["timeOnIce"] / by_player.size(),
                "timePerShift": icetime / sums["shifts"],
                "xGoals": sums["xGoals"] / sums["shotAttempts"],
                "shotsOnGoal": sums["shots"] / sums["shotAttempts"],
                "goals": (sums["goals"] - sums["xGoals"]) / sums["shotAttempts"],
                "rebounds": sums["shotRebounds"] / sums["shotAttempts"],
                "freeze": sums["shotFreezes"] / sums["shotAttempts"],
                "oZoneStarts": sums["oZoneStarts"] / (sums["oZoneStarts"] + sums["dZoneStarts"]),
                "flyStarts": sums["flyStarts"] / sums["shifts"],
                "shotAttempts": sums["shotAttempts"] * per_hour,
                "hits": sums["hits"] * per_hour,
                "takeaways": sums["takeaways"] * per_hour,
                "giveaways": sums["giveaways"] * per_hour,
                "assists": sums["assists"] * per_hour,
                "penaltyMinutes": sums["penaltyMinutes"] * per_hour,
                "penaltyMinutesDrawn": sums["penaltyMinutesDrawn"] * per_hour,
                "blockedShots": sums["blocked"] * per_hour,
            }
        )
        saves = sums["shotsAgainst"] - sums["goalsAgainst"]
        goalies = pd.DataFrame(
            {
                "timePerGame": skaters["timePerGame"],
                "savePct": saves / sums["shotsAgainst"],
                "freezeAgainst": (sums["freezesAgainst"] - sums["xFreezesAgainst"]) / saves,
                "reboundsAgainst": (sums["reboundsAgainst"] - sums["xReboundsAgainst"]) / saves,
                "goalsAgainst": (sums["goalsAgainst"] - sums["xGoalsAgainst"])
                / sums["shotsAgainst"],
            }
        )
        is_goalie = by_player["position"].first() == "G"
        return pd.concat([skaters[~is_goalie], goalies[is_goalie]])

    def _player_seasons_through(self, target_game_date: date) -> dict:
        """The comp pool at ``target_game_date`` as ``{player_id: stats_dict}``.

        Past seasons' stored rows merge oldest to newest, so a player's latest one wins. The
        running season's stored row is read for its ``PLAYER_BIO_FIELDS`` only; its stats are
        rebuilt from the games before the date (:meth:`_season_to_date_profile`), and a player
        without one keeps his latest past-season row. ``age`` is as of the date wherever a row
        carries ``birthDate``.
        """
        season = self._current_season_key(target_game_date)
        merged: dict = {}
        for season_key in sorted(key for key in self.players if key < season):
            merged.update(self.players[season_key])
        if season in self.players:
            rates = {
                pid: row.dropna().to_dict()
                for pid, row in self._season_to_date_profile(season, target_game_date).iterrows()
            }
            for pid, stored in self.players[season].items():
                if pid in rates:
                    bio = {field: stored[field] for field in PLAYER_BIO_FIELDS if field in stored}
                    merged[pid] = bio | rates[pid]
        return {
            pid: row
            | {"age": (target_game_date - date.fromisoformat(row["birthDate"])).days / 365.25}
            if "birthDate" in row
            else row
            for pid, row in merged.items()
        }

    def _compute_comps(self, target_game_date: date | None = None) -> None:
        """Build comps from loaded data at runtime (no JSON I/O).

        The pool is :meth:`_player_seasons_through` at ``target_game_date``: past seasons'
        stored summaries plus the running season rebuilt from the games before the date, so a
        training row's comps know neither later rookies nor later games. Today() default
        preserves inference / cron behavior.
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
        stats.update({name: float(player[src]) for name, src in SEASON_SUM_COLUMNS.items()})
        return n | stats

    def _update(self):
        latest_date = self.season_start
        if not self.gamelog.empty:
            latest_date = pd.to_datetime(self.gamelog["gameDate"]).max().date() + timedelta(days=1)
        today = datetime.today().date()

        ids = self._collect_game_ids(latest_date, today)
        nhl_gamelog, nhl_teamlog = self._parse_new_games(ids, today)
        self._merge_new_gamelogs(nhl_gamelog, nhl_teamlog)
        # Rows stored before these sums were logged hold NaN; the first update after the
        # deploy re-parses the running season's games so its comp profile can be rebuilt.
        self.backfill_season_sums(seasons={self.season_start.year})
        self._fetch_upcoming_games_nhl(today)

        player_df = self._fetch_player_bios_nhl()
        skater_df = self._fetch_skater_summary(player_df)
        goalie_df = self._fetch_goalie_summary(player_df)
        roster_pull = skater_df.to_dict("index") | goalie_df.to_dict("index")
        # A pull moneypuck blocks comes back empty. Merged, it leaves the season's rows as
        # they were; assigned, it would hand every player to last season's row.
        self._merge_season_roster(self.season_start.year, roster_pull)

        four_years_ago = today - timedelta(days=GAMELOG_RETENTION_DAYS)
        self.gamelog = self.gamelog[
            pd.to_datetime(self.gamelog["gameDate"]).dt.date >= four_years_ago
        ]
        self.gamelog.drop_duplicates(subset=["gameId", "playerName"], keep="last", inplace=True)
        self.teamlog = self.teamlog[
            pd.to_datetime(self.teamlog["gameDate"]).dt.date >= four_years_ago
        ]
        self.teamlog.drop_duplicates(subset=["gameId", "team"], keep="last", inplace=True)

        if self._full_clean_due():
            self.gamelog["playerName"] = self.gamelog["playerName"].apply(remove_accents)
            self._enrich_team_markets(self.gamelog, date_col="gameDate", team_col="team")

        write_gamelog("nhl", self.gamelog, self.teamlog, self.players)
        # Underdog's parser files an NHL fantasy line as a goalie's or a skater's by this list;
        # the hourly board run's update keeps it current.
        self.dump_goalie_list()

        # Raised after the writes, so the parsed games and the stored roster are kept. Before
        # opening night the season has no summary yet and an empty pull is normal.
        game_days = pd.to_datetime(self.gamelog["gameDate"]).dt.date
        if not roster_pull and (game_days >= self.season_start).any():
            raise RuntimeError(
                "NHL roster pull came back empty: moneypuck returned no skater or goalie "
                f"season summary for {self.season_start.year}"
            )

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

    def backfill_season_sums(self, seasons: set[int] | None = None) -> int:
        """Fill the SEASON_SUM_COLUMNS on gamelog rows that lack them by re-parsing their games.

        Args:
            seasons: Season start years to restrict to, e.g. ``{2025}``; ``None`` is every season.

        Returns:
            The number of games re-fetched. Rows whose game could not be parsed keep NaN.
        """
        columns = list(SEASON_SUM_COLUMNS)
        log = self.gamelog.reindex(columns=self.gamelog.columns.union(columns, sort=False))
        lacking = log[columns].isna().any(axis=1)
        if seasons is not None:
            # An NHL game id leads with its season's start year: 2025020001 -> 2025.
            lacking &= (log["gameId"] // 1_000_000).isin(seasons)
        games = log.loc[lacking, ["gameId", "gameDate"]].drop_duplicates()
        parsed = []
        for gameId, gameDate in tqdm(
            games.itertuples(index=False), total=len(games), desc="Backfilling NHL season sums"
        ):
            rows, _ = self.parse_game(gameId, gameDate)
            parsed.extend(rows)
        if parsed:
            # Keyed on the player id, not the name: moneypuck respells names between seasons
            # (Cam Atkinson, J.J. Moser), and a name key left 310 respelled 2024 rows lacking.
            fresh = pd.DataFrame(parsed)
            # A blank cell reads 0 as the hourly merge reads it; DataFrame.update skips NaN.
            fresh = fresh.set_index(_game_player_key(fresh))[columns].fillna(0)
            fresh = fresh[~fresh.index.duplicated(keep="last")]
            stored = log.set_index(_game_player_key(log))[columns]
            stored.update(fresh)
            log[columns] = stored.to_numpy()
        self.gamelog = log
        return len(games)

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

        self.playerProfile.fillna(0, inplace=True)

    def _fantasy_combo_spec(self, market, player):
        if market == "skater fantasy points underdog":
            return ComboSpec(
                marginals=NHL_SKATER_UNDERDOG_WEIGHTS,
                assumable=NHL_ASSUMABLE_SKATER_COMPONENTS,
            )
        if market == "fantasy points prizepicks":
            return ComboSpec(marginals=NHL_SKATER_PRIZEPICKS_WEIGHTS)
        if market == "goalie fantasy points underdog":
            return ComboSpec(
                marginals=NHL_GOALIE_UNDERDOG_WEIGHTS,
                bernoulli=NHL_GOALIE_UNDERDOG_BERNOULLI,
                analytics=("win_ml",),
            )
        return None

    def _combo_bernoulli_p(self, name, player, date):
        if name != "win":
            return super()._combo_bernoulli_p(name, player, date)
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        if date < datetime.today().date():
            todays_games = self._gamelog_on(date)
            player_game = todays_games.loc[todays_games[self.log_strings["player"]] == player]
            if player_game.empty:
                return np.nan
            team = player_game[self.log_strings["team"]].iloc[0]
        else:
            team = self.short_gamelog.loc[
                self.short_gamelog[self.log_strings["player"]] == player,
                self.log_strings["team"],
            ].iloc[-1]
        # A goalie starter takes essentially every team decision, so the raw team
        # moneyline is P(win) — no SP-style calibration map (unlike MLB pitchers).
        p = archive.get_moneyline("NHL", date.strftime("%Y-%m-%d"), team)
        return float(np.clip(p, _GOALIE_WIN_P_CLIP, 1 - _GOALIE_WIN_P_CLIP))

    def check_combo_markets(self, market, player, date=datetime.today().date()):
        if self.snapshot_only_rebuild:
            player_column = self.log_strings["player"]
            cache_key = (
                self.profile_latest_date,
                id(self.short_gamelog),
                len(self.short_gamelog),
                player_column,
            )
            if cache_key != self._combo_player_games_cache_key:
                self._combo_player_games_cache = dict(
                    tuple(self.short_gamelog.groupby(player_column, sort=False))
                )
                self._combo_player_games_cache_key = cache_key
            player_games = self._combo_player_games_cache.get(player, self.short_gamelog.iloc[:0])
        else:
            player_games = self.short_gamelog.loc[
                self.short_gamelog[self.log_strings["player"]] == player
            ]
        cv = stat_cv.get(self.league, {}).get(market, 1)
        dist = stat_dist.get(self.league, {}).get(market, "Gamma")
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()

        if date < datetime.today().date():
            todays_games = self._gamelog_on(date)
            player_game = todays_games.loc[todays_games[self.log_strings["player"]] == player]
            if player_game.empty:
                return 0

            team = player_game[self.log_strings["team"]].iloc[0]
            opponent = player_game[self.log_strings["opponent"]].iloc[0]

        else:
            team = player_games[self.log_strings["team"]].iloc[-1]
            opponent = self.upcoming_games[team]["Opponent"]

        date = date.strftime("%Y-%m-%d")
        if self.snapshot_only_rebuild:
            self._prepare_combo_archive_cache(market, date, todays_games)
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

    def _prepare_combo_archive_cache(self, market, date, todays_games):
        """Batch legacy submarket EV/line reads for one NHL fallback slate."""
        if not self.snapshot_only_rebuild:
            return
        cache_key = (market, date)
        if cache_key == self._combo_archive_cache_key:
            return

        if market in combo_props:
            submarkets = list(combo_props[market])
        elif "fantasy" in market:
            if "prizepicks" in market:
                submarkets = ["goals", "assists", "shots", "blocked"]
            elif ("underdog" in market) and ("skater" in market):
                submarkets = [
                    "goals",
                    "assists",
                    "shots",
                    "blocked",
                    "hits",
                    "powerPlayPoints",
                ]
            else:
                submarkets = ["saves", "goalsAgainst"]
        else:
            submarkets = []

        players = todays_games[self.log_strings["player"]].dropna().unique().tolist()
        self._combo_archive_cache = {
            submarket: archive.get_ev_line_inputs("NHL", submarket, date, players)
            for submarket in submarkets
        }
        self._combo_archive_cache_key = cache_key

    def _submarket_ev(self, submarket, date, player, dist, cv):
        cached = self._combo_archive_cache.get(submarket)
        if cached is None or player not in cached:
            return super()._submarket_ev(submarket, date, player, dist, cv)

        sub_cv = stat_cv.get(self.league, {}).get(submarket, 1)
        sub_dist = stat_dist.get(self.league, {}).get(submarket, "Gamma")
        value, subline = cached[player]
        value = self._convert_to_market_dist(value, subline, sub_cv, sub_dist, dist, cv)
        return value, subline, sub_cv, sub_dist

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
