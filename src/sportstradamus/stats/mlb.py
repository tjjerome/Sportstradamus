"""StatsMLB: MLB player stats loading, feature engineering, and prediction."""

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
import statsapi as mlb
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
from sportstradamus.stats.base import Stats, archive, clean_data, scraper

# Minimum Savant affinity match_score to include a player as a comparable.
# Scores below this are weak matches that add noise to comp feature sets.
_COMP_MATCH_SCORE_THRESHOLD: float = 0.6
# League-average ERA floor in the FIP formula (Fielding Independent Pitching).
# FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + FIP_CONSTANT, where the constant
# normalizes the scale so FIP ≈ ERA league-wide each season.
_FIP_CONSTANT: float = 3.2
# Minimum hit rate for a market across a player's game log for them to be
# included in the profiling group.  Filters out players with too few non-zero
# occurrences to provide meaningful signal.
_MARKET_HIT_RATE_MIN: float = 0.1
# MLB schedule game_type codes that are postseason (Wild Card, Division Series,
# League Championship, World Series, generic Playoff). "R" is regular season and
# the spring/exhibition/All-Star codes (S/E/A) are filtered out before they reach
# the gamelog, so anything in this set is a true bracket game.
_MLB_POSTSEASON_GAME_TYPES: frozenset[str] = frozenset({"F", "D", "L", "W", "P"})
# Wins to clinch each MLB postseason round, keyed by game_type: Wild Card best-of-3,
# Division Series best-of-5, League Championship + World Series best-of-7. A generic
# postseason code (P) and anything unmapped fall back to best-of-5.
_MLB_SERIES_GAMES_TO_WIN: dict[str, int] = {"F": 2, "D": 3, "L": 4, "W": 4, "P": 3}
_MLB_DEFAULT_SERIES_WINS: int = 3

# Batting-order plate-appearance structure, measured from the backfilled ~2-season
# MLB gamelog (scripts/measure_mlb_volume_constants.py). The batting slot fixes a
# hitter's PA regardless of who fills it, so a missing starter's PAs go to his
# replacement in the same slot -- no team budget redistribution is needed. Away
# teams bat a full ninth every game; home teams skip the bottom 9th when leading,
# so away slots carry ~0.18 PA more. Index 0 = leadoff (slot 1) .. index 8 = slot 9.
SLOT_PA_HOME = (4.404, 4.291, 4.208, 4.110, 3.971, 3.831, 3.688, 3.533, 3.358)
SLOT_PA_AWAY = (4.584, 4.488, 4.377, 4.284, 4.149, 4.008, 3.862, 3.705, 3.543)
SLOT_PA_ALL = tuple((h + a) / 2 for h, a in zip(SLOT_PA_HOME, SLOT_PA_AWAY, strict=True))
# Within-slot game-to-game PA spread (extra innings, blowouts, early removal): the
# genuinely unpredictable part of PA, so it stays in std and is not offense-adjusted.
SLOT_STD = (0.712, 0.698, 0.685, 0.662, 0.693, 0.731, 0.764, 0.784, 0.799)
# Fallback for a priced hitter whose slot cannot be resolved (no lineup, no history):
# mean starting-batter PA across slots, with a deliberately wide spread.
SLOT_PA_LEAGUE_AVG = sum(SLOT_PA_ALL) / len(SLOT_PA_ALL)
SLOT_STD_UNKNOWN = 1.0

# Team offense adjustment: a bounded per-team PA multiplier (nominal 1.0). OBP is the
# mechanistic driver (more base-runners -> more lineup turnover -> more PA); the
# book-implied team total is a sanity anchor. Clipped to +/-8% because the predictable
# offense signal is only ~1-2 PA on a ~36 PA base -- the rest is unpredictable (-> std).
LG_AVG_OBP = 0.315  # mean team OBP (teamlog scale; matches teamProfile["OBP"] the offense adjustment reads)
LG_AVG_TEAM_TOTAL = 4.671  # mirrors archive default_totals["MLB"] so unquoted games -> neutral 1.0
OBP_ADJ_WEIGHT = 0.70
MARKET_ADJ_WEIGHT = 0.30
OFFENSE_ADJ_CLIP = (0.92, 1.08)
_OBP_POLE_GUARD = 0.5  # cap expected OBP so the 1/(1-OBP) PA law stays finite
_TEAM_OBP_WINDOW = 10  # recent starts for opposing-starter OBP-allowed (matches teamProfile last-10)


def _mlb_team_abbr(mlb_teams, team_id):
    return next(
        team["abbreviation"].replace("AZ", "ARI")
        for team in mlb_teams["teams"]
        if team["id"] == team_id
    )


def _build_mlb_upcoming_games(mlb_games, mlb_teams):
    mlb_upcoming_games = {}
    for game in mlb_games:
        if game["status"] in ["Pre-Game", "Scheduled"] and game["game_type"] not in ["A"]:
            awayTeam = _mlb_team_abbr(mlb_teams, game["away_id"])
            homeTeam = _mlb_team_abbr(mlb_teams, game["home_id"])
            game_bs = mlb.boxscore_data(game["game_id"])
            players = {p["id"]: p["fullName"] for k, p in game_bs["playerInfo"].items()}
            playoff = game["game_type"] in _MLB_POSTSEASON_GAME_TYPES
            away_entry = {
                "Pitcher": remove_accents(game["away_probable_pitcher"]),
                "Home": False,
                "Opponent": homeTeam,
                "Opponent Pitcher": remove_accents(game["home_probable_pitcher"]),
                "Batting Order": [players[i] for i in game_bs["away"]["battingOrder"]],
                "Playoff": playoff,
                "game type": game["game_type"],
            }
            home_entry = {
                "Pitcher": remove_accents(game["home_probable_pitcher"]),
                "Home": True,
                "Opponent": awayTeam,
                "Opponent Pitcher": remove_accents(game["away_probable_pitcher"]),
                "Batting Order": [players[i] for i in game_bs["home"]["battingOrder"]],
                "Playoff": playoff,
                "game type": game["game_type"],
            }
            suffix = "" if game["game_num"] == 1 else str(game["game_num"])
            mlb_upcoming_games[awayTeam + suffix] = away_entry
            mlb_upcoming_games[homeTeam + suffix] = home_entry
    return mlb_upcoming_games


def _mlb_final_game_ids(mlb_games, prev_game_ids):
    return [
        game["game_id"]
        for game in mlb_games
        if game["status"] == "Final"
        and game["game_type"] not in ("E", "S", "A")
        and game["game_id"] not in prev_game_ids
    ]


class StatsMLB(Stats):
    """A class for handling and analyzing MLB statistics.
    Inherits from the Stats parent class.

    Additional Attributes:
        pitchers (mlb_pitchers): Object containing MLB pitcher data.
        gameIds (list): List of game ids in gamelog.

    Additional Methods:
        None
    """

    def __init__(self):
        """Initialize the StatsMLB instance."""
        super().__init__()
        self.season_start = datetime(2024, 3, 28).date()
        self.pitchers = get_mlb_pitchers()
        self.gameIds = []
        self.gamelog = pd.DataFrame()
        self.teamlog = pd.DataFrame()
        self.park_factors = {}
        self.players = {}
        self.comps = {}
        self.league = "MLB"
        self.stat_types = {
            "batting": ["OBP", "AVG", "SLG", "PASO", "BABIP"],
            "fielding": ["DER"],
            "pitching": ["FIP", "WHIP", "ERA", "K9", "BB9", "PA9", "IP"],
        }
        self.volume_stats = ["plateAppearances", "pitches thrown"]
        self.default_total = 4.671
        self.log_strings = {
            "game": "gameId",
            "date": "gameDate",
            "player": "playerName",
            "position": "position",
            "team": "team",
            "opponent": "opponent",
            "home": "home",
            "win": "WL",
            "score": "runs",
        }
        self._volume_model_cache = None

    def parse_game(self, gameId):
        """Fetch a baseballsavant box score and append rows to gamelog and teamlog."""
        game = scraper.get(f"https://baseballsavant.mlb.com/gf?game_pk={gameId}")
        if not game:
            return

        linescore = game["scoreboard"]["linescore"]
        boxscore = game["boxscore"]
        awayTeam = game["away_team_data"]["abbreviation"].replace("AZ", "ARI").replace("WSH", "WAS")
        homeTeam = game["home_team_data"]["abbreviation"].replace("AZ", "ARI").replace("WSH", "WAS")
        bpf = self.park_factors[homeTeam]
        game_date = game["game_date"]

        awayPitcherId = game["away_pitcher_lineup"][0]
        awayPitcher = remove_accents(
            boxscore["teams"]["away"]["players"]["ID" + str(awayPitcherId)]["person"]["fullName"]
        )
        awayPitcherHand = self._resolve_pitcher_hand(awayPitcherId, awayPitcher, game, "away")
        if awayPitcherHand is None:
            return
        homePitcherId = game["home_pitcher_lineup"][0]
        homePitcher = remove_accents(
            boxscore["teams"]["home"]["players"]["ID" + str(homePitcherId)]["person"]["fullName"]
        )
        homePitcherHand = self._resolve_pitcher_hand(homePitcherId, homePitcher, game, "home")
        if homePitcherHand is None:
            return

        away1 = linescore["innings"][0]["away"]
        home1 = linescore["innings"][0]["home"]

        # The away pitcher's "1st inning allowed" stats are the home team's first
        # inning, and vice versa.
        away_rows, away_bullpen = self._team_player_rows(
            boxscore["teams"]["away"]["players"],
            side="away",
            team=awayTeam,
            opponent=homeTeam,
            opp_pitcher=homePitcher,
            opp_pitcher_id=homePitcherId,
            opp_pitcher_hand=homePitcherHand,
            starting_pitcher_id=awayPitcherId,
            inning1_runs=home1["runs"],
            inning1_hits=home1["hits"],
            gameId=gameId,
            game_date=game_date,
            game=game,
            bpf=bpf,
        )
        home_rows, home_bullpen = self._team_player_rows(
            boxscore["teams"]["home"]["players"],
            side="home",
            team=homeTeam,
            opponent=awayTeam,
            opp_pitcher=awayPitcher,
            opp_pitcher_id=awayPitcherId,
            opp_pitcher_hand=awayPitcherHand,
            starting_pitcher_id=homePitcherId,
            inning1_runs=away1["runs"],
            inning1_hits=away1["hits"],
            gameId=gameId,
            game_date=game_date,
            game=game,
            bpf=bpf,
        )

        new_games = pd.DataFrame.from_records(away_rows + home_rows)
        self._enrich_team_markets(new_games, date_col="gameDate", team_col="team")
        self.gamelog = pd.concat([self.gamelog, new_games], ignore_index=True)

        home_adj = self._bullpen_adj(home_bullpen, bpf)
        away_adj = self._bullpen_adj(away_bullpen, bpf)
        teams = [
            self._team_row(
                homeTeam,
                awayTeam,
                gameId,
                game_date,
                boxscore,
                bpf,
                home_bullpen,
                home_adj,
                me="home",
                opp="away",
            ),
            self._team_row(
                awayTeam,
                homeTeam,
                gameId,
                game_date,
                boxscore,
                bpf,
                away_bullpen,
                away_adj,
                me="away",
                opp="home",
            ),
        ]
        self.teamlog = pd.concat(
            [self.teamlog, pd.DataFrame.from_records(teams)], ignore_index=True
        )

    def _team_player_rows(
        self,
        players,
        *,
        side,
        team,
        opponent,
        opp_pitcher,
        opp_pitcher_id,
        opp_pitcher_hand,
        starting_pitcher_id,
        inning1_runs,
        inning1_hits,
        gameId,
        game_date,
        game,
        bpf,
    ):
        """Build the per-player gamelog rows for one team and tally its bullpen.

        Returns ``(rows, bullpen)``; the away and home calls are identical apart
        from the team-context arguments threaded through here.
        """
        is_home = side == "home"
        rows = []
        bullpen = dict.fromkeys(
            [
                "pitches thrown",
                "pitcher strikeouts",
                "pitching outs",
                "batters faced",
                "walks allowed",
                "hits allowed",
                "home runs allowed",
                "runs allowed",
            ],
            0,
        )
        for v in players.values():
            if v["person"]["id"] == starting_pitcher_id or v.get("battingOrder"):
                n = self._player_game_row(
                    v,
                    gameId=gameId,
                    game_date=game_date,
                    team=team,
                    opponent=opponent,
                    opp_pitcher=opp_pitcher,
                    opp_pitcher_id=opp_pitcher_id,
                    opp_pitcher_hand=opp_pitcher_hand,
                    is_home=is_home,
                    starting_pitcher_id=starting_pitcher_id,
                    inning1_runs=inning1_runs,
                    inning1_hits=inning1_hits,
                )
                bat_side = 0
                if n["starting batter"]:
                    bat_side = self._resolve_bat_side(n, game, side)
                    if bat_side is None:
                        continue
                adj = self._park_adjusted(n, v, bpf)
                bip = (
                    n["atBats"]
                    - n["batter strikeouts"]
                    - n["home runs"]
                    - v["stats"]["batting"].get("sacFlies", 0)
                )
                n.update(self._pitching_rates(n, adj))
                n.update(self._batting_rates(n, adj, bip, bat_side, bpf))
                rows.append(n)
            elif v.get("position", {}).get("type", "") == "Pitcher":
                self._accumulate_bullpen(bullpen, v)
        return rows, bullpen

    @staticmethod
    def _player_game_row(
        v,
        *,
        gameId,
        game_date,
        team,
        opponent,
        opp_pitcher,
        opp_pitcher_id,
        opp_pitcher_hand,
        is_home,
        starting_pitcher_id,
        inning1_runs,
        inning1_hits,
    ):
        """Base gamelog row (batting + pitching counting stats) for one player."""
        bat = v["stats"]["batting"]
        pit = v["stats"]["pitching"]
        ip_whole = int(pit.get("inningsPitched", "0.0").split(".")[0])
        ip_frac = int(pit.get("inningsPitched", "0.0").split(".")[1])
        is_sp = v["person"]["id"] == starting_pitcher_id
        return {
            "gameId": gameId,
            "gameDate": game_date,
            "playerId": v["person"]["id"],
            "playerName": remove_accents(v["person"]["fullName"]),
            "position": v.get("position", {"abbreviation": ""})["abbreviation"],
            "team": team,
            "opponent": opponent,
            "opponent pitcher": opp_pitcher,
            "opponent pitcher id": opp_pitcher_id,
            "opponent pitcher hand": opp_pitcher_hand,
            "home": is_home,
            "starting pitcher": is_sp,
            "starting batter": int(v.get("battingOrder", "001")[2]) == 0,
            "battingOrder": int(v.get("battingOrder", "000")[0]),
            "hits": bat.get("hits", 0),
            "total bases": bat.get("hits", 0)
            + bat.get("doubles", 0)
            + 2 * bat.get("triples", 0)
            + 3 * bat.get("homeRuns", 0),
            "singles": bat.get("hits", 0)
            - bat.get("doubles", 0)
            - bat.get("triples", 0)
            - bat.get("homeRuns", 0),
            "doubles": bat.get("doubles", 0),
            "triples": bat.get("triples", 0),
            "home runs": bat.get("homeRuns", 0),
            "batter strikeouts": bat.get("strikeOuts", 0),
            "runs": bat.get("runs", 0),
            "rbi": bat.get("rbi", 0),
            "hits+runs+rbi": bat.get("hits", 0) + bat.get("runs", 0) + bat.get("rbi", 0),
            "walks": bat.get("baseOnBalls", 0) + bat.get("hitByPitch", 0),
            "stolen bases": bat.get("stolenBases", 0),
            "atBats": bat.get("atBats", 0),
            "plateAppearances": bat.get("plateAppearances", 0),
            "pitcher strikeouts": pit.get("strikeOuts", 0),
            "pitcher win": pit.get("wins", 0),
            "walks allowed": pit.get("baseOnBalls", 0) + pit.get("hitByPitch", 0),
            "pitches thrown": pit.get("numberOfPitches", 0),
            "runs allowed": pit.get("runs", 0),
            "hits allowed": pit.get("hits", 0),
            "home runs allowed": pit.get("homeRuns", 0),
            "pitching outs": 3 * ip_whole + ip_frac,
            "batters faced": pit.get("battersFaced", 0),
            "1st inning runs allowed": inning1_runs if is_sp else 0,
            "1st inning hits allowed": inning1_hits if is_sp else 0,
            "hitter fantasy score": 3 * bat.get("hits", 0)
            + 2 * bat.get("doubles", 0)
            + 5 * bat.get("triples", 0)
            + 7 * bat.get("homeRuns", 0)
            + 2 * bat.get("runs", 0)
            + 2 * bat.get("rbi", 0)
            + 2 * bat.get("baseOnBalls", 0)
            + 2 * bat.get("hitByPitch", 0)
            + 5 * bat.get("stolenBases", 0),
            "pitcher fantasy score": 6 * pit.get("wins", 0)
            + 3 * pit.get("strikeOuts", 0)
            - 3 * pit.get("earnedRuns", 0)
            + 3 * ip_whole
            + ip_frac
            + (4 if ip_whole > 5 and pit.get("earnedRuns", 0) < 4 else 0),
            "hitter fantasy points underdog": 3 * bat.get("hits", 0)
            + 3 * bat.get("doubles", 0)
            + 5 * bat.get("triples", 0)
            + 7 * bat.get("homeRuns", 0)
            + 2 * bat.get("runs", 0)
            + 2 * bat.get("rbi", 0)
            + 3 * bat.get("baseOnBalls", 0)
            + 3 * bat.get("hitByPitch", 0)
            + 4 * bat.get("stolenBases", 0),
            "pitcher fantasy points underdog": 5 * pit.get("wins", 0)
            + 3 * pit.get("strikeOuts", 0)
            - 3 * pit.get("earnedRuns", 0)
            + 3 * ip_whole
            + (5 if ip_whole > 5 and pit.get("earnedRuns", 0) < 4 else 0),
            "hitter fantasy points parlay": 3 * bat.get("hits", 0)
            + 3 * bat.get("doubles", 0)
            + 6 * bat.get("triples", 0)
            + 9 * bat.get("homeRuns", 0)
            + 3 * bat.get("runs", 0)
            + 3 * bat.get("rbi", 0)
            + 3 * bat.get("baseOnBalls", 0)
            + 3 * bat.get("hitByPitch", 0)
            + 6 * bat.get("stolenBases", 0),
            "pitcher fantasy points parlay": 6 * pit.get("wins", 0)
            + 3 * pit.get("strikeOuts", 0)
            - 3 * pit.get("earnedRuns", 0)
            + 3 * ip_whole
            + ip_frac,
        }

    def _resolve_pitcher_hand(self, pid, name, game, side):
        """Throwing hand for a starting pitcher, caching it on ``self.players``.

        Returns ``None`` when the pitcher is absent from the feed's lineup, which
        signals ``parse_game`` to skip the whole game (the box score is unusable).
        """
        if pid in self.players and "throws" in self.players[pid]:
            return self.players[pid]["throws"]
        if str(pid) not in game[f"{side}_pitchers"]:
            return None
        hand = game[f"{side}_pitchers"][str(pid)][0]["p_throws"]
        if pid not in self.players:
            self.players[pid] = {"name": name, "throws": hand}
        else:
            self.players[pid]["throws"] = hand
        return hand

    def _resolve_bat_side(self, n, game, side):
        """Batting side for a starting batter, caching it on ``self.players``.

        Returns ``None`` when the batter is absent from the feed's lineup, which
        signals the caller to skip that player's row.
        """
        pid = n["playerId"]
        if pid in self.players and "bats" in self.players[pid]:
            return self.players[pid]["bats"]
        if str(pid) not in game[f"{side}_batters"]:
            return None
        bat_side = game[f"{side}_batters"][str(pid)][0]["stand"]
        if pid not in self.players:
            self.players[pid] = {"name": n["playerName"], "bats": bat_side}
        else:
            self.players[pid]["bats"] = bat_side
        return bat_side

    @staticmethod
    def _park_adjusted(n, v, bpf):
        """Park-factor-normalized counting stats used by the rate metrics."""
        bat = v["stats"]["batting"]
        return {
            "R": n["runs"] / bpf["R"],
            "RBI": n["rbi"] / bpf["R"],
            "H": n["hits"] / bpf["H"],
            "1B": n["singles"] / bpf["1B"],
            "2B": bat.get("doubles", 0) / bpf["2B"],
            "3B": bat.get("triples", 0) / bpf["3B"],
            "HR": n["home runs"] / bpf["HR"],
            "W": n["walks"] / bpf["BB"],
            "SO": n["batter strikeouts"] / bpf["K"],
            "RA": n["runs allowed"] / bpf["R"],
            "HA": n["hits allowed"] / bpf["H"],
            "HRA": n["home runs allowed"] / bpf["HR"],
            "BB": n["walks allowed"] / bpf["BB"],
            "K": n["pitcher strikeouts"] / bpf["K"],
        }

    @staticmethod
    def _pitching_rates(n, adj):
        sp_outs = n["starting pitcher"] and n["pitching outs"]
        return {
            "FIP": (
                3 * (13 * adj["HRA"] + 3 * adj["BB"] - 2 * adj["K"]) / n["pitching outs"]
                + _FIP_CONSTANT
            )
            if sp_outs
            else 0,
            "WHIP": (3 * (adj["BB"] + adj["HA"]) / n["pitching outs"]) if sp_outs else 0,
            "ERA": (9 * adj["RA"] / n["pitching outs"]) if sp_outs else 0,
            "K9": (27 * adj["K"] / n["pitching outs"]) if sp_outs else 0,
            "BB9": (27 * adj["BB"] / n["pitching outs"]) if sp_outs else 0,
            "PA9": (27 * n["batters faced"] / n["pitching outs"]) if sp_outs else 0,
            "IP": (n["pitching outs"] / 3) if n["starting pitcher"] else 0,
        }

    @staticmethod
    def _batting_rates(n, adj, bip, bat_side, bpf):
        has_ab = n["atBats"] > 0
        return {
            "OBP": ((n["hits"] + n["walks"]) / n["atBats"] / bpf["OBP"]) if has_ab else 0,
            "AVG": (n["hits"] / n["atBats"]) if has_ab else 0,
            "SLG": (n["total bases"] / n["atBats"]) if has_ab else 0,
            "PASO": (n["plateAppearances"] / adj["SO"])
            if (n["starting batter"] and adj["SO"])
            else n["plateAppearances"],
            "BABIP": ((n["hits"] - n["home runs"]) / bip) if (n["starting batter"] and bip) else 0,
            "batSide": bat_side if n["starting batter"] else 0,
        }

    @staticmethod
    def _accumulate_bullpen(bullpen, v):
        pit = v["stats"]["pitching"]
        bullpen["pitches thrown"] += pit.get("numberOfPitches", 0)
        bullpen["pitcher strikeouts"] += pit.get("strikeOuts", 0)
        bullpen["pitching outs"] += 3 * int(pit.get("inningsPitched", "0.0").split(".")[0]) + int(
            pit.get("inningsPitched", "0.0").split(".")[1]
        )
        bullpen["batters faced"] += pit.get("battersFaced", 0)
        bullpen["walks allowed"] += pit.get("baseOnBalls", 0) + pit.get("hitByPitch", 0)
        bullpen["hits allowed"] += pit.get("hits", 0)
        bullpen["home runs allowed"] += pit.get("homeRuns", 0)
        bullpen["runs allowed"] += pit.get("runs", 0)

    @staticmethod
    def _bullpen_adj(bullpen, bpf):
        return {
            "RA": bullpen["runs allowed"] / bpf["R"],
            "HA": bullpen["hits allowed"] / bpf["H"],
            "HRA": bullpen["home runs allowed"] / bpf["HR"],
            "BB": bullpen["walks allowed"] / bpf["BB"],
            "K": bullpen["pitcher strikeouts"] / bpf["K"],
        }

    @staticmethod
    def _team_row(
        team, opponent, gameId, game_date, boxscore, bpf, bullpen, bullpen_adj, *, me, opp
    ):
        """One team-level teamlog row; ``me``/``opp`` select the box-score side."""
        me_bat = boxscore["teams"][me]["teamStats"]["batting"]
        opp_bat = boxscore["teams"][opp]["teamStats"]["batting"]
        me_errors = boxscore["teams"][me]["teamStats"]["fielding"]["errors"]
        outs = bullpen["pitching outs"]
        return {
            "team": team,
            "opponent": opponent,
            "gameId": gameId,
            "gameDate": game_date,
            "WL": "W" if float(me_bat["runs"]) > float(opp_bat["runs"]) else "L",
            "runs": float(me_bat["runs"]),
            "OBP": float(me_bat["obp"]) / bpf["OBP"],
            "AVG": float(me_bat["avg"]),
            "SLG": float(me_bat["slg"]),
            "PASO": (me_bat["plateAppearances"] / me_bat["strikeOuts"])
            if me_bat["strikeOuts"]
            else me_bat["plateAppearances"],
            "BABIP": (me_bat["hits"] - me_bat["homeRuns"])
            / (me_bat["atBats"] - me_bat["strikeOuts"] - me_bat["homeRuns"] - me_bat["sacFlies"]),
            "DER": 1
            - (
                (opp_bat["hits"] + me_errors - opp_bat["homeRuns"])
                / (
                    opp_bat["plateAppearances"]
                    - opp_bat["baseOnBalls"]
                    - opp_bat["hitByPitch"]
                    - opp_bat["homeRuns"]
                    - opp_bat["strikeOuts"]
                )
            ),
            "FIP": (
                3 * (13 * bullpen_adj["HRA"] + 3 * bullpen_adj["BB"] - 2 * bullpen_adj["K"]) / outs
                + _FIP_CONSTANT
            )
            if outs
            else 0,
            "WHIP": (3 * (bullpen_adj["BB"] + bullpen_adj["HA"]) / outs) if outs else 0,
            "ERA": (9 * bullpen_adj["RA"] / outs) if outs else 0,
            "K9": (27 * bullpen_adj["K"] / outs) if outs else 0,
            "BB9": (27 * bullpen_adj["BB"] / outs) if outs else 0,
            "IP": outs / 3,
            "PA9": (27 * bullpen["batters faced"] / outs) if outs else 0,
        }

    def load(self):
        """Read the MLB gamelog bundle and the auxiliary park-factor / comp tables.

        Calls :meth:`Stats.load` first to populate ``gamelog`` / ``teamlog`` /
        ``players`` from the standard per-league artifact directory, then layers
        on the MLB-specific extras: the static park-factor lookup and the
        baseballsavant affinity CSVs that seed ``self.comps``. Other leagues
        compute their comps from rolling z-scored profiles; MLB inherits a
        fixed match-score table written by :meth:`update_player_comps`, so this
        loader is responsible for materializing it into the in-memory mapping.
        """
        super().load()

        filepath = pkg_resources.files(data) / "config" / "park_factor.json"
        if os.path.isfile(filepath):
            with open(filepath) as infile:
                self.park_factors = json.load(infile)

        filepath = (
            pkg_resources.files(data) / "player_data/MLB/affinity_pitchersBySHV_matchScores.csv"
        )
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df = df.loc[
                (df.key1.str[-1] == df.key2.str[-1])
                & (df.match_score >= _COMP_MATCH_SCORE_THRESHOLD)
            ]
            df.key1 = df.key1.str[:-2].astype(int)
            df.key2 = df.key2.str[:-2].astype(int)
            self.comps["pitchers"] = df.groupby("key1").apply(lambda x: x.key2.to_list()).to_dict()

        filepath = (
            pkg_resources.files(data)
            / "player_data/MLB/affinity_hittersByHittingProfile_matchScores.csv"
        )
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df = df.loc[
                (df.key1.str[-1] == df.key2.str[-1])
                & (df.match_score >= _COMP_MATCH_SCORE_THRESHOLD)
            ]
            df.key1 = df.key1.str[:-2].astype(int)
            df.key2 = df.key2.str[:-2].astype(int)
            self.comps["hitters"] = df.groupby("key1").apply(lambda x: x.key2.to_list()).to_dict()

    def update_player_comps(self, year=None):
        url = "https://baseballsavant.mlb.com/app/affinity/affinity_hittersByHittingProfile_matchScores.csv"
        res = requests.get(url)
        filepath = (
            pkg_resources.files(data)
            / "player_data/MLB/affinity_hittersByHittingProfile_matchScores.csv"
        )
        with open(filepath, "w") as outfile:
            outfile.write(res.text)

        url = "https://baseballsavant.mlb.com/app/affinity/affinity_pitchersBySHV_matchScores.csv"
        res = requests.get(url)
        filepath = (
            pkg_resources.files(data) / "player_data/MLB/affinity_pitchersBySHV_matchScores.csv"
        )
        with open(filepath, "w") as outfile:
            outfile.write(res.text)

    def update(self):
        """Fetch and append new MLB game logs, then trim to the rolling 4-year window.

        Queries a 60-day schedule window starting from the last logged date (or
        season start if the gamelog is empty), parses each Final-status game, and
        writes the updated gamelog/teamlog/players bundle to disk.
        """
        today = datetime.today().date()
        if self.gamelog.empty:
            next_day = self.season_start
        else:
            next_day = pd.to_datetime(self.gamelog.gameDate).max().date()
        next_day = max(next_day, self.season_start)
        next_day = min(next_day, today)
        end_date = next_day + timedelta(days=60)
        end_date = min(end_date, today)
        mlb_games = mlb.schedule(
            start_date=next_day.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d")
        )
        mlb_teams = mlb.get("teams", {"sportId": 1})
        self.upcoming_games = _build_mlb_upcoming_games(mlb_games, mlb_teams)

        prev_game_ids = [] if self.gamelog.empty else self.gamelog.gameId.unique()
        mlb_game_ids = _mlb_final_game_ids(mlb_games, prev_game_ids)

        for id in tqdm(mlb_game_ids, desc="Getting MLB Stats"):
            self.parse_game(id)

        game_type_map = {g["game_id"]: g["game_type"] for g in mlb_games}
        for log_name in ("gamelog", "teamlog"):
            log = getattr(self, log_name)
            stamped = log["gameId"].map(game_type_map)
            if "game type" in log:
                stamped = stamped.fillna(log["game type"])
            log["game type"] = stamped

        self._trim_old_games(today)

        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog["playerName"] = self.gamelog["playerName"].apply(remove_accents)
            self._enrich_team_markets(self.gamelog, date_col="gameDate", team_col="team")

        write_gamelog("mlb", self.gamelog, self.teamlog, self.players)

    def _trim_old_games(self, today):
        four_years_ago = today - timedelta(days=1461)
        self.gamelog = self.gamelog[
            self.gamelog["gameDate"].apply(
                lambda x: four_years_ago <= datetime.strptime(x, "%Y-%m-%d").date() <= today
            )
        ]
        self.gamelog = self.gamelog[~self.gamelog["opponent"].isin(["AL", "NL"])]
        self.teamlog = self.teamlog[
            self.teamlog["gameDate"].apply(
                lambda x: four_years_ago <= datetime.strptime(x, "%Y-%m-%d").date() <= today
            )
        ]
        self.gamelog.drop_duplicates(subset=["gameId", "playerId"], keep="last", inplace=True)
        self.teamlog.drop_duplicates(subset=["gameId", "team"], keep="last", inplace=True)

    def _playoff_flag(self, stats, date, teams):
        """Flag postseason games from the stamped ``game type`` (historical) or the
        ``Playoff`` bool on ``upcoming_games`` (upcoming). MLB game IDs carry no
        season-type code, so the type is stamped at fetch; rows logged before the
        column existed (pre-backfill) carry no type and read as regular (0).
        """
        if date < datetime.today().date():
            todays = (
                self.gamelog.loc[pd.to_datetime(self.gamelog["gameDate"]).dt.date == date]
                .drop_duplicates("playerName")
                .set_index("playerName")
            )
            if "game type" in todays:
                playoff = todays["game type"].reindex(stats.index).isin(_MLB_POSTSEASON_GAME_TYPES)
            else:
                playoff = pd.Series(False, index=stats.index)
        else:
            ug = self.upcoming_games
            playoff = pd.Series(
                {p: bool(ug.get(teams.get(p), {}).get("Playoff")) for p in stats.index}
            )
        stats["Playoff"] = playoff.astype(int)

    def _playoff_teamlog(self):
        """MLB game IDs carry no season-type code, so scope the series teamlog by the
        stamped ``game type``. Empty until the historical backfill stamps it.
        """
        if "game type" not in self.teamlog:
            return self.teamlog.iloc[0:0]
        return self.teamlog.loc[self.teamlog["game type"].isin(_MLB_POSTSEASON_GAME_TYPES)]

    def _series_games_to_win(self, playoff_teamlog, team, opp, date):
        """Wins to clinch by MLB postseason round (Wild Card best-of-3 .. World Series
        best-of-7). The round is the matchup's ``game type`` -- read from the realized
        teamlog row (historical) or ``upcoming_games`` (upcoming).
        """
        date_col = self.log_strings["date"]
        tl_dates = pd.to_datetime(playoff_teamlog[date_col]).dt.date
        this_game = playoff_teamlog.loc[
            (playoff_teamlog[self.log_strings["team"]] == team)
            & (playoff_teamlog[self.log_strings["opponent"]] == opp)
            & (tl_dates == date)
        ]
        if not this_game.empty:
            game_type = this_game["game type"].iloc[0]
        else:
            game_type = self.upcoming_games.get(team, {}).get("game type")
        return _MLB_SERIES_GAMES_TO_WIN.get(game_type, _MLB_DEFAULT_SERIES_WINS)

    def profile_market(self, market, date=datetime.today().date()):
        date = self._begin_profile_market(market, date)
        if date is None:
            return

        self.pitcherProfile = pd.DataFrame(columns=["z", "home", "moneyline gain", "totals gain"])

        # Filter non-starting pitchers or non-starting batters depending on the market
        if any(string in market for string in ["allowed", "pitch"]):
            gamelog = self.short_gamelog[self.short_gamelog["starting pitcher"]].copy()
        else:
            gamelog = self.short_gamelog[self.short_gamelog["starting batter"]].copy()

        # Filter players with at least 2 entries
        playerGroups = (
            gamelog.groupby("playerName")
            .filter(
                lambda x: (
                    (x[market].clip(0, 1).mean() > _MARKET_HIT_RATE_MIN) & (x[market].count() > 1)
                )
            )
            .groupby("playerName")
        )

        # defenseGroups = gamelog.groupby('opponent')
        defenseGroups = gamelog.groupby(["opponent", "gameId"])
        defenseGames = defenseGroups[[market, self.log_strings["home"], "moneyline", "totals"]].agg(
            {
                market: "sum",
                self.log_strings["home"]: lambda x: np.mean(x) > 0.5,
                "moneyline": "mean",
                "totals": "mean",
            }
        )
        defenseGroups = defenseGames.groupby("opponent")

        pitcherGroups = gamelog.groupby(["opponent pitcher", "gameId"])
        pitcherGames = pitcherGroups[[market, self.log_strings["home"], "moneyline", "totals"]].agg(
            {
                market: "sum",
                self.log_strings["home"]: lambda x: np.mean(x) > 0.5,
                "moneyline": "mean",
                "totals": "mean",
            }
        )
        pitcherGroups = (
            pitcherGames.groupby("opponent pitcher")
            .filter(lambda x: x[market].count() > 1)
            .groupby("opponent pitcher")
        )

        # Compute league average
        leagueavg = playerGroups[market].mean().mean()
        leaguestd = playerGroups[market].mean().std()
        if np.isnan(leagueavg) or np.isnan(leaguestd):
            return

        # Compute playerProfile DataFrame
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.playerProfile[["z", "home", "moneyline gain", "totals gain"]] = 0
            self.playerProfile["z"] = (playerGroups[market].mean() - leagueavg).div(leaguestd)
            self.playerProfile["home"] = (
                playerGroups.apply(lambda x: x.loc[x["home"], market].mean() / x[market].mean()) - 1
            )

            leagueavg = defenseGroups[market].mean().mean()
            leaguestd = defenseGroups[market].mean().std()
            self.defenseProfile[["avg", "z", "home", "moneyline gain", "totals gain", "comps"]] = 0
            self.defenseProfile["avg"] = defenseGroups[market].mean().div(leagueavg) - 1
            self.defenseProfile["z"] = (defenseGroups[market].mean() - leagueavg).div(leaguestd)
            self.defenseProfile["home"] = (
                defenseGroups.apply(
                    lambda x: x.loc[x["home"] == 1, market].mean() / x[market].mean()
                )
                - 1
            )

            leagueavg = pitcherGroups[market].mean().mean()
            leaguestd = pitcherGroups[market].mean().std()
            self.pitcherProfile[["avg", "z", "home", "moneyline gain", "totals gain"]] = 0
            self.pitcherProfile["avg"] = pitcherGroups[market].mean().div(leagueavg) - 1
            self.pitcherProfile["z"] = (pitcherGroups[market].mean() - leagueavg).div(leaguestd)
            self.pitcherProfile["home"] = (
                pitcherGroups.apply(
                    lambda x: x.loc[x["home"] == 1, market].mean() / x[market].mean()
                )
                - 1
            )

            self.playerProfile["moneyline gain"] = playerGroups.apply(
                lambda x: np.polyfit(
                    x.moneyline.fillna(0.5).values.astype(float) / 0.5
                    - x.moneyline.fillna(0.5).mean(),
                    x[market].values / x[market].mean() - 1,
                    1,
                )[0]
            )

            self.playerProfile["totals gain"] = playerGroups.apply(
                lambda x: np.polyfit(
                    x.totals.fillna(self.default_total).values.astype(float) / self.default_total
                    - x.totals.fillna(self.default_total).mean(),
                    x[market].values / x[market].mean() - 1,
                    1,
                )[0]
            )

            self.defenseProfile["moneyline gain"] = defenseGroups.apply(
                lambda x: np.polyfit(
                    x.moneyline.fillna(0.5).values.astype(float) / 0.5
                    - x.moneyline.fillna(0.5).mean(),
                    x[market].values / x[market].mean() - 1,
                    1,
                )[0]
            )

            self.defenseProfile["totals gain"] = defenseGroups.apply(
                lambda x: np.polyfit(
                    x.totals.fillna(self.default_total).values.astype(float) / self.default_total
                    - x.totals.fillna(self.default_total).mean(),
                    x[market].values / x[market].mean() - 1,
                    1,
                )[0]
            )

            self.pitcherProfile["moneyline gain"] = pitcherGroups.apply(
                lambda x: np.polyfit(
                    x.moneyline.fillna(0.5).values.astype(float) / 0.5
                    - x.moneyline.fillna(0.5).mean(),
                    x[market].values / x[market].mean() - 1,
                    1,
                )[0]
            )

            self.pitcherProfile["totals gain"] = pitcherGroups.apply(
                lambda x: np.polyfit(
                    x.totals.fillna(self.default_total).values.astype(float) / self.default_total
                    - x.totals.fillna(self.default_total).mean(),
                    x[market].values / x[market].mean() - 1,
                    1,
                )[0]
            )

        if not any(string in market for string in ["allowed", "pitch"]):
            self.pitcherProfile = self.pitcherProfile.join(
                self.playerProfile[self.stat_types["pitching"]]
            )

        self.defenseProfile.fillna(0.0, inplace=True)
        self.pitcherProfile.fillna(0.0, inplace=True)
        self.teamProfile.fillna(0.0, inplace=True)
        self.playerProfile.fillna(0.0, inplace=True)

    def get_volume_stats(self, offers, date=datetime.today().date(), pitcher=False):
        if not pitcher:
            self._project_plate_appearances(offers, date)
            return
        market = "pitches thrown"
        self.load_volume_model_params(
            offers,
            market,
            date,
            {
                "loc": f"proj {market} mean",
                "rate": f"proj {market} mean",
                "scale": f"proj {market} std",
            },
        )

    def _project_plate_appearances(self, offers, date=datetime.today().date()):
        """Structural batting-order plate-appearance projection (no trained model).

        ``get_depth`` resolves each hitter's slot into ``playerProfile["depth"]``
        (actual for settled games, posted-lineup or modal slot upcoming). This maps
        that slot through the measured home/away PA curve, scales by a bounded
        per-team offense multiplier, and writes ``proj plateAppearances mean/std``
        into ``playerProfile`` -- the same contract the pitcher volume model produces.
        """
        self.get_depth(offers, date)
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        profile = self.playerProfile

        if date < datetime.today().date():
            day = self.gamelog[pd.to_datetime(self.gamelog["gameDate"]).dt.date == date]
            home_map = (
                day.drop_duplicates("playerName").set_index("playerName")["home"].to_dict()
            )
        else:
            home_map = {
                p: self.upcoming_games.get(t, {}).get("Home")
                for p, t in profile["team"].items()
            }

        teams = set(profile["team"].dropna())
        adjustment = self._mlb_offense_adjustment(teams, offers, date, home_map)

        means, stds = {}, {}
        for player in profile.index:
            raw = profile.at[player, "depth"]
            slot = int(raw) if pd.notna(raw) and 1 <= raw <= 9 else 0
            if slot == 0:
                means[player] = SLOT_PA_LEAGUE_AVG
                stds[player] = SLOT_STD_UNKNOWN
                continue
            is_home = home_map.get(player)
            curve = SLOT_PA_ALL if is_home is None else SLOT_PA_HOME if is_home else SLOT_PA_AWAY
            team = profile.at[player, "team"]
            means[player] = curve[slot - 1] * adjustment.get(team, 1.0)
            stds[player] = SLOT_STD[slot - 1]

        profile["proj plateAppearances mean"] = pd.Series(means)
        profile["proj plateAppearances std"] = pd.Series(stds)
        profile.fillna(
            {
                "proj plateAppearances mean": SLOT_PA_LEAGUE_AVG,
                "proj plateAppearances std": SLOT_STD_UNKNOWN,
            },
            inplace=True,
        )

    def _mlb_offense_adjustment(self, teams, offers, date, home_map):
        """Bounded per-team PA multiplier (nominal 1.0).

        Blends an OBP-driven factor (team on-base talent x opposing-starter
        OBP-allowed x park) with a market anchor (book-implied team runs), then
        clips to +/-8%. A team missing OBP history falls back to the market factor
        alone; an unquoted game yields a neutral market factor via the archive default.
        """
        records = offers if isinstance(offers, list) else list(offers.values())
        opponent_of = {r["Team"]: r["Opponent"] for r in records}
        home_by_team = {
            r["Team"]: home_map.get(r["Player"]) for r in records
        }

        if date < datetime.today().date():
            day = self.gamelog[pd.to_datetime(self.gamelog["gameDate"]).dt.date == date]
            starter_of = {}
            for team in teams:
                named = day.loc[day[self.log_strings["team"]] == team, "opponent pitcher"]
                starter_of[team] = named.mode().iloc[0] if not named.mode().empty else None
        else:
            starter_of = {
                team: self.upcoming_games.get(team, {}).get("Opponent Pitcher")
                for team in teams
            }

        adjustment = {}
        for team in teams:
            obp = self._obp_factor(
                team, opponent_of.get(team), home_by_team.get(team), starter_of.get(team), date
            )
            market = self._market_factor(team, date)
            blended = (
                OBP_ADJ_WEIGHT * obp + MARKET_ADJ_WEIGHT * market if obp is not None else market
            )
            adjustment[team] = float(np.clip(blended, *OFFENSE_ADJ_CLIP))
        return adjustment

    def _obp_factor(self, team, opponent, is_home, opp_starter, date):
        """Expected-OBP PA factor: team talent x park x opposing-starter OBP-allowed.

        Returns ``None`` when the team has no recent OBP so the caller can fall back
        to the market anchor alone.
        """
        if team not in self.teamProfile.index:
            return None
        obp_exp = self.teamProfile.at[team, "OBP"]
        park_team = team if is_home else opponent
        obp_exp *= self.park_factors.get(park_team, {}).get("OBP", 1.0)
        if opp_starter:
            # The gamelog OBP column is ~0 on pitcher rows (it holds the pitcher's own
            # batting), so compute OBP-allowed from the populated counting columns.
            starts = self.gamelog[
                (self.gamelog["playerName"] == opp_starter)
                & self.gamelog["starting pitcher"]
                & (pd.to_datetime(self.gamelog["gameDate"]).dt.date < date)
            ].tail(_TEAM_OBP_WINDOW)
            faced = pd.to_numeric(starts["batters faced"], errors="coerce").sum()
            if faced > 0:
                on_base = (
                    pd.to_numeric(starts["hits allowed"], errors="coerce").sum()
                    + pd.to_numeric(starts["walks allowed"], errors="coerce").sum()
                )
                obp_exp *= (on_base / faced) / LG_AVG_OBP
        obp_exp = min(obp_exp, _OBP_POLE_GUARD)
        return (1 - LG_AVG_OBP) / (1 - obp_exp)

    def _market_factor(self, team, date):
        """Book-implied team runs relative to league average (neutral 1.0 when unquoted)."""
        date_str = date.strftime("%Y-%m-%d") if not isinstance(date, str) else date
        return archive.get_total(self.league, date_str, team) / LG_AVG_TEAM_TOTAL

    def check_combo_markets(self, market, player, date=datetime.today().date()):
        player_games = self.short_gamelog.loc[
            self.short_gamelog[self.log_strings["player"]] == player
        ]
        cv = stat_cv.get(self.league, {}).get(market, 1)
        dist = stat_dist.get(self.league, {}).get(market, "Gamma")
        if not isinstance(date, str):
            date = date.strftime("%Y-%m-%d")
        ev = 0
        if market in combo_props:
            ev = self._combo_market_ev(market, date, player, dist, cv)
        elif "fantasy" in market:
            ev = self._check_mlb_fantasy(market, date, player, dist, cv, player_games)
        return 0 if np.isnan(ev) else ev

    @staticmethod
    def _mlb_fantasy_props(market):
        if "pitcher" in market:
            if "underdog" in market:
                return [
                    ("pitcher win", 5),
                    ("pitcher strikeouts", 3),
                    ("runs allowed", -3),
                    ("pitching outs", 1),
                    ("quality start", 5),
                ]
            return [
                ("pitcher win", 6),
                ("pitcher strikeouts", 3),
                ("runs allowed", -3),
                ("pitching outs", 1),
                ("quality start", 4),
            ]
        if "underdog" in market:
            return [
                ("singles", 3),
                ("doubles", 6),
                ("triples", 8),
                ("home runs", 10),
                ("walks", 3),
                ("rbi", 2),
                ("runs", 2),
                ("stolen bases", 4),
            ]
        return [
            ("singles", 3),
            ("doubles", 5),
            ("triples", 8),
            ("home runs", 10),
            ("walks", 2),
            ("rbi", 2),
            ("runs", 2),
            ("stolen bases", 5),
        ]

    @staticmethod
    def _keep_positive(new, old):
        return new if not np.isnan(new) and new > 0 else old

    @staticmethod
    def _mlb_quality_start_ev(v_outs, v_runs, weight):
        if v_outs <= 0:
            return 0
        std = stat_cv.get("MLB", {}).get("quality start", 1) * v_outs
        p = norm.sf(18, v_outs, std) + norm.pdf(18, v_outs, std)
        p *= poisson.cdf(3, v_runs) if v_runs > 0 else 0.5
        return p * weight

    def _mlb_hits_proportional_ev(self, submarket, date, player, dist, cv, player_games, weight):
        hits_cv = stat_cv.get("MLB", {}).get("hits", 1)
        hits_dist = stat_dist.get("MLB", {}).get("hits", "Gamma")
        v = archive.get_ev("MLB", "hits", date, player)
        subline = archive.get_line("MLB", "hits", date, player)
        v = get_ev(subline, get_odds(subline, v, hits_dist, cv=hits_cv), cv=cv, dist=dist)
        share = (
            player_games[submarket].sum() / player_games["hits"].sum()
            if player_games["hits"].sum()
            else 0
        )
        return v * share * weight

    def _check_mlb_fantasy(self, market, date, player, dist, cv, player_games):
        ev = 0
        book_odds = False
        v_outs = 0
        v_runs = 0
        for submarket, weight in self._mlb_fantasy_props(market):
            sub_cv = stat_cv.get("MLB", {}).get(submarket, 1)
            sub_dist = stat_dist.get("MLB", {}).get(submarket, "Gamma")
            v = archive.get_ev("MLB", submarket, date, player)
            subline = archive.get_line("MLB", submarket, date, player)
            if submarket == "pitcher win":
                ev += (1 - get_odds(subline, v, sub_dist, cv=sub_cv)) * weight
            elif submarket == "quality start":
                ev += self._mlb_quality_start_ev(v_outs, v_runs, weight)
            elif submarket in ["singles", "doubles", "triples", "home runs"] and np.isnan(v):
                ev += self._mlb_hits_proportional_ev(
                    submarket, date, player, dist, cv, player_games, weight
                )
            else:
                v = self._convert_to_market_dist(v, subline, sub_cv, sub_dist, dist, cv)
                contribution, from_book = self._fantasy_default_contribution(
                    submarket, weight, v, subline, sub_cv, sub_dist, player_games
                )
                ev += contribution
                book_odds |= from_book

            if submarket == "runs allowed":
                v_runs = self._keep_positive(v, v_runs)
            if submarket == "pitching outs":
                v_outs = self._keep_positive(v, v_outs)

        return ev if book_odds else 0

    def get_depth(self, offers, date=datetime.today().date()):
        if isinstance(offers, dict):
            players = list(offers.keys())
            teams = {k: v["Team"] for k, v in offers.items()}
        else:
            players = [x["Player"] for x in offers]
            teams = {x["Player"]: x["Team"] for x in offers}

        for player in players.copy():
            if " + " in player.replace(" vs. ", " + "):
                split_players = player.replace(" vs. ", " + ").split(" + ")
                players.append(split_players[0])
                players.append(split_players[1])
                players.remove(player)

                split_teams = teams.pop(player).split("/")
                if len(split_teams) == 1:
                    split_teams = split_teams * 2

                teams[split_players[0]] = split_teams[0]
                teams[split_players[1]] = split_teams[1]

        players = set(players)
        self.base_profile(date)

        if date < datetime.today().date():
            games = self.gamelog.loc[pd.to_datetime(self.gamelog.gameDate).dt.date == date]
            games.index = games["playerName"]
            self.playerProfile["depth"] = games.loc[~games.index.duplicated(), "battingOrder"]
            self.playerProfile.fillna(0, inplace=True)

        else:
            depth = {}
            for player in list(players):
                order = self.upcoming_games.get(teams[player], {}).get("Batting Order", [])

                if player in order:
                    depth[player] = order.index(player) + 1
                else:
                    mode = self.short_gamelog.loc[
                        self.short_gamelog["playerName"] == player, "battingOrder"
                    ].mode()
                    if mode.empty:
                        continue

                    depth[player] = int(mode.iloc[-1])

            self.playerProfile["depth"] = depth
