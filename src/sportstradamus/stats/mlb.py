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
        """Parses the game data for a given game ID.

        Args:
            gameId (int): The ID of the game to parse.
        """
        # game = mlb.boxscore_data(gameId)
        game = scraper.get(f"https://baseballsavant.mlb.com/gf?game_pk={gameId}")
        new_games = []
        if game:
            linescore = game["scoreboard"]["linescore"]
            boxscore = game["boxscore"]
            awayTeam = (
                game["away_team_data"]["abbreviation"].replace("AZ", "ARI").replace("WSH", "WAS")
            )
            homeTeam = (
                game["home_team_data"]["abbreviation"].replace("AZ", "ARI").replace("WSH", "WAS")
            )
            bpf = self.park_factors[homeTeam]
            awayPitcherId = game["away_pitcher_lineup"][0]
            awayPitcher = remove_accents(
                boxscore["teams"]["away"]["players"]["ID" + str(awayPitcherId)]["person"][
                    "fullName"
                ]
            )
            if awayPitcherId in self.players and "throws" in self.players[awayPitcherId]:
                awayPitcherHand = self.players[awayPitcherId]["throws"]
            else:
                if str(awayPitcherId) not in game["away_pitchers"]:
                    return
                awayPitcherHand = game["away_pitchers"][str(awayPitcherId)][0]["p_throws"]
                if awayPitcherId not in self.players:
                    self.players[awayPitcherId] = {"name": awayPitcher, "throws": awayPitcherHand}
                else:
                    self.players[awayPitcherId]["throws"] = awayPitcherHand
            homePitcherId = game["home_pitcher_lineup"][0]
            homePitcher = remove_accents(
                boxscore["teams"]["home"]["players"]["ID" + str(homePitcherId)]["person"][
                    "fullName"
                ]
            )
            if homePitcherId in self.players and "throws" in self.players[homePitcherId]:
                homePitcherHand = self.players[homePitcherId]["throws"]
            else:
                if str(homePitcherId) not in game["home_pitchers"]:
                    return
                homePitcherHand = game["home_pitchers"][str(homePitcherId)][0]["p_throws"]
                if homePitcherId not in self.players:
                    self.players[homePitcherId] = {"name": homePitcher, "throws": homePitcherHand}
                else:
                    self.players[homePitcherId]["throws"] = homePitcherHand
            awayInning1Runs = linescore["innings"][0]["away"]["runs"]
            homeInning1Runs = linescore["innings"][0]["home"]["runs"]
            awayInning1Hits = linescore["innings"][0]["away"]["hits"]
            homeInning1Hits = linescore["innings"][0]["home"]["hits"]

            away_bullpen = {
                k: 0
                for k in [
                    "pitches thrown",
                    "pitcher strikeouts",
                    "pitching outs",
                    "batters faced",
                    "walks allowed",
                    "hits allowed",
                    "home runs allowed",
                    "runs allowed",
                ]
            }
            for v in boxscore["teams"]["away"]["players"].values():
                if v["person"]["id"] == awayPitcherId or v.get("battingOrder"):
                    n = {
                        "gameId": gameId,
                        "gameDate": game["game_date"],
                        "playerId": v["person"]["id"],
                        "playerName": remove_accents(v["person"]["fullName"]),
                        "position": v.get("position", {"abbreviation": ""})["abbreviation"],
                        "team": awayTeam,
                        "opponent": homeTeam,
                        "opponent pitcher": homePitcher,
                        "opponent pitcher id": homePitcherId,
                        "opponent pitcher hand": homePitcherHand,
                        "home": False,
                        "starting pitcher": v["person"]["id"] == awayPitcherId,
                        "starting batter": int(v.get("battingOrder", "001")[2]) == 0,
                        "battingOrder": int(v.get("battingOrder", "000")[0]),
                        "hits": v["stats"]["batting"].get("hits", 0),
                        "total bases": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("doubles", 0)
                        + 2 * v["stats"]["batting"].get("triples", 0)
                        + 3 * v["stats"]["batting"].get("homeRuns", 0),
                        "singles": v["stats"]["batting"].get("hits", 0)
                        - v["stats"]["batting"].get("doubles", 0)
                        - v["stats"]["batting"].get("triples", 0)
                        - v["stats"]["batting"].get("homeRuns", 0),
                        "doubles": v["stats"]["batting"].get("doubles", 0),
                        "triples": v["stats"]["batting"].get("triples", 0),
                        "home runs": v["stats"]["batting"].get("homeRuns", 0),
                        "batter strikeouts": v["stats"]["batting"].get("strikeOuts", 0),
                        "runs": v["stats"]["batting"].get("runs", 0),
                        "rbi": v["stats"]["batting"].get("rbi", 0),
                        "hits+runs+rbi": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("runs", 0)
                        + v["stats"]["batting"].get("rbi", 0),
                        "walks": v["stats"]["batting"].get("baseOnBalls", 0)
                        + v["stats"]["batting"].get("hitByPitch", 0),
                        "stolen bases": v["stats"]["batting"].get("stolenBases", 0),
                        "atBats": v["stats"]["batting"].get("atBats", 0),
                        "plateAppearances": v["stats"]["batting"].get("plateAppearances", 0),
                        "pitcher strikeouts": v["stats"]["pitching"].get("strikeOuts", 0),
                        "pitcher win": v["stats"]["pitching"].get("wins", 0),
                        "walks allowed": v["stats"]["pitching"].get("baseOnBalls", 0)
                        + v["stats"]["pitching"].get("hitByPitch", 0),
                        "pitches thrown": v["stats"]["pitching"].get("numberOfPitches", 0),
                        "runs allowed": v["stats"]["pitching"].get("runs", 0),
                        "hits allowed": v["stats"]["pitching"].get("hits", 0),
                        "home runs allowed": v["stats"]["pitching"].get("homeRuns", 0),
                        "pitching outs": 3
                        * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1]),
                        "batters faced": v["stats"]["pitching"].get("battersFaced", 0),
                        "1st inning runs allowed": homeInning1Runs
                        if v["person"]["id"] == game["away_pitcher_lineup"][0]
                        else 0,
                        "1st inning hits allowed": homeInning1Hits
                        if v["person"]["id"] == game["away_pitcher_lineup"][0]
                        else 0,
                        "hitter fantasy score": 3 * v["stats"]["batting"].get("hits", 0)
                        + 2 * v["stats"]["batting"].get("doubles", 0)
                        + 5 * v["stats"]["batting"].get("triples", 0)
                        + 7 * v["stats"]["batting"].get("homeRuns", 0)
                        + 2 * v["stats"]["batting"].get("runs", 0)
                        + 2 * v["stats"]["batting"].get("rbi", 0)
                        + 2 * v["stats"]["batting"].get("baseOnBalls", 0)
                        + 2 * v["stats"]["batting"].get("hitByPitch", 0)
                        + 5 * v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy score": 6 * v["stats"]["pitching"].get("wins", 0)
                        + 3 * v["stats"]["pitching"].get("strikeOuts", 0)
                        - 3 * v["stats"]["pitching"].get("earnedRuns", 0)
                        + 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1])
                        + (
                            4
                            if int(
                                v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]
                            )
                            > 5
                            and v["stats"]["pitching"].get("earnedRuns", 0) < 4
                            else 0
                        ),
                        "hitter fantasy points underdog": 3 * v["stats"]["batting"].get("hits", 0)
                        + 3 * v["stats"]["batting"].get("doubles", 0)
                        + 5 * v["stats"]["batting"].get("triples", 0)
                        + 7 * v["stats"]["batting"].get("homeRuns", 0)
                        + 2 * v["stats"]["batting"].get("runs", 0)
                        + 2 * v["stats"]["batting"].get("rbi", 0)
                        + 3 * v["stats"]["batting"].get("baseOnBalls", 0)
                        + 3 * v["stats"]["batting"].get("hitByPitch", 0)
                        + 4 * v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points underdog": 5 * v["stats"]["pitching"].get("wins", 0)
                        + 3 * v["stats"]["pitching"].get("strikeOuts", 0)
                        - 3 * v["stats"]["pitching"].get("earnedRuns", 0)
                        + 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + (
                            5
                            if int(
                                v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]
                            )
                            > 5
                            and v["stats"]["pitching"].get("earnedRuns", 0) < 4
                            else 0
                        ),
                        "hitter fantasy points parlay": 3 * v["stats"]["batting"].get("hits", 0)
                        + 3 * v["stats"]["batting"].get("doubles", 0)
                        + 6 * v["stats"]["batting"].get("triples", 0)
                        + 9 * v["stats"]["batting"].get("homeRuns", 0)
                        + 3 * v["stats"]["batting"].get("runs", 0)
                        + 3 * v["stats"]["batting"].get("rbi", 0)
                        + 3 * v["stats"]["batting"].get("baseOnBalls", 0)
                        + 3 * v["stats"]["batting"].get("hitByPitch", 0)
                        + 6 * v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points parlay": 6 * v["stats"]["pitching"].get("wins", 0)
                        + 3 * v["stats"]["pitching"].get("strikeOuts", 0)
                        - 3 * v["stats"]["pitching"].get("earnedRuns", 0)
                        + 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1]),
                    }

                    if n["starting batter"]:
                        if n["playerId"] in self.players and "bats" in self.players[n["playerId"]]:
                            batSide = self.players[n["playerId"]]["bats"]
                        elif str(n["playerId"]) in game["away_batters"]:
                            batSide = game["away_batters"][str(n["playerId"])][0]["stand"]
                            if n["playerId"] not in self.players:
                                self.players[n["playerId"]] = {
                                    "name": n["playerName"],
                                    "bats": batSide,
                                }
                            else:
                                self.players[n["playerId"]]["bats"] = batSide
                        else:
                            continue

                    adj = {
                        "R": n["runs"] / bpf["R"],
                        "RBI": n["rbi"] / bpf["R"],
                        "H": n["hits"] / bpf["H"],
                        "1B": n["singles"] / bpf["1B"],
                        "2B": v["stats"]["batting"].get("doubles", 0) / bpf["2B"],
                        "3B": v["stats"]["batting"].get("triples", 0) / bpf["3B"],
                        "HR": n["home runs"] / bpf["HR"],
                        "W": n["walks"] / bpf["BB"],
                        "SO": n["batter strikeouts"] / bpf["K"],
                        "RA": n["runs allowed"] / bpf["R"],
                        "HA": n["hits allowed"] / bpf["H"],
                        "HRA": n["home runs allowed"] / bpf["HR"],
                        "BB": n["walks allowed"] / bpf["BB"],
                        "K": n["pitcher strikeouts"] / bpf["K"],
                    }

                    BIP = (
                        n["atBats"]
                        - n["batter strikeouts"]
                        - n["home runs"]
                        - v["stats"]["batting"].get("sacFlies", 0)
                    )
                    n.update(
                        {
                            "FIP": (
                                3
                                * (13 * adj["HRA"] + 3 * adj["BB"] - 2 * adj["K"])
                                / n["pitching outs"]
                                + 3.2
                            )
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "WHIP": (3 * (adj["BB"] + adj["HA"]) / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "ERA": (9 * adj["RA"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "K9": (27 * adj["K"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "BB9": (27 * adj["BB"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "PA9": (27 * n["batters faced"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "IP": (n["pitching outs"] / 3) if n["starting pitcher"] else 0,
                            "OBP": ((n["hits"] + n["walks"]) / n["atBats"] / bpf["OBP"])
                            if n["atBats"] > 0
                            else 0,
                            "AVG": (n["hits"] / n["atBats"]) if n["atBats"] > 0 else 0,
                            "SLG": (n["total bases"] / n["atBats"]) if n["atBats"] > 0 else 0,
                            "PASO": (n["plateAppearances"] / adj["SO"])
                            if (n["starting batter"] and adj["SO"])
                            else n["plateAppearances"],
                            "BABIP": ((n["hits"] - n["home runs"]) / BIP)
                            if (n["starting batter"] and BIP)
                            else 0,
                            "batSide": batSide if n["starting batter"] else 0,
                        }
                    )

                    new_games.append(n)

                elif v.get("position", {}).get("type", "") == "Pitcher":
                    away_bullpen["pitches thrown"] += v["stats"]["pitching"].get(
                        "numberOfPitches", 0
                    )
                    away_bullpen["pitcher strikeouts"] += v["stats"]["pitching"].get(
                        "strikeOuts", 0
                    )
                    away_bullpen["pitching outs"] += 3 * int(
                        v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]
                    ) + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1])
                    away_bullpen["batters faced"] += v["stats"]["pitching"].get("battersFaced", 0)
                    away_bullpen["walks allowed"] += v["stats"]["pitching"].get(
                        "baseOnBalls", 0
                    ) + v["stats"]["pitching"].get("hitByPitch", 0)
                    away_bullpen["hits allowed"] += v["stats"]["pitching"].get("hits", 0)
                    away_bullpen["home runs allowed"] += v["stats"]["pitching"].get("homeRuns", 0)
                    away_bullpen["runs allowed"] += v["stats"]["pitching"].get("runs", 0)

            home_bullpen = {
                k: 0
                for k in [
                    "pitches thrown",
                    "pitcher strikeouts",
                    "pitching outs",
                    "batters faced",
                    "walks allowed",
                    "hits allowed",
                    "home runs allowed",
                    "runs allowed",
                ]
            }
            for v in boxscore["teams"]["home"]["players"].values():
                if v["person"]["id"] == homePitcherId or v.get("battingOrder"):
                    n = {
                        "gameId": gameId,
                        "gameDate": game["game_date"],
                        "playerId": v["person"]["id"],
                        "playerName": remove_accents(v["person"]["fullName"]),
                        "position": v.get("position", {"abbreviation": ""})["abbreviation"],
                        "team": homeTeam,
                        "opponent": awayTeam,
                        "opponent pitcher": awayPitcher,
                        "opponent pitcher id": awayPitcherId,
                        "opponent pitcher hand": awayPitcherHand,
                        "home": True,
                        "starting pitcher": v["person"]["id"] == homePitcherId,
                        "starting batter": int(v.get("battingOrder", "001")[2]) == 0,
                        "battingOrder": int(v.get("battingOrder", "000")[0]),
                        "hits": v["stats"]["batting"].get("hits", 0),
                        "total bases": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("doubles", 0)
                        + 2 * v["stats"]["batting"].get("triples", 0)
                        + 3 * v["stats"]["batting"].get("homeRuns", 0),
                        "singles": v["stats"]["batting"].get("hits", 0)
                        - v["stats"]["batting"].get("doubles", 0)
                        - v["stats"]["batting"].get("triples", 0)
                        - v["stats"]["batting"].get("homeRuns", 0),
                        "doubles": v["stats"]["batting"].get("doubles", 0),
                        "triples": v["stats"]["batting"].get("triples", 0),
                        "home runs": v["stats"]["batting"].get("homeRuns", 0),
                        "batter strikeouts": v["stats"]["batting"].get("strikeOuts", 0),
                        "runs": v["stats"]["batting"].get("runs", 0),
                        "rbi": v["stats"]["batting"].get("rbi", 0),
                        "hits+runs+rbi": v["stats"]["batting"].get("hits", 0)
                        + v["stats"]["batting"].get("runs", 0)
                        + v["stats"]["batting"].get("rbi", 0),
                        "walks": v["stats"]["batting"].get("baseOnBalls", 0)
                        + v["stats"]["batting"].get("hitByPitch", 0),
                        "stolen bases": v["stats"]["batting"].get("stolenBases", 0),
                        "atBats": v["stats"]["batting"].get("atBats", 0),
                        "plateAppearances": v["stats"]["batting"].get("plateAppearances", 0),
                        "pitcher strikeouts": v["stats"]["pitching"].get("strikeOuts", 0),
                        "pitcher win": v["stats"]["pitching"].get("wins", 0),
                        "walks allowed": v["stats"]["pitching"].get("baseOnBalls", 0)
                        + v["stats"]["pitching"].get("hitByPitch", 0),
                        "pitches thrown": v["stats"]["pitching"].get("numberOfPitches", 0),
                        "runs allowed": v["stats"]["pitching"].get("runs", 0),
                        "hits allowed": v["stats"]["pitching"].get("hits", 0),
                        "home runs allowed": v["stats"]["pitching"].get("homeRuns", 0),
                        "pitching outs": 3
                        * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1]),
                        "batters faced": v["stats"]["pitching"].get("battersFaced", 0),
                        "1st inning runs allowed": awayInning1Runs
                        if v["person"]["id"] == game["home_pitcher_lineup"][0]
                        else 0,
                        "1st inning hits allowed": awayInning1Hits
                        if v["person"]["id"] == game["home_pitcher_lineup"][0]
                        else 0,
                        "hitter fantasy score": 3 * v["stats"]["batting"].get("hits", 0)
                        + 2 * v["stats"]["batting"].get("doubles", 0)
                        + 5 * v["stats"]["batting"].get("triples", 0)
                        + 7 * v["stats"]["batting"].get("homeRuns", 0)
                        + 2 * v["stats"]["batting"].get("runs", 0)
                        + 2 * v["stats"]["batting"].get("rbi", 0)
                        + 2 * v["stats"]["batting"].get("baseOnBalls", 0)
                        + 2 * v["stats"]["batting"].get("hitByPitch", 0)
                        + 5 * v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy score": 6 * v["stats"]["pitching"].get("wins", 0)
                        + 3 * v["stats"]["pitching"].get("strikeOuts", 0)
                        - 3 * v["stats"]["pitching"].get("earnedRuns", 0)
                        + 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1])
                        + (
                            4
                            if int(
                                v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]
                            )
                            > 5
                            and v["stats"]["pitching"].get("earnedRuns", 0) < 4
                            else 0
                        ),
                        "hitter fantasy points underdog": 3 * v["stats"]["batting"].get("hits", 0)
                        + 3 * v["stats"]["batting"].get("doubles", 0)
                        + 5 * v["stats"]["batting"].get("triples", 0)
                        + 7 * v["stats"]["batting"].get("homeRuns", 0)
                        + 2 * v["stats"]["batting"].get("runs", 0)
                        + 2 * v["stats"]["batting"].get("rbi", 0)
                        + 3 * v["stats"]["batting"].get("baseOnBalls", 0)
                        + 3 * v["stats"]["batting"].get("hitByPitch", 0)
                        + 4 * v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points underdog": 5 * v["stats"]["pitching"].get("wins", 0)
                        + 3 * v["stats"]["pitching"].get("strikeOuts", 0)
                        - 3 * v["stats"]["pitching"].get("earnedRuns", 0)
                        + 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + (
                            5
                            if int(
                                v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]
                            )
                            > 5
                            and v["stats"]["pitching"].get("earnedRuns", 0) < 4
                            else 0
                        ),
                        "hitter fantasy points parlay": 3 * v["stats"]["batting"].get("hits", 0)
                        + 3 * v["stats"]["batting"].get("doubles", 0)
                        + 6 * v["stats"]["batting"].get("triples", 0)
                        + 9 * v["stats"]["batting"].get("homeRuns", 0)
                        + 3 * v["stats"]["batting"].get("runs", 0)
                        + 3 * v["stats"]["batting"].get("rbi", 0)
                        + 3 * v["stats"]["batting"].get("baseOnBalls", 0)
                        + 3 * v["stats"]["batting"].get("hitByPitch", 0)
                        + 6 * v["stats"]["batting"].get("stolenBases", 0),
                        "pitcher fantasy points parlay": 6 * v["stats"]["pitching"].get("wins", 0)
                        + 3 * v["stats"]["pitching"].get("strikeOuts", 0)
                        - 3 * v["stats"]["pitching"].get("earnedRuns", 0)
                        + 3 * int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0])
                        + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1]),
                    }

                    if n["starting batter"]:
                        if n["playerId"] in self.players and "bats" in self.players[n["playerId"]]:
                            batSide = self.players[n["playerId"]]["bats"]
                        elif str(n["playerId"]) in game["home_batters"]:
                            batSide = game["home_batters"][str(n["playerId"])][0]["stand"]
                            if n["playerId"] not in self.players:
                                self.players[n["playerId"]] = {
                                    "name": n["playerName"],
                                    "bats": batSide,
                                }
                            else:
                                self.players[n["playerId"]]["bats"] = batSide
                        else:
                            continue

                    adj = {
                        "R": n["runs"] / bpf["R"],
                        "RBI": n["rbi"] / bpf["R"],
                        "H": n["hits"] / bpf["H"],
                        "1B": n["singles"] / bpf["1B"],
                        "2B": v["stats"]["batting"].get("doubles", 0) / bpf["2B"],
                        "3B": v["stats"]["batting"].get("triples", 0) / bpf["3B"],
                        "HR": n["home runs"] / bpf["HR"],
                        "W": n["walks"] / bpf["BB"],
                        "SO": n["batter strikeouts"] / bpf["K"],
                        "RA": n["runs allowed"] / bpf["R"],
                        "HA": n["hits allowed"] / bpf["H"],
                        "HRA": n["home runs allowed"] / bpf["HR"],
                        "BB": n["walks allowed"] / bpf["BB"],
                        "K": n["pitcher strikeouts"] / bpf["K"],
                    }

                    BIP = (
                        n["atBats"]
                        - n["batter strikeouts"]
                        - n["home runs"]
                        - v["stats"]["batting"].get("sacFlies", 0)
                    )
                    n.update(
                        {
                            "FIP": (
                                3
                                * (13 * adj["HRA"] + 3 * adj["BB"] - 2 * adj["K"])
                                / n["pitching outs"]
                                + 3.2
                            )
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "WHIP": (3 * (adj["BB"] + adj["HA"]) / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "ERA": (9 * adj["RA"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "K9": (27 * adj["K"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "BB9": (27 * adj["BB"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "PA9": (27 * n["batters faced"] / n["pitching outs"])
                            if (n["starting pitcher"] and n["pitching outs"])
                            else 0,
                            "IP": (n["pitching outs"] / 3) if n["starting pitcher"] else 0,
                            "OBP": ((n["hits"] + n["walks"]) / n["atBats"] / bpf["OBP"])
                            if n["atBats"] > 0
                            else 0,
                            "AVG": (n["hits"] / n["atBats"]) if n["atBats"] > 0 else 0,
                            "SLG": (n["total bases"] / n["atBats"]) if n["atBats"] > 0 else 0,
                            "PASO": (n["plateAppearances"] / adj["SO"])
                            if (n["starting batter"] and adj["SO"])
                            else n["plateAppearances"],
                            "BABIP": ((n["hits"] - n["home runs"]) / BIP)
                            if (n["starting batter"] and BIP)
                            else 0,
                            "batSide": batSide if n["starting batter"] else 0,
                        }
                    )

                    new_games.append(n)

                elif v.get("position", {}).get("type", "") == "Pitcher":
                    home_bullpen["pitches thrown"] += v["stats"]["pitching"].get(
                        "numberOfPitches", 0
                    )
                    home_bullpen["pitcher strikeouts"] += v["stats"]["pitching"].get(
                        "strikeOuts", 0
                    )
                    home_bullpen["pitching outs"] += 3 * int(
                        v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[0]
                    ) + int(v["stats"]["pitching"].get("inningsPitched", "0.0").split(".")[1])
                    home_bullpen["batters faced"] += v["stats"]["pitching"].get("battersFaced", 0)
                    home_bullpen["walks allowed"] += v["stats"]["pitching"].get(
                        "baseOnBalls", 0
                    ) + v["stats"]["pitching"].get("hitByPitch", 0)
                    home_bullpen["hits allowed"] += v["stats"]["pitching"].get("hits", 0)
                    home_bullpen["home runs allowed"] += v["stats"]["pitching"].get("homeRuns", 0)
                    home_bullpen["runs allowed"] += v["stats"]["pitching"].get("runs", 0)

        home_adj = {
            "RA": home_bullpen["runs allowed"] / bpf["R"],
            "HA": home_bullpen["hits allowed"] / bpf["H"],
            "HRA": home_bullpen["home runs allowed"] / bpf["HR"],
            "BB": home_bullpen["walks allowed"] / bpf["BB"],
            "K": home_bullpen["pitcher strikeouts"] / bpf["K"],
        }

        away_adj = {
            "RA": away_bullpen["runs allowed"] / bpf["R"],
            "HA": away_bullpen["hits allowed"] / bpf["H"],
            "HRA": away_bullpen["home runs allowed"] / bpf["HR"],
            "BB": away_bullpen["walks allowed"] / bpf["BB"],
            "K": away_bullpen["pitcher strikeouts"] / bpf["K"],
        }

        new_games = pd.DataFrame.from_records(new_games)
        new_games.loc[:, "moneyline"] = new_games.apply(
            lambda x: archive.get_moneyline(self.league, x["gameDate"], x["team"]), axis=1
        )
        new_games.loc[:, "totals"] = new_games.apply(
            lambda x: archive.get_total(self.league, x["gameDate"], x["team"]), axis=1
        )
        self.gamelog = pd.concat([self.gamelog, new_games], ignore_index=True)

        teams = [
            {
                "team": homeTeam,
                "opponent": awayTeam,
                "gameId": gameId,
                "gameDate": game["game_date"],
                "WL": "W"
                if float(boxscore["teams"]["home"]["teamStats"]["batting"]["runs"])
                > float(boxscore["teams"]["away"]["teamStats"]["batting"]["runs"])
                else "L",
                "runs": float(boxscore["teams"]["home"]["teamStats"]["batting"]["runs"]),
                "OBP": float(boxscore["teams"]["home"]["teamStats"]["batting"]["obp"]) / bpf["OBP"],
                "AVG": float(boxscore["teams"]["home"]["teamStats"]["batting"]["avg"]),
                "SLG": float(boxscore["teams"]["home"]["teamStats"]["batting"]["slg"]),
                "PASO": (
                    boxscore["teams"]["home"]["teamStats"]["batting"]["plateAppearances"]
                    / boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"]
                )
                if boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"]
                else boxscore["teams"]["home"]["teamStats"]["batting"]["plateAppearances"],
                "BABIP": (
                    boxscore["teams"]["home"]["teamStats"]["batting"]["hits"]
                    - boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"]
                )
                / (
                    boxscore["teams"]["home"]["teamStats"]["batting"]["atBats"]
                    - boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"]
                    - boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"]
                    - boxscore["teams"]["home"]["teamStats"]["batting"]["sacFlies"]
                ),
                "DER": 1
                - (
                    (
                        boxscore["teams"]["away"]["teamStats"]["batting"]["hits"]
                        + boxscore["teams"]["home"]["teamStats"]["fielding"]["errors"]
                        - boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"]
                    )
                    / (
                        boxscore["teams"]["away"]["teamStats"]["batting"]["plateAppearances"]
                        - boxscore["teams"]["away"]["teamStats"]["batting"]["baseOnBalls"]
                        - boxscore["teams"]["away"]["teamStats"]["batting"]["hitByPitch"]
                        - boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"]
                        - boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"]
                    )
                ),
                "FIP": (
                    3
                    * (13 * home_adj["HRA"] + 3 * home_adj["BB"] - 2 * home_adj["K"])
                    / home_bullpen["pitching outs"]
                    + 3.2
                )
                if home_bullpen["pitching outs"]
                else 0,
                "WHIP": (3 * (home_adj["BB"] + home_adj["HA"]) / home_bullpen["pitching outs"])
                if home_bullpen["pitching outs"]
                else 0,
                "ERA": (9 * home_adj["RA"] / home_bullpen["pitching outs"])
                if home_bullpen["pitching outs"]
                else 0,
                "K9": (27 * home_adj["K"] / home_bullpen["pitching outs"])
                if home_bullpen["pitching outs"]
                else 0,
                "BB9": (27 * home_adj["BB"] / home_bullpen["pitching outs"])
                if home_bullpen["pitching outs"]
                else 0,
                "IP": home_bullpen["pitching outs"] / 3,
                "PA9": (27 * home_bullpen["batters faced"] / home_bullpen["pitching outs"])
                if home_bullpen["pitching outs"]
                else 0,
            },
            {
                "team": awayTeam,
                "opponent": homeTeam,
                "gameId": gameId,
                "gameDate": game["game_date"],
                "WL": "W"
                if float(boxscore["teams"]["away"]["teamStats"]["batting"]["runs"])
                > float(boxscore["teams"]["home"]["teamStats"]["batting"]["runs"])
                else "L",
                "runs": float(boxscore["teams"]["away"]["teamStats"]["batting"]["runs"]),
                "OBP": float(boxscore["teams"]["away"]["teamStats"]["batting"]["obp"]) / bpf["OBP"],
                "AVG": float(boxscore["teams"]["away"]["teamStats"]["batting"]["avg"]),
                "SLG": float(boxscore["teams"]["away"]["teamStats"]["batting"]["slg"]),
                "PASO": (
                    boxscore["teams"]["away"]["teamStats"]["batting"]["plateAppearances"]
                    / boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"]
                )
                if boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"]
                else boxscore["teams"]["away"]["teamStats"]["batting"]["plateAppearances"],
                "BABIP": (
                    boxscore["teams"]["away"]["teamStats"]["batting"]["hits"]
                    - boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"]
                )
                / (
                    boxscore["teams"]["away"]["teamStats"]["batting"]["atBats"]
                    - boxscore["teams"]["away"]["teamStats"]["batting"]["strikeOuts"]
                    - boxscore["teams"]["away"]["teamStats"]["batting"]["homeRuns"]
                    - boxscore["teams"]["away"]["teamStats"]["batting"]["sacFlies"]
                ),
                "DER": 1
                - (
                    (
                        boxscore["teams"]["home"]["teamStats"]["batting"]["hits"]
                        + boxscore["teams"]["away"]["teamStats"]["fielding"]["errors"]
                        - boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"]
                    )
                    / (
                        boxscore["teams"]["home"]["teamStats"]["batting"]["plateAppearances"]
                        - boxscore["teams"]["home"]["teamStats"]["batting"]["baseOnBalls"]
                        - boxscore["teams"]["home"]["teamStats"]["batting"]["hitByPitch"]
                        - boxscore["teams"]["home"]["teamStats"]["batting"]["homeRuns"]
                        - boxscore["teams"]["home"]["teamStats"]["batting"]["strikeOuts"]
                    )
                ),
                "FIP": (
                    3
                    * (13 * away_adj["HRA"] + 3 * away_adj["BB"] - 2 * away_adj["K"])
                    / away_bullpen["pitching outs"]
                    + 3.2
                )
                if away_bullpen["pitching outs"]
                else 0,
                "WHIP": (3 * (away_adj["BB"] + away_adj["HA"]) / away_bullpen["pitching outs"])
                if away_bullpen["pitching outs"]
                else 0,
                "ERA": (9 * away_adj["RA"] / away_bullpen["pitching outs"])
                if away_bullpen["pitching outs"]
                else 0,
                "K9": (27 * away_adj["K"] / away_bullpen["pitching outs"])
                if away_bullpen["pitching outs"]
                else 0,
                "BB9": (27 * away_adj["BB"] / away_bullpen["pitching outs"])
                if away_bullpen["pitching outs"]
                else 0,
                "IP": away_bullpen["pitching outs"] / 3,
                "PA9": (27 * away_bullpen["batters faced"] / away_bullpen["pitching outs"])
                if away_bullpen["pitching outs"]
                else 0,
            },
        ]

        self.teamlog = pd.concat(
            [self.teamlog, pd.DataFrame.from_records(teams)], ignore_index=True
        )

    def load(self):
        """Loads MLB player statistics from a file."""
        filepath = pkg_resources.files(data) / "mlb_data.dat"
        if os.path.isfile(filepath):
            with open(filepath, "rb") as infile:
                mlb_data = pickle.load(infile)
                self.gamelog = mlb_data["gamelog"]
                self.teamlog = mlb_data["teamlog"]
                self.players = mlb_data["players"]

        filepath = pkg_resources.files(data) / "park_factor.json"
        if os.path.isfile(filepath):
            with open(filepath) as infile:
                self.park_factors = json.load(infile)

        filepath = (
            pkg_resources.files(data) / "player_data/MLB/affinity_pitchersBySHV_matchScores.csv"
        )
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df = df.loc[(df.key1.str[-1] == df.key2.str[-1]) & (df.match_score >= 0.6)]
            df.key1 = df.key1.str[:-2].astype(int)
            df.key2 = df.key2.str[:-2].astype(int)
            self.comps["pitchers"] = df.groupby("key1").apply(lambda x: x.key2.to_list()).to_dict()

        filepath = (
            pkg_resources.files(data)
            / "player_data/MLB/affinity_hittersByHittingProfile_matchScores.csv"
        )
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df = df.loc[(df.key1.str[-1] == df.key2.str[-1]) & (df.match_score >= 0.6)]
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
        """Updates the MLB player statistics."""
        # Get the current MLB schedule
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
        mlb_upcoming_games = {}
        for game in mlb_games:
            if game["status"] in ["Pre-Game", "Scheduled"] and game["game_type"] not in ["A"]:
                awayTeam = next(
                    team["abbreviation"].replace("AZ", "ARI")
                    for team in mlb_teams["teams"]
                    if team["id"] == game["away_id"]
                )
                homeTeam = next(
                    team["abbreviation"].replace("AZ", "ARI")
                    for team in mlb_teams["teams"]
                    if team["id"] == game["home_id"]
                )
                game_bs = mlb.boxscore_data(game["game_id"])
                players = {p["id"]: p["fullName"] for k, p in game_bs["playerInfo"].items()}
                if game["game_num"] == 1:
                    mlb_upcoming_games[awayTeam] = {
                        "Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Home": False,
                        "Opponent": homeTeam,
                        "Opponent Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Batting Order": [players[i] for i in game_bs["away"]["battingOrder"]],
                    }
                    mlb_upcoming_games[homeTeam] = {
                        "Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Home": True,
                        "Opponent": awayTeam,
                        "Opponent Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Batting Order": [players[i] for i in game_bs["home"]["battingOrder"]],
                    }
                elif game["game_num"] > 1:
                    mlb_upcoming_games[awayTeam + str(game["game_num"])] = {
                        "Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Home": False,
                        "Opponent": homeTeam,
                        "Opponent Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Batting Order": [players[i] for i in game_bs["away"]["battingOrder"]],
                    }
                    mlb_upcoming_games[homeTeam + str(game["game_num"])] = {
                        "Pitcher": remove_accents(game["home_probable_pitcher"]),
                        "Home": True,
                        "Opponent": awayTeam,
                        "Opponent Pitcher": remove_accents(game["away_probable_pitcher"]),
                        "Batting Order": [players[i] for i in game_bs["home"]["battingOrder"]],
                    }

        self.upcoming_games = mlb_upcoming_games

        prev_game_ids = [] if self.gamelog.empty else self.gamelog.gameId.unique()
        mlb_game_ids = [
            game["game_id"]
            for game in mlb_games
            if game["status"] == "Final"
            and game["game_type"] != "E"
            and game["game_type"] != "S"
            and game["game_type"] != "A"
            and game["game_id"] not in prev_game_ids
        ]

        # Parse the game stats
        for id in tqdm(mlb_game_ids, desc="Getting MLB Stats"):
            self.parse_game(id)

        # Remove old games to prevent file bloat
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

        if self.season_start < datetime.today().date() - timedelta(days=300) or clean_data:
            self.gamelog["playerName"] = self.gamelog["playerName"].apply(remove_accents)
            self.gamelog.loc[:, "moneyline"] = self.gamelog.apply(
                lambda x: archive.get_moneyline(self.league, x["gameDate"][:10], x["team"]), axis=1
            )
            self.gamelog.loc[:, "totals"] = self.gamelog.apply(
                lambda x: archive.get_total(self.league, x["gameDate"][:10], x["team"]), axis=1
            )

        # Write to file
        with open(pkg_resources.files(data) / "mlb_data.dat", "wb") as outfile:
            pickle.dump(
                {"players": self.players, "gamelog": self.gamelog, "teamlog": self.teamlog},
                outfile,
                -1,
            )


    def profile_market(self, market, date=datetime.today().date()):
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        elif isinstance(date, datetime):
            date = date.date()
        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.base_profile(date)
        self.profiled_market = market

        self.pitcherProfile = pd.DataFrame(columns=["z", "home", "moneyline gain", "totals gain"])

        # Filter non-starting pitchers or non-starting batters depending on the market
        if any([string in market for string in ["allowed", "pitch"]]):
            gamelog = self.short_gamelog[self.short_gamelog["starting pitcher"]].copy()
        else:
            gamelog = self.short_gamelog[self.short_gamelog["starting batter"]].copy()

        # Filter players with at least 2 entries
        playerGroups = (
            gamelog.groupby("playerName")
            .filter(lambda x: (x[market].clip(0, 1).mean() > 0.1) & (x[market].count() > 1))
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

        if not any([string in market for string in ["allowed", "pitch"]]):
            self.pitcherProfile = self.pitcherProfile.join(
                self.playerProfile[self.stat_types["pitching"]]
            )

        self.defenseProfile.fillna(0.0, inplace=True)
        self.pitcherProfile.fillna(0.0, inplace=True)
        self.teamProfile.fillna(0.0, inplace=True)
        self.playerProfile.fillna(0.0, inplace=True)


    def get_volume_stats(self, offers, date=datetime.today().date(), pitcher=False):
        flat_offers = {}
        if isinstance(offers, dict):
            for players in offers.values():
                flat_offers.update(players)
        else:
            flat_offers = offers

        market = "pitches thrown" if pitcher else "plateAppearances"

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

        # Drop gate column for ZI distributions — not needed for downstream use
        prob_params.drop(columns=["gate"], inplace=True, errors="ignore")
        self.playerProfile = self.playerProfile.join(
            prob_params.rename(
                columns={
                    "loc": f"proj {market} mean",
                    "rate": f"proj {market} mean",
                    "scale": f"proj {market} std",
                }
            ),
            lsuffix="_obs",
        )
        self.playerProfile.drop(
            columns=[col for col in self.playerProfile.columns if "_obs" in col], inplace=True
        )

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

        elif "fantasy" in market:
            book_odds = False
            if "pitcher" in market:
                if "underdog" in market:
                    fantasy_props = [
                        ("pitcher win", 5),
                        ("pitcher strikeouts", 3),
                        ("runs allowed", -3),
                        ("pitching outs", 1),
                        ("quality start", 5),
                    ]
                else:
                    fantasy_props = [
                        ("pitcher win", 6),
                        ("pitcher strikeouts", 3),
                        ("runs allowed", -3),
                        ("pitching outs", 1),
                        ("quality start", 4),
                    ]
            elif "underdog" in market:
                fantasy_props = [
                    ("singles", 3),
                    ("doubles", 6),
                    ("triples", 8),
                    ("home runs", 10),
                    ("walks", 3),
                    ("rbi", 2),
                    ("runs", 2),
                    ("stolen bases", 4),
                ]
            else:
                fantasy_props = [
                    ("singles", 3),
                    ("doubles", 5),
                    ("triples", 8),
                    ("home runs", 10),
                    ("walks", 2),
                    ("rbi", 2),
                    ("runs", 2),
                    ("stolen bases", 5),
                ]

            v_outs = 0
            v_runs = 0
            for submarket, weight in fantasy_props:
                sub_cv = stat_cv["MLB"].get(submarket, 1)
                sub_dist = stat_dist.get("MLB", {}).get(submarket, "Gamma")
                v = archive.get_ev("MLB", submarket, date, player)
                subline = archive.get_line("MLB", submarket, date, player)
                if submarket == "pitcher win":
                    p = 1 - get_odds(subline, v, sub_dist, cv=sub_cv)
                    ev += p * weight
                elif submarket == "quality start":
                    if v_outs > 0:
                        std = stat_cv["MLB"].get(submarket, 1) * v_outs
                        p = norm.sf(18, v_outs, std) + norm.pdf(18, v_outs, std)
                        p *= poisson.cdf(3, v_runs) if v_runs > 0 else 0.5
                        ev += p * weight
                elif submarket in ["singles", "doubles", "triples", "home runs"] and np.isnan(v):
                    _hits_cv = stat_cv.get("MLB", {}).get("hits", 1)
                    _hits_dist = stat_dist.get("MLB", {}).get("hits", "Gamma")
                    v = archive.get_ev("MLB", "hits", date, player)
                    subline = archive.get_line("MLB", "hits", date, player)
                    v = get_ev(
                        subline, get_odds(subline, v, _hits_dist, cv=_hits_cv), cv=cv, dist=dist
                    )
                    v *= (
                        (player_games[submarket].sum() / player_games["hits"].sum())
                        if player_games["hits"].sum()
                        else 0
                    )
                    ev += v * weight
                else:
                    if sub_dist != dist and not np.isnan(v):
                        v = get_ev(
                            subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist
                        )

                    if np.isnan(v) or v == 0:
                        if subline == 0 and not player_games.empty:
                            subline = np.floor(player_games.iloc[-10:][submarket].median()) + 0.5

                        if subline != 0:
                            under = (player_games[submarket] < subline).mean()
                            ev += get_ev(subline, under, sub_cv, dist=sub_dist) * weight
                    else:
                        book_odds = True
                        ev += v * weight

                if submarket == "runs allowed":
                    v_runs = v if not np.isnan(v) and v > 0 else v_runs
                if submarket == "pitching outs":
                    v_outs = v if not np.isnan(v) and v > 0 else v_outs

            if not book_odds:
                ev = 0

        return 0 if np.isnan(ev) else ev

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

