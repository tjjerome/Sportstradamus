from sportstradamus.helpers import archive, get_ev
from sportstradamus.stats import StatsNHL
import pandas as pd
from itertools import combinations
from scipy.stats import poisson, skellam
import numpy as np
from datetime import datetime
from tqdm import tqdm

NHL = StatsNHL()
NHL.load()
NHL.update()
archive.__init__(True)
markets = [
    "saves",
    "goalsAgainst",
    "goalie fantasy points underdog",
    "goalie fantasy points parlay",
    "faceOffWins",
    "timeOnIce",
    # "goals",
    # "assists",
    # "points",
    # "hits",
    # "shots",
    # "sogBS",
    # "blocked",
    # "fantasy score",
    # "skater fantasy points underdog",
    # "skater fantasy points parlay"
]
for market in tqdm(markets, unit="markets", position=1):

    for game in tqdm(NHL.gamelog, desc=market, unit='game'):
        if (
            any([string in market for string in ["saves", "goalie", "Against"]])
            and game["position"] != "G"
        ):
            continue
        elif (
            not any([string in market for string in [
                    "saves", "goalie", "Against"]])
            and game["position"] == "G"
        ):
            continue
        gameDate = game['gameDate']
        NHL.bucket_stats(market, date=datetime.strptime(gameDate, "%Y-%m-%d"))
        player = game['playerName']
        if player not in NHL.playerStats:
            continue
        # line = np.rint(NHL.playerStats[player]['avg'])
        line = NHL.playerStats[player]['line']
        if market == "1st inning hits allowed":
            line = 1.5
        stats = [0.5] * 4
        if market not in archive.archive["NHL"]:
            archive.archive["NHL"][market] = {}
        if gameDate not in archive.archive["NHL"][market]:
            archive.archive["NHL"][market][gameDate] = {}
        archive.archive["NHL"][market][gameDate][player] = {
            line: stats}

archive.write()
