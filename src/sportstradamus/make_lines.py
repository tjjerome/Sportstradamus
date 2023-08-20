from sportstradamus.helpers import archive, get_ev
from sportstradamus.stats import StatsMLB
import pandas as pd
from itertools import combinations
from scipy.stats import poisson, skellam
import numpy as np
from datetime import datetime
from tqdm import tqdm

MLB = StatsMLB()
MLB.load()
MLB.update()
archive.__init__(True)
markets = [
    "pitcher fantasy points underdog",
    "pitcher fantasy score"
    # "saves",
    # "goalsAgainst",
    # "goalie fantasy points underdog",
    # "goalie fantasy points parlay",
    # "faceOffWins",
    # "timeOnIce",
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

    for i, game in tqdm(MLB.gamelog.iterrows(), desc=market, unit='game', total=len(MLB.gamelog)):
        if (
            any([string in market for string in ["pitch", "allowed"]])
            and not game["starting pitcher"]
        ):
            continue
        elif (
            not any([string in market for string in ["pitch", "allowed"]])
            and not game["starting batter"]
        ):
            continue
        gameDate = game['gameId'][:10].replace("/", "-")
        MLB.bucket_stats(market, date=datetime.strptime(gameDate, "%Y-%m-%d"))
        player = game['playerName']
        if player not in MLB.playerStats:
            continue
        # line = np.rint(MLB.playerStats[player]['avg'])
        line = MLB.playerStats[player]['line']
        stats = [0.5] * 4
        if market not in archive.archive["MLB"]:
            archive.archive["MLB"][market] = {}
        if gameDate not in archive.archive["MLB"][market]:
            archive.archive["MLB"][market][gameDate] = {}
        archive.archive["MLB"][market][gameDate][player] = {
            line: stats}

archive.write()
