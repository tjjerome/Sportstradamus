from sportstradamus.helpers import archive, get_ev
from sportstradamus.stats import StatsNFL
import pandas as pd
from itertools import combinations
from scipy.stats import poisson, skellam
import numpy as np
from datetime import datetime
from tqdm import tqdm

NFL = StatsNFL()
NFL.load()
NFL.update()
archive.__init__("NFL")
markets = [
    "passing yards",
    "rushing yards",
    "receiving yards",
    "yards",
    "fantasy points prizepicks",
    "fantasy points underdog",
    "fantasy points parlayplay",
    "passing tds",
    "rushing tds",
    "receiving tds",
    "tds",
    "completions",
    "carries",
    "receptions",
    "interceptions",
    "attempts",
    "targets",
]
for market in tqdm(markets, unit="markets", position=1):

    for i, game in tqdm(NFL.gamelog.iterrows(), desc=market, unit='game', total=len(NFL.gamelog)):
        if (
            any([string in market for string in [
                "pass", "completions", "attempts", "interceptions"]])
            and game["position group"] != "QB"
        ):
            continue
        gameDate = game['gameday']
        if datetime.strptime(gameDate, '%Y-%m-%d') > datetime(2022, 9, 1):
            continue
        NFL.bucket_stats(market, date=datetime.strptime(gameDate, "%Y-%m-%d"))
        player = game['player display name']
        if player not in NFL.playerStats:
            continue
        # line = np.rint(NFL.playerStats[player]['avg'])
        line = NFL.playerStats[player]['line']
        stats = [0.5] * 4
        if market not in archive.archive["NFL"]:
            archive.archive["NFL"][market] = {}
        if gameDate not in archive.archive["NFL"][market]:
            archive.archive["NFL"][market][gameDate] = {}
        archive.archive["NFL"][market][gameDate][player] = {
            line: stats}

archive.write()
