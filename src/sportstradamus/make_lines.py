from sportstradamus.helpers import archive, get_ev
from sportstradamus.stats import StatsMLB
import pandas as pd
from itertools import combinations
from scipy.stats import poisson, skellam
import numpy as np
from datetime import datetime
from tqdm import tqdm

mlb = StatsMLB()
mlb.load()
mlb.update()

df = pd.DataFrame(mlb.gamelog)

markets = ["hitter fantasy score",
           "pitcher fantasy score",
           "hitter fantasy points underdog",
           "pitcher fantasy points underdog",
           "hitter fantasy points parlay",
           "pitcher fantasy points parlay",
           "1st inning hits allowed"]
for market in tqdm(markets, unit="markets", position=1):
    mlb.bucket_stats(market)

    for game in mlb.gamelog:
        if not game['starting batter']:
            continue
        gameDate = game['gameId'][:10].replace('/', '-')
        player = game['playerName']
        if player not in mlb.playerStats:
            continue
        # line = np.rint(mlb.playerStats[player]['avg'])
        line = mlb.playerStats[player]['line']
        if market == "1st inning hits allowed":
            line = 1.5
        stats = np.append([0.5] * 4)
        if market not in archive.archive["MLB"]:
            archive.archive["MLB"][market] = {}
        if gameDate not in archive.archive["MLB"][market]:
            archive.archive["MLB"][market][gameDate] = {}
        archive.archive["MLB"][market][gameDate][player] = {
            line: stats}

archive.write()
