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

markets = ["batter strikeouts"]
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
        stats = np.append(np.zeros(5), [0.5] * 4)
        if gameDate not in archive.archive["MLB"][market]:
            archive.archive["MLB"][market][gameDate] = {}
        archive.archive["MLB"][market][gameDate][player] = {
            line: stats}

archive.write()
