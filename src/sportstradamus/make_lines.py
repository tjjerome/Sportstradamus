from sportstradamus.helpers import archive, get_ev
from sportstradamus.stats import StatsNBA
import pandas as pd
from itertools import combinations
from scipy.stats import poisson, skellam
import numpy as np
from datetime import datetime
from tqdm import tqdm

NBA = StatsNBA()
NBA.load()
NBA.update()

markets = ["FG3A", "FTM", "FGM", "FGA", "OREB", "DREB", "PF", "MIN"]
for market in tqdm(markets, unit="markets", position=1):
    NBA.bucket_stats(market)

    for game in NBA.gamelog:
        # if (
        #     any([string in market for string in ["saves", "goalie", "Against"]])
        #     and game["position"] != "G"
        # ):
        #     continue
        # elif (
        #     not any([string in market for string in [
        #             "saves", "goalie", "Against"]])
        #     and game["position"] == "G"
        # ):
        #     continue
        gameDate = game['GAME_DATE']
        player = game['PLAYER_NAME']
        if player not in NBA.playerStats:
            continue
        # line = np.rint(NBA.playerStats[player]['avg'])
        line = NBA.playerStats[player]['line']
        if market == "1st inning hits allowed":
            line = 1.5
        stats = [0.5] * 4
        if market not in archive.archive["NBA"]:
            archive.archive["NBA"][market] = {}
        if gameDate not in archive.archive["NBA"][market]:
            archive.archive["NBA"][market][gameDate] = {}
        archive.archive["NBA"][market][gameDate][player] = {
            line: stats}

archive.write()
