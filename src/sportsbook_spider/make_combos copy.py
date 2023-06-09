from sportsbook_spider.helpers import archive, get_ev
from sportsbook_spider.stats import statsMLB
import pandas as pd
from itertools import combinations
from scipy.stats import poisson, skellam
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random

mlb = statsMLB()
mlb.load()
mlb.update()

df = pd.DataFrame(mlb.gamelog)
df['gameDate'] = df['gameId'].apply(lambda row: row[:10].replace('/', '-'))

markets = ['1st inning runs allowed']
for market in tqdm(markets, unit='markets', position=1):
    mlb.bucket_stats(market)

    games = df.gameDate.unique()

    for gameDate in tqdm(games, unit='games', position=2):
        sub_df = df.loc[df['gameDate'] == gameDate].loc[df['starting pitcher']]
        if sub_df.empty:
            continue

        players = [player for player in sub_df['playerName'].to_list()]

        combos = sub_df.groupby('gameId', group_keys=False).apply(
            lambda x: ' + '.join(x['playerName'])).to_list()

        if len(players) > 4:
            players = random.sample(players, 4)
            combos = combos + [
                ' + '.join(players[:2]), ' + '.join(players[2:])]
        for player in combos:
            if player in archive['MLB'][market].get(gameDate, {}) or ' + '.join(player.split(' + ')[::-1]) in archive['MLB'][market].get(gameDate, {}):
                continue

            opponents = []
            for c in player.split(' + '):
                opponents.append(
                    sub_df.loc[sub_df['playerName'] == c]['opponent'].to_list()[0])

            opponent = '/'.join(opponents)
            if market == '1st inning runs allowed':
                line = 0.5
            # stats = mlb.get_stats_date(player, opponent, datetime.strptime(
            #     gameDate, '%Y-%m-%d'), market, line)
            stats = np.append(np.zeros(5), [0.5]*4)
            if gameDate not in archive.archive['MLB'][market]:
                archive.archive['MLB'][market][gameDate] = {}
            archive.archive['MLB'][market][gameDate][player] = {
                line: stats}

archive.write()
