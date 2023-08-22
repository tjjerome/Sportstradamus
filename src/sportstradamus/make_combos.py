from sportstradamus.helpers import Archive, get_ev
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

archive = Archive("NFL")

df = pd.DataFrame(NFL.gamelog)

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

    games = df['game id'].unique()

    for gameId in tqdm(games, desc=market, unit="games", position=2):
        sub_df = df.loc[df["game id"] == gameId]
        gameDate = sub_df.iloc[0]["gameday"]
        if not archive["NFL"][market].get(gameDate):
            continue

        NFL.bucket_stats(market, date=datetime.strptime(gameDate, '%Y-%m-%d'))

        if any([string in market for string in ["pass", "completions", "attempts", "interceptions"]]):
            players = sub_df.loc[sub_df['position group'] == 'QB',
                                 ['player display name', market]].\
                sort_values(market, ascending=False)[
                'player display name'].head(2).to_list()
        else:
            players = [
                (player, NFL.playerStats.get(player, {}).get("avg", 0))
                for player in sub_df["player display name"].to_list()
            ]
            players.sort(reverse=True, key=lambda x: x[1])
            players = [i[0] for i in players[:4]]

        for combo in combinations(players, 2):
            try:
                EV = []
                for c in combo:
                    offer = archive["NFL"][market][gameDate][c]
                    ev = []
                    for line, stats in offer.items():
                        if line == "Closing Lines":
                            continue
                        stats = np.nan_to_num(stats, nan=0.5)
                        over = np.mean(
                            [i for i in stats[-4:] if not i == -1000])
                        ev.append(get_ev(line, 1 - over))

                    EV.append(np.mean(ev))
            except:
                continue

            player = " + ".join(combo)
            if not (
                player in archive["NFL"][market][gameDate]
                or " + ".join(player.split(" + ")[::-1])
                in archive["NFL"][market][gameDate]
            ):
                line = np.round((EV[1]+EV[0])*2)/2
                over = poisson.sf(np.floor(line), EV[1] + EV[0])
                if np.mod(line, 1) == 0:
                    over += poisson.pmf(line, EV[1] + EV[0]) / 2
                over = 0.5
                stats = [over] * 4
                archive.archive["NFL"][market][gameDate][player] = {
                    line: stats}

            player = " vs. ".join(combo)
            if not (
                player in archive["NFL"][market][gameDate]
                or " vs. ".join(player.split(" vs. ")[::-1])
                in archive["NFL"][market][gameDate]
            ):
                line = np.round((EV[1]-EV[0])*2)/2
                over = skellam.sf(np.floor(line), EV[1], EV[0])
                if np.mod(line, 1) == 0:
                    over += skellam.pmf(line, EV[1], EV[0]) / 2
                over = 0.5
                stats = [over] * 4
                archive.archive["NFL"][market][gameDate][player] = {
                    line: stats}

archive.write()
