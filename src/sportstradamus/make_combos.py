from sportstradamus.helpers import Archive, get_ev
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

archive = Archive(True)

df = pd.DataFrame(MLB.gamelog)

markets = [
    "pitcher fantasy score",
    "pitcher fantasy points underdog",
    # "passing tds",
    # "passing yards",
    # "completions",
    # "attempts",
    # "interceptions",
    # "carries",
    # "rushing yards",
    # "receiving yards",
    # "receptions",
    # "yards",
    # "rushing tds",
    # "receptions",
    # "targets",
    # "receiving tds",
    # "tds",
    # "fantasy points prizepicks",
    # "fantasy points underdog",
    # "fantasy points parlayplay"
]
for market in tqdm(markets, unit="markets", position=1):

    games = df['gameId'].unique()

    for gameId in tqdm(games, desc=market, unit="games", position=2):
        sub_df = df.loc[df["gameId"] == gameId]
        gameDate = sub_df.iloc[0]["gameId"][:10].replace("/", "-")
        if not archive["MLB"][market].get(gameDate):
            continue

        MLB.bucket_stats(market, date=datetime.strptime(gameDate, '%Y-%m-%d'))

        # if any([string in market for string in ["pass", "completions", "attempts", "interceptions"]]):
        #     players = sub_df.loc[sub_df['position group'] == 'QB',
        #                          'player display name'].to_list()
        # else:
        #     players = [
        #         (player, MLB.playerStats.get(player, {}).get("avg", 0))
        #         for player in sub_df["player display name"].to_list()
        #     ]
        #     players.sort(reverse=True, key=lambda x: x[1])
        #     players = [i[0] for i in players[:4]]

        players = sub_df.loc[sub_df['starting pitcher'],
                             'playerName'].to_list()

        for combo in combinations(players, 2):
            try:
                EV = []
                for c in combo:
                    offer = archive["MLB"][market][gameDate][c]
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
                player in archive["MLB"][market][gameDate]
                or " + ".join(player.split(" + ")[::-1])
                in archive["MLB"][market][gameDate]
            ):
                line = np.round((EV[1]+EV[0])*2)/2
                over = poisson.sf(np.floor(line), EV[1] + EV[0])
                if np.mod(line, 1) == 0:
                    over += poisson.pmf(line, EV[1] + EV[0]) / 2
                over = 0.5
                stats = [over] * 4
                archive.archive["MLB"][market][gameDate][player] = {
                    line: stats}

            player = " vs. ".join(combo)
            if not (
                player in archive["MLB"][market][gameDate]
                or " vs. ".join(player.split(" vs. ")[::-1])
                in archive["MLB"][market][gameDate]
            ):
                line = np.round((EV[1]-EV[0])*2)/2
                over = skellam.sf(np.floor(line), EV[1], EV[0])
                if np.mod(line, 1) == 0:
                    over += skellam.pmf(line, EV[1], EV[0]) / 2
                over = 0.5
                stats = [over] * 4
                archive.archive["MLB"][market][gameDate][player] = {
                    line: stats}

archive.write()
