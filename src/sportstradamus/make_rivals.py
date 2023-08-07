from sportstradamus.helpers import Archive, get_ev
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

archive = Archive(True)

df = pd.DataFrame(mlb.gamelog)

markets = [
    "pitcher strikeouts",
    "pitching outs",
    "pitches thrown",
    "walks allowed",
    "hits allowed",
    "runs allowed",
    "1st inning runs allowed",
    "1st inning hits allowed",
    "hits",
    "runs",
    "rbi",
    "hits+runs+rbi",
    "singles",
    "total bases",
    "batter strikeouts",
    "hitter fantasy score",
    "pitcher fantasy score",
    "hitter fantasy points underdog",
    "pitcher fantasy points underdog",
    "hitter fantasy points parlay",
    "pitcher fantasy points parlay"
]
for market in tqdm(markets, unit="markets", position=1):

    games = df.gameId.unique()

    for gameId in tqdm(games, unit="games", position=2):
        sub_df = df.loc[df["gameId"] == gameId]
        gameDate = sub_df.iloc[0]["gameId"][:10].replace("/", "-")
        if not archive["MLB"][market].get(gameDate):
            continue

        mlb.bucket_stats(market, date=datetime.strptime(gameDate, '%Y-%m-%d'))

        if any([string in market for string in ["allowed", "pitch"]]):
            players = sub_df.loc[sub_df['starting pitcher'],
                                 'playerName'].to_list()
        else:
            players = [
                (player, mlb.playerStats.get(player, {}).get("avg", 0))
                for player in sub_df["playerName"].to_list()
            ]
            players.sort(reverse=True, key=lambda x: x[1])
            players = [i[0] for i in players[:4]]

        for combo in combinations(players, 2):
            player = " vs. ".join(combo)
            if (
                player in archive["MLB"][market][gameDate]
                or " vs. ".join(player.split(" + ")[::-1])
                in archive["MLB"][market][gameDate]
            ):
                continue
            try:
                EV = []
                opponents = []
                for c in combo:
                    opponents.append(
                        sub_df.loc[sub_df["playerName"] == c]["opponent"].to_list()[
                            0]
                    )
                    offer = archive["MLB"][market][gameDate][c]
                    ev = []
                    for line, stats in offer.items():
                        over = np.mean(
                            [i for i in stats[-4:] if not i == -1000])
                        ev.append(get_ev(line, 1 - over))

                    EV.append(np.mean(ev))
            except:
                continue

            opponent = "/".join(opponents)
            line = np.round((EV[1]-EV[0])*2)/2
            over = skellam.sf(np.floor(line), EV[1], EV[0])
            if np.mod(line, 1) == 0:
                over += skellam.pmf(line, EV[1], EV[0]) / 2
            stats = [over] * 4
            archive.archive["MLB"][market][gameDate][player] = {line: stats}

archive.write()
