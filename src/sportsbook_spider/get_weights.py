from sportsbook_spider.stats import statsNBA, statsMLB, statsNHL
import pickle
import importlib.resources as pkg_resources
from sportsbook_spider import data
from datetime import datetime
from sportsbook_spider.helpers import likelihood
from scipy.optimize import minimize
import numpy as np

mlb = statsMLB()
mlb.load()
filepath = (pkg_resources.files(data) / "archive.dat")
with open(filepath, "rb") as infile:
    archive = pickle.load(infile)

market = 'total bases'
results = []
for game in mlb.gamelog:
    if any([string in market for string in ['allowed', 'pitch']]) and not game['starting pitcher']:
        continue
    elif not any([string in market for string in ['allowed', 'pitch']]) and not game['starting batter']:
        continue

    gameDate = datetime.strptime(game['gameId'][:10], '%Y/%m/%d')
    player = game['playerName']

    try:
        archiveData = archive['MLB'][market][gameDate.strftime(
            '%Y-%m-%d')][player]
    except:
        continue

    for line, stats in archiveData.items():
        if not line == 'Closing Lines':
            baseline = np.array([0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5])
            stats[stats == -1000] = baseline[stats == -1000]
            stats = stats - baseline
            y = int(game[market] > line)
            results.append({'stats': stats, 'result': y,
                            'player': player, 'date': gameDate})

res = minimize(lambda x: -likelihood(results, x),
               np.ones(9), method='l-bfgs-b', tol=1e-8, bounds=[(0, 100)]*9)

print(res.x)
