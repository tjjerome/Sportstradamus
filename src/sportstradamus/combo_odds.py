from sportstradamus.stats import StatsNFL
from sportstradamus.helpers import archive, get_ev
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import poisson

# NFL = StatsNFL()
# NFL.load()
# NFL.update()

markets = ["rushing-yards", "receiving-yards"]
new_market = "yards"
Date = datetime.strptime("2022-09-09T12:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
archive.__init__(True)
while Date < datetime.strptime("2023-05-26", "%Y-%m-%d"):
    date = Date.strftime("%Y-%m-%d")
    print(date)
    Date = Date + timedelta(days=1)
    market = markets[0]
    # and (not archive['NFL'].get(new_market, {}).get(date)):
    if archive["NFL"][market].get(date):
        players = archive["NFL"][market][date]
        for player, offer in players.items():
            if player not in archive["NFL"][markets[0]][date].keys():
                continue
            EV = []
            ev = []
            for line, stats in offer.items():
                over = np.mean([i for i in stats[-4:] if not i == -1000])
                ev.append(get_ev(line, 1 - over))
            EV.append(np.mean(ev))
            for market in markets[1:]:
                offer1 = archive["NFL"][market][date].get(player)
                if offer1 is None:
                    EV.append(0)
                    continue
                ev = []
                for line, stats in offer1.items():
                    over = np.mean([i for i in stats[-4:] if not i == -1000])
                    ev.append(get_ev(line, 1 - over))
                EV.append(np.mean(ev))
            if not np.prod(EV) == 0:
                EV = np.sum(EV)
                line = np.round(EV - 0.5) + 0.5
                over = poisson.sf(np.floor(line), EV)
                stats = np.array([over] * 4)
                if new_market not in archive.archive["NFL"]:
                    archive.archive["NFL"][new_market] = {}
                if date not in archive.archive["NFL"][new_market]:
                    archive.archive["NFL"][new_market][date] = {}
                archive.archive["NFL"][new_market][date][player] = {
                    line: stats}

archive.write()
