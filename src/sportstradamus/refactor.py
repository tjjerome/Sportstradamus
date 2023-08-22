from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsMLB
from datetime import datetime
from tqdm import tqdm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json

archive = Archive("NFL")

for market in [
    # "passing yards",
    # "rushing yards",
    # "receiving yards",
    # "yards",
    "fantasy points prizepicks",
    "fantasy points underdog",
    "fantasy points parlayplay",
    # "passing tds",
    # "rushing tds",
    # "receiving tds",
    # "tds",
    # "completions",
    # "carries",
    # "receptions",
    # "interceptions",
    # "attempts",
    # "targets"
]:
    for date in tqdm(list(archive["NFL"][market].keys())):
        # if datetime.strptime(date, "%Y-%m-%d") > datetime(2023, 6, 13):
        #     continue
        for player in list(archive["NFL"][market][date].keys()):
            if " + " in player or " vs. " in player:
                continue
            for line in list(archive["NFL"][market][date][player].keys()):
                if line == "Closing Lines":
                    continue
                if line < 6:
                    archive["NFL"][market][date][player].pop(line)

            if len(list(archive["NFL"][market][date][player].keys())) == 0:
                archive["NFL"][market][date].pop(player)


archive.write()

# mlb = StatsMLB()
# mlb.load()
# mlb.update()

# offer = {
#     "Player": "Jeremy Pena + Jose Altuve",
#     "Market": "hits+runs+rbi",
#     "Date": "2023-08-22",
#     "Line": 2.5,
#     "League": "MLB",
#     "Team": "HOU",
#     "Opponent": "BOS"
# }

# mlb.get_stats(offer, date=offer['Date'])
