from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsMLB
from datetime import datetime
from tqdm import tqdm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json

# archive = Archive("MLB")

# for market in ["batter strikeouts"]:
#     for date in tqdm(list(archive["MLB"][market].keys())):
#         # if datetime.strptime(date, "%Y-%m-%d") > datetime(2023, 6, 13):
#         #     continue
#         for player in list(archive["MLB"][market][date].keys()):
#             if " + " in player or " vs. " in player:
#                 continue

#             for line in list(archive["MLB"][market][date][player].keys()):
#                 if line == "Closing Lines":
#                     continue
#                 if line > 2:
#                     archive["MLB"][market][date][player].pop(line)

#             if len(archive["MLB"][market][date][player].keys()) == 0:
#                 archive["MLB"][market][date].pop(player)

# archive.write()

mlb = StatsMLB()
mlb.load()
mlb.update()

offer = {
    "Player": "Jeremy Pena",
    "Market": "hits+runs+rbi",
    "Date": "2023-08-22",
    "Line": 1.5,
    "League": "MLB",
    "Team": "HOU",
    "Opponent": "BOS"
}

mlb.get_stats(offer, date=offer['Date'])
