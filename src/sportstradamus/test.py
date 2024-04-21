from sportstradamus.stats import StatsNBA, StatsMLB, StatsNFL, StatsNHL
from sportstradamus.helpers import scraper, archive, stat_cv
from urllib.parse import urlencode
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
import json
from scipy.stats import norm, poisson
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import requests

# filepath = pkg_resources.files(data) / "banned_combos.json"
# with open(filepath, "r") as infile:
#     banned = json.load(infile)

# for platform in banned.keys():
#     for league in list(banned[platform].keys()):
#         if "modified" in list(banned[platform][league].keys()):
#             for market in list(banned[platform][league]["modified"].keys()):
#                 for submarket in list(banned[platform][league]["modified"][market].keys()):
#                     market2 = market
#                     submarket2 = submarket
#                     if "_OPP_" in submarket:
#                         market2 = "_OPP_"+market2
#                         submarket2 = submarket2.replace("_OPP_", "")
#                     banned[platform][league]["modified"].setdefault(submarket2, {})
#                     banned[platform][league]["modified"][submarket2][market2] = banned[platform][league]["modified"][market][submarket]

# with open(filepath, "w") as outfile:
#     json.dump(banned, outfile, indent=4)

# url = "https://api.prizepicks.com/projections?league_id=7"
# params = {
#     "api_key": "82ccbf28-ddd6-4e37-b3a1-0097b10fd412",
#     "url": url,
#     "optimize_request": True
# }

# then = time()
# response1 = requests.get(
#     "https://proxy.scrapeops.io/v1/",
#     params=params
# )
# scrapeops = time() - then

# params = {
#     "api_key": "ElO5mYYVEiyFzpb7VasAdBQvJiaxKTQ1khUQtV7bwkdykhwwpSJhBD1NoKCDrDd1YtGKCOnKjoNkG17Y0b",
#     "url": url
# }
# then = time()
# response2 = requests.get(
#     "https://scraping.narf.ai/api/v1/",
#     params=params
# )
# scrapefish = time() - then

# (scrapeops, scrapefish)

# NFL = StatsNFL()
# NFL.season_start = datetime(2018, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2019, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2020, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2021, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2022, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2023, 9, 1).date()
# NFL.update()

NBA = StatsNBA()
# NBA.load()
# NBA.update()
# NBA.profile_market("BLST")
# stat=NBA.get_stats({
#     "Player": "Tim Hardaway",
#     "Market": "BLST",
#     "Line": 0.5,
#     "Date": datetime(2024, 3, 19).date(),
#     "Team": "DAL",
#     "Opponent": "SAS",
#     "League": "NBA"
# })
# pass
NBA.season = "2021-22"
NBA.season_start = datetime(2021, 10, 1).date()
NBA.update()
NBA.season = "2022-23"
NBA.season_start = datetime(2022, 10, 1).date()
NBA.update()
NBA.season = "2023-24"
NBA.season_start = datetime(2023, 10, 1).date()
NBA.update()

# NHL = StatsNHL()
# NHL.season_start = datetime(2021, 10, 12).date()
# NHL.update()
# NHL.season_start = datetime(2022, 10, 7).date()
# NHL.update()
# NHL.season_start = datetime(2023, 10, 10).date()
# NHL.update()

# MLB = StatsMLB()
# MLB.load()
# MLB.gamelog = pd.DataFrame()
# MLB.teamlog = pd.DataFrame()
# MLB.season_start = datetime(2021, 3, 1).date()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.season_start = datetime(2022, 3, 1).date()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.season_start = datetime(2023, 3, 30).date()
# MLB.update()
# MLB.update()
# MLB.update()
# MLB.update()
