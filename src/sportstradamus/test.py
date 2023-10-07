from sportstradamus.stats import StatsNFL, StatsMLB, StatsNBA, StatsNHL
from sportstradamus.helpers import scraper
from urllib.parse import urlencode
import nba_api.stats.endpoints as nba
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
from scipy.stats import norm

stats = StatsNBA()
# stats.season = "2021-22"
# stats.season_start = datetime(2021, 10, 1).date()
stats.load()
stats.season = "2022-23"
stats.season_start = datetime(2022, 10, 1).date()
stats.update()
stats.season = "2023-24"
stats.season_start = datetime(2023, 10, 1).date()
stats.update()

stats = StatsNHL()
stats.season_start = datetime(2021, 10, 1).date()
stats.update()
stats.season_start = datetime(2022, 10, 1).date()
stats.update()
stats.season_start = datetime(2023, 10, 1).date()
stats.update()


# stats = StatsMLB()
# stats.load()
# stats.update()
# stats.profile_market('pitcher strikeouts')
# offer = {
#     'Player': 'Zac Gallen',
#     'Line': 3.5,
#     'Team': 'ARI',
#     'Opponent': 'MIL',
#     'Market': 'pitcher strikeouts',
#     'League': 'MLB',
#     'Date': '2023-10-04'
# }
# n = stats.get_stats(offer, date=offer['Date'])
# stats.profile_market('total bases')
# offer = {
#     'Player': 'Kyle Schwarber',
#     'Line': 0.5,
#     'Team': 'PHI',
#     'Opponent': 'MIA',
#     'Market': 'total bases',
#     'League': 'MLB',
#     'Pitcher': 'Jesus Luzardo',
#     'Date': '2023-10-03'
# }
# stats.get_stats(offer, date=offer['Date'])
# print(stats.defenseProfile)
