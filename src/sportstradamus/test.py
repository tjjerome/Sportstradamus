from sportstradamus.stats import StatsNFL, StatsMLB, StatsNBA, StatsNHL
from sportstradamus.helpers import scraper
from urllib.parse import urlencode
import nba_api.stats.endpoints as nba
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
from scipy.stats import norm

stats = StatsNHL()
stats.load()
stats.update()
M = stats.get_training_matrix("saves")
print(M)

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
