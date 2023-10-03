from sportstradamus.stats import StatsNFL, StatsMLB, StatsNBA, StatsNHL
from sportstradamus.helpers import scraper
from urllib.parse import urlencode
import nba_api.stats.endpoints as nba
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
from scipy.stats import norm

# stats = StatsNBA()
# stats.load()
# stats.update()
# M = stats.get_training_matrix("PRA")
# print(M)

stats = StatsNFL()
stats.season_start = datetime(2020, 9, 1)
stats.update()
stats.season_start = datetime(2021, 9, 1)
stats.update()
stats.season_start = datetime(2022, 9, 1)
stats.update()
stats.season_start = datetime(2023, 9, 1)
stats.update()

# NBA = StatsNBA()
# NBA.load()
# NBA.profile_market('PTS')
# print(NBA.defenseProfile)

# NHL = StatsNHL()
# NHL.load()
# NHL.profile_market('shots')
# print(NHL.defenseProfile)
