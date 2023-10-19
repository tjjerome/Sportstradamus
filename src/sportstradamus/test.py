from sportstradamus.stats import StatsNFL, StatsMLB, StatsNBA, StatsNHL
from sportstradamus.helpers import scraper
from urllib.parse import urlencode
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
from scipy.stats import norm
import pandas as pd
import numpy as np
from tqdm import tqdm

nfl = StatsNFL()
# nfl.season_start = datetime(2020, 9, 1)
# nfl.update()
# nfl.season_start = datetime(2021, 9, 1)
# nfl.update()
# nfl.season_start = datetime(2022, 9, 1)
# nfl.update()
# nfl.season_start = datetime(2023, 9, 1)
nfl.load()
nfl.update()
nfl.profile_market("passing yards")

nfl.playerProfile
