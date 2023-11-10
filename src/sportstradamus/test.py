from sportstradamus.stats import StatsNBA, StatsMLB, StatsNFL, StatsNHL
from sportstradamus.helpers import scraper, Archive
from urllib.parse import urlencode
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
from scipy.stats import norm
import pandas as pd
import numpy as np
from tqdm import tqdm

NFL = StatsNFL()
NFL.load()
# NFL.season_start = datetime(2020, 9, 1).date()
NFL.update()
# NFL.season_start = datetime(2021, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2022, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2023, 9, 1).date()
# NFL.update()
NFL.profile_market("passing yards")
NFL.get_training_matrix("passing yards")

# NBA = StatsNBA()
# NBA.season = "2021-22"
# NBA.season_start = datetime(2021, 10, 1).date()
# NBA.update()
# NBA.season = "2022-23"
# NBA.season_start = datetime(2022, 10, 1).date()
# NBA.update()
# NBA.season = "2023-24"
# NBA.season_start = datetime(2023, 10, 1).date()
# NBA.update()

# NHL = StatsNHL()
# NHL.season_start = datetime(2021, 10, 1).date()
# NHL.update()
# NHL.season_start = datetime(2022, 10, 1).date()
# NHL.update()
# NHL.season_start = datetime(2023, 10, 1).date()
# NHL.update()

# MLB = StatsMLB()
# MLB.load()
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
