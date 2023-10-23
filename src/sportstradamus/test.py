from sportstradamus.stats import StatsMLB, StatsMLB, StatsNBA, StatsNHL
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

MLB = StatsMLB()
MLB.load()
MLB.season_start = datetime(2021, 3, 1).date()
MLB.update()
MLB.update()
MLB.update()
MLB.season_start = datetime(2022, 3, 1).date()
MLB.update()
MLB.update()
MLB.update()
MLB.season_start = datetime(2023, 3, 1).date()
MLB.update()
MLB.update()
MLB.update()
MLB.profile_market("pitcher strikeouts")

MLB.playerProfile
