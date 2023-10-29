from sportstradamus.stats import StatsNBA, StatsMLB, StatsNFL, StatsNHL
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

# NFL = StatsNFL()
# NFL.season_start = datetime(2020, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2021, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2022, 9, 1).date()
# NFL.update()
# NFL.season_start = datetime(2023, 9, 1).date()
# NFL.update()

# NHL = StatsNHL()
# NHL.season_start = datetime(2021, 10, 1).date()
# NHL.update()
# NHL.season_start = datetime(2022, 10, 1).date()
# NHL.update()
# NHL.season_start = datetime(2023, 10, 1).date()
# NHL.update()

# NBA = StatsNBA()
# NBA.load()
# cols = ['SEASON_YEAR', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE',
#         'WL', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB',
#         'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
#         'PLUS_MINUS', 'POS', 'HOME', 'OPP', 'PRA', 'PR', 'PA', 'RA', 'BLST',
#         'fantasy points prizepicks', 'fantasy points underdog', 'fantasy points parlay',
#         'OFF_RATING', 'DEF_RATING', 'AST_PCT', 'OREB_PCT', 'DREB_PCT', 'REB_PCT',
#         'EFG_PCT', 'TS_PCT', 'USG_PCT', 'PIE', 'FTR']
# NBA.gamelog = pd.DataFrame(columns=cols)
# team_cols = ['SEASON_YEAR', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'OPP',
#              'OFF_RATING', 'DEF_RATING', 'EFG_PCT', 'OREB_PCT', 'DREB_PCT',
#              'TM_TOV_PCT', 'PACE', 'OPP_OFF_RATING', 'OPP_DEF_RATING',
#              'OPP_EFG_PCT', 'OPP_OREB_PCT', 'OPP_DREB_PCT',
#              'OPP_TM_TOV_PCT', 'OPP_PACE']
# NBA.teamlog = pd.DataFrame(columns=team_cols)
# NBA.season = "2021-22"
# NBA.season_start = datetime(2021, 10, 1).date()
# NBA.update()
# NBA.season = "2022-23"
# NBA.season_start = datetime(2022, 10, 1).date()
# NBA.update()
# NBA.season = "2023-24"
# NBA.season_start = datetime(2023, 10, 1).date()
# NBA.update()

MLB = StatsMLB()
MLB.load()
MLB.season_start = datetime(2021, 3, 1).date()
MLB.update()
MLB.update()
MLB.update()
MLB.update()
MLB.season_start = datetime(2022, 3, 1).date()
MLB.update()
MLB.update()
MLB.update()
MLB.update()
MLB.season_start = datetime(2023, 3, 30).date()
MLB.update()
MLB.update()
MLB.update()
MLB.update()
