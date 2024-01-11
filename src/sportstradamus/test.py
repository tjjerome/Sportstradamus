from sportstradamus.stats import StatsNBA, StatsMLB, StatsNFL, StatsNHL
from sportstradamus.helpers import scraper, Archive, stat_cv
from urllib.parse import urlencode
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
from scipy.stats import norm, poisson
import pandas as pd
import numpy as np
from tqdm import tqdm

archive = Archive("All")
filepath = pkg_resources.files(data) / "history.dat"
history = pd.read_pickle(filepath)
history["Books"] = np.nan
# history = history[["Player", "League", "Team", "Date", "Market", "Line", "Bet", "Books", "Model"]]
# for i, row in history.iterrows():
#     ev = archive[row["League"]][row["Market"]].get(row["Date"], {}).get(row["Player"], {}).get("EV", np.array([None]))
#     ev = np.nanmean(ev.astype(float))
#     cv = stat_cv.get(row["League"], {}).get(row["Market"], 1)
#     if not np.isnan(ev):
#         if cv == 1:
#             odds = poisson.sf(row["Line"], ev) + poisson.pmf(row["Line"], ev)/2
#         else:
#             odds = norm.sf(row["Line"], ev, ev*cv)
#     else:
#         odds = 0.5

#     if row["Bet"] == "Under":
#         odds = 1-odds

#     history.loc[i, "Books"] = odds

# history.to_pickle(filepath)

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
# NHL.season_start = datetime(2021, 10, 12).date()
# NHL.update()
# NHL.season_start = datetime(2022, 10, 7).date()
# NHL.update()
# NHL.season_start = datetime(2023, 10, 10).date()
# NHL.update()

MLB = StatsMLB()
MLB.load()
MLB.gamelog = pd.DataFrame()
MLB.teamlog = pd.DataFrame()
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
