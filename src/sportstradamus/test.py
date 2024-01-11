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

nba = StatsNBA()
nba.load()
nba.update()
mlb = StatsMLB()
mlb.load()
mlb.update()
nhl = StatsNHL()
nhl.load()
nhl.update()
nfl = StatsNFL()
nfl.load()
nfl.update()

stats = {"NBA": nba, "MLB": mlb, "NHL": nhl, "NFL": nfl}
filepath = pkg_resources.files(data) / "history.dat"
history = pd.read_pickle(filepath)
nameStr = {"MLB": "playerName", "NBA": "PLAYER_NAME",
            "NFL": "player display name", "NHL": "playerName"}
dateStr = {"MLB": "gameDate", "NBA": "GAME_DATE",
            "NFL": "gameday", "NHL": "gameDate"}
for i, row in tqdm(history.loc[history.isna().any(axis=1) & (pd.to_datetime(history.Date).dt.date < datetime.today().date())].iterrows(), desc="Checking history", total=len(history)):
    if np.isnan(row["Result"]):
        gamelog = stats[row["League"]].gamelog
        if " + " in row["Player"]:
            players = row["Player"].split(" + ")
            game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            if not game1.empty and not game2.empty and not game1[row["Market"]].isna().any() and not game2[row["Market"]].isna().any():
                history.at[i, "Result"] = "Over" if ((game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) > row["Line"]) else ("Under" if ((game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) < row["Line"]) else "Push")

        elif " vs. " in row["Player"]:
            players = row["Player"].split(" vs. ")
            game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            if not game1.empty and not game2.empty and not game1[row["Market"]].isna().any() and not game2[row["Market"]].isna().any():
                history.at[i, "Result"] = "Over" if ((game1.iloc[0][row["Market"]] + row["Line"]) > game2.iloc[0][row["Market"]]) else ("Under" if ((game1.iloc[0][row["Market"]] + row["Line"]) < game2.iloc[0][row["Market"]]) else "Push")

        else:
            game = gamelog.loc[(gamelog[nameStr[row["League"]]] == row["Player"]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            if not game.empty and not game[row["Market"]].isna().any():
                history.at[i, "Result"] = "Over" if (game.iloc[0][row["Market"]] > row["Line"]) else ("Under" if (game.iloc[0][row["Market"]] < row["Line"]) else "Push")

history.to_pickle(filepath)

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
