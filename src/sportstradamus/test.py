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

history = pd.read_pickle(pkg_resources.files(data) / "history.dat")
history.loc[(history['Market'] == 'AST') & (
    history['League'] == 'NHL'), 'Market'] = "points"
history.loc[(history['Market'] == 'PTS') & (
    history['League'] == 'NHL'), 'Market'] = "assists"
history = history.dropna(subset='Market')
nameStr = {"MLB": "playerName", "NBA": "PLAYER_NAME",
           "NFL": "player display name", "NHL": "playerName"}
dateStr = {"MLB": "gameDate", "NBA": "GAME_DATE",
           "NFL": "gameday", "NHL": "gameDate"}
for i, row in tqdm(history.iterrows(), desc="Checking history", total=len(history)):
    if np.isnan(row["Correct"]):
        gamelog = stats[row["League"]].gamelog
        if " + " in row["Player"]:
            players = row["Player"].split(" +")
            game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            if not game1.empty and not game2.empty:
                history.at[i, "Correct"] = int(((game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) > row["Line"] and row["Bet"] == "Over") or (
                    (game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) < row["Line"] and row["Bet"] == "Under"))

        elif " vs. " in row["Player"]:
            players = row["Player"].split(" vs. ")
            game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            if not game1.empty and not game2.empty:
                history.at[i, "Correct"] = int(((game1.iloc[0][row["Market"]] + row["Line"]) > game2.iloc[0][row["Market"]] and row["Bet"] == "Over") or (
                    (game1.iloc[0][row["Market"]] + row["Line"]) < game2.iloc[0][row["Market"]] and row["Bet"] == "Under"))

        else:
            game = gamelog.loc[(gamelog[nameStr[row["League"]]] == row["Player"]) & (
                pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
            if not game.empty:
                history.at[i, "Correct"] = int((game.iloc[0][row["Market"]] > row["Line"] and row["Bet"] == "Over") or (
                    game.iloc[0][row["Market"]] < row["Line"] and row["Bet"] == "Under"))

history.to_pickle(pkg_resources.files(data) / "history.dat")
