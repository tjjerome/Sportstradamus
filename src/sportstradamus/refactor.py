from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsNBA
from datetime import datetime
from tqdm import tqdm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json
import numpy as np

archive = Archive()
stats = StatsNBA()
stats.load()
stats.update()

for league in ["NBA"]:
    for date in tqdm(list(archive[league]["Moneyline"].keys()), desc=league):
        if datetime.strptime(date, "%Y-%m-%d") < datetime(2022, 10, 1):
            continue
        market = "Totals"
        new_dict = {}
        for team in list(archive[league][market][date].keys()):
            if team == "UTAH":
                team = "UTA"
            opponent = stats.gamelog.loc[(stats.gamelog["GAME_DATE"].str[:10] == date) & (
                stats.gamelog["TEAM_ABBREVIATION"] == team), "OPP"].max()
            new_dict.update({team: archive[league][market][date][opponent]})

        archive[league][market][date] = new_dict

archive.write()
