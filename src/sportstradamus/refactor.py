from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsNFL
from datetime import datetime
from tqdm import tqdm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json
import numpy as np

archive = Archive("NFL")
stats = StatsNFL()
stats.load()
stats.update()

for market in [
    "tds",
    "rushing tds",
    "receiving tds"
]:
    for date in tqdm(list(archive["NFL"][market].keys()), desc=market):
        if datetime.strptime(date, "%Y-%m-%d") < datetime(2023, 9, 1):
            archive["NFL"][market].pop(date)

for league in ["NFL", "NBA", "NHL", "MLB"]:
    for date in tqdm(list(archive[league]["Moneyline"].keys()), desc=league):
        for market in ["Moneyline", "Totals"]:
            if "WSH" in archive[league][market][date]:
                archive[league][market][date]["WAS"] = archive[league][market][date].pop(
                    "WSH")

archive.write()
