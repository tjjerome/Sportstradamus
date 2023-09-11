from sportstradamus.helpers import Archive
from sportstradamus.stats import StatsNFL
from datetime import datetime
from tqdm import tqdm
import requests
import importlib.resources as pkg_resources
from sportstradamus import creds, data
import json
import numpy as np

archive = Archive("All")

for league in ["NFL", "NBA", "NHL", "MLB"]:
    for date in tqdm(list(archive[league]["Moneyline"].keys()), desc=league):
        for market in ["Moneyline", "Totals"]:
            if "CHW" in archive[league][market][date]:
                archive[league][market][date]["CWS"] = archive[league][market][date].pop(
                    "CHW")
            if "WSH" in archive[league][market][date]:
                archive[league][market][date]["WAS"] = archive[league][market][date].pop(
                    "WSH")
            if "AZ" in archive[league][market][date]:
                archive[league][market][date]["ARI"] = archive[league][market][date].pop(
                    "AZ")

archive.write()
