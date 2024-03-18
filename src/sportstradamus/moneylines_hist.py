from sportstradamus.helpers import scraper, no_vig_odds, abbreviations, remove_accents, Archive
from sportstradamus.moneylines import get_moneylines, get_props
import pickle
import json
import numpy as np
from datetime import datetime, timedelta, tzinfo
import pytz
import importlib.resources as pkg_resources
from sportstradamus import data, creds
from itertools import cycle

archive = Archive("All")

# Load prop markets
filepath = pkg_resources.files(data) / "stat_map.json"
with open(filepath, "r") as infile:
    stat_map = json.load(infile)

filepath = pkg_resources.files(creds) / "keys.json"
with open(filepath, "r") as infile:
    keys = json.load(infile)
    apikey = keys["odds_api"]
    apikey_plus = keys["odds_api_plus"]

Date = datetime(2023, 10, 25, 9)
Date = pytz.timezone("America/Chicago").localize(Date)

sport="NBA"
key="basketball_nba"

while Date.astimezone(pytz.utc).date() < datetime(2024, 2, 12).date():
    if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
        Date = Date + timedelta(days=1)
        continue

    print(Date)

    archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

    Date = Date + timedelta(days=1)

archive.write()
