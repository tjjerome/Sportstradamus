from sportstradamus.helpers import Archive
from sportstradamus.moneylines import get_moneylines, get_props
import json
from datetime import datetime, timedelta
import pytz
import importlib.resources as pkg_resources
from sportstradamus import data, creds

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

Date = datetime(2023, 9, 7, 9)
Date = pytz.timezone("America/Chicago").localize(Date)

sport="NFL"
key="americanfootball_nfl"

while Date.astimezone(pytz.utc).date() < datetime(2024, 2, 12).date():
    if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
        Date = Date + timedelta(days=1)
        continue

    print(Date)

    # archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
    archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

    Date = Date + timedelta(days=1)


archive.write()
