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

# sport="NCAAF"
# key="americanfootball_ncaaf"
# stat_map["Odds API"][sport] = stat_map["Odds API"]["NFL"]

# print(sport)
# Date = datetime(2022, 8, 27, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# while Date.astimezone(pytz.utc).date() < datetime(2023, 1, 10).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2023, 8, 26, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# while Date.astimezone(pytz.utc).date() < datetime(2024, 1, 9).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# sport="NCAAB"
# key="basketball_nba"
# stat_map["Odds API"][sport] = stat_map["Odds API"]["NBA"]

# Date = datetime(2022, 11, 7, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2023, 4, 4).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2023, 11, 6, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# while Date.astimezone(pytz.utc).date() < datetime(2024, 4, 9).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# sport="WNBA"
# key="basketball_wnba"
# stat_map["Odds API"][sport] = stat_map["Odds API"]["NBA"]

# Date = datetime(2022, 9, 8, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2022, 9, 19).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2023, 5, 19, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# while Date.astimezone(pytz.utc).date() < datetime(2023, 9, 11).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# sport="NBA"
# key="basketball_nba"

# Date = datetime(2021, 10, 19, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2022, 6, 17).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2022, 10, 18, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2023, 6, 13).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2023, 10, 24, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2024, 5, 13).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# sport="NHL"
# key="icehockey_nhl"

# Date = datetime(2021, 10, 12, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2022, 6, 27).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2022, 10, 7, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2023, 6, 14).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2023, 10, 10, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2024, 5, 13).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# sport="NFL"
# key="americanfootball_nfl"

# Date = datetime(2020, 9, 10, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2021, 2, 8).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2021, 9, 9, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2022, 2, 14).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2022, 9, 9, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2023, 2, 13).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2023, 9, 7, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2024, 2, 12).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

sport="MLB"
key="baseball_mlb"

# Date = datetime(2021, 4, 1, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2021, 11, 3).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2022, 4, 7, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2022, 11, 6).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     # archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

# Date = datetime(2023, 3, 30, 12)
# Date = pytz.timezone("America/Chicago").localize(Date)

# print(sport)
# while Date.astimezone(pytz.utc).date() < datetime(2023, 2, 3).date():
#     if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
#         Date = Date + timedelta(days=1)
#         continue

#     print(Date)

#     archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
#     archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

#     Date = Date + timedelta(days=1)

Date = datetime(2024, 5, 10, 12)
Date = pytz.timezone("America/Chicago").localize(Date)

print(sport)
while Date.astimezone(pytz.utc).date() < datetime(2024, 5, 20).date():
    if sport == "NFL" and Date.weekday() not in [0,3,5,6]:
        Date = Date + timedelta(days=1)
        continue

    print(Date)

    archive = get_moneylines(archive, keys, date=Date, sport=sport, key=key)
    archive = get_props(archive, apikey_plus, stat_map["Odds API"], date=Date, sport=sport, key=key)

    Date = Date + timedelta(days=1)

archive.write()
