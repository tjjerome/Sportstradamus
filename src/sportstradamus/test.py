from sportstradamus.stats import StatsNHL
from datetime import datetime
import pandas as pd

NHL = StatsNHL()
NHL.load()
NHL.update()

offer = {
    "Player": "Jake Oettinger",
    "Market": "goalie fantasy points underdog",
    "Date": "2022-01-18",
    "Line": 15.5,
    "League": "NHL",
    "Team": "DAL",
    "Opponent": "MTL"
}
NHL.profile_market(offer["Market"], date=offer["Date"])
NHL.get_stats(offer, date=offer['Date'])
