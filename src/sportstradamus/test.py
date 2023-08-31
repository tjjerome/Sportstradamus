from sportstradamus.stats import StatsNHL
from datetime import datetime
import pandas as pd

NHL = StatsNHL()
NHL.load()
NHL.update()

offer = {
    "Player": "Jack Hughes",
    "Market": "shots",
    "Date": "2021-12-31",
    "Line": 5.5,
    "League": "NHL",
    "Team": "NJ",
    "Opponent": "EDM"
}
NHL.profile_market(offer["Market"], date=offer["Date"])
NHL.get_stats(offer, date=offer['Date'])
