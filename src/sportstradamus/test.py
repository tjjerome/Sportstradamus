from sportstradamus.stats import StatsMLB

MLB = StatsMLB()
MLB.load()
MLB.update()

offer = {
    "Player": "Jorge Soler",
    "Market": "runs",
    "Date": "2023-08-25",
    "Line": 0.5,
    "League": "MLB",
    "Team": "MIA",
    "Opponent": "WSH"
}
MLB.profile_market(offer["Market"])
MLB.get_stats(offer, date=offer['Date'])
