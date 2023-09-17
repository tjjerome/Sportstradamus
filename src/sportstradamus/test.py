from sportstradamus.stats import StatsNFL, StatsMLB, StatsNBA, StatsNHL
from datetime import datetime

# NFL = StatsNFL()
# NFL.season_start = datetime(2020, 9, 1)
# NFL.update()
# NFL.season_start = datetime(2021, 9, 1)
# NFL.update()
# NFL.season_start = datetime(2022, 9, 1)
# NFL.update()
# NFL.season_start = datetime(2023, 9, 1)
# NFL.update()

NFL = StatsNFL()
NFL.load()
NFL.update()
NFL.get_fantasy()

# MLB = StatsMLB()
# MLB.load()
# MLB.profile_market('pitcher strikeouts')
# print(MLB.defenseProfile)

# NBA = StatsNBA()
# NBA.load()
# NBA.profile_market('PTS')
# print(NBA.defenseProfile)

# NHL = StatsNHL()
# NHL.load()
# NHL.profile_market('shots')
# print(NHL.defenseProfile)
