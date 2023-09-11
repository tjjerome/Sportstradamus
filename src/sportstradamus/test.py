from sportstradamus.stats import StatsNFL, StatsMLB, StatsNBA, StatsNHL

NFL = StatsNFL()
NFL.load()
NFL.profile_market('passing yards')
print(NFL.defenseProfile)

MLB = StatsMLB()
MLB.load()
MLB.profile_market('pitcher strikeouts')
print(MLB.defenseProfile)

NBA = StatsNBA()
NBA.load()
NBA.profile_market('PTS')
print(NBA.defenseProfile)

NHL = StatsNHL()
NHL.load()
NHL.profile_market('shots')
print(NHL.defenseProfile)
