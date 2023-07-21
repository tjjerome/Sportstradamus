from sportstradamus.stats import StatsNHL

nhl = StatsNHL()
nhl.load()
nhl.get_training_matrix('saves')
