from sportsbook_spider.stats import statsNBA, statsMLB, statsNHL
nba = statsNBA()
nba.load()
nba.update()
nba.get_stats('Jaylen Brown + Bam Adebayo', 'MIA/BOS', 'AST', 6.5)
mlb = statsMLB()
mlb.load()
mlb.get_stats('Bryce Miller + James Paxton',
              'OAK/LAA', 'pitcher strikeouts', 10.5)
nhl = statsNHL()
nhl.load()
nhl.get_stats('Miro Heiskanen + Jack Eichel', 'VGK/DAL', 'shots', 4.5)
