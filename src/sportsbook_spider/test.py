from sportsbook_spider.stats import statsNBA, statsMLB, statsNHL
nba = statsNBA()
nba.load()
nba.update()
nba.get_stats('Jaylen Brown vs. Bam Adebayo', 'MIA/BOS', 'AST', -1.5)
mlb = statsMLB()
mlb.load()
mlb.update()
mlb.get_stats('Spencer Steer vs. Nolan Gorman',
              'STL/CIN', 'hits+runs+rbi', 0.5)
nhl = statsNHL()
nhl.load()
nhl.update()
nhl.get_stats('Miro Heiskanen vs. Jack Eichel', 'VGK/DAL', 'shots', 0.5)
