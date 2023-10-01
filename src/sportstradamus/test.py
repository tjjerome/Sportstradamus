from sportstradamus.stats import StatsNFL, StatsMLB, StatsNBA, StatsNHL
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
import pickle
from scipy.stats import norm

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
filepath = pkg_resources.files(data) / "NFL_fantasy-points-underdog.mdl"
with open(filepath, "rb") as infile:
    filedict = pickle.load(infile)
model = filedict["model"]

playerStats = NFL.get_fantasy()
categories = ["Home", "Position"]
for c in categories:
    playerStats[c] = playerStats[c].astype('category')

prob_params = model.predict(playerStats, pred_type="parameters")
prob_params.index = playerStats.index
prob_params['Player'] = playerStats.index
positions = {0: "QB", 1: "WR", 2: "RB", 3: "TE"}
prob_params['Position'] = playerStats.Position.map(positions)
prob_params['Projection'] = prob_params['loc'].round(1)
prob_params['Floor'] = norm.ppf(.1395, loc=prob_params['loc'],
                                scale=prob_params['scale'])
prob_params['Floor'] = prob_params['Floor'].clip(0).round(1)
prob_params['Ceiling'] = norm.ppf(.9561, loc=prob_params['loc'],
                                  scale=prob_params['scale'])
prob_params['Ceiling'] = prob_params['Ceiling'].clip(0).round(1)
prob_params['Rank'] = prob_params.groupby('Position').rank(
    ascending=False, method='dense')['Ceiling']
prob_params.loc[prob_params['Position'] == "QB", 'VORP'] = prob_params.loc[prob_params['Position'] == "QB",
                                                                           'Ceiling'] - prob_params.loc[(prob_params['Position'] == "QB") & (prob_params["Rank"] == 13), 'Ceiling'].mean()
prob_params.loc[prob_params['Position'] == "WR", 'VORP'] = prob_params.loc[prob_params['Position'] == "WR",
                                                                           'Ceiling'] - prob_params.loc[(prob_params['Position'] == "WR") & (prob_params["Rank"] == 31), 'Ceiling'].mean()
prob_params.loc[prob_params['Position'] == "RB", 'VORP'] = prob_params.loc[prob_params['Position'] == "RB",
                                                                           'Ceiling'] - prob_params.loc[(prob_params['Position'] == "RB") & (prob_params["Rank"] == 19), 'Ceiling'].mean()
prob_params.loc[prob_params['Position'] == "TE", 'VORP'] = prob_params.loc[prob_params['Position'] == "TE",
                                                                           'Ceiling'] - prob_params.loc[(prob_params['Position'] == "TE") & (prob_params["Rank"] == 13), 'Ceiling'].mean()

prob_params = prob_params[['Player', 'Position',
                           'Projection', 'Floor', 'Ceiling', 'Rank', 'VORP']].sort_values("VORP", ascending=False)

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
