import pandas as pd
import numpy as np
import os.path
import re
import json
from tqdm import tqdm
import importlib.resources as pkg_resources
from sportstradamus import data
from sportstradamus.stats import StatsNBA, StatsMLB, StatsNHL, StatsNFL
from sportstradamus.helpers import Archive, get_odds, stat_cv, remove_accents
from itertools import combinations
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import log_loss
from scipy.optimize import minimize

archive = Archive("All")

nba = StatsNBA()
nba.load()
mlb = StatsMLB()
mlb.load()
nhl = StatsNHL()
nhl.load()
nfl = StatsNFL()
nfl.load()

stats = {
    "MLB": mlb,
    "NBA": nba,
    "NHL": nhl,
    "NFL": nfl
}

filepath = pkg_resources.files(data) / "parlay_hist.dat"
if os.path.isfile(filepath):
    parlays = pd.read_pickle(filepath)

parlays.dropna(inplace=True)
parlays.sort_values("Rec Bet", ascending=False, inplace=True)

payout_table = { # using equivalent payouts when insured picks are better
    "Underdog": [3, 6, 10.9, 20.2],
    "PrizePicks": [3, 5.3, 10, 20.8, 38.8]
}

log_strings = {
    "NFL": {
        "game": "game id",
        "date": "gameday",
        "player": "player display name",
        "usage": "snap pct",
        "usage_sec": "route participation",
        "position": "position group",
        "team": "team",
        "home": "home",
        "win": "WL",
        "score": "points"
    },
    "NBA": {
        "game": "GAME_ID",
        "date": "GAME_DATE",
        "player": "PLAYER_NAME",
        "usage": "MIN",
        "usage_sec": "USG_PCT",
        "position": "POS",
        "team": "TEAM_ABBREVIATION",
        "home": "HOME",
        "win": "WL",
        "score": "PTS"
    },
    "NHL": {
        "game": "gameId",
        "date": "gameDate",
        "player": "playerName",
        "usage": "TimeShare",
        "usage_sec": "Fenwick",
        "position": "position",
        "team": "team",
        "home": "home",
        "win": "WL",
        "score": "goals"
    },
    "MLB": {
        "game": "gameId",
        "date": "gameDate",
        "player": "playerName",
        "position": "position",
        "team": "team",
        "home": "home",
        "win": "WL",
        "score": "runs"
    },
}

corr_modifiers = {}
for league in ["NBA"]:# parlays.League.unique():
    c = pd.read_csv(pkg_resources.files(data) / (f"{league}_corr.csv"), index_col = [0,1,2])
    c.rename_axis(["team", "market", "correlation"], inplace=True)
    c.columns = ["R"]

    league_df = parlays.loc[parlays.League == league]

    model_results = []
    book_results = []
    for n, row in tqdm(league_df.iterrows(), desc=f"Checking {league} parlays", total=len(league_df)):
        legs = row[[col for col in parlays.columns if "Leg " in col and row[col]!=""]]
        if legs.str.contains(" vs. ").any() or legs.str.contains(" \+ ").any():
            continue
        bet_size = len(legs)
        date = row["Date"]
        teams = row["Game"].split("/")
        players = [" ".join(remove_accents(player).split(" ")[:-1]) for player in row["Players"].split(", ")[:bet_size]]
        for i in range(bet_size):
            test_team = stats[league].gamelog.loc[(stats[league].gamelog[log_strings[league]["date"]].str[:10]==date) & (stats[league].gamelog[log_strings[league]["player"]]==players[i]), log_strings[league]["team"]]
            if not test_team.empty:
                test_team = test_team.iloc[0]
                break

        if not isinstance(test_team, str):
            continue

        if "_OPP_" in row.Markets[i][0]:
            opp = test_team
            teams.remove(test_team)
            team = teams[0]
        else:
            team = test_team
            teams.remove(test_team)
            opp = teams[0]

        markets = [market[0].split(".")[1] for market in row["Markets"]]
        evs = [archive.get_ev(league, markets[x], date, players[x]) for x in range(bet_size)]
        lines = [float(re.search(r'[\d,.]+(?!%)', leg).group(0)) for leg in legs]

        model_odds = [float(re.search(r'[\d,.]+(?=%)', leg).group(0))/100 for leg in legs]
        book_odds = [get_odds(evs[i], lines[i], stat_cv[league][markets[i]]) for i in range(bet_size)]

        for i in range(bet_size):
            if np.isnan(book_odds[i]):
                offer = {
                    "Player": players[i],
                    "Date": date,
                    "Market": markets[i],
                    "Line": lines[i],
                    "Team": opp if "_OPP_" in row.Markets[i][0] else team,
                    "Opponent": team if "_OPP_" in row.Markets[i][0] else opp
                }
                playerStats = stats[league].get_stats(offer)
                if playerStats == 0:
                    continue
                else:
                    odds = playerStats["Odds"]
                    book_odds[i] = np.nan if odds==0 else odds

        p = np.product(model_odds)
        pb = np.product(book_odds)

        # get correlation matrix
        bet_indices = np.arange(bet_size)
        SIG = np.zeros([bet_size, bet_size])
        team_c = c.loc[team]
        opp_c = c.loc[opp]
        opp_c.index = pd.MultiIndex.from_tuples([(f"_OPP_{x}".replace("_OPP__OPP_", ""), f"_OPP_{y}".replace("_OPP__OPP_", "")) for x, y in opp_c.index], names=("market", "correlation"))
        c_map = team_c["R"].add(opp_c["R"], fill_value=0).div(2).to_dict()

        # Iterate over combinations of bets
        for i, j in combinations(bet_indices, 2):
            cm1 = [row["Markets"][i][0]]
            b1 = [row["Markets"][i][1]] * len(cm1)
            if "vs." in players[i]:
                b1[1] = "Under" if b1[0] == "Over" else "Over"

            cm2 = [row["Markets"][j][0]]
            b2 = [row["Markets"][j][1]] * len(cm2)
            if "vs." in players[j]:
                b2[1] = "Under" if b2[0] == "Over" else "Over"

            # Vectorized operation to compute rho for all combinations of x and y
            rho_matrix = np.zeros((len(cm1), len(cm2)))
            for xi, x in enumerate(cm1):
                for yi, y in enumerate(cm2):
                    rho = c_map.get((x, y), c_map.get((y, x), 0))
                    if b1[xi] != b2[yi]:
                        rho = -rho
                    rho_matrix[xi, yi] = rho

            # Sum rho_matrix to update SIG
            SIG[i, j] = np.sum(rho_matrix)

        SIG = SIG + SIG.T + np.eye(bet_size)
        
        corr_p = multivariate_normal.cdf([norm.ppf(o) for o in model_odds], np.zeros(bet_size), SIG)
        corr_pb = multivariate_normal.cdf([norm.ppf(o) for o in book_odds], np.zeros(bet_size), SIG)

        model_results.append({
            "P": p,
            "C": corr_p,
            "R": int(row.Misses==0)
        })
        if not any(np.isnan(evs)):
            book_results.append({
                "P": pb,
                "C": corr_pb,
                "R": int(row.Misses==0)
            })

    x = pd.DataFrame(model_results)
    y = pd.DataFrame(book_results)

    def objective(w, x):
        return log_loss(x.R, (x.P+w*x.C)/(1+w))

    res1 = minimize(objective, [1], args=x, bounds=[(0, 1)], tol=1e-8, method='TNC')
    res2 = minimize(objective, [1], args=y, bounds=[(0, 1)], tol=1e-8, method='TNC')
    corr_modifiers[league] = {
        "Model": res1.x[0],
        "Books": res2.x[0]
        }

with open(pkg_resources.files(data) / ("corr_modifiers.json"), "w") as outfile:
    json.dump(corr_modifiers, outfile, indent=4)