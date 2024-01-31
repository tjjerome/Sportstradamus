import pandas as pd
import numpy as np
import os.path
import json
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportstradamus import data, creds
from sportstradamus.stats import StatsNBA, StatsMLB, StatsNHL, StatsNFL
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

def reflect():
    tqdm.pandas()
    payout_table = {
        "Underdog": [(1, 1),
                    (1, 1),
                    (3, 0, 0),
                    (6, 0, 0, 0),
                    (6, 1.5, 0, 0, 0),
                    (10, 2.5, 0, 0, 0, 0)],
        "PrizePicks": [(1, 1),
                    (1, 1),
                    (3, 0, 0),
                    (2.25, 1.25, 0, 0),
                    (10, 0, 0, 0, 0),
                    (10, 2, 0.4, 0, 0, 0),
                    (25, 2, 0.4, 0, 0, 0, 0)]
    }
    pt_simp = { # using equivalent payouts when insured picks are better
        "Underdog": [0, 0, 3, 6, 10.9, 20.2],
        "PrizePicks": [0, 0, 3, 5.3, 10, 20.8, 38.8]
    }

    nba = StatsNBA()
    nba.load()
    nba.update()
    mlb = StatsMLB()
    mlb.load()
    mlb.update()
    nhl = StatsNHL()
    nhl.load()
    nhl.update()
    nfl = StatsNFL()
    nfl.load()
    nfl.update()

    stats = {"NBA": nba, "MLB": mlb, "NHL": nhl, "NFL": nfl}

    nameStr = {"MLB": "playerName", "NBA": "PLAYER_NAME",
                "NFL": "player display name", "NHL": "playerName"}
    dateStr = {"MLB": "gameDate", "NBA": "GAME_DATE",
                "NFL": "gameday", "NHL": "gameDate"}
    teamStr = {"MLB": "team", "NBA": "TEAM_ABBREVIATION",
                "NFL": "recent team", "NHL": "team"}

    with open((pkg_resources.files(data) / "stat_map.json"), "r") as infile:
        stat_map = json.load(infile)

    filepath = pkg_resources.files(data) / "parlay_hist.dat"
    if os.path.isfile(filepath):
        parlays_clean = pd.read_pickle(filepath)

    parlays = parlays_clean.copy()

    def check_bet(bet):
        new_map = stat_map[bet.Platform]
        if bet.League == "NHL":
            new_map.update({
                "Points": "points",
                "Blocked Shots": "blocked",
                "Assists": "assists"
            })
        if bet.League == "NBA":
            new_map.update({
                "Fantasy Points": "fantasy points prizepicks",
                "Points": "PTS",
                "Blocked Shots": "BLK",
                "Assists": "AST"
            })
        gamelog = stats[bet.League].gamelog
        gamelog[dateStr[bet.League]] = gamelog[dateStr[bet.League]].str[:10]
        game = gamelog.loc[(gamelog[dateStr[bet.League]]==bet.Date) & (gamelog[teamStr[bet.League]].isin(bet.Game.split('/')))]
        if game.empty:
            return np.nan, np.nan
        else:
            legs = 0
            misses = 0
            for leg in bet:
                if isinstance(leg, str) and "%" in leg:
                    legs += 1
                    if " Under " in leg:
                        split_leg = leg.split(" Under ")
                        over = False
                    else:
                        split_leg = leg.split(" Over ")
                        over = True
                    player = split_leg[0]
                    rest = split_leg[1].split(" - ")[0]
                    line = rest.split(" ")[0]
                    market = rest[(len(line)+1):].replace("H2H ", "")
                    line = float(line)
                    market = new_map.get(market, market)

                    if " + " in player:
                        players = player.split(" + ")
                        result1 = game.loc[game[nameStr[bet.League]] == players[0], market]
                        result2 = game.loc[game[nameStr[bet.League]] == players[1], market]
                        if result1.empty or result2.empty:
                            result = result1
                        else:
                            result = pd.Series(result1.iat[0] + result2.iat[0])
                    elif " vs. " in player:
                        players = player.split(" vs. ")
                        result1 = game.loc[game[nameStr[bet.League]] == players[0], market]
                        result2 = game.loc[game[nameStr[bet.League]] == players[1], market]
                        if result1.empty or result2.empty:
                            result = result1
                        else:
                            result = pd.Series(result1.iat[0] + result2.iat[0])
                    else:
                        result = game.loc[game[nameStr[bet.League]] == player, market]

                    if result.empty:
                        legs -= 1
                    elif result.iat[0] == line:
                        legs -= 1
                    elif result.iat[0] < line and over:
                        misses += 1
                    elif result.iat[0] > line and not over:
                        misses += 1

            return legs, misses

    parlays.loc[parlays.Legs.isna(), ["Legs", "Misses"]] = parlays.loc[parlays.Legs.isna()].progress_apply(check_bet, axis=1).to_list()
    parlays.dropna(subset=['Legs'], inplace=True)
    parlays[["Legs", "Misses"]] = parlays[["Legs", "Misses"]].astype(int)
    parlays["Profit"] = parlays.apply(lambda x: np.clip(payout_table[x.Platform][x.Legs][x.Misses]*(x.Boost if x.Boost < 2 or x.Misses==0 else 1),None,100)-1, axis=1)

    pd.concat([parlays, parlays_clean]).drop_duplicates(subset=parlays.columns[:-3]).to_pickle(filepath)

    parlays["Actual Legs"] = parlays.apply(lambda x: np.sum([1 for y in x.index if "Leg " in y and x[y] != ""]), axis=1)

    units = []
    profits = []
    unit_size = np.arange(0.01, 0.1, .001)
    control = parlays.sort_values("Model EV", ascending=False).groupby(["Game", "Date", "Platform"]).apply(lambda x: x["Profit"].head(5).mean()).sum()
    for unit in tqdm(unit_size):
        parlays["Bet"] = (parlays["Model EV"] - 1)/(parlays.apply(lambda x: pt_simp[x["Platform"]][x["Actual Legs"]]*x["Boost"] - 1, axis=1))
        # unit = parlays.loc[parlays["Profit"] > 0, "Bet"].mean()
        parlays["Bet"] = np.round(parlays["Bet"]/unit*2)/2
        parlays["Actual Profit"] = parlays["Profit"]*parlays["Bet"]
        profits.append(parlays.sort_values("Model EV", ascending=False).groupby(["Game", "Date", "Platform"]).apply(lambda x: x["Actual Profit"].head(5).mean()).sum())

    plt.plot(unit_size, profits)
    plt.plot(unit_size, np.ones(len(profits))*control, '-')
    plt.xlabel("Unit Size")
    plt.ylabel("Profit")

    
if __name__ == "__main__":
    reflect()