import pandas as pd
import numpy as np
import os.path
import json
from datetime import datetime
import importlib.resources as pkg_resources
from sportstradamus import data
from sportstradamus.stats import StatsNBA, StatsMLB, StatsNHL, StatsNFL
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

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
parlays["Profit"] = parlays.apply(lambda x: payout_table[x.Platform][x.Legs][x.Misses]*x.Boost-1, axis=1)

pd.concat([parlays, parlays_clean]).drop_duplicates(subset=parlays.columns[:-3]).to_pickle(filepath)

profits = {
    "Underdog": {},
    "PrizePicks": {}
}

for platform in profits.keys():
    platform_df = parlays.loc[parlays["Platform"] == platform]
    for league in ["All", "NBA", "NFL", "NHL", "MLB"]:
        if league == "All":
            df = platform_df
        else:
            df = platform_df.loc[platform_df["League"] == league]

        if not df.empty:
            n = .1
            mean_profit = np.zeros(1/n, 5/n)
            highest_book = np.zeros(1/n, 5/n)
            highest_model = np.zeros(1/n, 5/n)
            for i, bt in enumerate(np.arange(1, 2, n)):
                for j, mt in enumerate(np.arange(1, 6, n)):
                    test_df = df.loc[(df["Books EV"] > bt) & (df["Model EV"] > mt)]
                    mean_profit[i, j] = test_df.Profit.mean()
                    highest_model[i, j] = test_df.sort_values("Model EV", ascending=False).drop_duplicates(["Game", "Date"]).Profit.sum()
                    highest_book[i, j] = test_df.sort_values("Books EV", ascending=False).drop_duplicates(["Game", "Date"]).Profit.sum()

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1 = sns.heatmap(mean_profit)
            ax1.set_title('Mean, All Parlays')
            ax2 = sns.heatmap(highest_book)
            ax2.set_title('Sum, Highest Book Parlay')
            ax2 = sns.heatmap(highest_book)
            ax2.set_title('Sum, Highest Model Parlay')
            fig.suptitle(f"{league} Parlay Profits")
            fig.show()

        pass