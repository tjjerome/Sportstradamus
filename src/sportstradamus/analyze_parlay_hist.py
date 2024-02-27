import pandas as pd
import numpy as np
import os.path
import json
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportstradamus import data, creds
from sportstradamus.stats import StatsNBA, StatsMLB, StatsNHL, StatsNFL
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import gspread

def reflect():
    # Authorize the gspread API
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]
    cred = None

    # Check if token.json file exists and load credentials
    if os.path.exists((pkg_resources.files(creds) / "token.json")):
        cred = Credentials.from_authorized_user_file(
            (pkg_resources.files(creds) / "token.json"), SCOPES
        )

    # If no valid credentials found, let the user log in
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                (pkg_resources.files(creds) / "credentials.json"), SCOPES
            )
            cred = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open((pkg_resources.files(creds) / "token.json"), "w") as token:
            token.write(cred.to_json())

    gc = gspread.authorize(cred)

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
    parlays["Profit"] = parlays.apply(lambda x: np.clip(payout_table[x.Platform][x.Legs][x.Misses]*(x.Boost if x.Boost < 2 or x.Misses==0 else 1),None,100)-1, axis=1)
    parlays["Profit"] = parlays["Profit"]*parlays["Rec Bet"]

    pd.concat([parlays, parlays_clean]).drop_duplicates(subset=parlays.columns[:-3]).to_pickle(filepath)

    parlays.loc[parlays["Model EV"] >= 6, "Model EV"] = 5.99

    profits = {}
    masks = {}

    for platform in payout_table.keys():
        platform_df = parlays.loc[parlays["Platform"] == platform]
        for league in ["All", "NBA", "NFL", "NHL", "MLB"]:
            if league == "All":
                league_df = platform_df
            else:
                league_df = platform_df.loc[platform_df["League"] == league]

            if not league_df.empty:
                profits.setdefault(f"{platform}, {league}", {})

                # if platform == "Underdog":
                    # n = .01
                    # mean_profit = np.zeros([int(1/n), int(5/n)])
                    # for i, bt in enumerate(tqdm(np.arange(1, 2, n))):
                    #     for j, mt in enumerate(np.arange(1, 6, n)):
                    #         test_df = league_df.loc[(league_df["Books EV"] > bt) & (league_df["Model EV"] > mt)]
                    #         bets = test_df.sort_values("Model EV", ascending=False).groupby(["Game", "Date"]).apply(lambda x: x.Profit.head(5).mean())
                    #         if len(bets):
                    #             mean_profit[i, j] = bets.sum()
                            
                    # model_threshold = np.argmax((mean_profit-np.nanmean(mean_profit))/np.nanstd(mean_profit) >= 1, axis=1)*n+1
                    # book_threshold = np.arange(1,2,n)
                    # book_threshold = book_threshold[model_threshold > 1]
                    # model_threshold = model_threshold[model_threshold > 1]
                    # p = np.polyfit(np.log(6-model_threshold), book_threshold, 1)
                    # masks[league] = list(p)

            for tf, days in [("Last Week", 7), ("Last Month", 30), ("Three Months", 91), ("Six Months", 183), ("Last Year", 365)]:
                df = league_df.loc[pd.to_datetime(league_df.Date).dt.date > datetime.today().date() - timedelta(days=days)]

                if not df.empty:
                    profits[f"{platform}, {league}"][tf] = df.sort_values("Model EV", ascending=False).groupby(["Game", "Date"]).apply(lambda x: x.Profit.head(10).mean()).sum()

    # filepath = pkg_resources.files(data) / "parlay_masks.json"
    # with open(filepath, 'w') as of:
    #     json.dump(masks, of, indent=4)

    profits = pd.DataFrame(profits).T.reset_index(names='Split').fillna(0.0)
    wks = gc.open("Sportstradamus").worksheet("Parlay Profit")
    wks.clear()
    wks.update([profits.columns.values.tolist()] +
            profits.values.tolist())
    

    filepath = pkg_resources.files(data) / "history.dat"
    if os.path.isfile(filepath):
        history = pd.read_pickle(filepath)

        nameStr = {"MLB": "playerName", "NBA": "PLAYER_NAME",
                "NFL": "player display name", "NHL": "playerName"}
        dateStr = {"MLB": "gameDate", "NBA": "GAME_DATE",
                "NFL": "gameday", "NHL": "gameDate"}
        for i, row in tqdm(history.loc[history.isna().any(axis=1) & (pd.to_datetime(history.Date).dt.date < datetime.datetime.today().date())].iterrows(), desc="Checking history", total=len(history)):
            if np.isnan(row["Result"]):
                gamelog = stats[row["League"]].gamelog
                if " + " in row["Player"]:
                    players = row["Player"].split(" + ")
                    game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                        pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                    game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                        pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                    if not game1.empty and not game2.empty and not game1[row["Market"]].isna().any() and not game2[row["Market"]].isna().any():
                        history.at[i, "Result"] = "Over" if ((game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) > row["Line"]) else ("Under" if ((game1.iloc[0][row["Market"]] + game2.iloc[0][row["Market"]]) < row["Line"]) else "Push")

                elif " vs. " in row["Player"]:
                    players = row["Player"].split(" vs. ")
                    game1 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[0]) & (
                        pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                    game2 = gamelog.loc[(gamelog[nameStr[row["League"]]] == players[1]) & (
                        pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                    if not game1.empty and not game2.empty and not game1[row["Market"]].isna().any() and not game2[row["Market"]].isna().any():
                        history.at[i, "Result"] = "Over" if ((game1.iloc[0][row["Market"]] + row["Line"]) > game2.iloc[0][row["Market"]]) else ("Under" if ((game1.iloc[0][row["Market"]] + row["Line"]) < game2.iloc[0][row["Market"]]) else "Push")

                else:
                    game = gamelog.loc[(gamelog[nameStr[row["League"]]] == row["Player"]) & (
                        pd.to_datetime(gamelog[dateStr[row["League"]]]).dt.date == pd.to_datetime(row["Date"]).date())]
                    if not game.empty and not game[row["Market"]].isna().any():
                        history.at[i, "Result"] = "Over" if (game.iloc[0][row["Market"]] > row["Line"]) else ("Under" if (game.iloc[0][row["Market"]] < row["Line"]) else "Push")

        history.to_pickle(filepath)
        history.dropna(inplace=True, ignore_index=True)

        if len(history) > 0:
            wks = gc.open("Sportstradamus").worksheet("History")
            wks.clear()
            wks.update([history.columns.values.tolist()] +
                    history.values.tolist())
            wks.set_basic_filter()

            history = history.loc[history["Result"] != "Push"]
            hist_stats = pd.DataFrame(columns=["Accuracy", "Balance", "LogLoss", "Brier", "Samples"])
            hist_stats.loc["All"] = {
                "Accuracy": accuracy_score(history["Bet"], history["Result"]),
                "Balance": (history["Bet"] == "Over").mean() - (history["Result"] == "Over").mean(),
                "LogLoss": log_loss((history["Bet"] == history["Result"]).astype(int), history["Model"], labels=[0,1]),
                "Brier": brier_score_loss((history["Bet"] == history["Result"]).astype(int), history["Model"], pos_label=1),
                "Samples": len(history)
            }
            hist_filt = history.loc[history["Books"] > .52]
            hist_stats.loc["All, Book Filtered"] = {
                "Accuracy": accuracy_score(hist_filt["Bet"], hist_filt["Result"]),
                "Balance": (hist_filt["Bet"] == "Over").mean() - (hist_filt["Result"] == "Over").mean(),
                "LogLoss": log_loss((hist_filt["Bet"] == hist_filt["Result"]).astype(int), hist_filt["Model"], labels=[0,1]),
                "Brier": brier_score_loss((hist_filt["Bet"] == hist_filt["Result"]).astype(int), hist_filt["Model"], pos_label=1),
                "Samples": len(hist_filt)
            }
            for league in history["League"].unique():
                league_hist = history.loc[history["League"] == league]
                hist_stats.loc[league] = {
                    "Accuracy": accuracy_score(league_hist["Bet"], league_hist["Result"]),
                    "Balance": (league_hist["Bet"] == "Over").mean() - (league_hist["Result"] == "Over").mean(),
                    "LogLoss": log_loss((league_hist["Bet"] == league_hist["Result"]).astype(int), league_hist["Model"], labels=[0,1]),
                    "Brier": brier_score_loss((league_hist["Bet"] == league_hist["Result"]).astype(int), league_hist["Model"], pos_label=1),
                    "Samples": len(league_hist)
                }
                hist_filt = league_hist.loc[league_hist["Books"] > .52]
                hist_stats.loc[f"{league}, Book Filtered"] = {
                    "Accuracy": accuracy_score(hist_filt["Bet"], hist_filt["Result"]),
                    "Balance": (hist_filt["Bet"] == "Over").mean() - (hist_filt["Result"] == "Over").mean(),
                    "LogLoss": log_loss((hist_filt["Bet"] == hist_filt["Result"]).astype(int), hist_filt["Model"], labels=[0,1]),
                    "Brier": brier_score_loss((hist_filt["Bet"] == hist_filt["Result"]).astype(int), hist_filt["Model"], pos_label=1),
                    "Samples": len(hist_filt)
                }
                for market in league_hist["Market"].unique():
                    market_hist = league_hist.loc[league_hist["Market"] == market]
                    hist_stats.loc[f"{league} - {market}"] = {
                        "Accuracy": accuracy_score(market_hist["Bet"], market_hist["Result"]),
                        "Balance": (market_hist["Bet"] == "Over").mean() - (market_hist["Result"] == "Over").mean(),
                        "LogLoss": log_loss((market_hist["Bet"] == market_hist["Result"]).astype(int), market_hist["Model"], labels=[0,1]),
                        "Brier": brier_score_loss((market_hist["Bet"] == market_hist["Result"]).astype(int), market_hist["Model"], pos_label=1),
                        "Samples": len(market_hist)
                    }
                    
            hist_stats["Split"] = hist_stats.index
            hist_stats = hist_stats[["Split", "Accuracy", "Balance", "LogLoss", "Brier", "Samples"]].sort_values('Samples', ascending=False)
            
            wks = gc.open("Sportstradamus").worksheet("Model Stats")
            wks.clear()
            wks.update([hist_stats.columns.values.tolist()] +
                    hist_stats.values.tolist())
            wks.set_basic_filter()
    
if __name__ == "__main__":
    reflect()