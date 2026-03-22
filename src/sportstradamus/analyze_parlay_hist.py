import pandas as pd
import numpy as np
import os.path
import json
import re
import click
from datetime import datetime, timedelta
import importlib.resources as pkg_resources
from sportstradamus import data, creds
from sportstradamus.helpers import remove_accents
from sportstradamus.stats import StatsNBA, StatsWNBA, StatsMLB, StatsNHL, StatsNFL
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

LEG_PATTERN = re.compile(
    r"^(.+?)\s+(Over|Under)\s+([\d.]+)\s+(.+?)\s+-\s+[\d.]+%"
)

LEAGUE_CLASSES = {
    "NBA": StatsNBA,
    "WNBA": StatsWNBA,
    "MLB": StatsMLB,
    "NFL": StatsNFL,
    "NHL": StatsNHL,
}

PAYOUT_TABLE = {
    "Underdog": [
        (1, 1),
        (1, 1),
        (3.5, 0, 0),
        (6.5, 0, 0, 0),
        (6, 1.5, 0, 0, 0),
        (10, 2.5, 0, 0, 0, 0),
        (25, 2.6, 0.25, 0, 0, 0, 0),
    ],
}

TIMEFRAMES = [
    ("7d", 7),
    ("30d", 30),
    ("3m", 91),
    ("6m", 183),
    ("1y", 365),
]


def load_stats(league_filter):
    """Load Stats objects for available leagues, filtered by league_filter."""
    stats = {}
    leagues = [league_filter] if league_filter != "All" else list(LEAGUE_CLASSES.keys())
    for lg in tqdm(leagues, desc="Loading stats"):
        if lg not in LEAGUE_CLASSES:
            continue
        obj = LEAGUE_CLASSES[lg]()
        obj.load()
        if hasattr(obj, "gamelog") and not obj.gamelog.empty:
            if datetime.today().date() > (obj.season_start - timedelta(days=7)):
                try:
                    obj.update()
                except Exception:
                    pass
            stats[lg] = obj
    return stats


def check_bet(bet, stats, stat_map):
    """Resolve a single parlay bet against actual game logs.

    Returns (legs, misses) where legs is valid legs count
    and misses is number of legs that missed.
    """
    if bet.League not in stats:
        return np.nan, np.nan

    stat_obj = stats[bet.League]
    ls = stat_obj.log_strings

    new_map = stat_map.get(bet.Platform, {}).copy()
    if bet.League == "NHL":
        new_map.update({
            "Points": "points",
            "Blocked Shots": "blocked",
            "Assists": "assists",
        })
    if bet.League in ("NBA", "WNBA"):
        new_map.update({
            "Fantasy Points": "fantasy points prizepicks",
        })

    gamelog = stat_obj.gamelog
    game_dates = pd.to_datetime(gamelog[ls["date"]]).dt.date
    bet_date = pd.to_datetime(bet.Date).date()
    game = gamelog.loc[
        (game_dates == bet_date) &
        (gamelog[ls["team"]].isin(bet.Game.split("/")))
    ]

    if game.empty:
        return np.nan, np.nan

    legs = 0
    misses = 0

    for col in [c for c in bet.index if c.startswith("Leg ") and c[4:].strip().isdigit()]:
        leg = bet[col]
        if not isinstance(leg, str) or leg == "":
            continue

        m = LEG_PATTERN.match(leg)
        if not m:
            continue

        player, direction, line_str, market = m.groups()
        over = direction == "Over"
        line = float(line_str)
        market = market.replace("H2H ", "")
        market = new_map.get(market, market)

        try:
            if " + " in player:
                players = player.split(" + ")
                r1 = game.loc[game[ls["player"]] == players[0], market]
                r2 = game.loc[game[ls["player"]] == players[1], market]
                if r1.empty or r2.empty:
                    continue
                result_val = r1.iat[0] + r2.iat[0]
            elif " vs. " in player:
                players = player.split(" vs. ")
                r1 = game.loc[game[ls["player"]] == players[0], market]
                r2 = game.loc[game[ls["player"]] == players[1], market]
                if r1.empty or r2.empty:
                    continue
                result_val = r1.iat[0] + r2.iat[0]
            else:
                result = game.loc[game[ls["player"]] == player, market]
                if result.empty:
                    continue
                result_val = result.iat[0]
        except KeyError:
            continue

        if result_val == line:
            continue  # push, skip leg

        legs += 1
        if (over and result_val < line) or (not over and result_val > line):
            misses += 1

    return legs, misses


def resolve_history(history, stats):
    """Fill in Result column for individual bets in history.dat."""
    pending = history.loc[
        history["Result"].isna() &
        (pd.to_datetime(history["Date"], errors="coerce").dt.date < datetime.today().date())
    ]

    for i, row in tqdm(pending.iterrows(), desc="Checking history", total=len(pending)):
        if row["League"] not in stats:
            continue

        stat_obj = stats[row["League"]]
        ls = stat_obj.log_strings
        gamelog = stat_obj.gamelog
        bet_date = pd.to_datetime(row["Date"]).date()

        try:
            if " + " in row["Player"]:
                players = row["Player"].split(" + ")
                g1 = gamelog.loc[
                    (gamelog[ls["player"]] == players[0]) &
                    (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                g2 = gamelog.loc[
                    (gamelog[ls["player"]] == players[1]) &
                    (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                if g1.empty or g2.empty or g1[row["Market"]].isna().any() or g2[row["Market"]].isna().any():
                    continue
                val = g1.iloc[0][row["Market"]] + g2.iloc[0][row["Market"]]
                history.at[i, "Result"] = "Over" if val > row["Line"] else ("Under" if val < row["Line"] else "Push")

            elif " vs. " in row["Player"]:
                players = row["Player"].split(" vs. ")
                g1 = gamelog.loc[
                    (gamelog[ls["player"]] == players[0]) &
                    (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                g2 = gamelog.loc[
                    (gamelog[ls["player"]] == players[1]) &
                    (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                if g1.empty or g2.empty or g1[row["Market"]].isna().any() or g2[row["Market"]].isna().any():
                    continue
                history.at[i, "Result"] = "Over" if (g1.iloc[0][row["Market"]] + row["Line"]) > g2.iloc[0][row["Market"]] else (
                    "Under" if (g1.iloc[0][row["Market"]] + row["Line"]) < g2.iloc[0][row["Market"]] else "Push")

            else:
                g = gamelog.loc[
                    (gamelog[ls["player"]] == row["Player"]) &
                    (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                if g.empty or g[row["Market"]].isna().any():
                    continue
                val = g.iloc[0][row["Market"]]
                history.at[i, "Result"] = "Over" if val > row["Line"] else ("Under" if val < row["Line"] else "Push")
        except KeyError:
            continue

    return history


def _compute_stats_row(subset, prob_col):
    """Compute a single row of accuracy metrics for a subset of history."""
    if len(subset) == 0:
        return None
    sub_hit = (subset["Bet"] == subset["Result"]).astype(int)
    wins = sub_hit.sum()
    profit = wins * (100 / 110) - (len(subset) - wins)
    row = {
        "Accuracy": round(accuracy_score(subset["Bet"], subset["Result"]), 4),
        "Balance": round((subset["Bet"] == "Over").mean() - (subset["Result"] == "Over").mean(), 4),
        "LogLoss": round(log_loss(sub_hit, subset[prob_col].clip(0.01, 0.99), labels=[0, 1]), 4),
        "Brier": round(brier_score_loss(sub_hit, subset[prob_col].clip(0, 1)), 4),
        "ROI": round(profit / len(subset), 4),
        "Samples": len(subset),
    }
    return row


def compute_individual_metrics(history):
    """Compute accuracy, calibration, and ROI metrics from resolved history.

    Returns (hist_stats, daily, calibration, roi) DataFrames.
    """
    history = history.loc[history["Result"] != "Push"].copy()
    if len(history) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prob_col = "Model P" if "Model P" in history.columns and history["Model P"].notna().any() else "Model"
    today = datetime.today().date()
    history["_date"] = pd.to_datetime(history["Date"], errors="coerce").dt.date
    history = history.loc[history["_date"].notna()].copy()

    # --- Summary table with timeframe splits ---
    rows = []
    filtered = history.loc[history["Model"] > 0.58]

    for tf_label, tf_days in TIMEFRAMES:
        cutoff = today - timedelta(days=tf_days)
        tf_data = filtered.loc[filtered["_date"] >= cutoff]

        # All
        r = _compute_stats_row(tf_data, prob_col)
        if r:
            rows.append({"Period": tf_label, "Split": "All"} | r)

        # Book filtered
        r = _compute_stats_row(tf_data.loc[tf_data["Books"] > 0.52], prob_col)
        if r:
            rows.append({"Period": tf_label, "Split": "All, Book Filtered"} | r)

        # By league
        for league in sorted(tf_data["League"].unique()):
            league_data = tf_data.loc[tf_data["League"] == league]
            r = _compute_stats_row(league_data, prob_col)
            if r:
                rows.append({"Period": tf_label, "Split": league} | r)

            # By market within league
            for market in sorted(league_data["Market"].unique()):
                market_data = league_data.loc[league_data["Market"] == market]
                r = _compute_stats_row(market_data, prob_col)
                if r:
                    rows.append({"Period": tf_label, "Split": f"{league} - {market}"} | r)

    hist_stats = pd.DataFrame(rows)
    if not hist_stats.empty:
        hist_stats = hist_stats[["Period", "Split", "Accuracy", "Balance", "LogLoss", "Brier", "ROI", "Samples"]]

    # --- Daily granular data for time series charting ---
    # One row per Date x League x Market with daily metrics
    one_year = filtered.loc[filtered["_date"] >= today - timedelta(days=365)].copy()
    one_year["Hit"] = (one_year["Bet"] == one_year["Result"]).astype(int)
    one_year["Profit Unit"] = one_year["Hit"] * (100 / 110) - (1 - one_year["Hit"])

    daily = one_year.groupby(["_date", "League", "Market"]).agg(
        Bets=("Hit", "count"),
        Hits=("Hit", "sum"),
        Avg_Model_P=(prob_col, "mean"),
        Profit=("Profit Unit", "sum"),
    ).reset_index()
    daily.rename(columns={"_date": "Date"}, inplace=True)
    daily["Date"] = daily["Date"].astype(str)
    daily = daily.sort_values(["Date", "League", "Market"])

    # --- Calibration ---
    cal_data = one_year.copy()
    bins = np.linspace(0.5, 1.0, 11)
    cal_data["bin"] = pd.cut(cal_data[prob_col], bins=bins)
    calibration = cal_data.groupby("bin", observed=False).agg(
        Predicted=(prob_col, "mean"),
        Actual=("Hit", "mean"),
        Count=("Hit", "count"),
    ).reset_index()
    calibration["Bin"] = calibration["bin"].astype(str)
    calibration = calibration[["Bin", "Predicted", "Actual", "Count"]]

    # --- ROI by threshold ---
    roi_rows = []
    for threshold in [0.55, 0.58, 0.60, 0.65, 0.70]:
        for tf_label, tf_days in TIMEFRAMES:
            cutoff = today - timedelta(days=tf_days)
            for label_filter, subset in [
                ("All", history.loc[history["_date"] >= cutoff]),
                ("Book Filtered", history.loc[(history["_date"] >= cutoff) & (history["Books"] > 0.52)]),
            ]:
                t_sub = subset.loc[subset["Model"] > threshold]
                if len(t_sub) == 0:
                    continue
                wins = (t_sub["Bet"] == t_sub["Result"]).sum()
                profit = wins * (100 / 110) - (len(t_sub) - wins)
                roi_rows.append({
                    "Period": tf_label,
                    "Threshold": threshold,
                    "Filter": label_filter,
                    "Bets": len(t_sub),
                    "Win%": round(wins / len(t_sub), 4),
                    "ROI": round(profit / len(t_sub), 4),
                })
    roi = pd.DataFrame(roi_rows)

    return hist_stats, daily, calibration, roi


def compute_parlay_metrics(parlays, stats, stat_map):
    """Compute parlay P&L, hit rates, Sleeper outcomes, and correlation calibration.

    Returns (profit_df, daily_parlays, size_stats, sleeper_outcomes, corr_cal) DataFrames.
    """
    tqdm.pandas()
    today = datetime.today().date()

    # Filter to last year
    parlays["_date"] = pd.to_datetime(parlays["Date"], errors="coerce").dt.date
    parlays = parlays.loc[parlays["_date"].notna()].copy()
    parlays = parlays.loc[parlays["_date"] >= today - timedelta(days=365)].copy()

    # Resolve unresolved parlays
    unresolved = parlays.loc[parlays["Legs"].isna()]
    if len(unresolved) > 0:
        results = unresolved.progress_apply(
            lambda bet: check_bet(bet, stats, stat_map), axis=1
        ).to_list()
        parlays.loc[parlays["Legs"].isna(), ["Legs", "Misses"]] = results

    parlays.dropna(subset=["Legs"], inplace=True)
    if len(parlays) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    parlays[["Legs", "Misses"]] = parlays[["Legs", "Misses"]].astype(int)
    parlays["Hit"] = (parlays["Misses"] == 0).astype(int)

    # --- Underdog profit ---
    profit_rows = []
    ud_parlays = parlays.loc[parlays["Platform"] == "Underdog"].copy()
    if len(ud_parlays) > 0:
        ud_parlays["Profit"] = ud_parlays.apply(
            lambda x: np.clip(
                PAYOUT_TABLE["Underdog"][x.Legs][x.Misses] *
                (x.Boost if x.Boost < 2 or x.Misses == 0 else 1),
                None, 100
            ) - 1, axis=1
        )
        ud_parlays["Profit"] = ud_parlays["Profit"] * np.round(ud_parlays["Rec Bet"] * 2) / 2

        for league in ["All"] + sorted(ud_parlays["League"].unique()):
            ldf = ud_parlays if league == "All" else ud_parlays.loc[ud_parlays["League"] == league]
            if ldf.empty:
                continue
            for tf_label, tf_days in TIMEFRAMES:
                cutoff = today - timedelta(days=tf_days)
                tf_df = ldf.loc[ldf["_date"] >= cutoff]
                if not tf_df.empty:
                    p = tf_df.sort_values("Model EV", ascending=False).groupby(
                        ["Game", "Date"]).apply(lambda x: x.Profit.mean()).sum()
                    profit_rows.append({
                        "Platform": "Underdog",
                        "League": league,
                        "Period": tf_label,
                        "Profit": round(p, 2),
                        "Parlays": len(tf_df),
                        "Hit Rate": round(tf_df["Hit"].mean(), 4),
                    })

    profit_df = pd.DataFrame(profit_rows)

    # --- Daily parlay data for time series ---
    daily_rows = []
    for platform in parlays["Platform"].unique():
        plat_df = parlays.loc[parlays["Platform"] == platform]
        daily_plat = plat_df.groupby(["_date", "League"]).agg(
            Parlays=("Hit", "count"),
            Hits=("Hit", "sum"),
            Misses_0=("Misses", lambda x: (x == 0).sum()),
            Misses_1=("Misses", lambda x: (x == 1).sum()),
            Misses_2_plus=("Misses", lambda x: (x >= 2).sum()),
        ).reset_index()
        daily_plat["Platform"] = platform
        daily_plat.rename(columns={"_date": "Date"}, inplace=True)
        daily_rows.append(daily_plat)

    if daily_rows:
        daily_parlays = pd.concat(daily_rows, ignore_index=True)
        daily_parlays["Date"] = daily_parlays["Date"].astype(str)
        daily_parlays = daily_parlays[["Date", "Platform", "League", "Parlays", "Hits",
                                       "Misses_0", "Misses_1", "Misses_2_plus"]].sort_values(["Date", "Platform", "League"])
    else:
        daily_parlays = pd.DataFrame()

    # --- Hit rate by parlay size with timeframe splits ---
    size_rows = []
    for tf_label, tf_days in TIMEFRAMES:
        cutoff = today - timedelta(days=tf_days)
        tf_parlays = parlays.loc[parlays["_date"] >= cutoff]
        for size in sorted(tf_parlays["Bet Size"].unique()):
            size_df = tf_parlays.loc[tf_parlays["Bet Size"] == size]
            if len(size_df) == 0:
                continue
            row = {
                "Period": tf_label,
                "Size": int(size),
                "Actual Rate": round(size_df["Hit"].mean(), 4),
                "Count": len(size_df),
            }
            if "P" in parlays.columns:
                row["Predicted P"] = round(size_df["P"].mean(), 4)
            if "Leg Probs" in parlays.columns and size_df["Leg Probs"].notna().any():
                indep = size_df["Leg Probs"].apply(
                    lambda lp: np.prod(lp) if isinstance(lp, (list, tuple)) and len(lp) > 0 else np.nan
                )
                row["Independent Rate"] = round(indep.mean(), 4)
            size_rows.append(row)
    size_stats = pd.DataFrame(size_rows)

    # --- Sleeper outcome tracking with timeframe splits ---
    sl_parlays = parlays.loc[parlays["Platform"] == "Sleeper"].copy()
    sleeper_rows = []
    if len(sl_parlays) > 0:
        for tf_label, tf_days in TIMEFRAMES:
            cutoff = today - timedelta(days=tf_days)
            tf_sl = sl_parlays.loc[sl_parlays["_date"] >= cutoff]
            for size in sorted(tf_sl["Bet Size"].unique()):
                sdf = tf_sl.loc[tf_sl["Bet Size"] == size]
                if len(sdf) == 0:
                    continue
                row = {"Period": tf_label, "Size": int(size), "Count": len(sdf)}
                row["Hit All"] = int((sdf["Misses"] == 0).sum())
                row["Missed 1"] = int((sdf["Misses"] == 1).sum())
                if size >= 5:
                    row["Missed 2"] = int((sdf["Misses"] == 2).sum())
                    row["Missed 3+"] = int((sdf["Misses"] >= 3).sum())
                else:
                    row["Missed 2+"] = int((sdf["Misses"] >= 2).sum())
                sleeper_rows.append(row)
    sleeper_outcomes = pd.DataFrame(sleeper_rows)

    # --- Correlation calibration ---
    if "P" in parlays.columns and len(parlays) > 0:
        cal_df = parlays.copy()
        bins = np.linspace(0, 1, 11)
        cal_df["p_bin"] = pd.cut(cal_df["P"], bins=bins)
        corr_cal = cal_df.groupby("p_bin", observed=False).agg(
            Predicted=("P", "mean"),
            Actual=("Hit", "mean"),
            Count=("Hit", "count"),
        ).reset_index()
        corr_cal["Bin"] = corr_cal["p_bin"].astype(str)
        corr_cal = corr_cal[["Bin", "Predicted", "Actual", "Count"]]
    else:
        corr_cal = pd.DataFrame()

    # Clean up temp column
    parlays.drop(columns=["_date", "Hit"], inplace=True, errors="ignore")

    return profit_df, daily_parlays, size_stats, sleeper_outcomes, corr_cal


def write_section(wks, data_frames, start_row=1):
    """Write multiple DataFrames to a worksheet, separated by blank rows.

    Returns the next available row.
    """
    row = start_row
    for label, df in data_frames:
        if df is None or df.empty:
            continue
        wks.update(f"A{row}", [[label]])
        row += 1
        values = [df.columns.values.tolist()] + df.fillna("").values.tolist()
        wks.update(f"A{row}", values)
        row += len(values) + 1
    return row


@click.command()
@click.option("--league", type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL", "WNBA"]),
              default="All", help="Select league to analyze")
def reflect(league):
    # Authorize gspread
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]
    cred = None
    token_path = pkg_resources.files(creds) / "token.json"

    if os.path.exists(token_path):
        cred = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                pkg_resources.files(creds) / "credentials.json", SCOPES
            )
            cred = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(cred.to_json())

    gc = gspread.authorize(cred)

    with open(pkg_resources.files(data) / "stat_map.json", "r") as infile:
        stat_map = json.load(infile)

    stats = load_stats(league)

    # --- Job 2: Parlay analysis ---
    filepath = pkg_resources.files(data) / "parlay_hist.dat"
    if os.path.isfile(filepath):
        parlays = pd.read_pickle(filepath)
        if league != "All":
            parlays = parlays.loc[parlays["League"] == league]

        if len(parlays) > 0:
            profit_df, daily_parlays, size_stats, sleeper_outcomes, corr_cal = compute_parlay_metrics(
                parlays, stats, stat_map
            )

            # Save resolved results back
            if league == "All":
                parlays.to_pickle(filepath)
            else:
                full_parlays = pd.read_pickle(filepath)
                full_parlays.update(parlays)
                full_parlays.to_pickle(filepath)

            wks = gc.open("Sportstradamus").worksheet("Parlay Profit")
            wks.clear()
            write_section(wks, [
                ("Underdog P&L", profit_df),
                ("Hit Rate by Parlay Size", size_stats),
                ("Sleeper Outcomes", sleeper_outcomes),
                ("Parlay Calibration", corr_cal),
            ])

            # Daily parlay data on its own sheet for time series
            if not daily_parlays.empty:
                wks = gc.open("Sportstradamus").worksheet("Daily Parlays")
                wks.clear()
                wks.update("A1", [daily_parlays.columns.values.tolist()] + daily_parlays.values.tolist())
                wks.set_basic_filter()
        else:
            click.echo("No parlay data for selected league.")
    else:
        click.echo("No parlay history found. Run prophecize first.")

    # --- Job 1: Individual prediction analysis ---
    filepath = pkg_resources.files(data) / "history.dat"
    if os.path.isfile(filepath):
        history = pd.read_pickle(filepath)
        if league != "All":
            history = history.loc[history["League"] == league]

        history = resolve_history(history, stats)

        # Save resolved results back
        full_history = pd.read_pickle(pkg_resources.files(data) / "history.dat")
        full_history.update(history)
        full_history.to_pickle(pkg_resources.files(data) / "history.dat")

        resolved = history.dropna(subset=["Result"])
        if len(resolved) > 0:
            hist_stats, daily, calibration, roi = compute_individual_metrics(resolved)

            wks = gc.open("Sportstradamus").worksheet("Model Stats")
            wks.clear()
            write_section(wks, [
                ("Accuracy by Split", hist_stats),
                ("Calibration", calibration),
                ("ROI by Threshold", roi),
            ])

            # Daily data on its own sheet for time series
            if not daily.empty:
                wks = gc.open("Sportstradamus").worksheet("Daily Model")
                wks.clear()
                wks.update("A1", [daily.columns.values.tolist()] + daily.values.tolist())
                wks.set_basic_filter()
        else:
            click.echo("No resolved history data yet.")
    else:
        click.echo("No history found. Run prophecize first.")

    click.echo("Done!")


if __name__ == "__main__":
    reflect()
