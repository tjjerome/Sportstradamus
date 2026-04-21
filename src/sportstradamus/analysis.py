"""Shared analysis and metric functions for the Sportstradamus dashboard.

Extracted from analyze_parlay_hist.py with additional professional forecasting
metrics (CRPS, Brier Skill Score, Murphy decomposition, prediction intervals).
"""

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import nbinom
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from tqdm import tqdm

from sportstradamus.helpers import get_odds

LEG_PATTERN = re.compile(r"^(.+?)\s+(Over|Under)\s+([\d.]+)\s+(.+?)\s+-\s+[\d.]+%")

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
        new_map.update(
            {
                "Points": "points",
                "Blocked Shots": "blocked",
                "Assists": "assists",
            }
        )
    if bet.League in ("NBA", "WNBA"):
        new_map.update(
            {
                "Fantasy Points": "fantasy points prizepicks",
            }
        )

    gamelog = stat_obj.gamelog
    game_dates = pd.to_datetime(gamelog[ls["date"]]).dt.date
    bet_date = pd.to_datetime(bet.Date).date()
    game = gamelog.loc[(game_dates == bet_date) & (gamelog[ls["team"]].isin(bet.Game.split("/")))]

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


def _migrate_flat_history(history):
    """Convert old flat history schema (one row per offer) to normalized schema.

    Groups by (Player, League, Date, Market) and collects per-offer columns
    into an Offers list of tuples: (Line, Boost, Platform, Bet, ModelP, BooksP).
    """
    pred_key = ["Player", "League", "Date", "Market"]
    offer_cols = ["Line", "Boost", "Platform", "Bet", "Model P", "Books P"]
    pred_cols = [
        "Team",
        "Model EV",
        "Books EV",
        "Dist",
        "CV",
        "Model Param",
        "Gate",
        "Temperature",
        "Disp Cal",
        "Step",
        "Actual",
    ]

    # Ensure columns exist
    for col in offer_cols + pred_cols + ["Actual"]:
        if col not in history.columns:
            history[col] = np.nan

    # If old schema had "Model" and "Books" (boosted values), derive Model P / Books P
    if "Model P" not in history.columns or history["Model P"].isna().all():
        if "Model" in history.columns:
            history["Model P"] = history["Model"] / history.get("Boost", 1)
    if "Books P" not in history.columns or history["Books P"].isna().all():
        if "Books" in history.columns:
            history["Books P"] = history["Books"] / history.get("Boost", 1)

    rows = []
    for key, grp in history.groupby(pred_key, dropna=False):
        player, league, date, market = key
        # Take prediction-level cols from the most recent row (last in group)
        latest = grp.iloc[-1]

        offers = []
        for _, r in grp.iterrows():
            line = r.get("Line")
            if pd.isna(line):
                continue
            offers.append(
                (
                    float(line),
                    float(r.get("Boost", 1)),
                    str(r.get("Platform", "")),
                    str(r.get("Bet", "")),
                    float(r["Model P"]) if pd.notna(r.get("Model P")) else np.nan,
                    float(r["Books P"]) if pd.notna(r.get("Books P")) else np.nan,
                )
            )

        # Dedup offers by (Line, Platform)
        offers = _dedup_offers(offers)

        row = {
            "Player": player,
            "League": league,
            "Date": date,
            "Market": market,
            "Offers": offers,
        }
        for col in pred_cols:
            row[col] = latest.get(col, np.nan)

        # Try to recover Actual from old Result column
        if pd.isna(row.get("Actual")) and "Result" in grp.columns:
            latest.get("Result")
            # Can't recover numeric Actual from Over/Under, but mark as resolved
            # so resolve_history will skip it if Actual is still NaN
            pass

        rows.append(row)

    return pd.DataFrame(rows)


def _dedup_offers(offers):
    """Deduplicate offers by (Line, Platform), keeping the last occurrence."""
    seen = {}
    for offer in offers:
        key = (offer[0], offer[2])  # (Line, Platform)
        seen[key] = offer
    return list(seen.values())


def _merge_offers(old_offers, new_offers):
    """Merge old and new offer lists, deduplicating by (Line, Platform).

    New offers overwrite old ones with the same (Line, Platform) key.
    """
    merged = {}
    if old_offers:
        for offer in old_offers:
            key = (offer[0], offer[2])  # (Line, Platform)
            merged[key] = offer
    for offer in new_offers:
        key = (offer[0], offer[2])
        merged[key] = offer
    return list(merged.values())


def explode_offers(history):
    """Expand Offers column into one row per offer, inheriting prediction-level cols.

    Returns DataFrame with columns: all prediction-level cols + Line, Boost,
    Platform, Bet, Model P, Books P, Result, Hit.
    """
    if "Offers" not in history.columns:
        return history

    pred_cols = [c for c in history.columns if c not in ("Offers",)]
    rows = []
    for _, pred in history.iterrows():
        offers = pred.get("Offers")
        if not isinstance(offers, list) or len(offers) == 0:
            continue
        actual = pred.get("Actual")
        for offer in offers:
            line, boost, platform, bet, model_p, books_p = offer
            # Derive result from actual vs line
            if pd.notna(actual) and pd.notna(line):
                if " vs. " in str(pred.get("Player", "")):
                    result = (
                        "Over"
                        if (actual + line) > 0
                        else ("Under" if (actual + line) < 0 else "Push")
                    )
                else:
                    result = "Over" if actual > line else ("Under" if actual < line else "Push")
            else:
                result = np.nan

            row = {col: pred[col] for col in pred_cols}
            row.update(
                {
                    "Line": line,
                    "Boost": boost,
                    "Platform": platform,
                    "Bet": bet,
                    "Model P": model_p,
                    "Books P": books_p,
                    "Result": result,
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    exploded = pd.DataFrame(rows)
    if "Result" in exploded.columns and "Bet" in exploded.columns:
        resolved = exploded.dropna(subset=["Result", "Bet"])
        if not resolved.empty:
            exploded.loc[resolved.index, "Hit"] = (resolved["Bet"] == resolved["Result"]).astype(
                int
            )
    # Derive Kelly-relevant columns
    if "Model P" in exploded.columns and "Boost" in exploded.columns:
        exploded["Model"] = exploded["Model P"] * exploded["Boost"]
        exploded["Books"] = exploded["Books P"].fillna(0.5) * exploded["Boost"]
        exploded["K"] = (exploded["Model"] - 1) / (exploded["Boost"] - 1).replace(0, np.nan)
    return exploded


def resolve_history(history, stats):
    """Fill in Actual column for predictions in history.dat.

    Only sets the Actual numeric value. Result (Over/Under/Push) is derived
    per-offer in explode_offers() from Actual vs Line.
    """
    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    pending = history.loc[
        history["Actual"].isna()
        & (pd.to_datetime(history["Date"], errors="coerce").dt.date < datetime.today().date())
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
                    (gamelog[ls["player"]] == players[0])
                    & (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                g2 = gamelog.loc[
                    (gamelog[ls["player"]] == players[1])
                    & (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                if (
                    g1.empty
                    or g2.empty
                    or g1[row["Market"]].isna().any()
                    or g2[row["Market"]].isna().any()
                ):
                    continue
                history.at[i, "Actual"] = g1.iloc[0][row["Market"]] + g2.iloc[0][row["Market"]]

            elif " vs. " in row["Player"]:
                players = row["Player"].split(" vs. ")
                g1 = gamelog.loc[
                    (gamelog[ls["player"]] == players[0])
                    & (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                g2 = gamelog.loc[
                    (gamelog[ls["player"]] == players[1])
                    & (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                if (
                    g1.empty
                    or g2.empty
                    or g1[row["Market"]].isna().any()
                    or g2[row["Market"]].isna().any()
                ):
                    continue
                history.at[i, "Actual"] = g1.iloc[0][row["Market"]] - g2.iloc[0][row["Market"]]

            else:
                g = gamelog.loc[
                    (gamelog[ls["player"]] == row["Player"])
                    & (pd.to_datetime(gamelog[ls["date"]]).dt.date == bet_date)
                ]
                if g.empty or g[row["Market"]].isna().any():
                    continue
                history.at[i, "Actual"] = g.iloc[0][row["Market"]]
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

    prob_col = (
        "Model P" if "Model P" in history.columns and history["Model P"].notna().any() else "Model"
    )
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
        hist_stats = hist_stats[
            ["Period", "Split", "Accuracy", "Balance", "LogLoss", "Brier", "ROI", "Samples"]
        ]

    # --- Daily granular data for time series charting ---
    one_year = filtered.loc[filtered["_date"] >= today - timedelta(days=365)].copy()
    one_year["Hit"] = (one_year["Bet"] == one_year["Result"]).astype(int)
    one_year["Profit Unit"] = one_year["Hit"] * (100 / 110) - (1 - one_year["Hit"])

    daily = (
        one_year.groupby(["_date", "League", "Market"])
        .agg(
            Bets=("Hit", "count"),
            Hits=("Hit", "sum"),
            Avg_Model_P=(prob_col, "mean"),
            Profit=("Profit Unit", "sum"),
        )
        .reset_index()
    )
    daily.rename(columns={"_date": "Date"}, inplace=True)
    daily["Date"] = daily["Date"].astype(str)
    daily = daily.sort_values(["Date", "League", "Market"])

    # --- Calibration ---
    cal_data = one_year.copy()
    bins = np.linspace(0.5, 1.0, 11)
    cal_data["bin"] = pd.cut(cal_data[prob_col], bins=bins)
    calibration = (
        cal_data.groupby("bin", observed=False)
        .agg(
            Predicted=(prob_col, "mean"),
            Actual=("Hit", "mean"),
            Count=("Hit", "count"),
        )
        .reset_index()
    )
    calibration["Bin"] = calibration["bin"].astype(str)
    calibration = calibration[["Bin", "Predicted", "Actual", "Count"]]

    # --- ROI by threshold ---
    roi_rows = []
    for threshold in [0.55, 0.58, 0.60, 0.65, 0.70]:
        for tf_label, tf_days in TIMEFRAMES:
            cutoff = today - timedelta(days=tf_days)
            for label_filter, subset in [
                ("All", history.loc[history["_date"] >= cutoff]),
                (
                    "Book Filtered",
                    history.loc[(history["_date"] >= cutoff) & (history["Books"] > 0.52)],
                ),
            ]:
                t_sub = subset.loc[subset["Model"] > threshold]
                if len(t_sub) == 0:
                    continue
                wins = (t_sub["Bet"] == t_sub["Result"]).sum()
                profit = wins * (100 / 110) - (len(t_sub) - wins)
                roi_rows.append(
                    {
                        "Period": tf_label,
                        "Threshold": threshold,
                        "Filter": label_filter,
                        "Bets": len(t_sub),
                        "Win%": round(wins / len(t_sub), 4),
                        "ROI": round(profit / len(t_sub), 4),
                    }
                )
    roi = pd.DataFrame(roi_rows)

    return hist_stats, daily, calibration, roi


def compute_parlay_metrics(parlays, stats, stat_map):
    """Compute parlay P&L, hit rates, and correlation calibration.

    Returns (profit_df, daily_parlays, size_stats, corr_cal) DataFrames.
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    parlays[["Legs", "Misses"]] = parlays[["Legs", "Misses"]].astype(int)
    parlays["Hit"] = (parlays["Misses"] == 0).astype(int)

    # --- Underdog profit ---
    profit_rows = []
    ud_parlays = parlays.loc[parlays["Platform"] == "Underdog"].copy()
    if len(ud_parlays) > 0:
        ud_parlays["Profit"] = ud_parlays.apply(
            lambda x: np.clip(
                PAYOUT_TABLE["Underdog"][x.Legs][x.Misses]
                * (x.Boost if x.Boost < 2 or x.Misses == 0 else 1),
                None,
                100,
            )
            - 1,
            axis=1,
        )
        ud_parlays["Profit"] = ud_parlays["Profit"] * np.round(ud_parlays["Rec Bet"] * 2) / 2

        for league in ["All", *sorted(ud_parlays["League"].unique())]:
            ldf = ud_parlays if league == "All" else ud_parlays.loc[ud_parlays["League"] == league]
            if ldf.empty:
                continue
            for tf_label, tf_days in TIMEFRAMES:
                cutoff = today - timedelta(days=tf_days)
                tf_df = ldf.loc[ldf["_date"] >= cutoff]
                if not tf_df.empty:
                    p = (
                        tf_df.sort_values("Model EV", ascending=False)
                        .groupby(["Game", "Date"])
                        .apply(lambda x: x.Profit.mean())
                        .sum()
                    )
                    profit_rows.append(
                        {
                            "Platform": "Underdog",
                            "League": league,
                            "Period": tf_label,
                            "Profit": round(p, 2),
                            "Parlays": len(tf_df),
                            "Hit Rate": round(tf_df["Hit"].mean(), 4),
                        }
                    )

    profit_df = pd.DataFrame(profit_rows)

    # --- Daily parlay data for time series ---
    daily_rows = []
    for platform in parlays["Platform"].unique():
        plat_df = parlays.loc[parlays["Platform"] == platform]
        daily_plat = (
            plat_df.groupby(["_date", "League"])
            .agg(
                Parlays=("Hit", "count"),
                Hits=("Hit", "sum"),
                Misses_0=("Misses", lambda x: (x == 0).sum()),
                Misses_1=("Misses", lambda x: (x == 1).sum()),
                Misses_2_plus=("Misses", lambda x: (x >= 2).sum()),
            )
            .reset_index()
        )
        daily_plat["Platform"] = platform
        daily_plat.rename(columns={"_date": "Date"}, inplace=True)
        daily_rows.append(daily_plat)

    if daily_rows:
        daily_parlays = pd.concat(daily_rows, ignore_index=True)
        daily_parlays["Date"] = daily_parlays["Date"].astype(str)
        daily_parlays = daily_parlays[
            [
                "Date",
                "Platform",
                "League",
                "Parlays",
                "Hits",
                "Misses_0",
                "Misses_1",
                "Misses_2_plus",
            ]
        ].sort_values(["Date", "Platform", "League"])
    else:
        daily_parlays = pd.DataFrame()

    # --- Hit rate by parlay size with miss distributions (both platforms) ---
    size_rows = []
    for platform in parlays["Platform"].unique():
        plat_parlays = parlays.loc[parlays["Platform"] == platform]
        for tf_label, tf_days in TIMEFRAMES:
            cutoff = today - timedelta(days=tf_days)
            tf_parlays = plat_parlays.loc[plat_parlays["_date"] >= cutoff]
            for size in sorted(tf_parlays["Bet Size"].unique()):
                size_df = tf_parlays.loc[tf_parlays["Bet Size"] == size]
                if len(size_df) == 0:
                    continue
                row = {
                    "Platform": platform,
                    "Period": tf_label,
                    "Size": int(size),
                    "Actual Rate": round(size_df["Hit"].mean(), 4),
                    "Hit All": int((size_df["Misses"] == 0).sum()),
                    "Missed 1": int((size_df["Misses"] == 1).sum()),
                    "Missed 2+": int((size_df["Misses"] >= 2).sum()),
                    "Count": len(size_df),
                }
                if "P" in parlays.columns:
                    row["Predicted P"] = round(size_df["P"].mean(), 4)
                if "Indep P" in parlays.columns and size_df["Indep P"].notna().any():
                    row["Independent Rate"] = round(size_df["Indep P"].mean(), 4)
                elif "Leg Probs" in parlays.columns and size_df["Leg Probs"].notna().any():
                    indep = size_df["Leg Probs"].apply(
                        lambda lp: np.prod(lp)
                        if isinstance(lp, list | tuple) and len(lp) > 0
                        else np.nan
                    )
                    row["Independent Rate"] = round(indep.mean(), 4)
                size_rows.append(row)
    size_stats = pd.DataFrame(size_rows)

    # --- Correlation calibration ---
    if "P" in parlays.columns and len(parlays) > 0:
        cal_df = parlays.copy()
        bins = np.linspace(0, 1, 11)
        cal_df["p_bin"] = pd.cut(cal_df["P"], bins=bins)
        corr_cal = (
            cal_df.groupby("p_bin", observed=False)
            .agg(
                Predicted=("P", "mean"),
                Actual=("Hit", "mean"),
                Count=("Hit", "count"),
            )
            .reset_index()
        )
        corr_cal["Bin"] = corr_cal["p_bin"].astype(str)
        corr_cal = corr_cal[["Bin", "Predicted", "Actual", "Count"]]
    else:
        corr_cal = pd.DataFrame()

    # Clean up temp column
    parlays.drop(columns=["_date", "Hit"], inplace=True, errors="ignore")

    return profit_df, daily_parlays, size_stats, corr_cal


# ---------------------------------------------------------------------------
# Professional forecasting metrics
# ---------------------------------------------------------------------------


def reconstruct_prob(row, line=None):
    """Reconstruct P(outcome <= line) from saved distribution parameters.

    If line is None, uses the row's saved Line. Returns NaN if distribution
    params are missing (old data).
    """
    if pd.isna(row.get("Dist")):
        return np.nan

    if line is None:
        line = row["Line"]

    dist = row["Dist"]
    cv = row["CV"]
    ev = row["Model EV"]
    step = row.get("Step", 1)
    gate = row.get("Gate")
    if pd.isna(gate):
        gate = None
    param = row.get("Model Param")

    r = param if dist in ("NegBin", "ZINB") else None
    alpha = param if dist in ("Gamma", "ZAGamma") else None

    raw_under = get_odds(line, ev, dist, cv, alpha=alpha, r=r, gate=gate, step=step)
    return raw_under


def reconstruct_quantile(row, q):
    """Return the q-th quantile of the predicted distribution.

    Uses the saved distribution parameters to compute quantiles via
    the inverse CDF (PPF).
    """
    if pd.isna(row.get("Dist")):
        return np.nan

    dist = row["Dist"]
    cv = row["CV"]
    ev = row["Model EV"]
    param = row.get("Model Param")
    gate = row.get("Gate")
    if pd.isna(gate):
        gate = None

    if dist in ("NegBin", "ZINB"):
        r = param if param is not None and not np.isnan(param) else 1 / cv
        p = r / (r + ev)
        if gate is not None and dist == "ZINB":
            # For ZI distributions: if q <= gate, quantile is 0
            if q <= gate:
                return 0.0
            # Adjust quantile for the base distribution
            q_adj = (q - gate) / (1 - gate)
            return nbinom.ppf(q_adj, r, p)
        return nbinom.ppf(q, r, p)
    else:
        alpha = param if param is not None and not np.isnan(param) else 1 / cv**2
        if gate is not None and dist == "ZAGamma":
            if q <= gate:
                return 0.0
            q_adj = (q - gate) / (1 - gate)
            return gamma_dist.ppf(q_adj, alpha, scale=ev / alpha)
        return gamma_dist.ppf(q, alpha, scale=ev / alpha)


def compute_crps_row(row):
    """Compute CRPS for a single prediction against the actual outcome.

    Uses closed-form expressions:
    - Gamma: Gneiting & Raftery (2007), eq. 21
    - NegBin: Finite sum over PMF (Jordan et al. 2019)

    Returns NaN if distribution params or actual value are missing.
    """
    if pd.isna(row.get("Dist")) or pd.isna(row.get("Actual")):
        return np.nan

    y = row["Actual"]
    dist = row["Dist"]
    cv = row["CV"]
    ev = row["Model EV"]
    param = row.get("Model Param")
    gate = row.get("Gate")
    if pd.isna(gate):
        gate = None

    if dist in ("NegBin", "ZINB"):
        r = param if param is not None and not np.isnan(param) else 1 / cv
        p_nb = r / (r + ev)  # scipy parameterization: p = success prob

        # CRPS via finite sum for NegBin (Jordan et al. 2019)
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        # Compute numerically up to a reasonable upper bound
        upper = int(max(y, ev) * 3 + 10 * np.sqrt(ev * (1 + ev / r)))
        upper = min(upper, 500)  # cap for performance
        k = np.arange(0, upper + 1)
        pmf = nbinom.pmf(k, r, p_nb)
        cdf = np.cumsum(pmf)

        if gate is not None and dist == "ZINB":
            # ZI adjustment
            pmf_zi = pmf.copy()
            pmf_zi[0] = gate + (1 - gate) * pmf[0]
            pmf_zi[1:] = (1 - gate) * pmf[1:]
            cdf = np.cumsum(pmf_zi)

        # CRPS = sum_k (F(k) - 1(k >= y))^2 for discrete distributions
        indicator = (k >= y).astype(float)
        crps = np.sum((cdf - indicator) ** 2)
        return crps

    else:
        alpha = param if param is not None and not np.isnan(param) else 1 / cv**2
        beta = alpha / ev  # rate parameter

        if gate is not None and dist == "ZAGamma":
            # ZA-Gamma CRPS: gate * |y| + (1-gate) * CRPS_gamma
            # plus cross term for the mixture
            # Simplified: numerical integration
            from scipy.integrate import quad

            def _cdf(x):
                if x < 0:
                    return 0.0
                base = gamma_dist.cdf(x, alpha, scale=1 / beta)
                return gate + (1 - gate) * base

            def integrand(x):
                return (_cdf(x) - (1 if x >= y else 0)) ** 2

            crps, _ = quad(integrand, 0, max(y, ev) * 5 + 50, limit=200)
            return crps

        # Closed-form Gamma CRPS (Gneiting & Raftery 2007, eq. 21):
        # CRPS(Gamma(alpha, beta), y) = y*(2*F(y) - 1) - alpha/beta*(2*F_a1(y) - 1)
        #   - 1/(beta * B(0.5, alpha))
        # where F is Gamma(alpha, beta) CDF, F_a1 is Gamma(alpha+1, beta) CDF,
        # and B is the Beta function
        from scipy.special import beta as beta_fn

        scale = 1 / beta
        cdf_y = gamma_dist.cdf(y, alpha, scale=scale)
        cdf_y_a1 = gamma_dist.cdf(y, alpha + 1, scale=scale)
        crps = (
            y * (2 * cdf_y - 1)
            - (alpha / beta) * (2 * cdf_y_a1 - 1)
            - 1 / (beta * beta_fn(0.5, alpha))
        )
        return crps


def compute_brier_skill_score(subset, base_rate=0.5):
    """Compute Brier Skill Score: 1 - Brier/Brier_ref.

    base_rate: climatological base rate (default 0.5 for Over/Under).
    Returns NaN if subset is empty.
    """
    if len(subset) == 0:
        return np.nan
    hits = (subset["Bet"] == subset["Result"]).astype(int)
    prob_col = (
        "Model P" if "Model P" in subset.columns and subset["Model P"].notna().any() else "Model"
    )
    brier = brier_score_loss(hits, subset[prob_col].clip(0, 1))
    brier_ref = base_rate * (1 - base_rate)
    if brier_ref == 0:
        return np.nan
    return 1 - brier / brier_ref


def murphy_decomposition(subset):
    """Decompose Brier score into Reliability, Resolution, and Uncertainty.

    Murphy (1973) decomposition:
    BS = Reliability - Resolution + Uncertainty

    Returns dict with keys: Reliability, Resolution, Uncertainty, Brier.
    """
    if len(subset) == 0:
        return {"Reliability": np.nan, "Resolution": np.nan, "Uncertainty": np.nan, "Brier": np.nan}

    prob_col = (
        "Model P" if "Model P" in subset.columns and subset["Model P"].notna().any() else "Model"
    )
    hits = (subset["Bet"] == subset["Result"]).astype(float)
    probs = subset[prob_col].clip(0, 1).values
    bar_o = hits.mean()

    # Bin predictions
    bins = np.linspace(0.5, 1.0, 11)
    bin_idx = np.digitize(probs, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    reliability = 0.0
    resolution = 0.0
    n = len(subset)

    for k in range(len(bins) - 1):
        mask = bin_idx == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        f_k = probs[mask].mean()  # mean predicted probability in bin
        bar_o_k = hits.values[mask].mean()  # actual hit rate in bin
        reliability += n_k * (f_k - bar_o_k) ** 2
        resolution += n_k * (bar_o_k - bar_o) ** 2

    reliability /= n
    resolution /= n
    uncertainty = bar_o * (1 - bar_o)
    brier = reliability - resolution + uncertainty

    return {
        "Reliability": round(reliability, 6),
        "Resolution": round(resolution, 6),
        "Uncertainty": round(uncertainty, 6),
        "Brier": round(brier, 6),
    }


def compute_prediction_intervals(row, levels=(0.5, 0.8, 0.9)):
    """Return prediction interval bounds for given coverage levels.

    Returns dict of {level: (lower, upper)} or None if params missing.
    """
    if pd.isna(row.get("Dist")):
        return None

    intervals = {}
    for level in levels:
        tail = (1 - level) / 2
        lower = reconstruct_quantile(row, tail)
        upper = reconstruct_quantile(row, 1 - tail)
        intervals[level] = (lower, upper)
    return intervals


def compute_coverage(subset, levels=(0.5, 0.8, 0.9)):
    """Compute prediction interval coverage for resolved predictions.

    Returns dict of {level: coverage_fraction}.
    Only uses rows with both distribution params and Actual values.
    """
    valid = subset.dropna(subset=["Dist", "Actual"])
    if len(valid) == 0:
        return {level: np.nan for level in levels}

    results = {level: 0 for level in levels}
    count = 0

    for _, row in valid.iterrows():
        intervals = compute_prediction_intervals(row, levels)
        if intervals is None:
            continue
        count += 1
        actual = row["Actual"]
        for level in levels:
            lo, hi = intervals[level]
            if lo <= actual <= hi:
                results[level] += 1

    if count == 0:
        return {level: np.nan for level in levels}

    return {level: results[level] / count for level in levels}
