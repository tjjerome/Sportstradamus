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

from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, get_odds
from sportstradamus.history_schema import (
    CLOSING_FIELD_INDICES,
    LEGACY_OFFER_ARITY,
    OFFER_FIELDS,
    PREDICTION_LEVEL_COLS,
)

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

# Underdog applies a boost of 2x or more only to a clean entry (zero misses); a
# boosted entry that drops a leg pays the un-boosted table value instead.
_UNDERDOG_PERFECT_BOOST_MULT = 2

# Underdog platform caps per-entry payout at 100x stake regardless of parlay size.
_UNDERDOG_PAYOUT_CAP = 100

TIMEFRAMES = [
    ("7d", 7),
    ("30d", 30),
    ("3m", 91),
    ("6m", 183),
    ("1y", 365),
]

# Pick-quality floors for the dashboard summary splits: keep only legs the model
# rates above _MODEL_PROB_PICK_FLOOR, and (for the "Book Filtered" split) that the
# book also priced above _BOOK_PROB_PICK_FLOOR.
_MODEL_PROB_PICK_FLOOR = 0.58
_BOOK_PROB_PICK_FLOOR = 0.52

# Model-probability cut points for the ROI threshold sweep (compute_individual_metrics).
_ROI_THRESHOLDS = (0.55, 0.58, 0.60, 0.65, 0.70)

# Upper bound on the NegBin CRPS finite-sum support; the PMF tail is negligible
# well before this, so the cap only guards pathological (huge-EV) inputs.
_CRPS_NEGBIN_SUM_CAP = 500


def _leg_market_map(league, platform, stat_map):
    """Per-platform market-name map plus the league-specific leg-name overrides."""
    new_map = stat_map.get(platform, {}).copy()
    if league == "NHL":
        new_map.update({"Points": "points", "Blocked Shots": "blocked", "Assists": "assists"})
    if league in ("NBA", "WNBA"):
        new_map.update({"Fantasy Points": "fantasy points prizepicks"})
    return new_map


def _bet_gameday_rows(gamelog, ls, bet):
    game_dates = pd.to_datetime(gamelog[ls["date"]]).dt.date
    bet_date = pd.to_datetime(bet.Date).date()
    return gamelog.loc[(game_dates == bet_date) & (gamelog[ls["team"]].isin(bet.Game.split("/")))]


def _leg_result_value(game, ls, player, market):
    """Realized ``market`` value for a leg's player(s), or ``None`` if unavailable.

    Combo (``" + "``) and matchup (``" vs. "``) legs sum the two players' values;
    an absent player or market column yields ``None`` so the caller skips the leg.
    """
    try:
        if " + " in player:
            players = player.split(" + ")
        elif " vs. " in player:
            players = player.split(" vs. ")
        else:
            result = game.loc[game[ls["player"]] == player, market]
            return None if result.empty else result.iat[0]
        r1 = game.loc[game[ls["player"]] == players[0], market]
        r2 = game.loc[game[ls["player"]] == players[1], market]
        if r1.empty or r2.empty:
            return None
        return r1.iat[0] + r2.iat[0]
    except KeyError:
        return None


def _resolve_leg(game, ls, new_map, leg):
    """Resolve one leg string to ``1`` (miss), ``0`` (hit), or ``None`` (skip).

    ``None`` covers an unparseable leg, an unavailable result, and a push
    (``result == line``).
    """
    m = LEG_PATTERN.match(leg)
    if not m:
        return None
    player, direction, line_str, market = m.groups()
    over = direction == "Over"
    line = float(line_str)
    market = market.replace("H2H ", "")
    market = new_map.get(market, market)

    result_val = _leg_result_value(game, ls, player, market)
    if result_val is None or result_val == line:
        return None
    missed = (over and result_val < line) or (not over and result_val > line)
    return 1 if missed else 0


def check_bet(bet, stats, stat_map):
    """Resolve a single parlay bet against actual game logs.

    Returns (legs, misses) where legs is valid legs count
    and misses is number of legs that missed.
    """
    if bet.League not in stats:
        return np.nan, np.nan

    stat_obj = stats[bet.League]
    ls = stat_obj.log_strings
    new_map = _leg_market_map(bet.League, bet.Platform, stat_map)

    game = _bet_gameday_rows(stat_obj.gamelog, ls, bet)
    if game.empty:
        return np.nan, np.nan

    legs = 0
    misses = 0
    for col in [c for c in bet.index if c.startswith("Leg ") and c[4:].strip().isdigit()]:
        leg = bet[col]
        if not isinstance(leg, str) or leg == "":
            continue
        outcome = _resolve_leg(game, ls, new_map, leg)
        if outcome is None:
            continue
        legs += 1
        misses += outcome

    return legs, misses


def _migrate_flat_history(history):
    """Convert old flat history schema (one row per offer) to normalized schema.

    Groups by (Player, League, Date, Market) and collects per-offer columns
    into an Offers list of 9-tuples: (Line, Boost, Platform, Bet, ModelP, BooksP,
    CloseBooksP, MarketCLV, ModelCLV). The trailing three are NaN-padded for
    pre-CLV rows; ``reflect`` populates them from the archive on resolution.
    """
    pred_cols = [*PREDICTION_LEVEL_COLS, "Actual"]
    _prep_legacy_columns(history)

    rows = []
    for key, grp in history.groupby(["Player", "League", "Date", "Market"], dropna=False):
        player, league, date, market = key
        # Prediction-level cols come from the most recent row (last in group).
        latest = grp.iloc[-1]
        row = {
            "Player": player,
            "League": league,
            "Date": date,
            "Market": market,
            "Offers": _group_offers(grp),
        }
        for col in pred_cols:
            row[col] = latest.get(col, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def _prep_legacy_columns(history):
    """Ensure offer/prediction columns exist and back-fill Model P / Books P.

    The legacy flat schema stored boosted ``Model`` / ``Books``; the normalized
    schema wants de-boosted per-offer ``Model P`` / ``Books P``. Mutates
    ``history`` in place.
    """
    offer_cols = OFFER_FIELDS[:LEGACY_OFFER_ARITY]
    pred_cols = [*PREDICTION_LEVEL_COLS, "Actual"]
    for col in offer_cols + pred_cols + ["Actual"]:
        if col not in history.columns:
            history[col] = np.nan

    if "Model P" not in history.columns or history["Model P"].isna().all():
        if "Model" in history.columns:
            history["Model P"] = history["Model"] / history.get("Boost", 1)
    if "Books P" not in history.columns or history["Books P"].isna().all():
        if "Books" in history.columns:
            history["Books P"] = history["Books"] / history.get("Boost", 1)


def _offer_tuple(r):
    """Build one normalized 9-tuple offer from a legacy flat row, or None.

    Returns None when the row has no line (NaN). The trailing closing trio
    (Close Books P, Market CLV, Model CLV) is NaN-padded; ``reflect`` fills it.
    """
    line = r.get("Line")
    if pd.isna(line):
        return None
    return (
        float(line),
        float(r.get("Boost", 1)),
        str(r.get("Platform", "")),
        str(r.get("Bet", "")),
        float(r["Model P"]) if pd.notna(r.get("Model P")) else np.nan,
        float(r["Books P"]) if pd.notna(r.get("Books P")) else np.nan,
        np.nan,  # Close Books P — populated by reflect.
        np.nan,  # Market CLV
        np.nan,  # Model CLV
    )


def _group_offers(grp):
    """Collect a prediction group's rows into a deduped Offers list."""
    offers = [t for _, r in grp.iterrows() if (t := _offer_tuple(r)) is not None]
    return _dedup_offers(offers)


def _dedup_offers(offers):
    """Deduplicate offers by (Line, Platform), keeping the last occurrence."""
    seen = {}
    for offer in offers:
        key = (offer[0], offer[2])  # (Line, Platform)
        seen[key] = offer
    return list(seen.values())


def _merge_offers(old_offers, new_offers):
    """Merge old and new offer lists, deduplicating by (Line, Platform).

    New offers overwrite old ones with the same (Line, Platform) key, except
    in the closing-trio positions (CloseBooksP, MarketCLV, ModelCLV): a
    captured non-NaN closing value on the old offer is preserved when the
    new offer is NaN there. Without this, the next ``prophecize`` run would
    clobber any close-line snapshot ``reflect`` had already filled in.
    """
    merged = {}
    if old_offers:
        for offer in old_offers:
            key = (offer[0], offer[2])  # (Line, Platform)
            merged[key] = _pad_legacy(offer)
    for offer in new_offers:
        key = (offer[0], offer[2])
        new_padded = _pad_legacy(offer)
        if key in merged:
            old_padded = merged[key]
            merged[key] = _preserve_closing(old_padded, new_padded)
        else:
            merged[key] = new_padded
    return list(merged.values())


def _pad_legacy(offer):
    """Return ``offer`` padded to the canonical 9-field shape with NaN."""
    if isinstance(offer, tuple | list) and len(offer) == LEGACY_OFFER_ARITY:
        return (*tuple(offer), np.nan, np.nan, np.nan)
    return tuple(offer)


def _preserve_closing(old_offer, new_offer):
    """Carry old closing-trio values forward when ``new_offer`` lacks them."""
    merged = list(new_offer)
    for idx in CLOSING_FIELD_INDICES:
        if idx < len(merged) and pd.isna(merged[idx]) and idx < len(old_offer):
            merged[idx] = old_offer[idx]
    return tuple(merged)


def _offer_result(actual, line, player):
    """Over / Under / Push for one offer, or NaN if actual or line is missing.

    Matchup (``" vs. "``) players resolve on the sign of ``actual + line``;
    everyone else on ``actual`` vs ``line``.
    """
    if not (pd.notna(actual) and pd.notna(line)):
        return np.nan
    if " vs. " in str(player):
        diff = actual + line
        return "Over" if diff > 0 else ("Under" if diff < 0 else "Push")
    return "Over" if actual > line else ("Under" if actual < line else "Push")


def _offer_row(pred, pred_cols, offer):
    line, boost, platform, bet, model_p, books_p, close_p, market_clv, model_clv = _pad_legacy(
        offer
    )
    result = _offer_result(pred.get("Actual"), line, pred.get("Player", ""))
    row = {col: pred[col] for col in pred_cols}
    row.update(
        {
            "Line": line,
            "Boost": boost,
            "Platform": platform,
            "Bet": bet,
            "Model P": model_p,
            "Books P": books_p,
            "Close Books P": close_p,
            "Market CLV": market_clv,
            "Model CLV": model_clv,
            "Result": result,
        }
    )
    return row


def _add_kelly_columns(exploded):
    """Add Hit and Kelly-relevant Model / Books / K columns in place.

    ``Boost`` on persisted rows is the raw promo multiplier (1.00 = no promo).
    For Underdog the per-$1 payout multiplier is ``Boost * UNDERDOG_BOOST_BASELINE``;
    other platforms are treated as already payout-inclusive (back-compat with
    the pre-fix callers that read ``Model = Model P x Boost`` directly).
    """
    if "Result" in exploded.columns and "Bet" in exploded.columns:
        resolved = exploded.dropna(subset=["Result", "Bet"])
        if not resolved.empty:
            exploded.loc[resolved.index, "Hit"] = (resolved["Bet"] == resolved["Result"]).astype(
                int
            )
    if "Model P" in exploded.columns and "Boost" in exploded.columns:
        platform = exploded.get("Platform", pd.Series(dtype=str))
        baseline = np.where(platform == "Underdog", UNDERDOG_BOOST_BASELINE, 1.0)
        payout = exploded["Boost"] * baseline
        exploded["Model"] = exploded["Model P"] * payout
        exploded["Books"] = exploded["Books P"].fillna(0.5) * payout
        exploded["K"] = (exploded["Model"] - 1) / (payout - 1).replace(0, np.nan)
    return exploded


def explode_offers(history):
    """Expand Offers column into one row per offer, inheriting prediction-level cols.

    Returns DataFrame with columns: all prediction-level cols + Line, Boost,
    Platform, Bet, Model P, Books P, Close Books P, Market CLV, Model CLV,
    Result, Hit. Legacy 6-tuples are read transparently — the closing trio
    surfaces as NaN.
    """
    if "Offers" not in history.columns:
        return history

    pred_cols = [c for c in history.columns if c not in ("Offers",)]
    rows = []
    for _, pred in history.iterrows():
        offers = pred.get("Offers")
        if not isinstance(offers, list) or not offers:
            continue
        for offer in offers:
            rows.append(_offer_row(pred, pred_cols, offer))

    if not rows:
        return pd.DataFrame()
    return _add_kelly_columns(pd.DataFrame(rows))


def _player_date_value(gamelog, ls, player, market, date):
    """Realized ``market`` value for one player on ``date``, or None.

    None when the player has no game that date or the value is NaN. A missing
    ``market`` column raises KeyError, which the caller turns into a skip.
    """
    g = gamelog.loc[
        (gamelog[ls["player"]] == player) & (pd.to_datetime(gamelog[ls["date"]]).dt.date == date)
    ]
    if g.empty or g[market].isna().any():
        return None
    return g.iloc[0][market]


def _resolved_actual(gamelog, ls, row):
    """Actual value for one history row, or None if it cannot be resolved.

    Combo (``" + "``) sums the two players' market values; matchup (``" vs. "``)
    subtracts; a plain player uses its own. None on an unavailable player /
    value or a missing market column (KeyError).
    """
    player = row["Player"]
    market = row["Market"]
    date = pd.to_datetime(row["Date"]).date()
    try:
        if " + " in player:
            players = player.split(" + ")
            v1 = _player_date_value(gamelog, ls, players[0], market, date)
            v2 = _player_date_value(gamelog, ls, players[1], market, date)
            return None if v1 is None or v2 is None else v1 + v2
        if " vs. " in player:
            players = player.split(" vs. ")
            v1 = _player_date_value(gamelog, ls, players[0], market, date)
            v2 = _player_date_value(gamelog, ls, players[1], market, date)
            return None if v1 is None or v2 is None else v1 - v2
        return _player_date_value(gamelog, ls, player, market, date)
    except KeyError:
        return None


def resolve_history(history, stats):
    """Fill in Actual column for predictions in history.parquet.

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
        actual = _resolved_actual(stat_obj.gamelog, stat_obj.log_strings, row)
        if actual is not None:
            history.at[i, "Actual"] = actual

    return history


def _compute_stats_row(subset, prob_col):
    """Compute a single row of accuracy metrics for a subset of history."""
    if len(subset) == 0:
        return None
    sub_hit = (subset["Bet"] == subset["Result"]).astype(int)
    wins = sub_hit.sum()
    profit = wins * (100 / 110) - (len(subset) - wins)
    return {
        "Accuracy": round(accuracy_score(subset["Bet"], subset["Result"]), 4),
        "Balance": round((subset["Bet"] == "Over").mean() - (subset["Result"] == "Over").mean(), 4),
        "LogLoss": round(log_loss(sub_hit, subset[prob_col].clip(0.01, 0.99), labels=[0, 1]), 4),
        "Brier": round(brier_score_loss(sub_hit, subset[prob_col].clip(0, 1)), 4),
        "ROI": round(profit / len(subset), 4),
        "Samples": len(subset),
    }


def _append_stats_row(rows, subset, prob_col, period, split):
    """Append a hist-stats row for ``subset`` if it is non-empty."""
    r = _compute_stats_row(subset, prob_col)
    if r:
        rows.append({"Period": period, "Split": split} | r)


def _hist_stats_table(filtered, prob_col, today):
    """Accuracy/calibration/ROI rows per timeframe x {All, Book, league, market}."""
    rows = []
    for tf_label, tf_days in TIMEFRAMES:
        cutoff = today - timedelta(days=tf_days)
        tf_data = filtered.loc[filtered["_date"] >= cutoff]
        _append_stats_row(rows, tf_data, prob_col, tf_label, "All")
        _append_stats_row(
            rows,
            tf_data.loc[tf_data["Books"] > _BOOK_PROB_PICK_FLOOR],
            prob_col,
            tf_label,
            "All, Book Filtered",
        )
        for league in sorted(tf_data["League"].unique()):
            league_data = tf_data.loc[tf_data["League"] == league]
            _append_stats_row(rows, league_data, prob_col, tf_label, league)
            for market in sorted(league_data["Market"].unique()):
                market_data = league_data.loc[league_data["Market"] == market]
                _append_stats_row(rows, market_data, prob_col, tf_label, f"{league} - {market}")

    hist_stats = pd.DataFrame(rows)
    if not hist_stats.empty:
        hist_stats = hist_stats[
            ["Period", "Split", "Accuracy", "Balance", "LogLoss", "Brier", "ROI", "Samples"]
        ]
    return hist_stats


def _daily_calibration_tables(filtered, prob_col, today):
    """Daily per-(date, league, market) P&L and the model-probability calibration."""
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
    return daily, calibration


def _roi_table(history, today):
    """Win%/ROI by Model-probability threshold x timeframe x {All, Book Filtered}."""
    roi_rows = []
    for threshold in _ROI_THRESHOLDS:
        for tf_label, tf_days in TIMEFRAMES:
            cutoff = today - timedelta(days=tf_days)
            for label_filter, subset in [
                ("All", history.loc[history["_date"] >= cutoff]),
                (
                    "Book Filtered",
                    history.loc[
                        (history["_date"] >= cutoff) & (history["Books"] > _BOOK_PROB_PICK_FLOOR)
                    ],
                ),
            ]:
                t_sub = subset.loc[subset["Model"] > threshold]
                if t_sub.empty:
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
    return pd.DataFrame(roi_rows)


def compute_individual_metrics(history):
    """Compute accuracy, calibration, and ROI metrics from resolved history.

    Returns (hist_stats, daily, calibration, roi) DataFrames.
    """
    history = history.loc[history["Result"] != "Push"].copy()
    if history.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prob_col = (
        "Model P" if "Model P" in history.columns and history["Model P"].notna().any() else "Model"
    )
    today = datetime.today().date()
    history["_date"] = pd.to_datetime(history["Date"], errors="coerce").dt.date
    history = history.loc[history["_date"].notna()].copy()
    filtered = history.loc[history["Model"] > _MODEL_PROB_PICK_FLOOR]

    hist_stats = _hist_stats_table(filtered, prob_col, today)
    daily, calibration = _daily_calibration_tables(filtered, prob_col, today)
    roi = _roi_table(history, today)
    return hist_stats, daily, calibration, roi


def _prep_parlays(parlays, stats, stat_map, today):
    """Date-filter to the last year, resolve unresolved bets, drop the rest."""
    parlays["_date"] = pd.to_datetime(parlays["Date"], errors="coerce").dt.date
    parlays = parlays.loc[parlays["_date"].notna()].copy()
    parlays = parlays.loc[parlays["_date"] >= today - timedelta(days=365)].copy()

    unresolved = parlays.loc[parlays["Legs"].isna()]
    if not unresolved.empty:
        results = unresolved.progress_apply(
            lambda bet: check_bet(bet, stats, stat_map), axis=1
        ).to_list()
        parlays.loc[parlays["Legs"].isna(), ["Legs", "Misses"]] = results

    parlays.dropna(subset=["Legs"], inplace=True)
    return parlays


def _parlay_profit_df(parlays, today):
    """Underdog P&L by league x timeframe, EV-ranked and game-day averaged."""
    profit_rows = []
    ud_parlays = parlays.loc[parlays["Platform"] == "Underdog"].copy()
    if ud_parlays.empty:
        return pd.DataFrame()

    ud_parlays["Profit"] = ud_parlays.apply(
        lambda x: (
            np.clip(
                PAYOUT_TABLE["Underdog"][x.Legs][x.Misses]
                * (x.Boost if x.Boost < _UNDERDOG_PERFECT_BOOST_MULT or x.Misses == 0 else 1),
                None,
                _UNDERDOG_PAYOUT_CAP,
            )
            - 1
        ),
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
            if tf_df.empty:
                continue
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
    return pd.DataFrame(profit_rows)


def _parlay_daily_table(parlays):
    """Per-platform daily parlay counts and miss-bucket breakdown."""
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

    if not daily_rows:
        return pd.DataFrame()
    daily_parlays = pd.concat(daily_rows, ignore_index=True)
    daily_parlays["Date"] = daily_parlays["Date"].astype(str)
    return daily_parlays[
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


def _parlay_size_row(size_df, parlays, platform, tf_label, size):
    """One parlay-size summary row for a (platform, period, size) cell.

    ``Independent Rate`` prefers the stored ``Indep P``; absent that it falls
    back to the product of each leg's probability in ``Leg Probs``.
    """
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
            lambda lp: np.prod(lp) if isinstance(lp, list | tuple) and len(lp) > 0 else np.nan
        )
        row["Independent Rate"] = round(indep.mean(), 4)
    return row


def _parlay_size_stats(parlays, today):
    """Hit-rate and independence calibration by platform / timeframe / size."""
    size_rows = []
    for platform in parlays["Platform"].unique():
        plat_parlays = parlays.loc[parlays["Platform"] == platform]
        for tf_label, tf_days in TIMEFRAMES:
            cutoff = today - timedelta(days=tf_days)
            tf_parlays = plat_parlays.loc[plat_parlays["_date"] >= cutoff]
            for size in sorted(tf_parlays["Bet Size"].unique()):
                size_df = tf_parlays.loc[tf_parlays["Bet Size"] == size]
                if size_df.empty:
                    continue
                size_rows.append(_parlay_size_row(size_df, parlays, platform, tf_label, size))
    return pd.DataFrame(size_rows)


def _parlay_corr_cal(parlays):
    """Correlation-probability calibration: predicted parlay P vs realized hit."""
    if "P" not in parlays.columns or len(parlays) == 0:
        return pd.DataFrame()
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
    return corr_cal[["Bin", "Predicted", "Actual", "Count"]]


def compute_parlay_metrics(parlays, stats, stat_map):
    """Compute parlay P&L, hit rates, and correlation calibration.

    Returns (profit_df, daily_parlays, size_stats, corr_cal) DataFrames.
    """
    tqdm.pandas()
    today = datetime.today().date()

    parlays = _prep_parlays(parlays, stats, stat_map, today)
    if len(parlays) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    parlays[["Legs", "Misses"]] = parlays[["Legs", "Misses"]].astype(int)
    parlays["Hit"] = (parlays["Misses"] == 0).astype(int)

    profit_df = _parlay_profit_df(parlays, today)
    daily_parlays = _parlay_daily_table(parlays)
    size_stats = _parlay_size_stats(parlays, today)
    corr_cal = _parlay_corr_cal(parlays)

    parlays.drop(columns=["_date", "Hit"], inplace=True, errors="ignore")
    return profit_df, daily_parlays, size_stats, corr_cal


# ---------------------------------------------------------------------------
# Professional forecasting metrics
# ---------------------------------------------------------------------------


def _dist_params(row):
    """Extract (Dist, CV, Model EV, Model Param, Gate) from a saved-prediction row.

    Gate is normalized to None when absent (NaN), matching how the reconstruct
    and CRPS math treat "no zero-inflation gate".
    """
    gate = row.get("Gate")
    return (
        row["Dist"],
        row["CV"],
        row["Model EV"],
        row.get("Model Param"),
        None if pd.isna(gate) else gate,
    )


def reconstruct_prob(row, line=None):
    """Reconstruct P(outcome <= line) from saved distribution parameters.

    If line is None, uses the row's saved Line. Returns NaN if distribution
    params are missing (old data).
    """
    if pd.isna(row.get("Dist")):
        return np.nan
    if line is None:
        line = row["Line"]

    dist, cv, ev, param, gate = _dist_params(row)
    r = param if dist in ("NegBin", "ZINB") else None
    alpha = param if dist in ("Gamma", "ZAGamma") else None
    return get_odds(line, ev, dist, cv, alpha=alpha, r=r, gate=gate, step=row.get("Step", 1))


def reconstruct_quantile(row, q):
    """Return the q-th quantile of the predicted distribution.

    Uses the saved distribution parameters to compute quantiles via
    the inverse CDF (PPF).
    """
    if pd.isna(row.get("Dist")):
        return np.nan
    dist, cv, ev, param, gate = _dist_params(row)
    if dist in ("NegBin", "ZINB"):
        return _negbin_quantile(q, ev, param, cv, gate, dist)
    return _gamma_quantile(q, ev, param, cv, gate, dist)


def _negbin_quantile(q, ev, param, cv, gate, dist):
    """q-th quantile of the (zero-inflated) negative-binomial count model."""
    r = param if param is not None and not np.isnan(param) else 1 / cv
    p = r / (r + ev)
    if gate is not None and dist == "ZINB":
        if q <= gate:
            return 0.0
        q = (q - gate) / (1 - gate)
    return nbinom.ppf(q, r, p)


def _gamma_quantile(q, ev, param, cv, gate, dist):
    """q-th quantile of the (zero-augmented) gamma continuous model."""
    alpha = param if param is not None and not np.isnan(param) else 1 / cv**2
    if gate is not None and dist == "ZAGamma":
        if q <= gate:
            return 0.0
        q = (q - gate) / (1 - gate)
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
    dist, cv, ev, param, gate = _dist_params(row)
    if dist in ("NegBin", "ZINB"):
        return _negbin_crps(y, ev, param, cv, gate, dist)
    return _gamma_crps(y, ev, param, cv, gate, dist)


def _zero_inflate_pmf(pmf, gate):
    """Zero-inflate a PMF: add mass ``gate`` at 0, scale the rest by (1 - gate)."""
    pmf_zi = pmf.copy()
    pmf_zi[0] = gate + (1 - gate) * pmf[0]
    pmf_zi[1:] = (1 - gate) * pmf[1:]
    return pmf_zi


def _negbin_crps(y, ev, param, cv, gate, dist):
    """CRPS of the (zero-inflated) negative-binomial via finite PMF sum (Jordan et al. 2019).

    CRPS = sum_k (F(k) - 1(k >= y))^2 over a support truncated where the tail is
    negligible (capped by _CRPS_NEGBIN_SUM_CAP).
    """
    r = param if param is not None and not np.isnan(param) else 1 / cv
    p_nb = r / (r + ev)  # scipy parameterization: p = success prob
    upper = int(max(y, ev) * 3 + 10 * np.sqrt(ev * (1 + ev / r)))
    upper = min(upper, _CRPS_NEGBIN_SUM_CAP)
    k = np.arange(0, upper + 1)
    pmf = nbinom.pmf(k, r, p_nb)
    if gate is not None and dist == "ZINB":
        pmf = _zero_inflate_pmf(pmf, gate)
    cdf = np.cumsum(pmf)
    indicator = (k >= y).astype(float)
    return np.sum((cdf - indicator) ** 2)


def _gamma_crps(y, ev, param, cv, gate, dist):
    """CRPS of the (zero-augmented) gamma model.

    Closed-form for plain Gamma (Gneiting & Raftery 2007, eq. 21); the
    zero-augmented mixture has no closed form, so integrate the squared CDF error.
    """
    alpha = param if param is not None and not np.isnan(param) else 1 / cv**2
    beta = alpha / ev
    if gate is not None and dist == "ZAGamma":
        return _zagamma_crps(y, ev, alpha, beta, gate)

    from scipy.special import beta as beta_fn

    scale = 1 / beta
    cdf_y = gamma_dist.cdf(y, alpha, scale=scale)
    cdf_y_a1 = gamma_dist.cdf(y, alpha + 1, scale=scale)
    return (
        y * (2 * cdf_y - 1) - (alpha / beta) * (2 * cdf_y_a1 - 1) - 1 / (beta * beta_fn(0.5, alpha))
    )


def _zagamma_crps(y, ev, alpha, beta, gate):
    """ZA-Gamma CRPS via numerical integration of the squared CDF error."""
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


def backfill_crps(history: pd.DataFrame) -> bool:
    """Fill the per-prediction ``CRPS`` column in place for newly-resolved rows.

    CRPS is a pure function of a prediction's distribution and its realized
    ``Actual``, so it never changes once the prediction resolves. The nightly
    resolve computes it once here and stores it in history.parquet; the dashboard
    then only aggregates the column instead of re-integrating per row on every
    rerun. Returns True if any cell was filled, so the caller knows whether the
    parquet needs rewriting.
    """
    if "Dist" not in history.columns or "Actual" not in history.columns:
        return False
    if "CRPS" not in history.columns:
        history["CRPS"] = np.nan
    need = history["Dist"].notna() & history["Actual"].notna() & history["CRPS"].isna()
    if not need.any():
        return False
    history.loc[need, "CRPS"] = history.loc[need].apply(compute_crps_row, axis=1)
    return True


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


def compute_book_brier_skill_score(subset):
    """Brier Skill Score with the bookmaker as the reference baseline.

    Mirrors training-side ``brier_skill_score``: ``1 - brier(model)/brier(book)``.
    Hits are ``(Bet == Result)``; the model probability column is ``Model P`` with
    a fallback to ``Model``; the book probability column is ``Books P``. Returns
    NaN if subset is empty, ``Books P`` is missing or all-NaN, or the book's
    Brier is zero (degenerate baseline).
    """
    if len(subset) == 0:
        return np.nan
    if "Books P" not in subset.columns or subset["Books P"].isna().all():
        return np.nan
    hits = (subset["Bet"] == subset["Result"]).astype(int)
    prob_col = (
        "Model P" if "Model P" in subset.columns and subset["Model P"].notna().any() else "Model"
    )
    brier_model = brier_score_loss(hits, subset[prob_col].clip(0, 1))
    brier_book = brier_score_loss(hits, subset["Books P"].clip(0, 1))
    if brier_book == 0:
        return np.nan
    return 1 - brier_model / brier_book


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
        return dict.fromkeys(levels, np.nan)

    results = dict.fromkeys(levels, 0)
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
        return dict.fromkeys(levels, np.nan)

    return {level: results[level] / count for level in levels}
