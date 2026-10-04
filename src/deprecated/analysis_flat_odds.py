# ARCHIVED 2026-10-04 from src/sportstradamus/analysis.py
# Reason: Receipts, Lab Diagnostics and reflect grade realized performance at the
#         platform payout through realized.py, so the flat -110 record and
#         calibration helpers and the hand-mirrored Underdog parlay payout table
#         were retired with no caller left.
# Last live SHA: 0f35e617
# Original imports (now unresolved here):
#   from datetime import datetime, timedelta
#   import numpy as np
#   import pandas as pd
#   from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
#   from tqdm import tqdm
#   from sportstradamus.analysis import TIMEFRAMES, check_bet

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

# Pick-quality floors for the dashboard summary splits: keep only legs the model
# rates above _MODEL_PROB_PICK_FLOOR, and (for the "Book Filtered" split) that the
# book also priced above _BOOK_PROB_PICK_FLOOR.
_MODEL_PROB_PICK_FLOOR = 0.58
_BOOK_PROB_PICK_FLOOR = 0.52

# Model-probability cut points for the ROI threshold sweep (compute_individual_metrics).
_ROI_THRESHOLDS = (0.55, 0.58, 0.60, 0.65, 0.70)

# Flat -110 ("standard juice") accounting: risk 110 to win 100. The Receipts hero and
# skeptic checks price every rec as one flat -110 unit so the numbers stay comparable.
JUICE_PAYOUT = 100 / 110
# Decimal odds of that flat bet (stake 1 → return 1 + 100/110 on a win).
_FLAT_DECIMAL_ODDS = 1 + JUICE_PAYOUT
# A rec clears the "EV>5%" skeptic check when its model edge at the flat reference price
# clears this: Win Prob * _FLAT_DECIMAL_ODDS - 1 >= 0.05  <=>  Win Prob >= 0.55.
_EV_EDGE_MIN = 0.05
# One real-world bet = one (player, market, line, side, date); the snapshot lists the same
# prop under every book that posts it, so the Receipts hero dedups on this before counting.
_BET_KEY = ["Date", "Player", "Market", "Line", "Bet"]

# Reliability-diagram bin edges for calibration_summary — wider than
# _daily_calibration_tables's [0.5, 1.0] floor to give the alt/ladder split room to
# show tail bins (alt lines carry more extreme predicted probabilities).
_CAL_BINS = np.arange(0.40, 1.01, 0.05)


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
            tf_data.loc[tf_data["Market EV"] > _BOOK_PROB_PICK_FLOOR],
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


def _with_profit_units(exploded):
    """Return a copy of the settled rows carrying ``Hit`` and the flat -110 ``Profit Unit``.

    Pushes and unresolved rows (``Result`` NaN) are dropped, not graded — a push
    refunds the stake and an unresolved row is not a settled bet, so counting
    either as a loss fakes the record. ``Hit`` is derived from ``Bet == Result``
    when absent; an upstream NaN-safe ``Hit`` is kept as-is. A hit pays
    ``JUICE_PAYOUT``, a miss ``-1`` — the flat-unit basis the Receipts hero and
    skeptic checks share so every number is comparable.
    """
    df = exploded.loc[exploded["Result"].notna() & (exploded["Result"] != "Push")].copy()
    if "Hit" not in df.columns:
        df["Hit"] = (df["Bet"] == df["Result"]).astype(int)
    df["Profit Unit"] = df["Hit"] * JUICE_PAYOUT - (1 - df["Hit"])
    return df


def _daily_calibration_tables(filtered, prob_col, today):
    """Daily per-(date, league, market) P&L and calibration, over settled bets only."""
    one_year = _with_profit_units(filtered.loc[filtered["_date"] >= today - timedelta(days=365)])

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
                        (history["_date"] >= cutoff)
                        & (history["Market EV"] > _BOOK_PROB_PICK_FLOOR)
                    ],
                ),
            ]:
                t_sub = subset.loc[subset["Model EV"] > threshold]
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
    # isin, not != "Push": a NaN Result (legacy row) passes != and would grade as a loss.
    history = history.loc[history["Result"].isin(("Over", "Under"))].copy()
    if history.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prob_col = (
        "Win Prob"
        if "Win Prob" in history.columns and history["Win Prob"].notna().any()
        else "Model EV"
    )
    today = datetime.today().date()
    history["_date"] = pd.to_datetime(history["Date"], errors="coerce").dt.date
    history = history.loc[history["_date"].notna()].copy()
    filtered = history.loc[history["Model EV"] > _MODEL_PROB_PICK_FLOOR]

    hist_stats = _hist_stats_table(filtered, prob_col, today)
    daily, calibration = _daily_calibration_tables(filtered, prob_col, today)
    roi = _roi_table(history, today)
    return hist_stats, daily, calibration, roi


def dedup_bets(exploded):
    """Collapse the same real-world bet posted under multiple books to a single row.

    The snapshot lists one ``(player, market, line, side, date)`` prop under every book
    that offers it; tailing it once — not once per book — is the honest unit. A different
    line is a different bet and survives. Keeps the first occurrence.
    """
    key = [c for c in _BET_KEY if c in exploded.columns]
    return exploded.drop_duplicates(subset=key) if key else exploded


def tailed_record(exploded) -> dict:
    """Receipts hero — win/loss record and flat-unit P&L if you'd tailed every rec."""
    if exploded.empty:
        return {"units": 0.0, "wins": 0, "losses": 0, "n": 0, "roi": 0.0, "win_pct": 0.0}
    df = _with_profit_units(exploded)
    n = len(df)
    wins = int(df["Hit"].sum())
    units = float(df["Profit Unit"].sum())
    return {
        "units": units,
        "wins": wins,
        "losses": n - wins,
        "n": n,
        "roi": units / n,
        "win_pct": wins / n,
    }


def ev_threshold_record(exploded, *, edge_min: float = _EV_EDGE_MIN) -> dict:
    """Record over recs whose flat -110 model edge clears ``edge_min``.

    A rec qualifies when ``prob * _FLAT_DECIMAL_ODDS - 1 >= edge_min``, the same
    accounting basis as the hero — so "record at EV>5%" stays coherent with the P&L
    beside it. ``prob`` is ``Win Prob`` (falling back to ``Model EV``).
    """
    if exploded.empty:
        return tailed_record(exploded)
    prob_col = (
        "Win Prob"
        if "Win Prob" in exploded.columns and exploded["Win Prob"].notna().any()
        else "Model EV"
    )
    qualifying = exploded.loc[exploded[prob_col] * _FLAT_DECIMAL_ODDS - 1 >= edge_min]
    return tailed_record(qualifying)


def worst_month(exploded) -> dict:
    """The single worst calendar month by flat-unit P&L — losers shown, never hidden.

    Returns ``{month, units, n, win_pct}`` for the ``YYYY-MM`` with the lowest summed
    units (ties broken to the earliest month); ``{}`` when nothing is resolved.
    """
    if exploded.empty:
        return {}
    df = _with_profit_units(exploded)
    df["_month"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m")
    df = df.loc[df["_month"].notna()]
    if df.empty:
        return {}
    grouped = (
        df.groupby("_month")
        .agg(units=("Profit Unit", "sum"), n=("Hit", "count"), wins=("Hit", "sum"))
        .sort_index()  # chronological order makes idxmin keep the earliest tie
    )
    worst = grouped["units"].idxmin()
    row = grouped.loc[worst]
    return {
        "month": worst,
        "units": float(row["units"]),
        "n": int(row["n"]),
        "win_pct": float(row["wins"] / row["n"]),
    }


def record_grid(exploded, by: str) -> pd.DataFrame:
    """Per-group record (``Bets`` / ``Win%`` / ``Units`` / ``ROI``), sorted Units-desc.

    ``by`` is one of ``League`` / ``Market`` / ``Platform``. ``Win%`` and ``ROI`` are
    fractions; the surface scales them to percentage points for the themed grid.
    """
    cols = [by, "Bets", "Win%", "Units", "ROI"]
    if exploded.empty:
        return pd.DataFrame(columns=cols)
    df = _with_profit_units(exploded)
    grouped = (
        df.groupby(by)
        .agg(Bets=("Hit", "count"), wins=("Hit", "sum"), Units=("Profit Unit", "sum"))
        .reset_index()
    )
    grouped["Win%"] = grouped["wins"] / grouped["Bets"]
    grouped["ROI"] = grouped["Units"] / grouped["Bets"]
    return grouped.sort_values("Units", ascending=False)[cols].reset_index(drop=True)


_CAL_SUMMARY_COLS = ["Alt Line", "Bin", "Predicted", "Actual", "N", "ECE", "ROI"]


def _split_calibration_rows(split: pd.DataFrame, prob_col: str, alt_line: bool) -> pd.DataFrame:
    """One split's binned reliability rows, each carrying that split's ECE and ROI.

    ``ECE`` is computed over the binned rows only (values outside ``_CAL_BINS`` after
    clipping get no bin and drop out via ``groupby(observed=True)``); ``ROI`` is flat
    -110 accounting over every resolved row in the split, whether or not it landed in
    a bin — the two denominators can legitimately differ.
    """
    binned = split.assign(_bin=pd.cut(split[prob_col].clip(0, 1), bins=_CAL_BINS))
    table = (
        binned.groupby("_bin", observed=True)
        .agg(Predicted=(prob_col, "mean"), Actual=("Hit", "mean"), N=("Hit", "count"))
        .reset_index()
    )
    if table.empty:
        return pd.DataFrame(columns=_CAL_SUMMARY_COLS)
    ece = float(
        (table["N"] / table["N"].sum() * (table["Predicted"] - table["Actual"]).abs()).sum()
    )
    roi = float(split["Profit Unit"].sum() / len(split))
    table["Bin"] = table["_bin"].astype(str)
    table["Alt Line"] = alt_line
    table["ECE"] = ece
    table["ROI"] = roi
    return table[_CAL_SUMMARY_COLS]


def calibration_summary(exploded: pd.DataFrame) -> pd.DataFrame:
    """Reliability frame: one row per (prob bin x alt split), resolved rows only.

    Carries the bin's ``Predicted`` mean / ``Actual`` hit rate / ``N`` count plus
    each split's ``ECE`` and flat-juice ``ROI``. Pure — nightly persists it,
    Receipts renders it.
    """
    if exploded.empty:
        return pd.DataFrame(columns=_CAL_SUMMARY_COLS)
    resolved = exploded.dropna(subset=["Result"])
    if resolved.empty:
        return pd.DataFrame(columns=_CAL_SUMMARY_COLS)

    df = _with_profit_units(resolved)
    prob_col = (
        "Win Prob" if "Win Prob" in df.columns and df["Win Prob"].notna().any() else "Model EV"
    )
    splits = [
        _split_calibration_rows(split, prob_col, alt_line)
        for alt_line, split in df.groupby("Alt Line")
    ]
    return (
        pd.concat(splits, ignore_index=True) if splits else pd.DataFrame(columns=_CAL_SUMMARY_COLS)
    )


def _prep_parlays(parlays, stats, stat_map, today):
    """Date-filter to the last year, resolve unresolved bets, drop the rest."""
    parlays["_date"] = pd.to_datetime(parlays["Date"], errors="coerce").dt.date
    parlays = parlays.loc[parlays["_date"].notna()].copy()
    parlays = parlays.loc[parlays["_date"] >= today - timedelta(days=365)].copy()

    unresolved = parlays.loc[parlays["Legs Resolved"].isna()]
    if not unresolved.empty:
        results = unresolved.progress_apply(
            lambda bet: check_bet(bet, stats, stat_map), axis=1
        ).to_list()
        parlays.loc[parlays["Legs Resolved"].isna(), ["Legs Resolved", "Misses"]] = results

    parlays.dropna(subset=["Legs Resolved"], inplace=True)
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
                PAYOUT_TABLE["Underdog"][x["Legs Resolved"]][x.Misses]
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

    parlays[["Legs Resolved", "Misses"]] = parlays[["Legs Resolved", "Misses"]].astype(int)
    parlays["Hit"] = (parlays["Misses"] == 0).astype(int)

    profit_df = _parlay_profit_df(parlays, today)
    daily_parlays = _parlay_daily_table(parlays)
    size_stats = _parlay_size_stats(parlays, today)
    corr_cal = _parlay_corr_cal(parlays)

    parlays.drop(columns=["_date", "Hit"], inplace=True, errors="ignore")
    return profit_df, daily_parlays, size_stats, corr_cal
