# ARCHIVED 2026-10-11 from src/sportstradamus/clv.py
# Reason: Nothing called the close-snapshot freshness warning, and no code builds
#         the sample-timestamp frame it takes; the 90-minute threshold only it read
#         goes with it. Its two tests were deleted from
#         tests/golden/test_clv_close_sanity.py (git keeps them).
# Last live SHA: a63b9107
# Original imports (now unresolved here):
#   import pandas as pd
#   from sportstradamus.clv import _logger

# From src/sportstradamus/clv.py.
#
# An archive snapshot taken more than this many minutes before kickoff is too
# stale to credibly stand in for the closing line; warn once per segment.
CLOSE_LOOKBACK_WARN_MINUTES = 90


def check_close_sample_freshness(
    samples: pd.DataFrame,
    *,
    logger=None,
) -> pd.DataFrame:
    """Warn when archive close-snapshots lag too far behind ``commence_time``.

    Iterates the supplied ``samples`` frame — one row per offer — and emits
    one WARNING per ``(league, market, date)`` segment when any snapshot's
    ``sample_ts`` precedes ``commence_time`` by more than
    :data:`CLOSE_LOOKBACK_WARN_MINUTES`. Rows missing either timestamp are
    skipped silently.

    Args:
        samples: DataFrame with columns ``league``, ``market``, ``date``,
            ``sample_ts``, ``commence_time``. The latter two must be
            timezone-aware datetimes (or NaT).
        logger: Optional override for the structured ``reflect`` logger;
            tests pass a captured logger to assert on warnings.

    Returns:
        A frame of the segments that produced warnings, with columns
        ``league``, ``market``, ``date``, ``max_lag_minutes``, ``n``.
        Empty when nothing was stale.
    """
    log = logger if logger is not None else _logger
    required = {"league", "market", "date", "sample_ts", "commence_time"}
    if samples.empty or not required.issubset(samples.columns):
        return pd.DataFrame(columns=["league", "market", "date", "max_lag_minutes", "n"])

    df = samples.copy()
    df["sample_ts"] = pd.to_datetime(df["sample_ts"], utc=True, errors="coerce")
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["sample_ts", "commence_time"])
    if df.empty:
        return pd.DataFrame(columns=["league", "market", "date", "max_lag_minutes", "n"])

    df["lag_minutes"] = (df["commence_time"] - df["sample_ts"]).dt.total_seconds() / 60.0
    stale = df.loc[df["lag_minutes"] > CLOSE_LOOKBACK_WARN_MINUTES]
    if stale.empty:
        return pd.DataFrame(columns=["league", "market", "date", "max_lag_minutes", "n"])

    grouped = (
        stale.groupby(["league", "market", "date"], dropna=False)["lag_minutes"]
        .agg(["max", "count"])
        .reset_index()
        .rename(columns={"max": "max_lag_minutes", "count": "n"})
    )
    for _, seg in grouped.iterrows():
        log.warning(
            "stale close snapshot (lag=%.0fmin > %dmin) for %s/%s on %s across %d legs",
            seg["max_lag_minutes"],
            CLOSE_LOOKBACK_WARN_MINUTES,
            seg["league"],
            seg["market"],
            seg["date"],
            int(seg["n"]),
        )
    return grouped
