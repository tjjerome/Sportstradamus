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


# ARCHIVED 2026-11-26 from src/sportstradamus/clv.py
# Reason: Its one caller, underdog_pickem.resolve_market_shrinkage, was archived when
#         pick'em and the paper ledger came to read the fitted number (policy_v5), so
#         nothing reads a segment's live trust. Its four tests were deleted from
#         tests/golden/test_clv_close_sanity.py (git keeps them).
# Last live SHA: 41d03f93
# Original imports (now unresolved here):
#   import numpy as np
#   import pandas as pd
#   from sportstradamus.clv import CLV_SEGMENT_MIN_N, _segments_parquet_path

# From src/sportstradamus/clv.py.
def get_segment_calibration(league: str, market: str) -> tuple[float, int]:
    """Return ``(shrinkage, n)`` for one ``(league, market)`` segment.

    Reads ``data/runtime/clv_segments.parquet`` (written by :func:`persist_segments`).
    The shrinkage weight is ``frac_beat_close`` linearly remapped from
    ``[0.5, 1.0]`` onto ``[0.0, 1.0]`` and clamped — a segment that beats
    the close 50% of the time gets no credit; one that beats it every
    time gets full credit. ``n`` is the number of resolved legs in the
    segment, surfaced for the Kelly blending ramp.

    The shrinkage is NaN — what ``kelly.resolve_shrinkage`` reads as "no live
    evidence" — when no parquet exists, the segment is absent (``n`` is 0) or it
    holds fewer than :data:`CLV_SEGMENT_MIN_N` legs (``n`` is that count). Never
    ``1.0`` there: that reads as full trust in exactly the cells nothing has measured.
    """
    try:
        seg_df = pd.read_parquet(_segments_parquet_path())
    except (OSError, ValueError):
        return np.nan, 0
    rows = seg_df.loc[(seg_df["League"] == league) & (seg_df["Market"] == market)]
    if rows.empty:
        return np.nan, 0
    n = int(rows["n"].sum())
    if n < CLV_SEGMENT_MIN_N:
        return np.nan, n
    weighted_beat = float((rows["frac_beat_close"] * rows["n"]).sum() / n)
    shrinkage = float(np.clip(2.0 * (weighted_beat - 0.5), 0.0, 1.0))
    return shrinkage, n


# ARCHIVED 2026-11-27 from src/sportstradamus/clv.py
# Reason: Nothing has read data/runtime/clv_segments.parquet since get_segment_calibration
#         (above) was archived, so reflect stopped writing it; summarize's "all_segments"
#         key, which only this writer took, went with it. Its test was deleted with
#         tests/golden/test_clv_close_sanity.py (git keeps it).
# Last live SHA: c7432a95
# Original imports (now unresolved here):
#   import importlib.resources as pkg_resources
#   from pathlib import Path
#   import pandas as pd
#   from sportstradamus import data as _data_pkg

# From src/sportstradamus/clv.py.
def _segments_parquet_path() -> Path:
    return Path(str(pkg_resources.files(_data_pkg) / "runtime" / "clv_segments.parquet"))


# From src/sportstradamus/clv.py.
def persist_segments(grouped: pd.DataFrame) -> None:
    """Write the full per-segment frame to ``data/runtime/clv_segments.parquet``.

    All segments are persisted, not just the reportable ones. Called explicitly from
    ``reflect`` after :func:`summarize`.
    """
    if grouped is None or grouped.empty:
        return
    path = _segments_parquet_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(path, index=False)
