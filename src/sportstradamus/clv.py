"""Closing-line value (CLV) computation for resolved predictions.

Reads the post-lock archive snapshot (whatever ``Archive.get_ev`` returns by
the time ``reflect`` runs — the Odds API stops surfacing prematch odds for
in-progress games, so the last sample before kickoff is the close) and
folds it into each offer in ``history`` as ``Close Books P``,
``Market CLV``, and ``Model CLV``. Two helpers and a one-shot summary
printer are all that's needed; ``reflect`` orchestrates the call.

Definitions, in no-vig probability units:

    sign       = +1 if Bet in {"Over",  "Higher"} else -1
    Market CLV = sign * (Close Books P - Open Books P)
    Model CLV  = sign * (Close Books P - Open Model P)
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from pathlib import Path

import numpy as np
import pandas as pd

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.logging import get_logger

# Per-(League, Market, Platform) CLV summary segments require this many legs
# to be reported. Smaller segments are statistical noise.
CLV_SEGMENT_MIN_N = 20

# An archive snapshot taken more than this many minutes before kickoff is too
# stale to credibly stand in for the closing line; warn once per segment.
CLOSE_LOOKBACK_WARN_MINUTES = 90

_OVER_BETS = {"Over", "Higher", "over", "higher"}

_logger = get_logger("reflect")


def _segments_parquet_path() -> Path:
    """Return the persistence path for the per-segment CLV parquet."""
    return Path(str(pkg_resources.files(_data_pkg) / "clv_segments.parquet"))


def _signed_clv(open_p: float, close_p: float, bet: str) -> float:
    """Return signed CLV in probability points, or NaN when undefined.

    Args:
        open_p: Probability the bettor implicitly bought at placement time.
        close_p: Closing book probability at game-lock.
        bet: Direction of the bet — Over/Higher get +1, Under/Lower get -1.

    Returns:
        Signed CLV (close − open, flipped for Under). NaN if either
        probability is NaN.
    """
    if pd.isna(open_p) or pd.isna(close_p):
        return np.nan
    sign = 1 if bet in _OVER_BETS else -1
    return sign * (float(close_p) - float(open_p))


def fill_from_archive(history: pd.DataFrame, archive) -> pd.DataFrame:
    """Populate the closing trio in each ``Offers`` tuple from ``archive``.

    For every offer in every history row, query
    ``archive.get_ev(league, market, date, player)`` once and rewrite the
    9-tuple in-place with ``Close Books P``, ``Market CLV``, and
    ``Model CLV``. Offers whose archive lookup returns NaN are left with
    NaN closing fields and excluded from CLV aggregates downstream.

    Skips offers that already carry a non-NaN ``Close Books P`` so a
    re-run doesn't redundantly hit archive.

    Args:
        history: DataFrame in the normalized 9-tuple ``Offers`` schema.
        archive: A loaded ``Archive`` instance.

    Returns:
        The same DataFrame, mutated in place. Returned for chaining.
    """
    if "Offers" not in history.columns or history.empty:
        return history

    for idx, row in history.iterrows():
        offers = row.get("Offers")
        if not isinstance(offers, list) or not offers:
            continue
        league = row.get("League")
        market = row.get("Market")
        date = row.get("Date")
        player = row.get("Player")
        if not (isinstance(league, str) and isinstance(market, str)):
            continue

        date_str = _normalize_date(date)
        close_p = _safe_get_ev(archive, league, market, date_str, player)

        new_offers = []
        for offer in offers:
            if isinstance(offer, tuple | list) and len(offer) == 6:
                offer = (*tuple(offer), np.nan, np.nan, np.nan)
            elif not (isinstance(offer, tuple | list) and len(offer) == 9):
                new_offers.append(offer)
                continue
            line, boost, platform, bet, model_p, books_p, prev_close, _, _ = offer
            if not pd.isna(prev_close):
                new_offers.append(tuple(offer))
                continue
            market_clv = _signed_clv(books_p, close_p, bet)
            model_clv = _signed_clv(model_p, close_p, bet)
            new_offers.append(
                (line, boost, platform, bet, model_p, books_p, close_p, market_clv, model_clv)
            )

        history.at[idx, "Offers"] = new_offers

    return history


def summarize(history: pd.DataFrame) -> dict:
    """Return aggregate CLV stats for logging by ``reflect``.

    Iterates exploded offers and computes overall n / mean Market CLV /
    mean Model CLV / fraction of legs that beat the close. Drops legs
    with NaN closing values from the count.
    """
    legs = []
    for _, row in history.iterrows():
        offers = row.get("Offers")
        if not isinstance(offers, list):
            continue
        platform_default = row.get("Platform")
        for offer in offers:
            if not (isinstance(offer, tuple | list) and len(offer) >= 9):
                continue
            _, _, platform, _, _, _, close_p, market_clv, _model_clv = offer[:9]
            if pd.isna(close_p) or pd.isna(market_clv):
                continue
            legs.append(
                {
                    "League": row.get("League"),
                    "Market": row.get("Market"),
                    "Platform": platform or platform_default,
                    "Market CLV": float(market_clv),
                    "Model CLV": float(_model_clv) if not pd.isna(_model_clv) else np.nan,
                }
            )

    if not legs:
        return {
            "n": 0,
            "market_clv_mean": np.nan,
            "model_clv_mean": np.nan,
            "frac_beat_close": np.nan,
            "segments": pd.DataFrame(),
        }

    df = pd.DataFrame(legs)
    market_mean = float(df["Market CLV"].mean())
    model_series = df["Model CLV"].dropna()
    model_mean = float(model_series.mean()) if not model_series.empty else np.nan
    frac_beat = float((df["Market CLV"] > 0).mean())

    df["_beat"] = (df["Market CLV"] > 0).astype(float)
    grouped = (
        df.groupby(["League", "Market", "Platform"], dropna=False)
        .agg(n=("Market CLV", "count"),
             market_clv=("Market CLV", "mean"),
             frac_beat_close=("_beat", "mean"))
        .reset_index()
    )
    segments = grouped.loc[grouped["n"] >= CLV_SEGMENT_MIN_N].sort_values(
        "market_clv", ascending=False
    )

    return {
        "n": len(df),
        "market_clv_mean": market_mean,
        "model_clv_mean": model_mean,
        "frac_beat_close": frac_beat,
        "segments": segments,
        "all_segments": grouped,
    }


def persist_segments(grouped: pd.DataFrame) -> None:
    """Write the full per-segment frame to ``data/clv_segments.parquet``.

    All segments are persisted (not just the reportable ones) so the Kelly
    shrinkage getter can reason about sample size for low-n segments too.
    Called explicitly from ``reflect`` after :func:`summarize`.
    """
    if grouped is None or grouped.empty:
        return
    path = _segments_parquet_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(path, index=False)


def get_segment_calibration(league: str, market: str) -> tuple[float, int]:
    """Return ``(shrinkage, n)`` for one ``(league, market)`` segment.

    Reads ``data/clv_segments.parquet`` (written by :func:`summarize`).
    The shrinkage weight is ``frac_beat_close`` linearly remapped from
    ``[0.5, 1.0]`` onto ``[0.0, 1.0]`` and clamped — a segment that beats
    the close 50% of the time gets no credit; one that beats it every
    time gets full credit. ``n`` is the number of resolved legs in the
    segment, surfaced for the Kelly blending ramp.

    Returns ``(1.0, 0)`` when the segment has fewer than
    :data:`CLV_SEGMENT_MIN_N` legs or no parquet exists. The
    ``shrinkage=1.0`` fallback means "no information, don't shrink".
    """
    path = _segments_parquet_path()
    if not path.exists():
        return 1.0, 0
    try:
        seg_df = pd.read_parquet(path)
    except (OSError, ValueError):
        return 1.0, 0
    rows = seg_df.loc[(seg_df["League"] == league) & (seg_df["Market"] == market)]
    if rows.empty:
        return 1.0, 0
    n = int(rows["n"].sum())
    if n < CLV_SEGMENT_MIN_N:
        return 1.0, n
    weighted_beat = float((rows["frac_beat_close"] * rows["n"]).sum() / n)
    shrinkage = float(np.clip(2.0 * (weighted_beat - 0.5), 0.0, 1.0))
    return shrinkage, n


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


def _normalize_date(date) -> str:
    """Return ``date`` as ``YYYY-MM-DD`` for archive lookup."""
    if isinstance(date, str):
        return date
    try:
        return pd.to_datetime(date).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def _safe_get_ev(archive, league: str, market: str, date: str, player) -> float:
    """Wrap ``archive.get_ev`` so any lookup miss surfaces as NaN."""
    if not date or not isinstance(player, str):
        return np.nan
    try:
        return float(archive.get_ev(league, market, date, player))
    except (KeyError, ValueError, TypeError):
        return np.nan
