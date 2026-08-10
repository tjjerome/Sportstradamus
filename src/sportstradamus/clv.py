"""Closing-line value (CLV) computation for resolved predictions.

Reads the closing snapshot from the time-series archive — the latest
observation per book at-or-before the row's nominal kickoff — and folds
it into each offer row in ``history`` as ``Close Market Prob``, ``Market CLV``,
and ``Model CLV``. The ``commence_time`` used as the ``at=`` cutoff is
derived from the row date; until per-row kickoff timestamps are wired in
the default sits at game-day evening UTC, which guarantees the cutoff is
after every league's kickoff window.

Definitions, in no-vig probability units:

    sign       = +1 if Bet in {"Over",  "Higher"} else -1
    Market CLV = sign * (Close Market Prob - Market Prob)
    Model CLV  = sign * (Close Market Prob - Win Prob)
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.config import book_skewnormal_shape
from sportstradamus.helpers.distributions import get_odds
from sportstradamus.helpers.logging import get_logger
from sportstradamus.history_schema import PREDICTION_KEY

# Per-(League, Market, Platform) CLV summary segments require this many legs
# to be reported. Smaller segments are statistical noise.
CLV_SEGMENT_MIN_N = 20

# An archive snapshot taken more than this many minutes before kickoff is too
# stale to credibly stand in for the closing line; warn once per segment.
CLOSE_LOOKBACK_WARN_MINUTES = 90

# Stand-in for true commence_time when the offer row carries only a date.
# Evening UTC is comfortably past every league's typical kickoff window so
# `at=commence_time` cuts off after every pre-game observation has landed.
_COMMENCE_DEFAULT_OFFSET = timedelta(hours=20)

_OVER_BETS = {"Over", "Higher", "over", "higher"}

_logger = get_logger("reflect")


def _segments_parquet_path() -> Path:
    return Path(str(pkg_resources.files(_data_pkg) / "runtime" / "clv_segments.parquet"))


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


def _fill_offer_rows(history: pd.DataFrame, idx, close_under: float) -> None:
    """Assign the closing trio onto ``history.loc[idx]`` in place.

    ``close_under`` is ``P(under offer_line)``, shared by every offer row in the
    ``PREDICTION_KEY`` group; each row flips it to its own side via ``Bet`` before
    computing CLV, since a single prediction can carry both Over and Under offers
    across platforms.
    """
    rows = history.loc[idx]
    close_p = rows["Bet"].apply(
        lambda bet: close_under if bet not in _OVER_BETS else 1.0 - close_under
    )
    market_clv = [
        _signed_clv(mp, cp, bet)
        for mp, cp, bet in zip(rows["Market Prob"], close_p, rows["Bet"], strict=True)
    ]
    model_clv = [
        _signed_clv(wp, cp, bet)
        for wp, cp, bet in zip(rows["Win Prob"], close_p, rows["Bet"], strict=True)
    ]
    history.loc[idx, "Close Market Prob"] = close_p
    history.loc[idx, "Market CLV"] = market_clv
    history.loc[idx, "Model CLV"] = model_clv


def _fetch_close_ev_or_composite(
    archive,
    league: str,
    market: str,
    date_str: str,
    player,
    *,
    at: datetime | None,
    dist,
    cv,
) -> tuple[float, bool]:
    """Fetch the archive's closing read for one ``PREDICTION_KEY`` group, once.

    Returns ``(value, is_composite_under_prob)``. ``Dist``/``CV`` are NaN for
    book-fallback cells with no trained model; those read
    :meth:`Archive.get_composite_under_prob` directly (a line-INEXACT devigged
    under-probability, accepted only because there's no shape to invert against).
    Otherwise reads :meth:`Archive.get_ev` (a book stat-mean, still needing the
    per-line ``get_odds`` inversion the caller applies afterwards).

    Both values are group-invariant — one consensus archive read per player/market/
    date regardless of how many distinct offer lines the group's rows quote — so this
    runs once per group, not once per line.

    Returns ``(nan, False)`` when the lookup misses.
    """
    if pd.isna(dist) or pd.isna(cv):
        composite = _safe_get_composite_under_prob(archive, league, market, date_str, player, at=at)
        return composite, True
    return _safe_get_ev(archive, league, market, date_str, player, at=at), False


def _close_prob_at_line(
    fetched: float,
    is_composite: bool,
    league: str,
    market: str,
    offer_line: float,
    dist,
    cv,
    gate,
    step,
) -> float:
    """Convert one group's fetched archive value into a probability at ``offer_line``.

    Mirrors the book-EV-to-probability conversion the live pipeline already runs at
    open (:func:`sportstradamus.prediction.model_prob._book_over_prob`): a book
    stat-mean must be inverted through ``get_odds`` at the offer's own line under the
    prediction's stored ``(Dist, CV)``. When ``fetched`` is already the composite
    under-probability (NaN-Dist/CV fallback), it's returned as-is — line-invariant by
    construction, so ``offer_line`` isn't used on that path.

    Returns ``P(under offer_line)`` — sign-flipping for Over vs. Under happens later,
    in :func:`_fill_offer_rows`. This function only needs to run once per distinct
    line in the group, not once per offer row.

    Returns NaN when ``fetched`` is NaN (archive miss).
    """
    if is_composite or pd.isna(fetched):
        return fetched

    close_ev = fetched
    gate_val = None if pd.isna(gate) else float(gate)
    step_val = 1.0 if pd.isna(step) else float(step)
    if dist == "SkewNormal":
        sigma, skew_alpha = book_skewnormal_shape(league, market, close_ev, float(cv))
        return get_odds(
            offer_line,
            close_ev,
            dist,
            cv=float(cv),
            gate=gate_val,
            step=step_val,
            sigma=float(sigma),
            skew_alpha=float(skew_alpha),
        )
    return get_odds(offer_line, close_ev, dist, cv=float(cv), gate=gate_val, step=step_val)


def _fill_one_group(history: pd.DataFrame, archive, key, idx) -> None:
    """Resolve one ``PREDICTION_KEY`` group's closing probability and write it in place.

    ``Dist``/``CV``/``Gate``/``Step`` are prediction-level columns (constant across the
    whole group), but ``Line`` is offer-level — different offer rows in the same group
    (different books, or an explicit Alt Line) can legitimately quote different lines.
    The archive fetch (:func:`_fetch_close_ev_or_composite`) is genuinely group-invariant
    and runs once; the mean-to-probability conversion
    (:func:`_close_prob_at_line`) depends on the line, so it re-runs once per distinct
    ``Line`` in the group, each writing only its own matching subset of rows.

    No-op (leaves a row NaN) when the key isn't a real (league, market) pair, the
    archive lookup misses, or the resolved probability falls outside ``[0, 1]`` — the
    last case is a clamp-to-NaN quarantine, matching ``migrate_leg_schema.py``'s
    existing handling of out-of-range ``Close Market Prob`` values. A bad conversion at
    one line does not blank out other lines in the same group that resolved fine.
    """
    player, league, date, market = key
    if not (isinstance(league, str) and isinstance(market, str)):
        return
    date_str = _normalize_date(date)
    group = history.loc[idx]
    dist = group["Dist"].iloc[0]
    cv = group["CV"].iloc[0]
    gate = group["Gate"].iloc[0]
    step = group["Step"].iloc[0]
    fetched, is_composite = _fetch_close_ev_or_composite(
        archive, league, market, date_str, player, at=_commence_time(date_str), dist=dist, cv=cv
    )
    if pd.isna(fetched):
        return

    for offer_line in group["Line"].unique():
        close_under = _close_prob_at_line(
            fetched, is_composite, league, market, float(offer_line), dist, cv, gate, step
        )
        if pd.isna(close_under) or not 0.0 <= close_under <= 1.0:
            continue
        line_idx = group.index[group["Line"] == offer_line]
        _fill_offer_rows(history, line_idx, close_under)


def fill_from_archive(history: pd.DataFrame, archive) -> pd.DataFrame:
    """Populate the closing trio on every offer row from ``archive``.

    Groups rows whose ``Close Market Prob`` is still NaN by
    :data:`~sportstradamus.history_schema.PREDICTION_KEY` and resolves each group's
    closing probability via :func:`_fill_one_group` (see :func:`_fetch_close_ev_or_composite`
    and :func:`_close_prob_at_line` for the archive-EV-to-probability conversion). Pinning
    ``at=commence_time`` makes the closing read reproducible regardless of when ``reflect``
    runs. Groups whose archive lookup returns NaN, or resolves outside ``[0, 1]``, are left
    with NaN closing fields and excluded from CLV aggregates downstream.

    Skips rows that already carry a non-NaN ``Close Market Prob`` so a
    re-run doesn't redundantly hit archive.

    Args:
        history: Flat one-row-per-offer DataFrame.
        archive: A loaded ``Archive`` instance.

    Returns:
        The same DataFrame, mutated in place. Returned for chaining.
    """
    if history.empty or "Close Market Prob" not in history.columns:
        return history

    pending = history.loc[history["Close Market Prob"].isna()]
    if pending.empty:
        return history

    for key, idx in pending.groupby(PREDICTION_KEY, dropna=False).groups.items():
        _fill_one_group(history, archive, key, idx)

    return history


def _row_movement(archive, league_val, market_val, date_val, player_val, movement_cache):
    if archive is None or not date_val or not isinstance(player_val, str):
        return None
    cache_key = (league_val, market_val, date_val, player_val)
    if cache_key not in movement_cache:
        try:
            movement_cache[cache_key] = archive.get_movement(
                league_val,
                market_val,
                date_val,
                player_val,
                until=_commence_time(date_val),
            )
        except (KeyError, ValueError, TypeError):
            movement_cache[cache_key] = None
    return movement_cache[cache_key]


def _offer_to_leg(row, movement) -> dict | None:
    close_p = row.get("Close Market Prob")
    market_clv = row.get("Market CLV")
    if pd.isna(close_p) or pd.isna(market_clv):
        return None
    model_clv = row.get("Model CLV")
    aligned = _movement_alignment(movement, row.get("Win Prob"), row.get("Bet"))
    return {
        "League": row.get("League"),
        "Market": row.get("Market"),
        "Platform": row.get("Platform"),
        "Market CLV": float(market_clv),
        "Model CLV": float(model_clv) if not pd.isna(model_clv) else np.nan,
        "MoveAligned": aligned,
    }


def _collect_clv_legs(history: pd.DataFrame, archive) -> list[dict]:
    legs: list[dict] = []
    movement_cache: dict = {}
    for _, row in history.iterrows():
        league_val = row.get("League")
        market_val = row.get("Market")
        date_val = _normalize_date(row.get("Date"))
        player_val = row.get("Player")
        movement = _row_movement(
            archive, league_val, market_val, date_val, player_val, movement_cache
        )
        leg = _offer_to_leg(row, movement)
        if leg is not None:
            legs.append(leg)
    return legs


def _aggregate_clv_segments(legs: list[dict], archive) -> dict:
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
    agg_dict: dict = {
        "n": ("Market CLV", "count"),
        "market_clv": ("Market CLV", "mean"),
        "frac_beat_close": ("_beat", "mean"),
    }
    if archive is not None:
        agg_dict["frac_lines_moved_toward_model"] = ("MoveAligned", "mean")
    grouped = (
        df.groupby(["League", "Market", "Platform"], dropna=False).agg(**agg_dict).reset_index()
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


def summarize(history: pd.DataFrame, archive=None) -> dict:
    """Return aggregate CLV stats for logging by ``reflect``.

    Iterates offer rows and computes overall n / mean Market CLV / mean
    Model CLV / fraction of legs that beat the close. Drops legs with NaN
    closing values from the count.

    When ``archive`` is supplied, augments segments with
    ``frac_lines_moved_toward_model`` — fraction of legs where the line
    movement direction matches the model's lean (model_p vs 0.5 × bet
    direction). Looked up per row via :meth:`Archive.get_movement` and
    reused across all offers on the row.
    """
    legs = _collect_clv_legs(history, archive)
    return _aggregate_clv_segments(legs, archive)


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
    if date is None:
        return ""
    if isinstance(date, str):
        return date
    try:
        return pd.to_datetime(date).strftime("%Y-%m-%d")
    except (ValueError, TypeError, AttributeError):
        return ""


def _commence_time(date_str: str) -> datetime | None:
    """Best-effort kickoff timestamp for ``date_str``.

    Until offer rows carry an explicit kickoff timestamp this is the
    midnight-of-game-date plus :data:`_COMMENCE_DEFAULT_OFFSET`, which
    guarantees the resulting ``at=`` snapshot includes every pre-game
    observation regardless of league.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d") + _COMMENCE_DEFAULT_OFFSET
    except (ValueError, TypeError):
        return None


def _safe_get_ev(
    archive,
    league: str,
    market: str,
    date: str,
    player,
    *,
    at: datetime | None = None,
) -> float:
    """Wrap ``archive.get_ev`` so any lookup miss surfaces as NaN."""
    if not date or not isinstance(player, str):
        return np.nan
    try:
        return float(archive.get_ev(league, market, date, player, at=at))
    except (KeyError, ValueError, TypeError):
        return np.nan


def _safe_get_composite_under_prob(
    archive,
    league: str,
    market: str,
    date: str,
    player,
    *,
    at: datetime | None = None,
) -> float:
    """Wrap ``archive.get_composite_under_prob`` so any lookup miss surfaces as NaN."""
    if not date or not isinstance(player, str):
        return np.nan
    try:
        return float(archive.get_composite_under_prob(league, market, date, player, at=at))
    except (KeyError, ValueError, TypeError):
        return np.nan


def _movement_alignment(movement: dict | None, model_p, bet) -> float:
    """Return 1.0 if line moved toward the model's lean, 0.0 if away, NaN when undefined.

    "Toward the model" = sign(close_line − open_line) matches the
    direction the model implicitly predicts via ``(model_p − 0.5) × sign(bet)``.
    """
    if not movement:
        return np.nan
    open_line = movement.get("open_line")
    close_line = movement.get("close_line")
    if pd.isna(open_line) or pd.isna(close_line):
        return np.nan
    if pd.isna(model_p):
        return np.nan
    line_delta = float(close_line) - float(open_line)
    if line_delta == 0:
        return np.nan
    bet_sign = 1.0 if bet in _OVER_BETS else -1.0
    model_lean = (float(model_p) - 0.5) * bet_sign  # >0 = model thinks line is too low
    if model_lean == 0:
        return np.nan
    return 1.0 if (line_delta > 0) == (model_lean > 0) else 0.0
