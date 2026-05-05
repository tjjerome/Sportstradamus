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

import numpy as np
import pandas as pd

# Per-(League, Market, Platform) CLV summary segments require this many legs
# to be reported. Smaller segments are statistical noise.
CLV_SEGMENT_MIN_N = 20

_OVER_BETS = {"Over", "Higher", "over", "higher"}


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

    grouped = (
        df.groupby(["League", "Market", "Platform"], dropna=False)["Market CLV"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "market_clv"})
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
    }


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
