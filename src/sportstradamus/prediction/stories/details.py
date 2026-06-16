"""Offer-detail prerender helpers (P6a).

Pure functions run at ``prophecize`` time so the dashboard reads precomputed detail
rows instead of recomputing comps / volume / SHAP context live. Each is fed plain
frames extracted from the ``Stats`` object by the cli adapter, keeping the module
import-safe (no ``Archive()`` at module load) and unit-testable with synthetic data.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

_COMPS_WINDOW_DAYS = 300
_VOLUME_TREND_GAMES = 10  # sparkline window for the volume-stat trend
_OTHER_STATS_K = 4  # SHAP-ranked extra stats surfaced in the "Other stats" tab
# SHAP feature names are engineered (e.g. "Player TS_PCT short"); strip the profile
# prefix + rolling-window suffix to recover the backing gamelog stat ("TS_PCT").
_FEATURE_PREFIX = "Player "
_FEATURE_SUFFIXES = (" short", " growth", " long")

# One detail row per offer's player/market/opponent (platform- and line-independent —
# comps, volume, and SHAP context don't vary by book or line). The dashboard joins on
# the key.
_DETAIL_KEY = ["League", "Date", "Player", "Market", "Opponent"]
_DETAIL_COLS = [*_DETAIL_KEY, "comps_vs_opp", "volume_trend", "other_stats"]


def comps_vs_opponent(
    player: str,
    opponent: str,
    stat: str,
    comp_pairs: pd.DataFrame,
    gamelog: pd.DataFrame,
    cols: dict[str, str],
    *,
    today: dt.date,
    window_days: int = _COMPS_WINDOW_DAYS,
) -> list[dict]:
    """Comp rows for ``player`` who faced ``opponent`` within ``window_days``.

    One row per qualifying comp, closest comp first: ``{comp, n_games, avg_vs_opp,
    pct_diff}``. ``avg_vs_opp`` is the comp's mean ``stat`` in its games vs the
    opponent; ``pct_diff`` compares that to the comp's own mean over all games in
    the window (0.0 when that window mean is non-positive). Comps with no in-window
    game vs the opponent are dropped.
    """
    comps = list(dict.fromkeys(comp_pairs.loc[comp_pairs["player"] == player, "comp"]))
    if not comps or gamelog.empty or stat not in gamelog.columns:
        return []

    pcol, ocol, dcol = cols["player"], cols["opponent"], cols["date"]
    cutoff = pd.Timestamp(today - dt.timedelta(days=window_days))
    gl = gamelog[[pcol, ocol, dcol, stat]].copy()
    gl[dcol] = pd.to_datetime(gl[dcol])
    gl = gl[(gl[dcol] >= cutoff) & gl[pcol].isin(comps)]
    if gl.empty:
        return []

    overall = gl.groupby(pcol)[stat].mean()
    vs_opp = gl[gl[ocol] == opponent].groupby(pcol)[stat].agg(["mean", "size"])

    out = []
    for comp in comps:
        if comp not in vs_opp.index:
            continue
        avg = float(vs_opp.loc[comp, "mean"])
        base = float(overall.get(comp, 0.0))
        out.append(
            {
                "comp": comp,
                "n_games": int(vs_opp.loc[comp, "size"]),
                "avg_vs_opp": round(avg, 2),
                "pct_diff": round(avg / base - 1.0, 4) if base > 0 else 0.0,
            }
        )
    return out


def volume_trend(
    player: str,
    gamelog: pd.DataFrame,
    cols: dict[str, str],
    volume_stat: str,
    *,
    n: int = _VOLUME_TREND_GAMES,
) -> list[float]:
    """Player's last-``n`` ``volume_stat`` values, oldest first (sparkline series).

    Empty when the stat column or the player is absent from the gamelog.
    """
    pcol, dcol = cols["player"], cols["date"]
    if volume_stat not in gamelog.columns or gamelog.empty:
        return []
    pg = gamelog.loc[gamelog[pcol] == player, [dcol, volume_stat]].copy()
    if pg.empty:
        return []
    pg[dcol] = pd.to_datetime(pg[dcol])
    pg = pg.sort_values(dcol).tail(n)
    return [round(float(v), 2) for v in pg[volume_stat]]


def _feature_to_stat(feature: str) -> str:
    for suffix in _FEATURE_SUFFIXES:
        if feature.endswith(suffix):
            feature = feature[: -len(suffix)]
            break
    if feature.startswith(_FEATURE_PREFIX):
        feature = feature[len(_FEATURE_PREFIX) :]
    return feature


def rank_other_stats(
    market: str,
    importances: pd.DataFrame,
    *,
    exclude: set[str],
    available: set[str],
    k: int = _OTHER_STATS_K,
) -> list[str]:
    """Top-``k`` SHAP-ranked gamelog stats for ``market``, descending by importance.

    Each feature is normalized to its backing gamelog stat (:func:`_feature_to_stat`);
    only stats present in the gamelog (``available``) and not already shown elsewhere
    (``not in exclude``) are kept, deduped to the first (highest-SHAP) occurrence.
    Empty when the market has no importance column.
    """
    if market not in importances.columns:
        return []
    ranked = importances[market].dropna().sort_values(ascending=False)
    kept: list[str] = []
    for feature in ranked.index:
        stat = _feature_to_stat(feature)
        if stat in available and stat not in exclude and stat not in kept:
            kept.append(stat)
        if len(kept) >= k:
            break
    return kept


def position_percentile(value: float, cohort: Sequence[float]) -> float | None:
    """Fraction of ``cohort`` values ≤ ``value`` (0..1), or ``None`` for an empty cohort."""
    if len(cohort) == 0:
        return None
    return round(float((np.asarray(cohort, dtype=float) <= value).mean()), 4)


def _recent_means(gamelog: pd.DataFrame, cols: dict[str, str], stat: str) -> pd.Series:
    pcol, dcol = cols["player"], cols["date"]
    if stat not in gamelog.columns:
        return pd.Series(dtype=float)
    g = gamelog[[pcol, dcol, stat]].copy()
    g[dcol] = pd.to_datetime(g[dcol])
    g = g.sort_values(dcol)
    return g.groupby(pcol)[stat].apply(lambda s: s.tail(_VOLUME_TREND_GAMES).mean())


def _position_cohorts(
    mgrp: pd.DataFrame, ranked: list[str], means: dict[str, pd.Series]
) -> dict[tuple[str, str], list[float]]:
    """``(position, stat) -> [recent-mean of every slate player at that position]``."""
    cohorts: dict[tuple[str, str], list[float]] = {}
    for pos, pgrp in mgrp.groupby("Position"):
        players = pgrp["Player"].tolist()
        for stat in ranked:
            series = means[stat]
            cohorts[(pos, stat)] = [
                float(series[p]) for p in players if p in series.index and pd.notna(series[p])
            ]
    return cohorts


def _stat_entry(
    stat: str,
    player: str,
    pos: str,
    means: dict[str, pd.Series],
    cohorts: dict[tuple[str, str], list[float]],
    gamelog: pd.DataFrame,
    cols: dict[str, str],
) -> dict | None:
    """One ``{stat, value, series, percentile}`` row, or ``None`` when the player has
    no recent value for ``stat``. ``value`` is the last-``n`` mean; ``percentile`` ranks
    it against the same-position cohort.
    """
    raw = means[stat].get(player)
    if raw is None or pd.isna(raw):
        return None
    value = round(float(raw), 2)
    return {
        "stat": stat,
        "value": value,
        "series": volume_trend(player, gamelog, cols, stat),
        "percentile": position_percentile(value, cohorts.get((pos, stat), [])),
    }


def _resolve_volume_stat(volume_stat: str | dict[str, str], pos: str) -> str:
    """The player's volume stat — a flat per-league string, or a per-position map (NFL).

    Position labels carry a depth rank (``"QB1"``, ``"WR2"``); the map keys on the base
    position, so the trailing rank digits are stripped before lookup.
    """
    if isinstance(volume_stat, dict):
        return volume_stat.get(pos.rstrip("0123456789"), "")
    return volume_stat


def _other_stats_for(
    player: str,
    pos: str,
    ranked: list[str],
    means: dict[str, pd.Series],
    cohorts: dict[tuple[str, str], list[float]],
    gamelog: pd.DataFrame,
    cols: dict[str, str],
) -> list[dict]:
    entries = (_stat_entry(s, player, pos, means, cohorts, gamelog, cols) for s in ranked)
    return [e for e in entries if e is not None]


def _market_detail_rows(
    mgrp: pd.DataFrame,
    league: str,
    market: str,
    ld: dict,
    importances: pd.DataFrame,
    exclude: set[str],
    today: dt.date,
) -> list[dict]:
    gl, cols, comp_pairs, volume_stat = (
        ld["gamelog"],
        ld["cols"],
        ld["comp_pairs"],
        ld["volume_stat"],
    )
    stat = str(mgrp["Stat"].iloc[0])
    vol_by_pos = {p: _resolve_volume_stat(volume_stat, p) for p in mgrp["Position"].astype(str)}
    vol_present = [s for s in dict.fromkeys(vol_by_pos.values()) if s and s in gl.columns]
    ranked = rank_other_stats(
        f"{league}_{market}",
        importances,
        exclude=exclude | set(vol_by_pos.values()) | {stat},
        available=set(gl.columns),
    )
    cohort_stats = vol_present + ranked
    means = {s: _recent_means(gl, cols, s) for s in cohort_stats}
    cohorts = _position_cohorts(mgrp, cohort_stats, means)

    rows = []
    for off in mgrp.to_dict("records"):
        player, opp, pos = off["Player"], off["Opponent"], str(off.get("Position", ""))
        vol_stat = vol_by_pos.get(pos, "")
        volume = (
            _stat_entry(vol_stat, player, pos, means, cohorts, gl, cols)
            if vol_stat in means
            else None
        )
        rows.append(
            {
                "League": league,
                "Date": off["Date"],
                "Player": player,
                "Market": market,
                "Opponent": opp,
                "comps_vs_opp": json.dumps(
                    comps_vs_opponent(player, opp, stat, comp_pairs, gl, cols, today=today)
                ),
                "volume_trend": json.dumps(volume),
                "other_stats": json.dumps(
                    _other_stats_for(player, pos, ranked, means, cohorts, gl, cols)
                ),
            }
        )
    return rows


def build_offer_details(
    offers: pd.DataFrame,
    league_data: dict[str, dict],
    importances: pd.DataFrame,
    *,
    exclude: set[str],
    today: dt.date,
) -> pd.DataFrame:
    """One prerendered detail row per ``(League, Date, Player, Market, Opponent)``.

    ``league_data[league]`` carries the frames extracted from that league's ``Stats``
    object: ``comp_pairs`` (flattened ``_comp_pairs``), ``gamelog`` (short gamelog),
    ``cols`` (``log_strings`` player/opponent/date names), and ``volume_stat``. A
    league absent from ``league_data`` is skipped (no model / not scored).
    """
    if offers.empty:
        return pd.DataFrame(columns=_DETAIL_COLS)
    keyed = offers.drop_duplicates(subset=_DETAIL_KEY)
    rows: list[dict] = []
    for league, lgrp in keyed.groupby("League"):
        ld = league_data.get(league)
        if ld is None:
            continue
        for market, mgrp in lgrp.groupby("Market"):
            rows.extend(_market_detail_rows(mgrp, league, market, ld, importances, exclude, today))
    return pd.DataFrame(rows, columns=_DETAIL_COLS)
