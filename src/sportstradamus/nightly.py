"""Nightly resolution script for Sportstradamus.

Runs after games finish to:
1. Fetch latest stats from league APIs (update)
2. Fill in Actual column in history (parquet primary, .dat fallback)
3. Fill in Close Books P / Market CLV / Model CLV per offer
4. Fill in Legs/Misses columns in parlay_hist
5. Write resolve_meta.json with last-run timestamp
6. Compute and persist per-(league, market) live metrics (Gate 2)

Schedule with cron after games finish, e.g.:
    0 2 * * * cd /home/trevor/Sportstradamus && poetry run reflect
"""

import importlib.resources as pkg_resources
import json
from datetime import UTC, datetime, timedelta

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import clv, data
from sportstradamus.analysis import (
    check_bet,
    compute_book_brier_skill_score,
    explode_offers,
    resolve_history,
)
from sportstradamus.helpers import Archive, get_logger
from sportstradamus.helpers.io import (
    LIVE_METRICS_PATH,
    _atomic_write_parquet,
    read_history,
    read_parlay_hist,
    write_history,
    write_parlay_hist,
)
from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

logger = get_logger("reflect")

LEAGUE_CLASSES = {
    "NBA": StatsNBA,
    "WNBA": StatsWNBA,
    "MLB": StatsMLB,
    "NFL": StatsNFL,
    "NHL": StatsNHL,
}

# Rolling windows for live-metrics aggregation (Stage 0 deliverable 0.2).
# 7-day catches recency drift; 30-day is the graduation gate window.
LIVE_METRICS_WINDOWS = (7, 30)
# Canonical column order for live_metrics_per_market.parquet. The schema is
# fixed: do not reorder, rename, or add columns without coordinating with the
# downstream check_graduation CLI and backfill script.
LIVE_METRICS_COLUMNS = (
    "league",
    "market",
    "computed_at",
    "window_days",
    "n_settled",
    "book_bss",
    "empirical_over_rate",
    "predicted_over_rate",
    "top_decile_mae",
    "profit_sim_yield",
)
# Minimum offers needed in a window before _top_decile_mae returns a value;
# below this, the top decile would be a single row and the metric is noise.
_TOP_DECILE_MIN_OFFERS = 10
# Clip bookmaker probabilities to avoid divide-by-zero in fair-odds payouts.
_BOOKS_P_CLIP = 1e-3


def _empty_live_metrics_frame() -> pd.DataFrame:
    """Empty parquet-bound frame with the locked dtype layout for downstream joins."""
    return pd.DataFrame(
        {
            "league": pd.Series(dtype="object"),
            "market": pd.Series(dtype="object"),
            "computed_at": pd.Series(dtype="datetime64[ns]"),
            "window_days": pd.Series(dtype="int16"),
            "n_settled": pd.Series(dtype="int64"),
            "book_bss": pd.Series(dtype="float64"),
            "empirical_over_rate": pd.Series(dtype="float64"),
            "predicted_over_rate": pd.Series(dtype="float64"),
            "top_decile_mae": pd.Series(dtype="float64"),
            "profit_sim_yield": pd.Series(dtype="float64"),
        }
    )


def _top_decile_mae(group: pd.DataFrame) -> float:
    """MAE of |Model EV - Actual| on the top-decile offers by Line.

    Buckets by Line (no Stats loading at reflect time per Stage 0 locked
    decision #17 — the heavy MeanYr lookup lives in compression_eval's
    --live-window mode). Returns NaN when the group is too small for a
    meaningful decile.
    """
    if len(group) < _TOP_DECILE_MIN_OFFERS:
        return float("nan")
    threshold = group["Line"].quantile(0.9)
    top = group[group["Line"] >= threshold]
    if top.empty:
        return float("nan")
    return float((top["Model EV"] - top["Actual"]).abs().mean())


def _profit_sim_yield(group: pd.DataFrame) -> float:
    """Flat $1-stake realized ROI at fair-odds payouts (locked decision #7).

    Payout multiplier on win = boost / books_p for Over bets (since the bet
    cashes when Result == Over and the book's implied prob of Over is books_p);
    boost / (1 - books_p) for Under bets. ``yield`` = (sum(payout * Hit) - n) / n.
    Returns NaN when the window has no settled offers — distinguishable from
    "broke even" by the n_settled column.
    """
    n = len(group)
    if n == 0:
        return float("nan")
    books_p = group["Books P"].clip(_BOOKS_P_CLIP, 1 - _BOOKS_P_CLIP).to_numpy()
    boost = group["Boost"].fillna(1.0).to_numpy()
    hit = group["Hit"].fillna(0).to_numpy()
    is_over = (group["Bet"] == "Over").to_numpy()
    payout = np.where(is_over, boost / books_p, boost / (1 - books_p))
    return float((np.sum(payout * hit) - n) / n)


def _build_cell_row(cell, group: pd.DataFrame, *, now: pd.Timestamp, window_days: int) -> dict:
    """Compute one (league, market, window) live-metrics row."""
    n = len(group)
    if n == 0:
        return {
            "league": cell[0],
            "market": cell[1],
            "computed_at": now,
            "window_days": window_days,
            "n_settled": 0,
            "book_bss": float("nan"),
            "empirical_over_rate": float("nan"),
            "predicted_over_rate": float("nan"),
            "top_decile_mae": float("nan"),
            "profit_sim_yield": float("nan"),
        }
    return {
        "league": cell[0],
        "market": cell[1],
        "computed_at": now,
        "window_days": window_days,
        "n_settled": n,
        "book_bss": float(compute_book_brier_skill_score(group)),
        "empirical_over_rate": float((group["Result"] == "Over").mean()),
        "predicted_over_rate": float((group["Bet"] == "Over").mean()),
        "top_decile_mae": _top_decile_mae(group),
        "profit_sim_yield": _profit_sim_yield(group),
    }


def _enforce_live_metrics_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Pin parquet-bound dtypes so consumers (check_graduation, backfill) join cleanly."""
    df = df[list(LIVE_METRICS_COLUMNS)].copy()
    df["league"] = df["league"].astype("object")
    df["market"] = df["market"].astype("object")
    df["computed_at"] = pd.to_datetime(df["computed_at"])
    df["window_days"] = df["window_days"].astype("int16")
    df["n_settled"] = df["n_settled"].astype("int64")
    for col in ("book_bss", "empirical_over_rate", "predicted_over_rate",
                "top_decile_mae", "profit_sim_yield"):
        df[col] = df[col].astype("float64")
    return df


def _compute_live_metrics(history: pd.DataFrame, *, now: datetime | None = None) -> pd.DataFrame:
    """Per-(league, market) Gate 2 metrics over 7- and 30-day rolling windows.

    Stage 0 deliverable 0.2. Writes are owned by the reflect pipeline; the
    backfill CLI reuses this helper. Returns a frame with
    ``LIVE_METRICS_COLUMNS`` exactly — two rows per cell (one per window in
    ``LIVE_METRICS_WINDOWS``). Cells with no offers in the 7-day window still
    emit a row with ``n_settled=0`` so downstream joins find every active cell.
    """
    if history.empty:
        return _empty_live_metrics_frame()
    exploded = explode_offers(history)
    if exploded.empty or "Actual" not in exploded.columns:
        return _empty_live_metrics_frame()
    settled = exploded[exploded["Actual"].notna()].copy()
    if settled.empty:
        return _empty_live_metrics_frame()
    settled["_date"] = pd.to_datetime(settled["Date"], errors="coerce")
    settled = settled[settled["_date"].notna()]
    if settled.empty:
        return _empty_live_metrics_frame()

    now_ts = pd.Timestamp(now or datetime.now(UTC))
    if now_ts.tz is not None:
        now_ts = now_ts.tz_localize(None)

    catalog_cut = now_ts - pd.Timedelta(days=max(LIVE_METRICS_WINDOWS))
    catalog = settled[settled["_date"] >= catalog_cut]
    if catalog.empty:
        return _empty_live_metrics_frame()
    cells = list(catalog[["League", "Market"]].drop_duplicates().itertuples(index=False, name=None))

    rows: list[dict] = []
    for window_days in LIVE_METRICS_WINDOWS:
        cutoff = now_ts - pd.Timedelta(days=window_days)
        window = settled[settled["_date"] >= cutoff]
        for cell in cells:
            grp = window[(window["League"] == cell[0]) & (window["Market"] == cell[1])]
            rows.append(_build_cell_row(cell, grp, now=now_ts, window_days=window_days))

    return _enforce_live_metrics_dtypes(pd.DataFrame(rows))


@click.command()
@click.option("--league", default=None, help="Resolve only this league (default: all).")
@click.option(
    "--skip-update",
    is_flag=True,
    default=False,
    help="Skip stats.update() API calls — use cached gamelogs only.",
)
@click.option(
    "--history-only", is_flag=True, default=False, help="Skip parlay resolution (much faster)."
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Verbosity for the structured JSONL log.",
)
def run(league, skip_update, history_only, log_level):
    """Nightly resolution: update stats, fill Actual/Legs/Misses, save."""
    logger.setLevel(log_level)
    logger.info(
        "reflect invoked",
        extra={"league": league, "skip_update": skip_update, "history_only": history_only},
    )
    # ------------------------------------------------------------------
    # 1. Load league Stats objects (optionally with update)
    # ------------------------------------------------------------------
    stats = {}
    leagues_to_load = [league] if league else list(LEAGUE_CLASSES.keys())

    for lg in tqdm(leagues_to_load, desc="Loading league stats"):
        cls = LEAGUE_CLASSES[lg]
        try:
            obj = cls()
            obj.load()
            if not skip_update:
                try:
                    obj.update()
                    logger.info(f"{lg}: stats updated")
                except Exception:
                    logger.warning(f"{lg}: update() failed, using cached gamelog")
            if hasattr(obj, "gamelog") and not obj.gamelog.empty:
                stats[lg] = obj
            else:
                logger.warning(f"{lg}: gamelog empty, skipping")
        except Exception:
            logger.warning(f"{lg}: load() failed, skipping")

    if not stats:
        logger.error("No league stats loaded. Aborting.")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 2. Resolve history (fill Actual column + CLV trio)
    # ------------------------------------------------------------------
    history = read_history()
    if history.empty:
        logger.error("history.parquet/.dat is empty or missing. Aborting.")
        raise SystemExit(1)

    if "Offers" not in history.columns:
        from sportstradamus.analysis import _migrate_flat_history

        logger.info("Migrating history to normalized schema")
        history = _migrate_flat_history(history)

    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    n_before_hist = int(history["Actual"].isna().sum())
    logger.info(f"Resolving {n_before_hist} pending history rows")
    history = resolve_history(history, stats)
    n_resolved_hist = n_before_hist - int(history["Actual"].isna().sum())
    logger.info(f"History: resolved {n_resolved_hist} / {n_before_hist} pending rows")

    # ------------------------------------------------------------------
    # 3. Fill in Close Books P / Market CLV / Model CLV per offer
    # ------------------------------------------------------------------
    archive = Archive()
    history = clv.fill_from_archive(history, archive)
    write_history(history)

    clv_summary = clv.summarize(history, archive=archive)
    clv.persist_segments(clv_summary.get("all_segments"))
    if clv_summary["n"]:
        logger.info(
            "CLV legs: %d  Market CLV mean: %+.3f  Model CLV mean: %+.3f  beat-close: %.1f%%",
            clv_summary["n"],
            clv_summary["market_clv_mean"],
            clv_summary["model_clv_mean"],
            100.0 * clv_summary["frac_beat_close"],
        )
        if not clv_summary["segments"].empty:
            logger.info(
                "CLV segments (n>=%d):\n%s",
                clv.CLV_SEGMENT_MIN_N,
                clv_summary["segments"].to_string(index=False),
            )
    else:
        logger.info("CLV: no resolved legs with closing-line data")

    # ------------------------------------------------------------------
    # 4. Resolve parlay_hist (fill Legs/Misses columns)
    # ------------------------------------------------------------------
    n_resolved_parl = 0
    if not history_only:
        parlays = read_parlay_hist()
        stat_map = json.loads((pkg_resources.files(data) / "stat_map.json").read_text())

        unresolved = parlays.loc[parlays["Legs"].isna()]
        n_before_parl = len(unresolved)
        logger.info(f"Resolving {n_before_parl} pending parlay rows")

        if n_before_parl > 0:
            tqdm.pandas(desc="Resolving parlays")
            results = unresolved.progress_apply(
                lambda bet: check_bet(bet, stats, stat_map), axis=1
            ).tolist()
            parlays.loc[parlays["Legs"].isna(), ["Legs", "Misses"]] = results
            write_parlay_hist(parlays)
            n_resolved_parl = sum(
                1 for legs, _ in results if not (isinstance(legs, float) and np.isnan(legs))
            )
            logger.info(f"Parlays: resolved {n_resolved_parl} / {n_before_parl} pending rows")

    # ------------------------------------------------------------------
    # 5. Write resolve_meta.json with last-run timestamp
    # ------------------------------------------------------------------
    meta = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "history_resolved": n_resolved_hist,
        "parlays_resolved": n_resolved_parl,
        "history_total": len(history),
        "history_pending": int(history["Actual"].isna().sum()),
    }
    meta_path = pkg_resources.files(data) / "resolve_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    click.echo(
        f"Done. History: {n_resolved_hist} resolved. "
        f"Parlays: {n_resolved_parl} resolved. "
        f"Last run: {meta['last_run']}"
    )

    # ------------------------------------------------------------------
    # 6. Compute and persist per-(league, market) live metrics (Gate 2)
    # ------------------------------------------------------------------
    metrics = _compute_live_metrics(history)
    _atomic_write_parquet(metrics, LIVE_METRICS_PATH)
    logger.info(f"Live metrics: wrote {len(metrics)} rows to {LIVE_METRICS_PATH.name}")
