"""Nightly resolution script for Sportstradamus.

Runs after games finish to:
1. Fetch latest stats from league APIs (update)
2. Fill in Actual column in history (parquet primary, .dat fallback)
3. Fill in Close Books P / Market CLV / Model CLV per offer
4. Fill in Legs Resolved/Misses columns in parlay_hist
5. Write resolve_meta.json with last-run timestamp
6. Compute and persist per-(league, market) live metrics (Gate 2)

Schedule with cron after games finish, e.g.:
    0 2 * * * cd /home/trevor/Sportstradamus && poetry run reflect
"""

import importlib.resources as pkg_resources
import json
from datetime import UTC, date, datetime, timedelta

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import clv, data
from sportstradamus.analysis import (
    _gameday_rows_for,
    _resolve_leg,
    annotate_offer_outcomes,
    calibration_summary,
    check_bet,
    compute_book_brier_skill_score,
    precompute_profit_sim_summary,
    resolve_history,
)
from sportstradamus.helpers import Archive, get_logger
from sportstradamus.helpers.io import (
    CALIBRATION_SUMMARY_PATH,
    LIVE_METRICS_PATH,
    PROFIT_SIM_SUMMARY_PATH,
    _atomic_write_parquet,
    read_history,
    read_parlay_hist,
    read_user_slips,
    write_history,
    write_parlay_hist,
    write_user_slips,
)
from sportstradamus.helpers.parlay_modifiers import fold_overlay
from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA
from sportstradamus.strategies import _ledger_bankroll, _ledger_settlement, _ledger_store

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
    "precision_over_live",
    "precision_under_live",
    "top_decile_mae",
    "profit_sim_yield",
    "profit_sim_kelly_yield",
)
# Minimum offers needed in a window before _top_decile_mae returns a value;
# below this, the top decile would be a single row and the metric is noise.
_TOP_DECILE_MIN_OFFERS = 10
# Minimum bets-on-a-side needed before precision_{over,under}_live is reported.
# Below 30, the conditional hit rate is dominated by sampling noise (~9pp se).
# This is the per-side analogue of MIN_SETTLED_FOR_GRADUATION (which is the
# whole-cell threshold used by graduation.py).
_MIN_BETS_FOR_PRECISION = 30
# Clip bookmaker probabilities to avoid divide-by-zero in fair-odds payouts.
_BOOKS_P_CLIP = 1e-3
# Phase 4 live ship-gate (set-baseline -> main): Kelly stake fraction cap, mirroring
# the offline S3 sim's 5% per-bet ceiling. The live ROI is the dollar-weighted
# realized return on the staked dollars within the rolling window.
_KELLY_LIVE_CAP_FRAC: float = 0.05


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
            "precision_over_live": pd.Series(dtype="float64"),
            "precision_under_live": pd.Series(dtype="float64"),
            "top_decile_mae": pd.Series(dtype="float64"),
            "profit_sim_yield": pd.Series(dtype="float64"),
            "profit_sim_kelly_yield": pd.Series(dtype="float64"),
        }
    )


def _side_precision(group: pd.DataFrame, side: str) -> float:
    """Hit rate among ``Bet == side`` rows (i.e. precision of side recommendations).

    ``side`` is ``"Over"`` or ``"Under"``. Returns NaN when fewer than
    :data:`_MIN_BETS_FOR_PRECISION` rows recommended that side — the conditional
    hit rate is too noisy at small n to gate a graduation decision.
    """
    side_mask = (group["Bet"] == side).to_numpy()
    n_side = int(side_mask.sum())
    if n_side < _MIN_BETS_FOR_PRECISION:
        return float("nan")
    side_hits = ((group["Bet"] == side) & (group["Result"] == side)).sum()
    return float(side_hits / n_side)


def _top_decile_mae(group: pd.DataFrame) -> float:
    """MAE of |Projection - Actual| on the top-decile offers by Line.

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
    return float((top["Projection"] - top["Actual"]).abs().mean())


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
    books_p = group["Market Prob"].clip(_BOOKS_P_CLIP, 1 - _BOOKS_P_CLIP).to_numpy()
    boost = group["Boost"].fillna(1.0).to_numpy()
    hit = group["Hit"].fillna(0).to_numpy()
    is_over = (group["Bet"] == "Over").to_numpy()
    payout = np.where(is_over, boost / books_p, boost / (1 - books_p))
    return float((np.sum(payout * hit) - n) / n)


def _profit_sim_kelly_yield(group: pd.DataFrame) -> float:
    """Kelly-sized realized ROI — Phase 4 set-baseline -> main live gate.

    Per offer: ``decimal_odds = boost / books_p_for_bet_side`` (same convention as
    :func:`_profit_sim_yield`); Kelly fraction = ``clip((p * d - 1) / (d - 1), 0,
    0.05)`` where ``p`` is the model's probability on the bet side. ROI is the
    dollar-weighted realized return — ``sum(stake * (b * hit - (1 - hit))) /
    sum(stake)`` — so cells with different bet counts are comparable. Returns
    NaN when no offer attracted a positive Kelly stake (no +EV bets in the
    window).
    """
    n = len(group)
    if n == 0:
        return float("nan")
    books_p = group["Market Prob"].clip(_BOOKS_P_CLIP, 1 - _BOOKS_P_CLIP).to_numpy()
    boost = group["Boost"].fillna(1.0).to_numpy()
    hit = group["Hit"].fillna(0).to_numpy()
    is_over = (group["Bet"] == "Over").to_numpy()
    model_p = group["Win Prob"].clip(_BOOKS_P_CLIP, 1.0 - _BOOKS_P_CLIP).to_numpy()
    decimal_odds = np.where(is_over, boost / books_p, boost / (1 - books_p))
    b = decimal_odds - 1.0
    # b <= 0 ⇒ even-money or worse — Kelly always returns 0; skip those bets.
    raw_kelly = np.where(b > 0, (model_p * decimal_odds - 1.0) / b, 0.0)
    stake_frac = np.clip(raw_kelly, 0.0, _KELLY_LIVE_CAP_FRAC)
    total_stake = float(stake_frac.sum())
    if total_stake <= 0:
        return float("nan")
    pnl = stake_frac * (b * hit - (1.0 - hit))
    return float(pnl.sum() / total_stake)


def _build_cell_row(
    cell: tuple[str, str], group: pd.DataFrame, *, now: pd.Timestamp, window_days: int
) -> dict:
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
            "precision_over_live": float("nan"),
            "precision_under_live": float("nan"),
            "top_decile_mae": float("nan"),
            "profit_sim_yield": float("nan"),
            "profit_sim_kelly_yield": float("nan"),
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
        "precision_over_live": _side_precision(group, "Over"),
        "precision_under_live": _side_precision(group, "Under"),
        "top_decile_mae": _top_decile_mae(group),
        "profit_sim_yield": _profit_sim_yield(group),
        "profit_sim_kelly_yield": _profit_sim_kelly_yield(group),
    }


def _enforce_live_metrics_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Pin parquet-bound dtypes so consumers (check_graduation, backfill) join cleanly."""
    df = df[list(LIVE_METRICS_COLUMNS)].copy()
    df["league"] = df["league"].astype("object")
    df["market"] = df["market"].astype("object")
    df["computed_at"] = pd.to_datetime(df["computed_at"])
    df["window_days"] = df["window_days"].astype("int16")
    df["n_settled"] = df["n_settled"].astype("int64")
    for col in (
        "book_bss",
        "empirical_over_rate",
        "predicted_over_rate",
        "precision_over_live",
        "precision_under_live",
        "top_decile_mae",
        "profit_sim_yield",
        "profit_sim_kelly_yield",
    ):
        df[col] = df[col].astype("float64")
    return df


def _settled_offers(history: pd.DataFrame):
    """Annotate history with per-offer outcomes and parse ``_date``; None if none settled."""
    if history.empty:
        return None
    exploded = annotate_offer_outcomes(history)
    if exploded.empty or "Actual" not in exploded.columns:
        return None
    settled = exploded[exploded["Actual"].notna()].copy()
    if settled.empty:
        return None
    settled["_date"] = pd.to_datetime(settled["Date"], errors="coerce")
    settled = settled[settled["_date"].notna()]
    if settled.empty:
        return None
    return settled


def _compute_live_metrics(history: pd.DataFrame, *, now: datetime | None = None) -> pd.DataFrame:
    """Per-(league, market) Gate 2 metrics over 7- and 30-day rolling windows.

    Stage 0 deliverable 0.2. Writes are owned by the reflect pipeline; the
    backfill CLI reuses this helper. Returns a frame with
    ``LIVE_METRICS_COLUMNS`` exactly — two rows per cell (one per window in
    ``LIVE_METRICS_WINDOWS``). Cells with no offers in the 7-day window still
    emit a row with ``n_settled=0`` so downstream joins find every active cell.
    """
    settled = _settled_offers(history)
    if settled is None:
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


def _precompute_profit_sim(history):
    summary = precompute_profit_sim_summary(annotate_offer_outcomes(history))
    _atomic_write_parquet(summary, PROFIT_SIM_SUMMARY_PATH)
    logger.info(
        f"Profit sim: wrote {len(summary)} strategy-horizon rows to {PROFIT_SIM_SUMMARY_PATH.name}"
    )


def _precompute_calibration(history):
    summary = calibration_summary(annotate_offer_outcomes(history))
    _atomic_write_parquet(summary, CALIBRATION_SUMMARY_PATH)
    logger.info(
        f"Calibration: wrote {len(summary)} bin-split rows to {CALIBRATION_SUMMARY_PATH.name}"
    )


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
    """Nightly resolution: update stats, fill Actual/Legs Resolved/Misses, save."""
    logger.setLevel(log_level)
    logger.info(
        "reflect invoked",
        extra={"league": league, "skip_update": skip_update, "history_only": history_only},
    )
    stats = _load_league_stats(league, skip_update)
    history, n_resolved_hist = _resolve_and_clv_history(stats)
    n_resolved_parl = _resolve_parlays(stats, history_only)
    n_resolved_slips = _resolve_user_slips(stats, history_only)
    n_resolved_ledger = _resolve_ledger(stats, history_only)
    _write_resolve_meta(
        history, n_resolved_hist, n_resolved_parl, n_resolved_slips, n_resolved_ledger
    )

    metrics = _compute_live_metrics(history)
    _atomic_write_parquet(metrics, LIVE_METRICS_PATH)
    logger.info(f"Live metrics: wrote {len(metrics)} rows to {LIVE_METRICS_PATH.name}")

    _precompute_profit_sim(history)
    _precompute_calibration(history)

    folded = fold_overlay()
    if folded["pairs"] or folded["rakes"]:
        logger.info(
            "Modifier overlay: folded %d pair modifiers + %d rakes into the committed configs",
            folded["pairs"],
            folded["rakes"],
        )


def _load_one_league(lg, cls, skip_update):
    """Load one league's Stats object (optionally update); None on failure / empty."""
    try:
        obj = cls()
        obj.load()
        if not skip_update:
            try:
                obj.update()
                logger.info(f"{lg}: stats updated")
            except Exception:
                logger.warning(f"{lg}: update() failed, using cached gamelog")
        if not obj.gamelog.empty:
            return obj
        logger.warning(f"{lg}: gamelog empty, skipping")
        return None
    except Exception:
        logger.warning(f"{lg}: load() failed, skipping")
        return None


def _load_league_stats(league, skip_update):
    stats = {}
    leagues_to_load = [league] if league else list(LEAGUE_CLASSES.keys())
    for lg in tqdm(leagues_to_load, desc="Loading league stats"):
        obj = _load_one_league(lg, LEAGUE_CLASSES[lg], skip_update)
        if obj is not None:
            stats[lg] = obj
    if not stats:
        logger.error("No league stats loaded. Aborting.")
        raise SystemExit(1)
    return stats


def _resolve_and_clv_history(stats):
    history = read_history()
    if history.empty:
        logger.error("history.parquet is empty or missing. Aborting.")
        raise SystemExit(1)

    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    n_before_hist = int(history["Actual"].isna().sum())
    logger.info(f"Resolving {n_before_hist} pending history rows")
    history = resolve_history(history, stats)
    n_resolved_hist = n_before_hist - int(history["Actual"].isna().sum())
    logger.info(f"History: resolved {n_resolved_hist} / {n_before_hist} pending rows")

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
    return history, n_resolved_hist


def _resolve_parlays(stats, history_only):
    n_resolved_parl = 0
    if not history_only:
        parlays = read_parlay_hist()
        stat_map = json.loads((pkg_resources.files(data) / "config" / "stat_map.json").read_text())

        unresolved = parlays.loc[parlays["Legs Resolved"].isna()]
        n_before_parl = len(unresolved)
        logger.info(f"Resolving {n_before_parl} pending parlay rows")

        if n_before_parl > 0:
            tqdm.pandas(desc="Resolving parlays")
            results = unresolved.progress_apply(
                lambda bet: check_bet(bet, stats, stat_map), axis=1
            ).tolist()
            parlays.loc[parlays["Legs Resolved"].isna(), ["Legs Resolved", "Misses"]] = results
            write_parlay_hist(parlays)
            n_resolved_parl = sum(
                1 for legs, _ in results if not (isinstance(legs, float) and np.isnan(legs))
            )
            logger.info(f"Parlays: resolved {n_resolved_parl} / {n_before_parl} pending rows")
    return n_resolved_parl


# Flex / Sleeper slips still cash with up to this many missed legs (partial tier).
_FLEX_MAX_MISS = 2


def _resolve_user_slips(stats, history_only):
    """Grade pending dashboard-built slips against final game logs.

    Mirrors :func:`_resolve_parlays` but resolves each leg against its own
    league/game/date (a slip may span games), so it reuses the per-leg
    ``analysis._resolve_leg`` rather than the single-game ``check_bet``.
    """
    if history_only:
        return 0
    slips = read_user_slips()
    if slips.empty or "status" not in slips.columns:
        return 0
    pending = slips.loc[slips["status"] == "pending"]
    if pending.empty:
        return 0
    graded = 0
    for idx, row in pending.iterrows():
        outcome = _grade_slip(row, stats)
        if outcome is None:
            continue
        for col, value in outcome.items():
            slips.loc[idx, col] = value
        graded += 1
    if graded:
        write_user_slips(slips)
    logger.info(f"User slips: graded {graded} / {len(pending)} pending")
    return graded


def _grade_slip(row, stats):
    """Resolve one slip; return the writeback dict, or ``None`` while a game is unplayed."""
    per_leg = []
    for leg in row["legs"]:
        league = leg["league"]
        if league not in stats:
            return None
        stat_obj = stats[league]
        ls = stat_obj.log_strings
        game_rows = _gameday_rows_for(stat_obj.gamelog, ls, leg["date"], leg["game"])
        if game_rows.empty:
            return None
        per_leg.append(_resolve_leg(game_rows, ls, leg))
    hits = sum(1 for o in per_leg if o == 0)
    effective = sum(1 for o in per_leg if o is not None)
    status = _slip_status(row["play_type"], hits, effective)
    realized = {"won": float(row["payout_multiplier"]), "push": 1.0, "lost": 0.0}.get(
        status, np.nan
    )
    return {
        "status": status,
        "legs_hit": hits,
        "result": json.dumps(per_leg),
        "payout_realized": realized,
        "graded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _slip_status(play_type, hits, effective):
    """Slip outcome from hits: Power all-or-nothing, Flex/Sleeper partial-cash."""
    if effective == 0:
        return "push"
    if hits == effective:
        return "won"
    if play_type in ("Flex", "Sleeper") and hits >= 2 and effective - hits <= _FLEX_MAX_MISS:
        return "partial"
    return "lost"


_LEDGER_ENTRIES_DIR = _ledger_store.ENTRIES_DIR


def _ledger_entry_dates() -> list[date]:
    """Slate dates with an entries JSONL on disk, oldest first; ``[]`` before the ledger's first run."""
    if not _LEDGER_ENTRIES_DIR.exists():
        return []
    return sorted(date.fromisoformat(p.stem) for p in _LEDGER_ENTRIES_DIR.glob("*.jsonl"))


def _resolve_ledger(stats, history_only):
    """Settle the simulated-bettor ledger's committed entries against final game logs.

    Mirrors :func:`_resolve_user_slips`'s ``history_only`` gate; all
    resolution/payout logic is delegated to ``_ledger_settlement`` /
    ``_ledger_bankroll`` — this function is pure orchestration.
    """
    if history_only:
        return 0
    archive = Archive()
    dates = _ledger_entry_dates()
    total = 0
    for day in dates:
        rows = _ledger_settlement.settle_day(day, stats, archive)
        if not rows:
            continue
        _ledger_bankroll.write_settled_entries(rows)
        _ledger_bankroll.update_bankroll(day, rows)
        total += len(rows)
    logger.info(f"Ledger: settled {total} entries across {len(dates)} slate dates")
    return total


def _write_resolve_meta(
    history, n_resolved_hist, n_resolved_parl, n_resolved_slips, n_resolved_ledger
):
    meta = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "history_resolved": n_resolved_hist,
        "parlays_resolved": n_resolved_parl,
        "slips_resolved": n_resolved_slips,
        "ledger_resolved": n_resolved_ledger,
        "history_total": len(history),
        "history_pending": int(history["Actual"].isna().sum()),
    }
    meta_path = pkg_resources.files(data) / "runtime" / "resolve_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    click.echo(
        f"Done. History: {n_resolved_hist} resolved. "
        f"Parlays: {n_resolved_parl} resolved. "
        f"Slips: {n_resolved_slips} graded. "
        f"Ledger: {n_resolved_ledger} settled. "
        f"Last run: {meta['last_run']}"
    )
