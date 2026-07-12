"""Settlement layer for the simulated-bettor ledger (Policy v1).

Resolves one slate date's distinct legs exactly once and broadcasts the
outcome across every committed entry that cites it, joins closing-line value
(:mod:`sportstradamus.clv`) over the same distinct-leg set, and computes each
entry's settled P&L in ``Decimal``. Never mutates the entries JSONL files --
this module only reads them via :mod:`sportstradamus.strategies._ledger_store`.
Persistence to parquet is a sibling module's job
(:mod:`sportstradamus.strategies._ledger_bankroll`), not this one's; this
module does not touch :mod:`sportstradamus.strategies.profit_sim`.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from sportstradamus import clv
from sportstradamus.analysis import _gameday_rows_for, _resolve_leg
from sportstradamus.helpers.io import read_history
from sportstradamus.prediction.payouts import payout_curve_for
from sportstradamus.strategies import _ledger_bankroll, _ledger_store

# CLV frame columns fill_from_archive needs but LEG_FIELDS doesn't carry;
# left-joined from history.parquet keyed on PREDICTION_KEY. A join miss
# leaves these NaN, which fill_from_archive already treats as "no trained
# model -- fall back to the composite-probability path", not an error.
_CLV_JOIN_COLS = ["Player", "League", "Date", "Market", "Dist", "CV", "Gate", "Step"]


def distinct_leg_key(leg: dict) -> tuple[str, str, float, str, str, str]:
    """Identity of one proposition, shared by the resolve step and the CLV join.

    Excludes ``game`` (derivable from the other fields, not independent) and
    ``platform`` (irrelevant to leg identity) -- two legs with the same
    (player, stat, line, bet, league, date) are the same bet regardless of
    which platform or citing entry recorded them.
    """
    return (leg["player"], leg["stat"], leg["line"], leg["bet"], leg["league"], leg["date"])


def _game_rows_cache_key(leg: dict) -> tuple[str, str, str]:
    return (leg["league"], leg["game"], leg["date"])


def _build_game_rows_cache(entries: list[dict], stats: dict) -> dict[tuple, pd.DataFrame]:
    """One :func:`~sportstradamus.analysis._gameday_rows_for` lookup per distinct
    (league, game, date).

    Shared by :func:`settleable_entries` and :func:`resolve_distinct_legs` so
    a slate's gamelog is sliced once total, not once per caller.
    """
    cache: dict[tuple, pd.DataFrame] = {}
    for entry in entries:
        for leg in entry["canonical_legs"]:
            key = _game_rows_cache_key(leg)
            if key in cache:
                continue
            league, game, date = key
            stat_obj = stats.get(league)
            if stat_obj is None:
                cache[key] = pd.DataFrame()
                continue
            cache[key] = _gameday_rows_for(stat_obj.gamelog, stat_obj.log_strings, date, game)
    return cache


def settleable_entries(
    date: datetime.date, stats: dict, game_rows_cache: dict[tuple, pd.DataFrame] | None = None
) -> list[dict]:
    """Records for ``date`` whose every distinct leg's game has landed.

    All-legs-complete, not first-leg -- a cross-game entry can span games and
    leagues that finish at different times, so a partially-finished slate
    must not be settled early. "Landed" means
    :func:`~sportstradamus.analysis._gameday_rows_for` returns a non-empty
    frame for that leg's (league, game, date). Already-settled entries (per
    ``_ledger_bankroll.already_settled_ids``) are excluded up front.
    """
    settled_ids = _ledger_bankroll.already_settled_ids(date)
    records = [rec for rec in _ledger_store.read_records(date) if rec["id"] not in settled_ids]
    if not records:
        return []
    cache = (
        game_rows_cache if game_rows_cache is not None else _build_game_rows_cache(records, stats)
    )
    return [
        rec
        for rec in records
        if all(not cache[_game_rows_cache_key(leg)].empty for leg in rec["canonical_legs"])
    ]


def resolve_distinct_legs(
    entries: list[dict], stats: dict, game_rows_cache: dict[tuple, pd.DataFrame] | None = None
) -> dict[tuple, int | None]:
    """Resolve every distinct leg cited by ``entries`` exactly once.

    Cost is O(distinct legs), not O(entries): with ~1,200 entries/day sharing
    legs across the 40-replicate ensemble and 3 personas by construction, the
    distinct-leg set is far smaller than the entry count, and this is the
    mechanism that keeps settlement cost flat as replicate/persona counts grow.
    Returns ``{distinct_leg_key(leg): outcome}`` where outcome is 0 (hit), 1
    (miss), or None (push) per ``analysis._resolve_leg``.
    """
    cache = (
        game_rows_cache if game_rows_cache is not None else _build_game_rows_cache(entries, stats)
    )
    distinct_legs: dict[tuple, dict] = {
        distinct_leg_key(leg): leg for entry in entries for leg in entry["canonical_legs"]
    }
    outcomes: dict[tuple, int | None] = {}
    for key, leg in distinct_legs.items():
        stat_obj = stats[leg["league"]]
        game_rows = cache[_game_rows_cache_key(leg)]
        outcomes[key] = _resolve_leg(game_rows, stat_obj.log_strings, leg)
    return outcomes


def _clv_input_frame(distinct_legs: dict[tuple, dict]) -> pd.DataFrame:
    rows = [
        {
            "Player": leg["player"],
            "League": leg["league"],
            "Date": leg["date"],
            "Market": leg["stat"],
            "Line": leg["line"],
            "Bet": leg["bet"],
            "Win Prob": leg["win_prob"],
            "Market Prob": np.nan,
            "Close Market Prob": np.nan,
            "Market CLV": np.nan,
            "Model CLV": np.nan,
        }
        for leg in distinct_legs.values()
    ]
    return pd.DataFrame(rows)


def join_clv(distinct_legs: dict[tuple, dict], archive) -> dict[tuple, dict]:
    """CLV for every distinct leg, computed once over the whole slate.

    Builds one small frame over the distinct-leg set, left-joins
    ``Dist``/``CV``/``Gate``/``Step`` from ``history.parquet`` keyed on
    :data:`~sportstradamus.history_schema.PREDICTION_KEY` (every ledger leg
    traces back to the same day's ``process_offers`` scoring pass that
    populates that parquet under the same key), and calls
    ``clv.fill_from_archive`` exactly once -- the spec requirement of
    resolving the union of distinct legs once per day, not once per entry.
    A join miss leaves those four columns NaN, which ``fill_from_archive``
    already handles via its composite-probability fallback path.

    Returns ``{distinct_leg_key(leg): {"close_market_prob", "market_clv",
    "model_clv"}}``.
    """
    if not distinct_legs:
        return {}
    frame = _clv_input_frame(distinct_legs)
    history = read_history()
    if not history.empty:
        join_cols = [c for c in _CLV_JOIN_COLS if c in history.columns]
        lookup = history[join_cols].drop_duplicates(subset=["Player", "League", "Date", "Market"])
        frame = frame.merge(lookup, on=["Player", "League", "Date", "Market"], how="left")
    else:
        for col in ("Dist", "CV", "Gate", "Step"):
            frame[col] = np.nan

    resolved = clv.fill_from_archive(frame, archive)
    out: dict[tuple, dict] = {}
    for key, leg, (_, row) in zip(
        distinct_legs.keys(), distinct_legs.values(), resolved.iterrows(), strict=True
    ):
        out[key] = {
            "close_market_prob": row["Close Market Prob"],
            "market_clv": row["Market CLV"],
            "model_clv": row["Model CLV"],
        }
    return out


def realized_multiplier(
    contest_variant: str, entry_size: int, effective_size: int, misses: int
) -> float:
    """Deterministic realized payout multiplier -- no unknown outcome, no RNG.

    The realized-outcome counterpart of
    ``payouts.expected_payout_with_pushes``'s Monte-Carlo expectation: here
    the outcome (misses, effective_size) is already known, so this is a
    single curve lookup rather than an average over samples. Mirrors that
    function's push edge cases: an entry reduced below the 2-leg minimum by
    pushes refunds (x1) if nothing was lost, else busts (x0); a lookup past
    the curve's recorded miss count also busts.
    """
    if effective_size < 2:
        return 0.0 if misses > 0 else 1.0
    _, curve = payout_curve_for("Underdog", contest_variant, legacy=False)
    row = curve.get(effective_size)
    if row is None or misses >= len(row):
        return 0.0
    return row[misses]


def _entry_clv_stats(record: dict, clv_by_leg: dict[tuple, dict]) -> dict:
    model_clvs = [
        clv_by_leg[distinct_leg_key(leg)]["model_clv"]
        for leg in record["canonical_legs"]
        if distinct_leg_key(leg) in clv_by_leg
    ]
    market_clvs = [
        clv_by_leg[distinct_leg_key(leg)]["market_clv"]
        for leg in record["canonical_legs"]
        if distinct_leg_key(leg) in clv_by_leg
    ]
    model_valid = [v for v in model_clvs if pd.notna(v)]
    market_valid = [v for v in market_clvs if pd.notna(v)]
    return {
        "clv_leg_count": len(model_valid),
        "model_clv_mean": float(np.mean(model_valid)) if model_valid else float("nan"),
        "market_clv_mean": float(np.mean(market_valid)) if market_valid else float("nan"),
    }


def settle_entry(
    record: dict, leg_outcomes: dict[tuple, int | None], clv_by_leg: dict[tuple, dict]
) -> dict:
    """One entry's settled P&L in ``Decimal``.

    Every leg here has already passed :func:`settleable_entries`'s
    completeness gate (its game landed), so an outcome of ``None`` can only
    mean a genuine push -- never "ungraded". A future reorder that calls this
    on incomplete entries would silently misclassify ungraded legs as pushes.
    """
    misses = 0
    pushes = 0
    for leg in record["canonical_legs"]:
        outcome = leg_outcomes[distinct_leg_key(leg)]
        if outcome is None:
            pushes += 1
        elif outcome == 1:
            misses += 1
    effective_size = record["entry_size"] - pushes
    mult = realized_multiplier(
        record["contest_variant"], record["entry_size"], effective_size, misses
    )
    stake = Decimal(record["stake"])
    payout = Decimal(str(mult)) * stake
    pnl = payout - stake
    return {
        "id": record["id"],
        "date": record["date"],
        "persona": record["persona"],
        "run_slot": record["run_slot"],
        "replicate_id": record["replicate_id"],
        "contest_variant": record["contest_variant"],
        "entry_size": record["entry_size"],
        "effective_size": effective_size,
        "misses": misses,
        "pushes": pushes,
        "realized_multiplier": mult,
        "stake": stake,
        "payout": payout,
        "pnl": pnl,
        "game_span": record["game_span"],
        "committed_at": record["committed_at"],
        "policy_version": record["policy_version"],
        "git_sha": record["git_sha"],
        **_entry_clv_stats(record, clv_by_leg),
        "settled_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def settle_day(date: datetime.date, stats: dict, archive) -> list[dict]:
    """Settle every settleable entry for ``date``. The one public entry point.

    Composes :func:`settleable_entries`, :func:`resolve_distinct_legs`,
    :func:`join_clv`, and :func:`settle_entry`. Returns ``[]`` when nothing on
    the slate is settleable yet (empty entries, or games still in progress).
    """
    cache = _build_game_rows_cache(_ledger_store.read_records(date), stats)
    entries = settleable_entries(date, stats, game_rows_cache=cache)
    if not entries:
        return []
    leg_outcomes = resolve_distinct_legs(entries, stats, game_rows_cache=cache)
    distinct_legs = {
        distinct_leg_key(leg): leg for entry in entries for leg in entry["canonical_legs"]
    }
    clv_by_leg = join_clv(distinct_legs, archive)
    return [settle_entry(entry, leg_outcomes, clv_by_leg) for entry in entries]
