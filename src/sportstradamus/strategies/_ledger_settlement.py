"""Settlement layer for the simulated-bettor ledger.

Resolves one slate date's distinct legs exactly once and broadcasts the
outcome across every committed entry that cites it, joins closing-line value
(:mod:`sportstradamus.clv`) over the same distinct-leg set, and computes each
entry's settled P&L in ``Decimal`` at what the platform pays that outcome
(:func:`sportstradamus.prediction.payouts.outcome_payouts`, the rule the
pricers use). Never mutates the entries JSONL files --
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
from sportstradamus.history_schema import PREDICTION_KEY
from sportstradamus.prediction.payouts import (
    LEG_LOSS,
    LEG_PUSH,
    LEG_WIN,
    SLEEPER_FULL_REFUND_MAX_SIZE,
    outcome_payouts,
    payout_curve_for,
)
from sportstradamus.strategies import _ledger_bankroll, _ledger_store

# CLV frame columns fill_from_archive needs but LEG_FIELDS doesn't carry; join_clv
# left-joins them from history.parquet. Commence is the kickoff clv.commence_times
# resolves over all of history, so a leg's close is read at the instant its history
# rows' is; Team is the column that function borrows a kickoff by.
_CLV_JOIN_COLS = [*PREDICTION_KEY, "Dist", "CV", "Gate", "Step", "Team", "Commence"]

# analysis._resolve_leg's verdict (0 hit, 1 miss, None push) as a payout-rule leg outcome.
_LEG_OUTCOME = {0: LEG_WIN, 1: LEG_LOSS, None: LEG_PUSH}

# Records of this version were committed before an entry carried its pair modifier and
# its cross-game legs carried raw multipliers. They settle on the bare table tier, as
# they always have, so one version is scored by one rule for the life of the ledger.
_BARE_TABLE_POLICY = "policy_v1"

# Records of these versions match their settled row on the id alone, as they always have.
# Production's policy_v1 files of August 2026 give every entry of one size the same id;
# matched by copy, the records that id has held back since then would settle now.
_ID_KEYED_POLICIES = frozenset({"policy_v1", "policy_v2", "policy_v3"})


def _settlement_key(entry: dict) -> tuple[str | int, ...]:
    """What matches a committed record to its settled row within one slate date."""
    if entry["policy_version"] in _ID_KEYED_POLICIES:
        return (entry["id"],)
    return _ledger_store.entry_key(entry)


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
    frame for that leg's (league, game, date). Records the settled table
    already holds a row for (per :func:`_settlement_key`) are excluded up front.
    """
    settled = {_settlement_key(row) for row in _ledger_bankroll.read_settled_entries(date)}
    records = [
        rec for rec in _ledger_store.read_records(date) if _settlement_key(rec) not in settled
    ]
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
    legs across the 40-replicate ensemble and its personas by construction, the
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
    ``Dist``/``CV``/``Gate``/``Step`` and the leg's kickoff from
    ``history.parquet`` keyed on
    :data:`~sportstradamus.history_schema.PREDICTION_KEY` (every ledger leg
    traces back to the same day's ``process_offers`` scoring pass that
    populates that parquet under the same key), and calls
    ``clv.fill_from_archive`` exactly once -- the spec requirement of
    resolving the union of distinct legs once per day, not once per entry.
    A join miss leaves the four model columns NaN, which ``fill_from_archive``
    already handles via its composite-probability fallback path, and reads
    the leg's close at the stand-in hour.

    Returns ``{distinct_leg_key(leg): {"close_market_prob", "market_clv",
    "model_clv"}}``.
    """
    if not distinct_legs:
        return {}
    frame = _clv_input_frame(distinct_legs)
    history = read_history()
    if not history.empty:
        history = history.assign(Commence=clv.commence_times(history))
        join_cols = [c for c in _CLV_JOIN_COLS if c in history.columns]
        lookup = history[join_cols].drop_duplicates(subset=PREDICTION_KEY)
        frame = frame.merge(lookup, on=PREDICTION_KEY, how="left")
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
    """One entry's settled P&L in ``Decimal``, at what the platform pays its outcome.

    The entry pays its table tier for the real outcome, times the multipliers
    the platform quotes that tier on (each leg's raw ``boost`` in
    ``canonical_legs``), times its ``pair_modifier``. A ``policy_v1`` record
    pays the bare table tier.

    Every leg here has already passed :func:`settleable_entries`'s
    completeness gate (its game landed), so an outcome of ``None`` can only
    mean a genuine push -- never "ungraded". A future reorder that calls this
    on incomplete entries would silently misclassify ungraded legs as pushes.
    """
    legs = record["canonical_legs"]
    outcomes = np.array([[_LEG_OUTCOME[leg_outcomes[distinct_leg_key(leg)]] for leg in legs]])
    misses = int((outcomes == LEG_LOSS).sum())
    pushes = int((outcomes == LEG_PUSH).sum())
    effective_size = record["entry_size"] - pushes
    # .get() with a default, not record["platform"]: pre-existing immutable
    # records committed before the platform field existed have no such key,
    # and the append-only ledger must settle those old records forever.
    platform = record.get("platform", "Underdog")
    _, payout_curve = payout_curve_for(platform, record["contest_variant"])
    bare_table = record["policy_version"] == _BARE_TABLE_POLICY
    mult = float(
        outcome_payouts(
            outcomes,
            1.0 if bare_table else np.array([leg["boost"] for leg in legs]),
            payout_curve,
            full_refund_below_size=SLEEPER_FULL_REFUND_MAX_SIZE if platform == "Sleeper" else None,
            pair_modifier=1.0 if bare_table else record["pair_modifier"],
        )[0]
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
