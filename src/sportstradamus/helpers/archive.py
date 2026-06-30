"""DuckDB-backed odds/EV archive.

The :class:`Archive` singleton is the only authorized entrypoint for reading
and writing the on-disk odds archive at ``archive/archive.duckdb``.
Consumers share the single instance so that in-memory transactions stay
consistent across the scrape/predict pipeline.

Schema (created on first connect):

* ``odds(league, market, game_date, entity, book, ev, observed_at)`` — one
  row per (slate-entry, book, observation). ``entity`` is a player name for
  player props or a team name for moneyline / totals / spreads / team
  markets. ``observed_at`` is set at write time so successive polls accrue
  a time-series rather than overwriting per-book EVs.
* ``lines(league, market, game_date, entity, line, observed_at)`` — every
  observed line value with its observation timestamp. Skipped for
  moneyline / totals / spreads / team-only markets, which are pure EV.

:func:`clean_archive` drops dates older than ``cutoff_date`` and prunes
combo / matchup pseudo-entities (``" + "``, ``" vs. "``).
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import time
import warnings
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sportstradamus.helpers.config import (
    book_gate,
    book_skewnormal_shape,
    book_weights,
    stat_cv,
    stat_dist,
    stat_meta,
)
from sportstradamus.helpers.distributions import get_ev, get_odds, no_vig_odds
from sportstradamus.helpers.text import remove_accents


@dataclasses.dataclass
class ClosingLine:
    """Consensus line and implied probability from the latest pre-kickoff odds snapshot.

    Attributes:
        line: Consensus line (median of all book lines).
        devig_over: Implied probability of outcome exceeding the line (no-vig).
        sample_ts: Timestamp of the latest odds snapshot (None if unavailable).
        book_set: Set of books that quoted this entry.
    """

    line: float
    devig_over: float
    sample_ts: datetime.datetime | None
    book_set: frozenset[str]


# Markets whose value schema is per-book EV only (no Lines table rows).
_TEAM_ONLY_MARKETS = frozenset({"Moneyline", "Totals", "1st 1 innings"})

_DEFAULT_DB_PATH = Path("archive/archive.duckdb")

# Hours before commence_time treated as "the books' line" during training.
# Aligned with the typical Vegas closing-window inflection (~8h pre-game).
TRAINING_LOOKBACK_HOURS: int = 8
TRAINING_LOOKBACK = timedelta(hours=TRAINING_LOOKBACK_HOURS)

# Sharp books that anchor the movement-direction diagnostic in CLV.
SHARP_BOOKS: tuple[str, ...] = ("pinnacle", "circa", "bookmaker")

# Total seconds to wait for a conflicting writer (e.g. a confer pass that's
# still flushing) to release the DuckDB file lock before giving up. DuckDB
# holds the lock for the lifetime of the connection, so this is sized to
# absorb a typical concurrent job wrapping up while the next one starts.
# Override via ``SPORTSTRADAMUS_ARCHIVE_LOCK_WAIT_SECONDS`` for tuning;
# set to 0 to disable retries (the legacy behaviour).
_LOCK_WAIT_SECONDS_DEFAULT: float = 120.0
_LOCK_BACKOFF_INITIAL: float = 2.0
_LOCK_BACKOFF_MAX: float = 16.0

# No PRIMARY KEY: DuckDB's PK creates an ART index that bloats the DB ~10x for
# this row count. Lookups don't need the index — zone-map pruning on naturally
# sorted data scans 15M rows in ~1ms. ``observed_at`` is left nullable in DDL
# so the auto-migration path below can backfill in place against pre-rework
# DBs without rebuilding the table; the standalone migration script tightens
# it to NOT NULL when run.
#
# ``under_prob`` / ``line`` are the shape-free book quote (the de-vigged
# under-probability and the line it was quoted at), stored so the distribution
# shape need not be baked into ``ev``. They are nullable: pre-migration rows and
# team-market writes leave them NULL, and readers fall back to ``ev``. Forward
# player-prop writes populate them; the migration backfills history.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS odds (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    book         TEXT NOT NULL,
    ev           DOUBLE,
    observed_at  TIMESTAMP,
    under_prob   DOUBLE,
    line         DOUBLE
);
CREATE TABLE IF NOT EXISTS lines (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    line         DOUBLE NOT NULL,
    observed_at  TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ladder (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    book         TEXT NOT NULL,
    line         DOUBLE NOT NULL,
    p_over       DOUBLE NOT NULL,
    observed_at  TIMESTAMP
);
"""


def _safe_date(d: str | datetime.date | None) -> datetime.date | None:
    """Return a :class:`date` for ISO-format strings, ``None`` for junk input."""
    if d is None or d == "":
        return None
    if isinstance(d, datetime.date):
        return d
    try:
        return datetime.datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _dfs_under_boost(over, boost_under):
    return boost_under if boost_under and boost_under > 0 else over


def _dedup_offers_by_boost(offers) -> list[dict]:
    """Normalize ``offers`` to records, one per ``(Player, Market)``.

    Accepts a single dict or a list. Duplicates are resolved in favor of the
    offer whose ``Boost_Over`` is closest to a neutral 1.0 (filling ``Boost_Over``
    from ``Boost`` when absent). Empty list when no offers survive.
    """
    if not isinstance(offers, list):
        offers = [offers]
    df = pd.DataFrame(offers)
    if df.empty:
        return []
    if "Boost_Over" not in df.columns:
        df["Boost_Over"] = np.nan
    if "Boost" in df.columns:
        df.loc[df["Boost_Over"].isna(), "Boost_Over"] = df.loc[df["Boost_Over"].isna(), "Boost"]
    df["Boost Factor"] = np.abs(df["Boost_Over"] - 1)
    df = df.loc[~df.sort_values("Boost Factor").duplicated(["Player", "Market"])]
    return df.to_dict(orient="records")


def _resolve_market(league: str, raw_market: str, key: dict) -> str:
    """Rename a sportsbook-native market string to its canonical per-league name."""
    market = raw_market.replace("H2H ", "")
    market = key.get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
    if league in ("NBA", "WNBA"):
        market = market.replace("underdog", "prizepicks")
    return market


def _devig_over(line, evs, dist, cv) -> float:
    """Mean no-vig implied over-probability across the books' EVs.

    Each book EV inverts through ``get_odds`` to an under-probability; the
    over side is ``1 - under``. None / NaN EVs and prices that fail to invert
    are skipped; ``NaN`` when no book yields a usable probability.
    """
    over_probs = []
    for ev in evs:
        if ev is None or np.isnan(ev):
            continue
        try:
            under_prob = get_odds(line, ev, dist, cv=cv)
            over_probs.append(1.0 - under_prob)
        except (ValueError, RuntimeError):
            continue
    return np.nan if not over_probs else float(np.mean(over_probs))


def clean_archive(cutoff_date: datetime.date | None = None) -> None:
    """Drop stale dates and combo/matchup pseudo-entities from the archive.

    Operates on the singleton :class:`Archive`'s connection. ``cutoff_date``
    defaults to four years before today (the original klepto window).
    """
    if cutoff_date is None:
        cutoff_date = (datetime.datetime.today() - datetime.timedelta(days=365 * 4)).date()
    a = Archive()
    con = a._connection
    con.execute("DELETE FROM odds WHERE game_date < ?", [cutoff_date])
    con.execute("DELETE FROM lines WHERE game_date < ?", [cutoff_date])
    con.execute("DELETE FROM odds WHERE entity LIKE '% + %' OR entity LIKE '% vs. %'")
    con.execute("DELETE FROM lines WHERE entity LIKE '% + %' OR entity LIKE '% vs. %'")
    con.commit()


class Archive:
    """Singleton wrapper around the DuckDB-backed odds archive.

    On first instantiation the connection is opened against
    ``archive/archive.duckdb`` (created if missing) and the schema is
    applied. All public read methods are point-lookups by
    ``(league, market, game_date, entity)``; write methods accumulate into
    a transaction that is committed by :meth:`write`.
    """

    _instance: Archive | None = None

    @staticmethod
    def _connect_once(db_path: Path) -> duckdb.DuckDBPyConnection:
        # DuckDB <=1.1.x can leave a .wal that replays a bare CREATE TABLE
        # against an already-checkpointed catalog after a hard kill — the
        # connection then refuses to open at all. Quarantine the stale WAL
        # so the next run heals itself; the schema DDL below is idempotent.
        try:
            return duckdb.connect(str(db_path))
        except duckdb.CatalogException as exc:
            if "Failure while replaying WAL" not in str(exc):
                raise
            wal = Path(str(db_path) + ".wal")
            if not wal.exists():
                raise
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            quarantined = wal.with_name(wal.name + f".corrupt-{ts}")
            wal.rename(quarantined)
            warnings.warn(
                f"Discarded stale DuckDB WAL: moved {wal} -> {quarantined}. "
                "Any uncheckpointed odds writes from the previous run are lost; "
                "re-run the affected pipeline to repopulate.",
                stacklevel=2,
            )
            return duckdb.connect(str(db_path))

    @staticmethod
    def _connect_with_wal_recovery(db_path: Path) -> duckdb.DuckDBPyConnection:
        # DuckDB's file lock is held for the entire lifetime of a peer
        # connection (not just during writes), so two production jobs that
        # overlap by a few seconds will collide. Retry with bounded
        # exponential backoff before giving up so a confer pass wrapping up
        # while prophecize starts no longer crashes the cron entry.
        wait_budget = float(
            os.environ.get("SPORTSTRADAMUS_ARCHIVE_LOCK_WAIT_SECONDS", _LOCK_WAIT_SECONDS_DEFAULT)
        )
        deadline = time.monotonic() + wait_budget
        backoff = _LOCK_BACKOFF_INITIAL
        while True:
            try:
                return Archive._connect_once(db_path)
            except duckdb.IOException as exc:
                if "Could not set lock on file" not in str(exc):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0 or wait_budget <= 0:
                    raise
                sleep_for = min(backoff, remaining)
                warnings.warn(
                    f"DuckDB archive {db_path} is locked by another process; "
                    f"retrying in {sleep_for:.1f}s ({remaining:.0f}s remaining "
                    f"of {wait_budget:.0f}s budget): {exc}",
                    stacklevel=2,
                )
                time.sleep(sleep_for)
                backoff = min(backoff * 2, _LOCK_BACKOFF_MAX)

    def __new__(cls):
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Open (or reuse) the DuckDB connection and reset write buffers.

        Idempotent: the singleton is initialized once per process, so repeat
        constructions short-circuit on the ``_initialized`` guard.
        """
        if self._initialized:
            return
        self._initialized = True

        db_path = Path(os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", _DEFAULT_DB_PATH))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._connection = self._connect_with_wal_recovery(db_path)
        self._connection.execute(_SCHEMA_DDL)
        self._auto_migrate_observed_at()
        self._auto_migrate_shapefree_columns()

        self.default_totals = {
            "MLB": 4.671,
            "NBA": 111.667,
            "WNBA": 81.667,
            "NFL": 22.668,
            "NHL": 2.674,
        }

        # Pending-write buffers. Append-only since observed_at distinguishes
        # successive observations; flushed in bulk by :meth:`write`.
        self._pending_odds: list[tuple] = []
        self._pending_lines: list[tuple] = []
        self._pending_ladder: list[tuple] = []

    def _table_columns(self, table: str) -> set[str]:
        return {
            row[0]
            for row in self._connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                [table],
            ).fetchall()
        }

    def _auto_migrate_observed_at(self) -> None:
        """In-place upgrade for pre-time-series schemas.

        Adds ``observed_at`` to ``odds`` / ``lines`` if missing, backfills
        from any ``sample_ts`` column left over from earlier in-progress
        work (else from ``game_date`` midnight), and drops ``sample_ts``.
        Idempotent: a fresh DB or post-migration DB has no work to do.
        """
        for table in ("odds", "lines"):
            cols = self._table_columns(table)
            if "observed_at" not in cols:
                self._connection.execute(f"ALTER TABLE {table} ADD COLUMN observed_at TIMESTAMP")
                if "sample_ts" in cols:
                    self._connection.execute(
                        f"UPDATE {table} SET observed_at = sample_ts "
                        "WHERE observed_at IS NULL AND sample_ts IS NOT NULL"
                    )
                self._connection.execute(
                    f"UPDATE {table} SET observed_at = CAST(game_date AS TIMESTAMP) "
                    "WHERE observed_at IS NULL"
                )
            if "sample_ts" in self._table_columns(table):
                self._connection.execute(f"ALTER TABLE {table} DROP COLUMN sample_ts")
        self._connection.commit()

    def _auto_migrate_shapefree_columns(self) -> None:
        """Add the nullable shape-free ``under_prob`` / ``line`` columns to ``odds``.

        Idempotent. The columns stay NULL until the standalone migration backfills
        history; readers fall back to ``ev`` while they are NULL.
        """
        cols = self._table_columns("odds")
        for col in ("under_prob", "line"):
            if col not in cols:
                self._connection.execute(f"ALTER TABLE odds ADD COLUMN {col} DOUBLE")
        self._connection.commit()

    def _weighted_book_ev(self, league: str, market: str, rows: list[tuple[str, float]]) -> float:
        weights = book_weights.get(league, {}).get(market, {})
        evs = []
        ws = []
        for book, ev in rows:
            if ev is None:
                continue
            evs.append(ev)
            ws.append(weights.get(book, 1))
        if not evs:
            return float("nan")
        return float(np.average(evs, weights=ws))

    @staticmethod
    def _book_shape_fitted(league, market) -> bool:
        """True when the cell has fitted WS2 ``book_shape`` coefficients in ``stat_meta``."""
        return (
            stat_dist.get(league, {}).get(market) == "SkewNormal"
            and stat_meta.get(league, {}).get(market, {}).get("book_shape") is not None
        )

    def _reencode_ev(self, league, market, ev, under_prob, line) -> float | None:
        """Rebuild the book mean from its ``(under_prob, line)`` quote via the fitted SkewNormal shape.

        Falls back to the stored ``ev`` for unfitted cells (see :meth:`_book_shape_fitted`)
        or rows whose shape-free quote is absent — bit-identical to the pre-WS2 read until
        ``book_shape`` coefficients exist for the cell.
        """
        if under_prob is None or line is None:
            return ev
        if not (np.isfinite(under_prob) and np.isfinite(line)):
            return ev
        if not self._book_shape_fitted(league, market):
            return ev
        sigma, skew = book_skewnormal_shape(league, market, line)
        return get_ev(
            line,
            under_prob,
            dist="SkewNormal",
            sigma=float(sigma),
            skew_alpha=float(skew),
            gate=book_gate(league, market, "SkewNormal"),
        )

    def _book_rows(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        entity: str,
        *,
        at: datetime.datetime | None = None,
        case_insensitive_entity: bool = False,
    ) -> list[tuple[str, float]]:
        """Return ``[(book, ev), ...]`` — latest observation per book at-or-before ``at``.

        ``ev`` is the WS2 re-encode of each book's ``(under_prob, line)`` quote at the fitted
        per-cell shape (:meth:`_reencode_ev`); unfitted cells return the stored ``ev`` as-is.

        ``at=None`` means "latest available", i.e. as-of-now.

        ``case_insensitive_entity`` compares ``entity`` via ``UPPER()`` on
        both sides. Set by the team-market readers (:meth:`get_moneyline`,
        :meth:`get_total`, :meth:`get_team_market`) so callers passing
        uppercase abbreviations like ``"BUF"`` still match legacy
        title-case archive rows like ``"Buf"`` written under the klepto
        backend. Player-prop callers leave this off — player names are
        already canonicalized via :func:`remove_accents` and proper-case
        names like ``"DeAndre Hopkins"`` would be corrupted by a
        case-folding match.
        """
        d = _safe_date(date)
        if d is None:
            return []
        params: list = [league, market, d, entity]
        entity_clause = "UPPER(entity)=UPPER(?)" if case_insensitive_entity else "entity=?"
        sql = (
            "SELECT book, ev, under_prob, line FROM ("
            "  SELECT book, ev, under_prob, line, observed_at, "
            "         ROW_NUMBER() OVER (PARTITION BY book ORDER BY observed_at DESC) AS rn "
            "  FROM odds "
            f"  WHERE league=? AND market=? AND game_date=? AND {entity_clause}"
        )
        if at is not None:
            sql += " AND observed_at <= ?"
            params.append(at)
        sql += ") WHERE rn = 1"
        return [
            (book, self._reencode_ev(league, market, ev, under_prob, line))
            for book, ev, under_prob, line in self._connection.execute(sql, params).fetchall()
        ]

    def get_ev(self, league, market, date, player, *, at: datetime.datetime | None = None):
        """Weighted-average player-prop EV across books for one slate entry.

        ``at=None`` returns the most recent observation per book; pass a
        ``datetime`` to read the at-or-before-``at`` snapshot for each book.
        """
        rows = self._book_rows(league, market, date, player, at=at)
        if not rows:
            return np.nan
        return self._weighted_book_ev(league, market, rows)

    def get_team_market(self, league, market, date, team, *, at: datetime.datetime | None = None):
        """Weighted-average team-market EV (non-player, non-moneyline).

        Matches ``team`` case-insensitively: gamelogs carry uppercase
        abbreviations (``"BUF"``) while the legacy klepto-migrated rows
        store title case (``"Buf"``). See ``_book_rows``.
        """
        rows = self._book_rows(league, market, date, team, at=at, case_insensitive_entity=True)
        if not rows:
            return np.nan
        return self._weighted_book_ev(league, market, rows)

    def get_moneyline(self, league, date, team, *, at: datetime.datetime | None = None):
        """Weighted-average moneyline EV across books for ``team`` on ``date``.

        Falls back to ``0.5`` when no book has quoted the game.
        Case-insensitive team match — see :meth:`get_team_market`.
        """
        rows = self._book_rows(league, "Moneyline", date, team, at=at, case_insensitive_entity=True)
        if not rows:
            return 0.5
        return self._weighted_book_ev(league, "Moneyline", rows)

    def get_total(self, league, date, team, *, at: datetime.datetime | None = None):
        """Weighted-average game-total EV for ``team`` on ``date``.

        Falls back to the per-league default total when no book has quoted
        the game so callers always receive a numeric value.
        Case-insensitive team match — see :meth:`get_team_market`.
        """
        rows = self._book_rows(league, "Totals", date, team, at=at, case_insensitive_entity=True)
        if not rows:
            return self.default_totals.get(league, 1)
        return self._weighted_book_ev(league, "Totals", rows)

    def get_team_market_map(
        self,
        league: str,
        market: str,
        *,
        dates: Iterable[str | datetime.date] | None = None,
        at: datetime.datetime | None = None,
    ) -> dict[tuple[str, str], float]:
        """Bulk weighted-average EV map for one ``(league, market)`` across many entities.

        Returns ``{("YYYY-MM-DD", entity): ev}`` for every ``(date, entity)``
        the archive holds for the league/market. ``dates`` restricts the
        scan to the slice the caller cares about (typical: the unique
        ``game_date`` values present in a freshly fetched gamelog). When
        ``dates`` is omitted, every date is scanned.

        Missing keys mean "no book quoted that slot"; callers should pass
        a fallback to :py:meth:`dict.get` (e.g. ``0.5`` for moneyline,
        :attr:`default_totals` for totals) to preserve the per-row
        semantics of :meth:`get_moneyline` / :meth:`get_total` when
        replacing row-by-row ``DataFrame.apply`` loops.

        Team names in the returned dict are uppercased so callers using
        the canonical uppercase abbreviation from :data:`abbreviations`
        (e.g. ``"BUF"``) match both legacy title-case archive rows
        (``"Buf"``, written under the klepto backend) and the current
        uppercase write path. See ``_book_rows`` for the read-side
        rationale.

        Args:
            league: League code (e.g., 'NBA', 'NFL').
            market: Market name (e.g., 'Moneyline', 'Totals').
            dates: Restrict scan to these game dates. ``None`` scans all dates.
            at: Observation cutoff; ``None`` means "latest available" per book.

        Returns:
            Dict mapping ``("YYYY-MM-DD", UPPER(entity))`` to weighted-average EV.
            Keys absent from the dict mean no book quoted that slot.
        """
        params: list = [league, market]
        sql = (
            "SELECT game_date, entity, book, ev FROM ("
            "  SELECT game_date, entity, book, ev, observed_at, "
            "         ROW_NUMBER() OVER ("
            "             PARTITION BY game_date, entity, book "
            "             ORDER BY observed_at DESC"
            "         ) AS rn "
            "  FROM odds "
            "  WHERE league=? AND market=?"
        )
        if dates is not None:
            normalized = sorted({_safe_date(d) for d in dates} - {None})
            if not normalized:
                return {}
            placeholders = ",".join(["?"] * len(normalized))
            sql += f" AND game_date IN ({placeholders})"
            params.extend(normalized)
        if at is not None:
            sql += " AND observed_at <= ?"
            params.append(at)
        sql += ") WHERE rn = 1"

        grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for game_date, entity, book, ev in self._connection.execute(sql, params).fetchall():
            key = (game_date.isoformat(), entity.upper())
            grouped.setdefault(key, []).append((book, ev))
        return {key: self._weighted_book_ev(league, market, rows) for key, rows in grouped.items()}

    def get_line(self, league, market, date, player, *, at: datetime.datetime | None = None):
        """Consensus line for ``player`` on ``date``: median, floored to ½.

        ``at=None`` aggregates every distinct line ever observed for the
        entity (the legacy semantics). Pass a ``datetime`` to median over
        only the distinct lines observed at-or-before ``at``.
        """
        d = _safe_date(date)
        if d is None:
            return 0
        params: list = [league, market, d, player]
        sql = (
            "SELECT DISTINCT line FROM lines "
            "WHERE league=? AND market=? AND game_date=? AND entity=?"
        )
        if at is not None:
            sql += " AND observed_at <= ?"
            params.append(at)
        arr = [row[0] for row in self._connection.execute(sql, params).fetchall()]
        if not arr:
            return 0
        line = np.floor(2 * np.median(arr)) / 2
        return 0 if np.isnan(line) else float(line)

    def to_pandas(self, league, market):
        """Flatten one league/market into a wide DataFrame.

        Indexed by ``(date, player)``, one column per book + a ``Line``
        column carrying the consensus line. Drops pre-2023-05-03 rows for
        non-totals markets (stale format) to match the legacy behaviour.

        Selects the latest observation per ``(date, player, book)`` so
        time-series storage does not change the per-book column semantics
        downstream consumers like ``fit_book_weights`` rely on.
        """
        cutoff = pd.Timestamp("2023-05-03")
        odds_df = self._connection.execute(
            "SELECT game_date, entity, book, ev, under_prob, line FROM ("
            "  SELECT game_date, entity, book, ev, under_prob, line, "
            "         ROW_NUMBER() OVER ("
            "             PARTITION BY game_date, entity, book "
            "             ORDER BY observed_at DESC"
            "         ) AS rn "
            "  FROM odds WHERE league=? AND market=?"
            ") WHERE rn = 1",
            [league, market],
        ).fetchdf()
        if odds_df.empty:
            return pd.DataFrame()

        odds_df["game_date"] = pd.to_datetime(odds_df["game_date"])
        if market not in ("Moneyline", "Total"):
            odds_df = odds_df[odds_df["game_date"] >= cutoff]
            if odds_df.empty:
                return pd.DataFrame()

        if self._book_shape_fitted(league, market):
            odds_df["ev"] = [
                self._reencode_ev(league, market, ev, up, ln)
                for ev, up, ln in zip(
                    odds_df["ev"], odds_df["under_prob"], odds_df["line"], strict=True
                )
            ]
        odds_df["game_date"] = odds_df["game_date"].dt.strftime("%Y-%m-%d")
        wide = odds_df.pivot_table(
            index=["game_date", "entity"], columns="book", values="ev", aggfunc="first"
        )
        wide.columns.name = None
        wide.index.names = ["date", "player"]

        if market in _TEAM_ONLY_MARKETS:
            return wide

        lines_df = self._connection.execute(
            "SELECT game_date, entity, line FROM lines WHERE league=? AND market=?",
            [league, market],
        ).fetchdf()
        if lines_df.empty:
            wide["Line"] = 0.0
            return wide

        lines_df["game_date"] = pd.to_datetime(lines_df["game_date"]).dt.strftime("%Y-%m-%d")
        consensus = (
            lines_df.groupby(["game_date", "entity"])["line"]
            .apply(lambda s: float(np.floor(2 * np.median(s)) / 2))
            .rename("Line")
        )
        consensus.index.names = ["date", "player"]
        wide = wide.join(consensus, how="left")
        wide["Line"] = wide["Line"].fillna(0.0)
        return wide

    def archived_players_by_date(self, league: str, market: str) -> dict[str, set[str]]:
        """Return ``{"YYYY-MM-DD": {player, ...}}`` for one (league, market).

        Used by training/data.count_training_rows to size the training matrix.
        """
        rows = self._connection.execute(
            "SELECT DISTINCT game_date, entity FROM odds WHERE league=? AND market=?",
            [league, market],
        ).fetchall()
        out: dict[str, set[str]] = {}
        for d, entity in rows:
            out.setdefault(d.isoformat(), set()).add(entity)
        return out

    def get_closing_line(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        entity: str,
        *,
        at: datetime.datetime | None = None,
    ) -> ClosingLine | None:
        """Return consensus line and implied probability from the latest pre-kickoff snapshot.

        Wraps ``get_line`` and ``get_ev`` to return a dataclass with the
        consensus line, the no-vig implied over probability, the timestamp
        of the latest snapshot, and the set of books that provided data.
        ``at=None`` uses the most recent observation; pass a ``datetime``
        to pin to a specific snapshot (e.g. ``commence_time``).

        Args:
            league: League code (e.g., 'NBA', 'MLB').
            market: Market name (e.g., 'player_pass_yds').
            date: Game date.
            entity: Player or team name.
            at: Snapshot cutoff; defaults to "latest available".

        Returns:
            ClosingLine with (line, devig_over, sample_ts, book_set) or
            ``None`` if no data exists for this entry.
        """
        d = _safe_date(date)
        if d is None:
            return None

        line = self.get_line(league, market, d, entity, at=at)
        rows = self._book_rows(league, market, d, entity, at=at)
        if not rows:
            return None

        books = [row[0] for row in rows]
        evs = [row[1] for row in rows]
        book_set = frozenset(books)

        dist = stat_dist.get(league, {}).get(market, "Gamma")
        cv = stat_cv.get(league, {}).get(market, 1)

        devig_over = _devig_over(line, evs, dist, cv)
        sample_ts = self._latest_sample_ts(league, market, d, entity, at)

        return ClosingLine(
            line=line,
            devig_over=devig_over,
            sample_ts=sample_ts,
            book_set=book_set,
        )

    def _latest_sample_ts(self, league, market, d, entity, at):
        sample_sql = (
            "SELECT observed_at FROM odds WHERE league=? AND market=? AND game_date=? AND entity=?"
        )
        sample_params: list = [league, market, d, entity]
        if at is not None:
            sample_sql += " AND observed_at <= ?"
            sample_params.append(at)
        sample_sql += " ORDER BY observed_at DESC LIMIT 1"
        sample_rows = self._connection.execute(sample_sql, sample_params).fetchall()
        return sample_rows[0][0] if sample_rows and sample_rows[0][0] is not None else None

    def get_line_history(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        entity: str,
        *,
        books: list[str] | None = None,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """Return ``[observed_at, line]`` rows sorted by ``observed_at``.

        ``books`` is accepted for API symmetry with :meth:`get_ev_history`
        but is ignored here — ``lines`` rows have no book column. ``since``
        and ``until`` bound the time range (inclusive at both ends).
        """
        del books  # lines table has no book column
        d = _safe_date(date)
        if d is None:
            return pd.DataFrame(columns=["observed_at", "line"])
        sql = (
            "SELECT observed_at, line FROM lines "
            "WHERE league=? AND market=? AND game_date=? AND entity=?"
        )
        params: list = [league, market, d, entity]
        if since is not None:
            sql += " AND observed_at >= ?"
            params.append(since)
        if until is not None:
            sql += " AND observed_at <= ?"
            params.append(until)
        sql += " ORDER BY observed_at"
        return self._connection.execute(sql, params).fetchdf()

    def get_ev_history(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        entity: str,
        *,
        books: list[str] | None = None,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """Return ``[observed_at, book, ev]`` rows sorted by ``observed_at``."""
        d = _safe_date(date)
        if d is None:
            return pd.DataFrame(columns=["observed_at", "book", "ev"])
        sql = (
            "SELECT observed_at, book, ev FROM odds "
            "WHERE league=? AND market=? AND game_date=? AND entity=?"
        )
        params: list = [league, market, d, entity]
        if books:
            placeholders = ", ".join("?" * len(books))
            sql += f" AND book IN ({placeholders})"
            params.extend(books)
        if since is not None:
            sql += " AND observed_at >= ?"
            params.append(since)
        if until is not None:
            sql += " AND observed_at <= ?"
            params.append(until)
        sql += " ORDER BY observed_at"
        return self._connection.execute(sql, params).fetchdf()

    def get_movement(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        entity: str,
        *,
        books: list[str] | None = None,
        until: datetime.datetime | None = None,
    ) -> dict:
        """Summarize the line/EV movement across a (league, market, date, entity).

        Returns ``open_*`` (first observation), ``close_*`` (last observation
        at-or-before ``until``), counts of observations and direction
        changes, peak/trough lines, and time-span minutes. NaN-filled when
        no observations match.
        """
        line_hist = self.get_line_history(league, market, date, entity, until=until)
        ev_hist = self.get_ev_history(league, market, date, entity, books=books, until=until)

        out: dict = {
            "open_line": np.nan,
            "open_ev": np.nan,
            "close_line": np.nan,
            "close_ev": np.nan,
            "n_obs": 0,
            "n_moves": 0,
            "peak_line": np.nan,
            "trough_line": np.nan,
            "time_span_minutes": np.nan,
        }

        if not line_hist.empty:
            out["open_line"] = float(line_hist["line"].iloc[0])
            out["close_line"] = float(line_hist["line"].iloc[-1])
            out["peak_line"] = float(line_hist["line"].max())
            out["trough_line"] = float(line_hist["line"].min())
            out["n_obs"] = len(line_hist)
            out["n_moves"] = int(line_hist["line"].diff().fillna(0).ne(0).sum())
            span = line_hist["observed_at"].iloc[-1] - line_hist["observed_at"].iloc[0]
            out["time_span_minutes"] = float(span.total_seconds() / 60.0)

        if not ev_hist.empty:
            first_ts = ev_hist["observed_at"].iloc[0]
            last_ts = ev_hist["observed_at"].iloc[-1]
            open_evs = ev_hist.loc[ev_hist["observed_at"] == first_ts, "ev"].dropna()
            close_evs = ev_hist.loc[ev_hist["observed_at"] == last_ts, "ev"].dropna()
            if len(open_evs):
                out["open_ev"] = float(open_evs.mean())
            if len(close_evs):
                out["close_ev"] = float(close_evs.mean())

        return out

    def _stage_book_ev(
        self,
        league: str,
        market: str,
        date: datetime.date,
        entity: str,
        book: str,
        ev: float,
        observed_at: datetime.datetime | None = None,
        under_prob: float | None = None,
        line: float | None = None,
    ) -> None:
        """Buffer a per-book EV observation; flushed by :meth:`write`.

        ``observed_at`` defaults to now; the historical backfill passes the
        snapshot's as-of time so point-in-time training reads pick it up.
        ``under_prob`` / ``line`` are the shape-free book quote; left NULL when
        the caller has only the encoded ``ev`` (readers fall back to ``ev``).
        """
        self._pending_odds.append(
            (
                league,
                market,
                date,
                entity,
                book,
                float(ev),
                observed_at or datetime.datetime.utcnow(),
                None if under_prob is None else float(under_prob),
                None if line is None else float(line),
            )
        )

    def _stage_line(
        self,
        league: str,
        market: str,
        date: datetime.date,
        entity: str,
        line: float,
        observed_at: datetime.datetime | None = None,
    ) -> None:
        """Buffer a line observation; flushed by :meth:`write`."""
        self._pending_lines.append(
            (league, market, date, entity, float(line), observed_at or datetime.datetime.utcnow())
        )

    def _stage_ladder(
        self,
        league: str,
        market: str,
        date: datetime.date,
        entity: str,
        book: str,
        line: float,
        p_over: float,
        observed_at: datetime.datetime | None = None,
    ) -> None:
        """Buffer one alt-line rung's de-vigged over-probability; flushed by :meth:`write`."""
        self._pending_ladder.append(
            (
                league,
                market,
                date,
                entity,
                book,
                float(line),
                float(p_over),
                observed_at or datetime.datetime.utcnow(),
            )
        )

    def add_ladder(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        entity: str,
        book: str,
        rungs: list[tuple[float, float]],
        observed_at: datetime.datetime | None = None,
    ) -> None:
        """Append every offered alt-line rung for one ``(player, book)``.

        ``rungs`` is a list of ``(line, de-vigged over-prob)`` captured before the
        consensus collapse, so the full ladder survives for later book-CDF fitting.
        The consensus ``ev`` / ``Lines`` path is unaffected — this is additive.
        """
        d = _safe_date(date)
        if d is None:
            return
        entity = remove_accents(entity)
        for line, p_over in rungs:
            if line is None or p_over is None:
                continue
            self._stage_ladder(league, market, d, entity, book, line, p_over, observed_at)

    def add_dfs(self, offers, platform, key):
        """Add a batch of scraped offers to the archive for one ``platform``.

        ``offers`` is accepted as a list or single dict; duplicates per
        ``(Player, Market)`` are resolved in favor of the offer closest to
        a neutral 1.0 boost. The ``key`` mapping renames sportsbook-native
        market strings into the canonical per-league market names used
        elsewhere in the pipeline.
        """
        offers = _dedup_offers_by_boost(offers)

        for o in offers:
            if not o["Line"]:
                continue
            d = _safe_date(o["Date"])
            if d is None:
                continue

            league = o["League"]
            market = _resolve_market(league, o["Market"], key)

            cv = stat_cv.get(league, {}).get(market, 1)
            dist = stat_dist.get(league, {}).get(market, "Gamma")
            gate = book_gate(league, market, dist)

            player = remove_accents(o["Player"])
            line = float(o["Line"])

            over = o.get("Boost_Over", 0) if o.get("Boost_Over", 0) > 0 else o.get("Boost", 1)
            # A missing/zero under side fabricated a ~6.5%-vig under in no_vig_odds
            # that inverts to a blown count-cell ev; DFS picks are symmetric instead.
            odds = no_vig_odds(over, _dfs_under_boost(over, o.get("Boost_Under")))
            ev = get_ev(line, odds[1], cv, dist=dist, gate=gate)

            self._stage_book_ev(
                league, market, d, player, platform, ev, under_prob=odds[1], line=line
            )
            self._stage_line(league, market, d, player, line)

    def merge_player_books(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        player: str,
        book_evs: dict[str, float],
        lines: list[float] | None = None,
        observed_at: datetime.datetime | None = None,
        book_quotes: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Append per-book EVs and any new lines for one player entry.

        Append-only under the time-series schema: every call adds new
        ``observed_at`` rows; the latest-per-book reader returns the
        freshest observations. ``observed_at`` defaults to now; the
        historical backfill passes the snapshot's as-of time. ``book_quotes``
        maps each book to its ``(line, under_prob)`` shape-free quote; when
        present the per-book row stores it so readers need not invert ``ev``.
        """
        d = _safe_date(date)
        if d is None:
            return
        player = remove_accents(player)
        book_quotes = book_quotes or {}
        for book, ev in book_evs.items():
            if ev is None:
                continue
            q_line, q_under = book_quotes.get(book, (None, None))
            self._stage_book_ev(
                league, market, d, player, book, ev, observed_at, under_prob=q_under, line=q_line
            )
        for line in lines or []:
            if line is None:
                continue
            self._stage_line(league, market, d, player, line, observed_at)

    def set_team_books(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        team: str,
        book_evs: dict[str, float],
    ) -> None:
        """Append per-book EVs for a team-market entry (Moneyline / Totals / Spreads).

        With time-series storage every observation is preserved; the
        ``set_*`` name is retained for caller compatibility but semantics
        are append-only — the latest-per-book reader returns the freshest.
        """
        d = _safe_date(date)
        if d is None:
            return
        for book, ev in book_evs.items():
            if ev is None:
                continue
            self._stage_book_ev(league, market, d, team, book, ev)

    def write(self, all=False):
        """Flush pending writes to disk.

        Append-only: every staged observation is inserted with its
        ``observed_at`` timestamp. ``observed_at`` distinguishes successive
        polls of the same ``(league, market, date, entity, book)`` — readers
        pick the latest by default and an explicit ``at=`` snapshot when
        needed.

        The ``all`` flag is retained for signature compatibility with the
        legacy klepto-backed Archive — it has no effect.
        """
        del all  # legacy flag; no effect under DuckDB
        con = self._connection

        if self._pending_odds:
            _odds_cols = [
                "league",
                "market",
                "game_date",
                "entity",
                "book",
                "ev",
                "observed_at",
                "under_prob",
                "line",
            ]
            odds_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                self._pending_odds,
                columns=_odds_cols,
            )
            con.execute(f"INSERT INTO odds ({', '.join(_odds_cols)}) SELECT * FROM odds_df")

        if self._pending_lines:
            lines_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                self._pending_lines,
                columns=["league", "market", "game_date", "entity", "line", "observed_at"],
            )
            con.execute("INSERT INTO lines SELECT * FROM lines_df")

        if self._pending_ladder:
            ladder_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                self._pending_ladder,
                columns=[
                    "league",
                    "market",
                    "game_date",
                    "entity",
                    "book",
                    "line",
                    "p_over",
                    "observed_at",
                ],
            )
            con.execute("INSERT INTO ladder SELECT * FROM ladder_df")

        con.commit()
        self._pending_odds.clear()
        self._pending_lines.clear()
        self._pending_ladder.clear()


class LazyArchive:
    """Proxy that defers :class:`Archive` instantiation until first use.

    DuckDB takes an exclusive file lock the moment a read-write connection
    is opened, and holds it for the connection's lifetime. The production
    modules historically bound ``archive = Archive()`` at module top, which
    meant any process that merely imported them — most importantly the
    long-lived Streamlit dashboard — grabbed the lock at startup and held
    it forever, blocking every cron job that opened the archive.

    ``LazyArchive`` looks and quacks exactly like an :class:`Archive`
    instance — every attribute read forwards to the live singleton — but
    constructing it does not touch the database. The lock is only acquired
    on the first attribute access from a code path that actually queries
    or writes the archive. :class:`Archive` is itself a singleton, so the
    forwarded call is bit-identical to the legacy direct binding.
    """

    __slots__ = ()

    def __getattr__(self, name: str):
        """Forward attribute lookups to the live :class:`Archive` singleton."""
        return getattr(Archive(), name)
