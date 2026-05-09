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
import warnings
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sportstradamus.helpers.config import book_weights, stat_cv, stat_dist, stat_zi
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

# No PRIMARY KEY: DuckDB's PK creates an ART index that bloats the DB ~10x for
# this row count. Lookups don't need the index — zone-map pruning on naturally
# sorted data scans 15M rows in ~1ms. ``observed_at`` is left nullable in DDL
# so the auto-migration path below can backfill in place against pre-rework
# DBs without rebuilding the table; the standalone migration script tightens
# it to NOT NULL when run.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS odds (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    book         TEXT NOT NULL,
    ev           DOUBLE,
    observed_at  TIMESTAMP
);
CREATE TABLE IF NOT EXISTS lines (
    league       TEXT NOT NULL,
    market       TEXT NOT NULL,
    game_date    DATE NOT NULL,
    entity       TEXT NOT NULL,
    line         DOUBLE NOT NULL,
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
    def _connect_with_wal_recovery(db_path: Path) -> duckdb.DuckDBPyConnection:
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        db_path = Path(os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", _DEFAULT_DB_PATH))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._connection = self._connect_with_wal_recovery(db_path)
        self._connection.execute(_SCHEMA_DDL)
        self._auto_migrate_observed_at()

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
                self._connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN observed_at TIMESTAMP"
                )
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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _mark_changed(self, league, market):
        """Retained for backwards compatibility — DuckDB needs no change-tracking."""
        return

    # ------------------------------------------------------------------ #
    # Read API
    # ------------------------------------------------------------------ #

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

    def _book_rows(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        entity: str,
        *,
        at: datetime.datetime | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(book, ev), ...]`` — latest observation per book at-or-before ``at``.

        ``at=None`` means "latest available", i.e. as-of-now.
        """
        d = _safe_date(date)
        if d is None:
            return []
        params: list = [league, market, d, entity]
        sql = (
            "SELECT book, ev FROM ("
            "  SELECT book, ev, observed_at, "
            "         ROW_NUMBER() OVER (PARTITION BY book ORDER BY observed_at DESC) AS rn "
            "  FROM odds "
            "  WHERE league=? AND market=? AND game_date=? AND entity=?"
        )
        if at is not None:
            sql += " AND observed_at <= ?"
            params.append(at)
        sql += ") WHERE rn = 1"
        return [(book, ev) for book, ev in self._connection.execute(sql, params).fetchall()]

    def get_ev(self, league, market, date, player, *, at: datetime.datetime | None = None):
        """Weighted-average player-prop EV across books for one slate entry.

        ``at=None`` returns the most recent observation per book; pass a
        ``datetime`` to read the at-or-before-``at`` snapshot for each book.
        """
        rows = self._book_rows(league, market, date, player, at=at)
        if not rows:
            return np.nan
        return self._weighted_book_ev(league, market, rows)

    def get_team_market(
        self, league, market, date, team, *, at: datetime.datetime | None = None
    ):
        """Weighted-average team-market EV (non-player, non-moneyline)."""
        rows = self._book_rows(league, market, date, team, at=at)
        if not rows:
            return np.nan
        return self._weighted_book_ev(league, market, rows)

    def get_moneyline(self, league, date, team, *, at: datetime.datetime | None = None):
        """Weighted-average moneyline EV across books for ``team`` on ``date``.

        Falls back to ``0.5`` when no book has quoted the game.
        """
        rows = self._book_rows(league, "Moneyline", date, team, at=at)
        if not rows:
            return 0.5
        return self._weighted_book_ev(league, "Moneyline", rows)

    def get_total(self, league, date, team, *, at: datetime.datetime | None = None):
        """Weighted-average game-total EV for ``team`` on ``date``.

        Falls back to the per-league default total when no book has quoted
        the game so callers always receive a numeric value.
        """
        rows = self._book_rows(league, "Totals", date, team, at=at)
        if not rows:
            return self.default_totals.get(league, 1)
        return self._weighted_book_ev(league, "Totals", rows)

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
            "SELECT game_date, entity, book, ev FROM ("
            "  SELECT game_date, entity, book, ev, "
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

        over_probs = []
        for ev in evs:
            if ev is None or np.isnan(ev):
                continue
            try:
                under_prob = get_odds(line, ev, dist, cv=cv)
                over_probs.append(1.0 - under_prob)
            except (ValueError, RuntimeError):
                continue

        devig_over = np.nan if not over_probs else float(np.mean(over_probs))

        sample_sql = (
            "SELECT observed_at FROM odds "
            "WHERE league=? AND market=? AND game_date=? AND entity=?"
        )
        sample_params: list = [league, market, d, entity]
        if at is not None:
            sample_sql += " AND observed_at <= ?"
            sample_params.append(at)
        sample_sql += " ORDER BY observed_at DESC LIMIT 1"
        sample_rows = self._connection.execute(sample_sql, sample_params).fetchall()
        sample_ts = sample_rows[0][0] if sample_rows and sample_rows[0][0] is not None else None

        return ClosingLine(
            line=line,
            devig_over=devig_over,
            sample_ts=sample_ts,
            book_set=book_set,
        )

    # ------------------------------------------------------------------ #
    # History API
    # ------------------------------------------------------------------ #

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
            out["n_obs"] = int(len(line_hist))
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

    # ------------------------------------------------------------------ #
    # Write API
    # ------------------------------------------------------------------ #

    def _stage_book_ev(
        self,
        league: str,
        market: str,
        date: datetime.date,
        entity: str,
        book: str,
        ev: float,
    ) -> None:
        """Buffer a per-book EV observation; flushed by :meth:`write`."""
        self._pending_odds.append(
            (league, market, date, entity, book, float(ev), datetime.datetime.utcnow())
        )

    def _stage_line(
        self, league: str, market: str, date: datetime.date, entity: str, line: float
    ) -> None:
        """Buffer a line observation; flushed by :meth:`write`."""
        self._pending_lines.append(
            (league, market, date, entity, float(line), datetime.datetime.utcnow())
        )

    def add_dfs(self, offers, platform, key):
        """Add a batch of scraped offers to the archive for one ``platform``.

        ``offers`` is accepted as a list or single dict; duplicates per
        ``(Player, Market)`` are resolved in favor of the offer closest to
        a neutral 1.0 boost. The ``key`` mapping renames sportsbook-native
        market strings into the canonical per-league market names used
        elsewhere in the pipeline.
        """
        if not isinstance(offers, list):
            offers = [offers]

        df = pd.DataFrame(offers)
        if df.empty:
            return
        if "Boost_Over" not in df.columns:
            df["Boost_Over"] = np.nan
        if "Boost" in df.columns:
            df.loc[df["Boost_Over"].isna(), "Boost_Over"] = df.loc[df["Boost_Over"].isna(), "Boost"]
        df["Boost Factor"] = np.abs(df["Boost_Over"] - 1)
        df = df.loc[~df.sort_values("Boost Factor").duplicated(["Player", "Market"])]
        offers = df.to_dict(orient="records")

        for o in offers:
            if not o["Line"]:
                continue
            d = _safe_date(o["Date"])
            if d is None:
                continue

            league = o["League"]
            market = o["Market"].replace("H2H ", "")
            market = key.get(market, market)
            if league == "NHL":
                market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
            if league in ("NBA", "WNBA"):
                market = market.replace("underdog", "prizepicks")

            cv = stat_cv.get(league, {}).get(market, 1)
            dist = stat_dist.get(league, {}).get(market, "Gamma")
            gate = stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0

            player = remove_accents(o["Player"])
            line = float(o["Line"])

            over = o.get("Boost_Over", 0) if o.get("Boost_Over", 0) > 0 else o.get("Boost", 1)
            odds = no_vig_odds(over, o.get("Boost_Under"))
            ev = get_ev(line, odds[1], cv, dist=dist, gate=gate or None)

            self._stage_book_ev(league, market, d, player, platform, ev)
            self._stage_line(league, market, d, player, line)

    def merge_player_books(
        self,
        league: str,
        market: str,
        date: str | datetime.date,
        player: str,
        book_evs: dict[str, float],
        lines: list[float] | None = None,
    ) -> None:
        """Append per-book EVs and any new lines for one player entry.

        Append-only under the time-series schema: every call adds new
        ``observed_at`` rows; the latest-per-book reader returns the
        freshest observations.
        """
        d = _safe_date(date)
        if d is None:
            return
        player = remove_accents(player)
        for book, ev in book_evs.items():
            if ev is None:
                continue
            self._stage_book_ev(league, market, d, player, book, ev)
        for line in lines or []:
            if line is None:
                continue
            self._stage_line(league, market, d, player, line)

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

    # ------------------------------------------------------------------ #
    # Sync
    # ------------------------------------------------------------------ #

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
            odds_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                self._pending_odds,
                columns=["league", "market", "game_date", "entity", "book", "ev", "observed_at"],
            )
            con.execute("INSERT INTO odds SELECT * FROM odds_df")

        if self._pending_lines:
            lines_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                self._pending_lines,
                columns=["league", "market", "game_date", "entity", "line", "observed_at"],
            )
            con.execute("INSERT INTO lines SELECT * FROM lines_df")

        con.commit()
        self._pending_odds.clear()
        self._pending_lines.clear()
