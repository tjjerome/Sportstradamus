"""DuckDB-backed odds/EV archive.

The :class:`Archive` singleton is the only authorized entrypoint for reading
and writing the on-disk odds archive at ``archive/archive.duckdb``.
Consumers share the single instance so that in-memory transactions stay
consistent across the scrape/predict pipeline.

Schema (created on first connect):

* ``odds(league, market, game_date, entity, book, ev)`` — one row per
  (slate-entry, book). ``entity`` is a player name for player props or a
  team name for moneyline / totals / spreads / team markets.
* ``lines(league, market, game_date, entity, line)`` — distinct lines
  quoted by any book for a player prop. Skipped for moneyline / totals /
  spreads / team-only markets, which are pure EV.

:func:`clean_archive` drops dates older than ``cutoff_date`` and prunes
combo / matchup pseudo-entities (``" + "``, ``" vs. "``).
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sportstradamus.helpers.config import book_weights, stat_cv, stat_dist, stat_zi
from sportstradamus.helpers.distributions import get_ev, no_vig_odds
from sportstradamus.helpers.text import remove_accents

# Markets whose value schema is per-book EV only (no Lines table rows).
_TEAM_ONLY_MARKETS = frozenset({"Moneyline", "Totals", "1st 1 innings"})

_DEFAULT_DB_PATH = Path("archive/archive.duckdb")

# No PRIMARY KEY: DuckDB's PK creates an ART index that bloats the DB ~10x for
# this row count. Lookups don't need the index — zone-map pruning on naturally
# sorted data scans 15M rows in ~1ms. Dedup is enforced at write-flush time.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS odds (
    league      TEXT NOT NULL,
    market      TEXT NOT NULL,
    game_date   DATE NOT NULL,
    entity      TEXT NOT NULL,
    book        TEXT NOT NULL,
    ev          DOUBLE
);
CREATE TABLE IF NOT EXISTS lines (
    league      TEXT NOT NULL,
    market      TEXT NOT NULL,
    game_date   DATE NOT NULL,
    entity      TEXT NOT NULL,
    line        DOUBLE NOT NULL
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
        self._connection = duckdb.connect(str(db_path))
        self._connection.execute(_SCHEMA_DDL)

        self.default_totals = {
            "MLB": 4.671,
            "NBA": 111.667,
            "WNBA": 81.667,
            "NFL": 22.668,
            "NHL": 2.674,
        }

        # Pending-write buffers. Mutations accumulate here and are flushed in
        # bulk by :meth:`write` — this matches the legacy klepto "in-memory dict
        # + dump on demand" semantics and lets us run without an on-disk index
        # (which would 10x the DB size for this row count).
        self._pending_odds: list[tuple] = []
        self._pending_lines: list[tuple] = []
        self._replace_keys: set[tuple] = set()

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
        self, league: str, market: str, date: str | datetime.date, entity: str
    ) -> list[tuple[str, float]]:
        d = _safe_date(date)
        if d is None:
            return []
        return self._connection.execute(
            "SELECT book, ev FROM odds WHERE league=? AND market=? AND game_date=? AND entity=?",
            [league, market, d, entity],
        ).fetchall()

    def get_ev(self, league, market, date, player):
        """Weighted-average player-prop EV across books for one slate entry."""
        rows = self._book_rows(league, market, date, player)
        if not rows:
            return np.nan
        return self._weighted_book_ev(league, market, rows)

    def get_team_market(self, league, market, date, team):
        """Weighted-average team-market EV (non-player, non-moneyline)."""
        rows = self._book_rows(league, market, date, team)
        if not rows:
            return np.nan
        return self._weighted_book_ev(league, market, rows)

    def get_moneyline(self, league, date, team):
        """Weighted-average moneyline EV across books for ``team`` on ``date``.

        Falls back to ``0.5`` when no book has quoted the game.
        """
        rows = self._book_rows(league, "Moneyline", date, team)
        if not rows:
            return 0.5
        return self._weighted_book_ev(league, "Moneyline", rows)

    def get_total(self, league, date, team):
        """Weighted-average game-total EV for ``team`` on ``date``.

        Falls back to the per-league default total when no book has quoted
        the game so callers always receive a numeric value.
        """
        rows = self._book_rows(league, "Totals", date, team)
        if not rows:
            return self.default_totals.get(league, 1)
        return self._weighted_book_ev(league, "Totals", rows)

    def get_line(self, league, market, date, player):
        """Consensus line for ``player`` on ``date``: median, floored to ½."""
        d = _safe_date(date)
        if d is None:
            return 0
        arr = [
            row[0]
            for row in self._connection.execute(
                "SELECT line FROM lines WHERE league=? AND market=? AND game_date=? AND entity=?",
                [league, market, d, player],
            ).fetchall()
        ]
        if not arr:
            return 0
        line = np.floor(2 * np.median(arr)) / 2
        return 0 if np.isnan(line) else float(line)

    def to_pandas(self, league, market):
        """Flatten one league/market into a wide DataFrame.

        Indexed by ``(date, player)``, one column per book + a ``Line``
        column carrying the consensus line. Drops pre-2023-05-03 rows for
        non-totals markets (stale format) to match the legacy behaviour.
        """
        cutoff = pd.Timestamp("2023-05-03")
        odds_df = self._connection.execute(
            "SELECT game_date, entity, book, ev FROM odds WHERE league=? AND market=?",
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
        """Buffer a per-book EV update; flushed by :meth:`write`."""
        self._pending_odds.append((league, market, date, entity, book, float(ev)))

    def _stage_line(
        self, league: str, market: str, date: datetime.date, entity: str, line: float
    ) -> None:
        """Buffer a line entry; flushed by :meth:`write`."""
        self._pending_lines.append((league, market, date, entity, float(line)))

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
        """Merge per-book EVs and append unique lines for one player entry.

        Mirrors the legacy ``["EV"].update(...)`` + ``["Lines"].append(...)``
        semantics: existing book entries are overwritten on overlap, missing
        ones are inserted; existing lines are kept; new distinct lines are
        added.
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
        """Replace all per-book EVs for a team-market entry (Moneyline / Totals / Spreads).

        Mirrors the legacy ``archive[league][market][date][team] = book_evs``
        whole-dict assignment: every existing book row for this entity is
        deleted before the new ones are inserted.
        """
        d = _safe_date(date)
        if d is None:
            return
        self._replace_keys.add((league, market, d, team))
        for book, ev in book_evs.items():
            if ev is None:
                continue
            self._stage_book_ev(league, market, d, team, book, ev)

    # ------------------------------------------------------------------ #
    # Sync
    # ------------------------------------------------------------------ #

    def write(self, all=False):
        """Flush pending writes to disk.

        Drains the in-memory buffers populated by :meth:`add_dfs`,
        :meth:`merge_player_books`, and :meth:`set_team_books` into the
        underlying DuckDB tables in three bulk steps:

        1. Delete every ``(league, market, date, entity)`` recorded by a
           ``set_team_books`` call so its replacement rows can be inserted
           without conflict.
        2. Dedupe pending odds (last-write-wins per book), delete any
           existing rows that would conflict, then bulk-insert the staging.
        3. Bulk-insert pending lines, anti-joining against existing rows
           so duplicates are silently skipped.

        The ``all`` flag is retained for signature compatibility with the
        legacy klepto-backed Archive — it has no effect.
        """
        con = self._connection

        if self._replace_keys:
            replace_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                list(self._replace_keys),
                columns=["league", "market", "game_date", "entity"],
            )
            con.execute(
                "DELETE FROM odds USING replace_df WHERE "
                "odds.league = replace_df.league AND odds.market = replace_df.market AND "
                "odds.game_date = replace_df.game_date AND odds.entity = replace_df.entity"
            )

        if self._pending_odds:
            odds_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                self._pending_odds,
                columns=["league", "market", "game_date", "entity", "book", "ev"],
            ).drop_duplicates(
                subset=["league", "market", "game_date", "entity", "book"], keep="last"
            )
            # Remove the rows that the new staging will overwrite, then insert.
            con.execute(
                "DELETE FROM odds USING odds_df WHERE "
                "odds.league = odds_df.league AND odds.market = odds_df.market AND "
                "odds.game_date = odds_df.game_date AND odds.entity = odds_df.entity AND "
                "odds.book = odds_df.book"
            )
            con.execute("INSERT INTO odds SELECT * FROM odds_df")

        if self._pending_lines:
            lines_df = pd.DataFrame(  # noqa: F841 — referenced via DuckDB DataFrame replacement
                self._pending_lines,
                columns=["league", "market", "game_date", "entity", "line"],
            ).drop_duplicates()
            con.execute(
                "INSERT INTO lines "
                "SELECT s.* FROM lines_df s WHERE NOT EXISTS ("
                "  SELECT 1 FROM lines l WHERE "
                "  l.league = s.league AND l.market = s.market AND "
                "  l.game_date = s.game_date AND l.entity = s.entity AND l.line = s.line)"
            )

        con.commit()
        self._pending_odds.clear()
        self._pending_lines.clear()
        self._replace_keys.clear()
