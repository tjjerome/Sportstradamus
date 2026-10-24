# ARCHIVED 2026-08-29 from src/sportstradamus/helpers/archive.py
# Reason: add_dfs now prices one-sided DFS offers at their payout-implied
#         breakeven (dfs_boost_probs); fabricating the symmetric under twin
#         let moved/discounted lines enter the archive as fair 50/50 quotes.
# Last live SHA: e1246576
# Original imports (now unresolved here):
#   (none)


def _dfs_under_boost(over, boost_under):
    return boost_under if boost_under and boost_under > 0 else over


# ARCHIVED 2026-10-11 from src/sportstradamus/helpers/archive.py
# Reason: Nothing called Archive.get_closing_line, so it is archived with what only it read:
#         the ClosingLine dataclass, _devig_over and Archive._latest_sample_ts.
#         Archive.archived_players_by_date goes too: its one caller, count_training_rows,
#         was archived the same day (training_data_orphans.py).
# Last live SHA: a63b9107
# Original imports (now unresolved here):
#   import dataclasses
#   import datetime
#   import numpy as np
#   from sportstradamus.helpers.archive import _safe_date
#   from sportstradamus.helpers.config import stat_cv, stat_dist
#   from sportstradamus.helpers.distributions import get_odds


# From src/sportstradamus/helpers/archive.py. Only get_closing_line built one.
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


# From src/sportstradamus/helpers/archive.py. Only get_closing_line called it.
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


# From src/sportstradamus/helpers/archive.py. Was a method of Archive, a plain function here:
# ``self`` is the Archive instance and ``self._connection`` its DuckDB connection.
# count_training_rows (training_data_orphans.py) was its only caller.
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


# From src/sportstradamus/helpers/archive.py. Was a method of Archive, a plain function here;
# ``self.get_line`` and ``self._book_rows`` are still Archive methods, and
# ``self._latest_sample_ts`` is the function below. Nothing called it.
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


# From src/sportstradamus/helpers/archive.py. Was a method of Archive, a plain function here.
# Only get_closing_line called it.
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
