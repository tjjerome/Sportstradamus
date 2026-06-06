"""Characterization pins for the two over-CC ``Archive`` methods: ``get_closing_line``
(consensus line + de-vigged over-probability + latest-snapshot timestamp) and
``add_dfs`` (DFS offer ingestion: per-(Player,Market) boost dedup, market-name
resolution, symmetric no-vig EV).

Both are pinned with duck-typed ``self`` objects exposing only the seams each method
touches -- no DuckDB. ``get_closing_line`` uses league/market ``"ZZZ"``/``"zzz"`` which
miss ``stat_dist`` / ``stat_cv`` so the distribution defaults to Gamma / cv=1,
config-independent and deterministic. The expected ``devig_over`` is a literal captured
from the pre-refactor code (not recomputed via ``get_odds``, which the method calls).
"""

from __future__ import annotations

import datetime

import pytest

from sportstradamus.helpers.archive import Archive, _devig_over
from sportstradamus.helpers.distributions import SN_MAX_MEAN_FACTOR


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.calls: list[tuple[str, list]] = []

    def execute(self, sql, params):
        self.calls.append((sql, list(params)))
        return _FakeCursor(self._rows)


class _FakeClosing:
    """Duck-typed ``self`` for ``get_closing_line``: fakes the I/O seams (get_line /
    _book_rows / _connection) and borrows the real ``_devig_over`` / ``_latest_sample_ts``
    so the orchestration is exercised end-to-end without a DuckDB instance."""

    _devig_over = staticmethod(_devig_over)
    _latest_sample_ts = Archive._latest_sample_ts

    def __init__(self, line, book_rows, sample_rows):
        self._line = line
        self._rows = book_rows
        self._connection = _FakeConn(sample_rows)

    def get_line(self, league, market, d, entity, *, at=None):
        return self._line

    def _book_rows(self, league, market, d, entity, *, at=None):
        return self._rows


_BOOK_ROWS = [("BookA", 9.5), ("BookB", 10.5), ("BookC", None), ("BookD", float("nan"))]
_SAMPLE = [(datetime.datetime(2025, 3, 9, 18, 30),)]


def test_get_closing_line_devig_and_book_set():
    fake = _FakeClosing(10.0, _BOOK_ROWS, _SAMPLE)
    cl = Archive.get_closing_line(fake, "ZZZ", "zzz", "2025-03-10", "Someone")

    assert cl.line == 10.0
    assert cl.devig_over == pytest.approx(0.36926293071495014)
    assert cl.sample_ts == datetime.datetime(2025, 3, 9, 18, 30)
    # book_set carries EVERY book row, including the None/NaN-EV ones skipped in devig
    assert cl.book_set == frozenset({"BookA", "BookB", "BookC", "BookD"})


def test_get_closing_line_sample_query_no_at():
    fake = _FakeClosing(10.0, _BOOK_ROWS, _SAMPLE)
    Archive.get_closing_line(fake, "ZZZ", "zzz", "2025-03-10", "Someone")
    sql, params = fake._connection.calls[0]
    assert "observed_at <= ?" not in sql
    assert sql.endswith("ORDER BY observed_at DESC LIMIT 1")
    assert params == ["ZZZ", "zzz", datetime.date(2025, 3, 10), "Someone"]


def test_get_closing_line_sample_query_with_at():
    fake = _FakeClosing(10.0, _BOOK_ROWS, _SAMPLE)
    at = datetime.datetime(2025, 3, 9, 19, 0)
    Archive.get_closing_line(fake, "ZZZ", "zzz", "2025-03-10", "Someone", at=at)
    sql, params = fake._connection.calls[0]
    assert "AND observed_at <= ?" in sql
    assert params == ["ZZZ", "zzz", datetime.date(2025, 3, 10), "Someone", at]


def test_get_closing_line_empty_rows_returns_none():
    fake = _FakeClosing(10.0, [], [])
    assert Archive.get_closing_line(fake, "ZZZ", "zzz", "2025-03-10", "X") is None


def test_get_closing_line_bad_date_returns_none():
    fake = _FakeClosing(10.0, _BOOK_ROWS, _SAMPLE)
    assert Archive.get_closing_line(fake, "ZZZ", "zzz", "not-a-date", "X") is None


class _CaptureArchive:
    """Duck-typed ``self`` for ``add_dfs`` -- captures the staged book-EV / line rows."""

    def __init__(self):
        self.book_evs: list[tuple] = []
        self.lines: list[tuple] = []

    def _stage_book_ev(self, league, market, date, entity, book, ev, observed_at=None):
        self.book_evs.append((league, market, entity, book, ev))

    def _stage_line(self, league, market, date, entity, line, observed_at=None):
        self.lines.append((league, market, entity, line))


def test_add_dfs_dedup_market_resolution_and_ev():
    offers = [
        {
            "Player": "Nikola Jokic",
            "League": "NBA",
            "Market": "BLK",
            "Line": 1.5,
            "Date": "2025-12-23",
            "Boost": 1.0,
        },
        {
            "Player": "Nikola Jokic",
            "League": "NBA",
            "Market": "BLK",
            "Line": 1.5,
            "Date": "2025-12-23",
            "Boost": 1.3,
        },
        {
            "Player": "Connor McDavid",
            "League": "NHL",
            "Market": "PTS",
            "Line": 1.5,
            "Date": "2025-12-23",
            "Boost": 1.0,
        },
        {
            "Player": "Some Guy",
            "League": "WNBA",
            "Market": "underdog_pts",
            "Line": 10.5,
            "Date": "2025-12-23",
            "Boost": 1.0,
        },
    ]
    cap = _CaptureArchive()
    Archive.add_dfs(cap, offers, "Underdog", {})

    # the two Jokic BLK offers collapse to one (boost 1.0 closer to neutral than 1.3)
    assert len(cap.book_evs) == 3
    leagues_markets_entities = [(lg, mkt, ent) for lg, mkt, ent, _, _ in cap.book_evs]
    assert leagues_markets_entities == [
        ("NBA", "BLK", "Nikola Jokic"),  # market unchanged
        ("NHL", "points", "Connor Mcdavid"),  # PTS->points via NHL remap; accents stripped
        ("WNBA", "prizepicks_pts", "Some Guy"),  # underdog->prizepicks for NBA/WNBA
    ]
    evs = {ent: ev for _, _, ent, _, ev in cap.book_evs}
    # Jokic BLK (ZINB) and McDavid points (SkewNormal) both encode through a
    # calibrated cell (stat_calibration.json -- gitignored, meditate-recomputed),
    # so their exact ev moves with the live calibration: pin only the invariant the
    # book-fallback clamp guarantees -- positive and bounded by SN_MAX_MEAN_FACTOR *
    # line, never a runaway. Some Guy's WNBA pts cell is config-missing (default
    # dist/cv), so its ev is deterministic and stays an exact pin.
    assert 0 < evs["Nikola Jokic"] <= SN_MAX_MEAN_FACTOR * 1.5
    assert 0 < evs["Connor Mcdavid"] <= SN_MAX_MEAN_FACTOR * 1.5
    assert evs["Some Guy"] == pytest.approx(15.136385967, rel=1e-6)

    assert cap.lines == [
        ("NBA", "BLK", "Nikola Jokic", 1.5),
        ("NHL", "points", "Connor Mcdavid", 1.5),
        ("WNBA", "prizepicks_pts", "Some Guy", 10.5),
    ]


def test_add_dfs_empty_after_dedup_stages_nothing():
    cap = _CaptureArchive()
    Archive.add_dfs(cap, [], "Underdog", {})
    assert cap.book_evs == [] and cap.lines == []
