"""Unit tests for the time-series read/write paths in ``helpers.archive``.

Covers the four guarantees the rework introduced:

* round-trip writes carry an ``observed_at`` timestamp through to reads;
* point-in-time ``at=`` queries return the latest-per-book observation
  at-or-before the cutoff (and an empty result when ``at`` predates every
  observation);
* ``get_line_history`` / ``get_ev_history`` enumerate all observations in
  observation order, optionally bounded by ``since``/``until``;
* ``get_movement`` summarises a synthetic 5-observation series correctly;
* legacy single-point rows backfilled to ``game_date`` midnight remain
  visible to any ``at >= midnight`` query (the back-compat guarantee).
"""

from __future__ import annotations

import contextlib
import datetime
from datetime import date, timedelta
from datetime import datetime as dt

import pytest

from sportstradamus.helpers.archive import Archive


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """Yield a fresh ``Archive`` rooted at ``tmp_path``."""
    db_path = tmp_path / "archive.duckdb"
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(db_path))
    # Reset the singleton so ``__init__`` re-runs against the env var above.
    if Archive._instance is not None:
        with contextlib.suppress(Exception):
            Archive._instance._connection.close()
        Archive._instance._initialized = False
    a = Archive()
    yield a
    with contextlib.suppress(Exception):
        a._connection.close()
    Archive._instance._initialized = False


def _insert_odds(archive, *, league, market, d, entity, book, ev, observed_at):
    archive._connection.execute(
        "INSERT INTO odds VALUES (?, ?, ?, ?, ?, ?, ?)",
        [league, market, d, entity, book, float(ev), observed_at],
    )


def _insert_line(archive, *, league, market, d, entity, line, observed_at):
    archive._connection.execute(
        "INSERT INTO lines VALUES (?, ?, ?, ?, ?, ?)",
        [league, market, d, entity, float(line), observed_at],
    )


# --------------------------------------------------------------------------
# round-trip via the public stagers
# --------------------------------------------------------------------------


def test_write_path_stamps_observed_at_and_reads_back(archive):
    archive.merge_player_books(
        "WNBA",
        "PTS",
        "2026-05-08",
        "A'Ja Wilson",
        {"pinnacle": 0.55},
        lines=[22.5],
    )
    archive.write()

    rows = archive._connection.execute("SELECT book, ev, observed_at FROM odds").fetchall()
    assert len(rows) == 1
    book, ev, observed_at = rows[0]
    assert book == "pinnacle"
    assert ev == pytest.approx(0.55)
    assert isinstance(observed_at, dt)

    line_rows = archive._connection.execute("SELECT line, observed_at FROM lines").fetchall()
    assert len(line_rows) == 1
    assert line_rows[0][0] == pytest.approx(22.5)


# --------------------------------------------------------------------------
# point-in-time reads
# --------------------------------------------------------------------------


def test_get_ev_at_picks_latest_per_book_at_or_before_cutoff(archive):
    d = date(2026, 5, 8)
    base = dt(2026, 5, 8, 12, 0, 0)
    _insert_odds(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="A. Wilson",
        book="pinnacle",
        ev=0.50,
        observed_at=base,
    )
    _insert_odds(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="A. Wilson",
        book="pinnacle",
        ev=0.55,
        observed_at=base + timedelta(hours=1),
    )
    _insert_odds(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="A. Wilson",
        book="pinnacle",
        ev=0.60,
        observed_at=base + timedelta(hours=2),
    )

    # at=None → most recent observation
    assert archive.get_ev("WNBA", "PTS", "2026-05-08", "A. Wilson") == pytest.approx(0.60)
    # at = exact second observation timestamp → that one
    assert archive.get_ev(
        "WNBA", "PTS", "2026-05-08", "A. Wilson", at=base + timedelta(hours=1)
    ) == pytest.approx(0.55)
    # at before any observation → no row → NaN
    assert archive.get_ev(
        "WNBA", "PTS", "2026-05-08", "A. Wilson", at=base - timedelta(hours=1)
    ) != archive.get_ev("WNBA", "PTS", "2026-05-08", "A. Wilson")  # NaN != itself
    # at well after all observations → most recent
    assert archive.get_ev(
        "WNBA", "PTS", "2026-05-08", "A. Wilson", at=base + timedelta(days=1)
    ) == pytest.approx(0.60)


def test_get_line_at_aggregates_distinct_lines_observed_through_cutoff(archive):
    d = date(2026, 5, 8)
    base = dt(2026, 5, 8, 12, 0, 0)
    _insert_line(archive, league="WNBA", market="PTS", d=d, entity="P", line=22.0, observed_at=base)
    _insert_line(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="P",
        line=23.0,
        observed_at=base + timedelta(hours=1),
    )
    _insert_line(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="P",
        line=24.0,
        observed_at=base + timedelta(hours=2),
    )

    # All lines visible at end → median = 23.0 → floor(46)/2 = 23.0.
    assert archive.get_line("WNBA", "PTS", "2026-05-08", "P") == pytest.approx(23.0)
    # Only first two lines visible at +1h cutoff → median = 22.5 → floor(45)/2 = 22.5.
    assert archive.get_line(
        "WNBA", "PTS", "2026-05-08", "P", at=base + timedelta(hours=1)
    ) == pytest.approx(22.5)
    # Cutoff before any observation → empty result → 0.
    assert archive.get_line("WNBA", "PTS", "2026-05-08", "P", at=base - timedelta(hours=1)) == 0


# --------------------------------------------------------------------------
# history APIs
# --------------------------------------------------------------------------


def test_get_line_history_returns_observations_in_order(archive):
    d = date(2026, 5, 8)
    base = dt(2026, 5, 8, 12, 0, 0)
    for offset, line in [(0, 22.0), (1, 22.5), (2, 23.0)]:
        _insert_line(
            archive,
            league="WNBA",
            market="PTS",
            d=d,
            entity="P",
            line=line,
            observed_at=base + timedelta(hours=offset),
        )

    full = archive.get_line_history("WNBA", "PTS", "2026-05-08", "P")
    assert list(full["line"]) == [22.0, 22.5, 23.0]
    assert list(full["observed_at"]) == [
        base,
        base + timedelta(hours=1),
        base + timedelta(hours=2),
    ]

    bounded = archive.get_line_history(
        "WNBA",
        "PTS",
        "2026-05-08",
        "P",
        since=base + timedelta(minutes=30),
        until=base + timedelta(hours=1, minutes=30),
    )
    assert list(bounded["line"]) == [22.5]


def test_get_ev_history_filters_by_books(archive):
    d = date(2026, 5, 8)
    base = dt(2026, 5, 8, 12, 0, 0)
    _insert_odds(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="P",
        book="pinnacle",
        ev=0.55,
        observed_at=base,
    )
    _insert_odds(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="P",
        book="fanduel",
        ev=0.52,
        observed_at=base,
    )

    pin_only = archive.get_ev_history("WNBA", "PTS", "2026-05-08", "P", books=["pinnacle"])
    assert list(pin_only["book"]) == ["pinnacle"]
    assert pin_only["ev"].iloc[0] == pytest.approx(0.55)


def test_get_movement_synthetic_5_observation_series(archive):
    d = date(2026, 5, 8)
    base = dt(2026, 5, 8, 12, 0, 0)
    series = [
        (0, 22.0),
        (15, 22.5),
        (30, 22.5),  # no move
        (45, 23.0),
        (60, 22.5),  # back down
    ]
    for minutes, line in series:
        _insert_line(
            archive,
            league="WNBA",
            market="PTS",
            d=d,
            entity="P",
            line=line,
            observed_at=base + timedelta(minutes=minutes),
        )
    for minutes, ev in [(0, 0.50), (60, 0.58)]:
        _insert_odds(
            archive,
            league="WNBA",
            market="PTS",
            d=d,
            entity="P",
            book="pinnacle",
            ev=ev,
            observed_at=base + timedelta(minutes=minutes),
        )

    movement = archive.get_movement("WNBA", "PTS", "2026-05-08", "P")
    assert movement["open_line"] == pytest.approx(22.0)
    assert movement["close_line"] == pytest.approx(22.5)
    assert movement["peak_line"] == pytest.approx(23.0)
    assert movement["trough_line"] == pytest.approx(22.0)
    assert movement["n_obs"] == 5
    # 22.0→22.5 (move), 22.5→22.5 (no move), 22.5→23.0 (move), 23.0→22.5 (move) = 3.
    assert movement["n_moves"] == 3
    assert movement["time_span_minutes"] == pytest.approx(60.0)
    assert movement["open_ev"] == pytest.approx(0.50)
    assert movement["close_ev"] == pytest.approx(0.58)


# --------------------------------------------------------------------------
# legacy single-point fallback
# --------------------------------------------------------------------------


def test_legacy_single_point_row_visible_to_at_queries_after_midnight(archive):
    """Mimics what ``add_observed_at_to_archive`` produces for old rows."""
    d = date(2026, 5, 8)
    midnight = dt(2026, 5, 8, 0, 0, 0)
    _insert_odds(
        archive,
        league="WNBA",
        market="PTS",
        d=d,
        entity="P",
        book="pinnacle",
        ev=0.55,
        observed_at=midnight,
    )

    # Any at >= midnight returns the legacy observation.
    assert archive.get_ev(
        "WNBA", "PTS", "2026-05-08", "P", at=midnight
    ) == pytest.approx(0.55)
    assert archive.get_ev(
        "WNBA", "PTS", "2026-05-08", "P", at=midnight + timedelta(hours=12)
    ) == pytest.approx(0.55)
    # Default at=None still resolves it.
    assert archive.get_ev("WNBA", "PTS", "2026-05-08", "P") == pytest.approx(0.55)


def test_two_polls_accrue_distinct_observations(archive):
    """Append-only writes preserve every poll, not just the latest."""
    archive.merge_player_books(
        "WNBA",
        "PTS",
        "2026-05-08",
        "P",
        {"pinnacle": 0.50},
    )
    archive.write()
    archive.merge_player_books(
        "WNBA",
        "PTS",
        "2026-05-08",
        "P",
        {"pinnacle": 0.55},
    )
    archive.write()

    history = archive.get_ev_history("WNBA", "PTS", "2026-05-08", "P")
    assert len(history) == 2
    assert history["observed_at"].is_monotonic_increasing
    # Latest reader picks the last observation.
    assert archive.get_ev("WNBA", "PTS", "2026-05-08", "P") == pytest.approx(0.55)


def test_set_team_books_is_append_only(archive):
    """set_team_books no longer wipes prior rows; both observations survive."""
    archive.set_team_books(
        "MLB", "Moneyline", "2026-05-08", "NYY", {"pinnacle": 0.62}
    )
    archive.write()
    archive.set_team_books(
        "MLB", "Moneyline", "2026-05-08", "NYY", {"pinnacle": 0.65}
    )
    archive.write()

    history = archive.get_ev_history("MLB", "Moneyline", "2026-05-08", "NYY")
    assert len(history) == 2
    assert archive.get_moneyline("MLB", "2026-05-08", "NYY") == pytest.approx(0.65)


def test_archive_auto_migrates_pre_observed_at_schema(tmp_path, monkeypatch):
    """Opening a pre-rework DB adds observed_at and backfills it in place.

    Old rows are stamped to ``game_date`` midnight; rows with the prior
    ``sample_ts`` column carry over their real timestamp instead.
    """
    import duckdb

    db_path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE odds ("
        "league TEXT NOT NULL, market TEXT NOT NULL, game_date DATE NOT NULL, "
        "entity TEXT NOT NULL, book TEXT NOT NULL, ev DOUBLE, sample_ts TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE lines ("
        "league TEXT NOT NULL, market TEXT NOT NULL, game_date DATE NOT NULL, "
        "entity TEXT NOT NULL, line DOUBLE NOT NULL)"
    )
    con.execute(
        "INSERT INTO odds VALUES "
        "('WNBA', 'PTS', DATE '2026-05-08', 'P', 'pinnacle', 0.55, NULL), "
        "('WNBA', 'PTS', DATE '2026-05-08', 'P', 'fanduel', 0.52, "
        "  TIMESTAMP '2026-05-08 12:00:00')"
    )
    con.execute(
        "INSERT INTO lines VALUES ('WNBA', 'PTS', DATE '2026-05-08', 'P', 22.5)"
    )
    con.close()

    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(db_path))
    if Archive._instance is not None:
        with __import__("contextlib").suppress(Exception):
            Archive._instance._connection.close()
        Archive._instance._initialized = False

    a = Archive()
    try:
        cols = a._table_columns("odds")
        assert "observed_at" in cols
        assert "sample_ts" not in cols, "auto-migration should drop the old column"

        rows_by_book = {
            r[0]: r[1]
            for r in a._connection.execute(
                "SELECT book, observed_at FROM odds ORDER BY book"
            ).fetchall()
        }
        # fanduel had a real sample_ts → carried over verbatim.
        # pinnacle was NULL → backfilled to game_date midnight.
        assert rows_by_book["fanduel"] == dt(2026, 5, 8, 12, 0, 0)
        assert rows_by_book["pinnacle"] == dt(2026, 5, 8, 0, 0, 0)

        line_rows = a._connection.execute("SELECT observed_at FROM lines").fetchall()
        assert line_rows[0][0] == dt(2026, 5, 8, 0, 0, 0)
    finally:
        with __import__("contextlib").suppress(Exception):
            a._connection.close()
        Archive._instance._initialized = False


# Silence unused-import warnings — datetime is referenced via the dt alias.
_ = datetime
