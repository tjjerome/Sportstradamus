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

import pandas as pd
import pytest

from sportstradamus.helpers.archive import (
    GAME_LINE_TRAINING_CUTOFF,
    Archive,
    _resolve_market,
    archive_market,
)
from sportstradamus.helpers.config import stat_map
from sportstradamus.stats import base


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


def _insert_odds(archive, *, league, market, d, entity, book, ev, observed_at, line=None):
    archive._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, line) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [league, market, d, entity, book, float(ev), observed_at, line],
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


def test_archive_market_applies_the_league_fixups_and_no_second_alias_pass():
    """A reader holding a board label asks under the key the writer filed the row at."""
    assert archive_market("NHL", "AST") == "assists"
    assert archive_market("WNBA", "fantasy points underdog") == "fantasy points prizepicks"
    assert archive_market("MLB", "walks") == "walks"

    # The writer aliases a raw platform label first. Sleeper's alias is not idempotent,
    # so a label that has been through it once must not go through it again.
    assert _resolve_market("MLB", "bat_walks", stat_map["Sleeper"]) == "walks"
    assert _resolve_market("MLB", "walks", stat_map["Sleeper"]) == "walks allowed"


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


def test_reference_line_at_aggregates_distinct_logged_lines_through_cutoff(archive):
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

    # No sportsbook posted a line, so there is no consensus and the log is the reference.
    assert archive.get_line("WNBA", "PTS", "2026-05-08", "P") == 0.0
    # All lines visible at end → median = 23.0 → floor(46)/2 = 23.0.
    assert archive.get_reference_line("WNBA", "PTS", "2026-05-08", "P") == pytest.approx(23.0)
    # Only first two lines visible at +1h cutoff → median = 22.5 → floor(45)/2 = 22.5.
    assert archive.get_reference_line(
        "WNBA", "PTS", "2026-05-08", "P", at=base + timedelta(hours=1)
    ) == pytest.approx(22.5)
    # Cutoff before any observation → empty result → 0.
    assert (
        archive.get_reference_line("WNBA", "PTS", "2026-05-08", "P", at=base - timedelta(hours=1))
        == 0
    )


_PTS_KEY = ("WNBA", "PTS", "2026-05-08", "P")
_NOON = dt(2026, 5, 8, 12, 0, 0)


def _post_line(archive, book, line, observed_at=_NOON):
    """One ``odds`` row on ``_PTS_KEY``: ``book``'s posted ``line`` (NULL when ``None``)."""
    _insert_odds(
        archive,
        league="WNBA",
        market="PTS",
        d=date(2026, 5, 8),
        entity="P",
        book=book,
        ev=20.0,
        observed_at=observed_at,
        line=line,
    )


def _log_line(archive, line, observed_at=_NOON):
    _insert_line(
        archive,
        league="WNBA",
        market="PTS",
        d=date(2026, 5, 8),
        entity="P",
        line=line,
        observed_at=observed_at,
    )


def test_get_line_ignores_a_dfs_platform_line(archive):
    """The log holds the Underdog rung beside the sportsbooks' median; only the books count."""
    _post_line(archive, "fanduel", 22.5)
    _post_line(archive, "draftkings", 23.5)
    _post_line(archive, "Underdog", 30.5)
    _log_line(archive, 23.0)
    _log_line(archive, 30.5)

    assert archive.get_line(*_PTS_KEY) == 23.0


def test_dfs_only_entry_has_no_consensus_line_and_keeps_its_line_of_record(archive):
    _post_line(archive, "Underdog", 30.5)
    _log_line(archive, 30.5)

    assert archive.get_line(*_PTS_KEY) == 0.0
    assert archive.get_reference_line(*_PTS_KEY) == 30.5


def test_get_line_counts_each_sportsbook_once_at_its_latest_line(archive):
    moved = _NOON + timedelta(hours=2)
    _post_line(archive, "fanduel", 22.5)
    _post_line(archive, "draftkings", 23.5)
    _post_line(archive, "fanduel", 24.5, observed_at=moved)
    # Each poll logs the books' median: 23.0 at noon, 24.0 once fanduel moved.
    _log_line(archive, 23.0)
    _log_line(archive, 24.0, observed_at=moved)

    # fanduel counts at 24.5 alone; the median of every line seen would be 23.5.
    assert archive.get_line(*_PTS_KEY) == 24.0
    assert archive.get_line(*_PTS_KEY, at=_NOON + timedelta(hours=1)) == 23.0
    assert archive.get_line(*_PTS_KEY, at=_NOON - timedelta(hours=1)) == 0.0


def test_reference_line_reads_the_log_for_a_sportsbook_row_archived_without_its_line(archive):
    """The 2023-24 MLB shape: a sportsbook ``ev`` row whose line survives only in ``lines``."""
    _post_line(archive, "fanduel", None)
    _log_line(archive, 0.5)

    assert archive.get_line(*_PTS_KEY) == 0.0
    assert archive.get_reference_line(*_PTS_KEY) == 0.5


def _seed_game_lines(archive):
    """One book's MLB moneylines, each stamped on its own game date."""
    for team, stamp, ev in [
        ("NYY", dt(2026, 5, 8, 13, 30), 0.55),
        ("NYY", dt(2026, 5, 8, 23, 40), 0.80),
        ("BOS", dt(2026, 5, 8, 23, 40), 0.20),
        ("CHC", dt(2026, 5, 8, 0, 0), 0.48),
        ("NYY", dt(2026, 5, 9, 13, 30), 0.60),
    ]:
        _insert_odds(
            archive,
            league="MLB",
            market="Moneyline",
            d=stamp.date(),
            entity=team,
            book="pinnacle",
            ev=ev,
            observed_at=stamp,
        )


def test_team_market_map_stops_at_the_cutoff_on_each_game_date(archive):
    """Each key reads its newest quote at or before 15:00 UTC on its own game date."""
    _seed_game_lines(archive)

    lines = archive.get_team_market_map("MLB", "Moneyline", cutoff=GAME_LINE_TRAINING_CUTOFF)

    # BOS was quoted only during its game, so it gets no key. NYY's 23:40 quote on the
    # 8th falls before the 9th's cutoff: one instant for the whole query would either
    # admit it or drop the 9th's own 13:30 quote.
    assert lines == pytest.approx(
        {("2026-05-08", "NYY"): 0.55, ("2026-05-08", "CHC"): 0.48, ("2026-05-09", "NYY"): 0.60}
    )


def test_enrich_team_markets_writes_the_pre_game_line(archive, monkeypatch):
    """A quote taken while the game was being played never reaches the gamelog."""
    _seed_game_lines(archive)
    monkeypatch.setattr(base, "archive", archive)
    stats = base.Stats()
    stats.league = "MLB"
    games = pd.DataFrame({"gameDate": "2026-05-08", "team": ["NYY", "BOS", "CHC"]})

    stats._enrich_team_markets(games, date_col="gameDate", team_col="team")

    # NYY's 13:30 quote, the no-quote default for BOS, CHC's midnight placeholder.
    assert games["moneyline"].tolist() == pytest.approx([0.55, 0.5, 0.48])


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
    assert archive.get_ev("WNBA", "PTS", "2026-05-08", "P", at=midnight) == pytest.approx(0.55)
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
    archive.set_team_books("MLB", "Moneyline", "2026-05-08", "NYY", {"pinnacle": 0.62})
    archive.write()
    archive.set_team_books("MLB", "Moneyline", "2026-05-08", "NYY", {"pinnacle": 0.65})
    archive.write()

    history = archive.get_ev_history("MLB", "Moneyline", "2026-05-08", "NYY")
    assert len(history) == 2
    assert archive.get_moneyline("MLB", "2026-05-08", "NYY") == pytest.approx(0.65)


def test_set_team_books_keeps_a_given_stamp(archive):
    """A backfill's snapshot time survives the write, so the cutoff read sees the row."""
    snapshot = dt(2025, 6, 1, 6, 0)
    archive.set_team_books(
        "MLB", "Moneyline", "2025-06-01", "NYY", {"pinnacle": 0.62}, observed_at=snapshot
    )
    archive.write()

    assert archive._connection.execute("SELECT observed_at FROM odds").fetchall() == [(snapshot,)]
    lines = archive.get_team_market_map("MLB", "Moneyline", cutoff=GAME_LINE_TRAINING_CUTOFF)
    assert lines == pytest.approx({("2025-06-01", "NYY"): 0.62})


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
    con.execute("INSERT INTO lines VALUES ('WNBA', 'PTS', DATE '2026-05-08', 'P', 22.5)")
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


# --------------------------------------------------------------------------
# ladder capture (WS1 — alt-line rungs)
# --------------------------------------------------------------------------


def test_add_ladder_round_trips_all_rungs(archive):
    archive.add_ladder(
        "NBA",
        "PTS",
        "2026-05-08",
        "Nikola Jokic",
        "draftkings",
        [(19.5, 0.82), (24.5, 0.55), (29.5, 0.28)],
    )
    archive.write()

    rows = archive._connection.execute(
        "SELECT book, line, p_over FROM ladder ORDER BY line"
    ).fetchall()
    assert rows == [
        ("draftkings", 19.5, 0.82),
        ("draftkings", 24.5, 0.55),
        ("draftkings", 29.5, 0.28),
    ]


# Silence unused-import warnings — datetime is referenced via the dt alias.
_ = datetime
