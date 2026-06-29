"""WS1 shape-free storage: the ``odds`` table carries the book's ``(under_prob, line)``
quote alongside the legacy ``ev``, and every write/backfill stays an exact inverse of
``get_ev``.

These pin the storage layer, not the read path — WS1 leaves reads on ``ev`` untouched, so
the invariants here are calibration-*invariant*: they assert the round trip
``get_ev(line, stored_under, shape) == stored_ev`` (true for whatever ``cv``/``gate`` are
current) rather than any frozen EV, and that un-invertible rows (stored ``ev`` beyond
``get_ev``'s ``SN_MAX_MEAN_FACTOR*line`` cap) and team markets are left NULL so readers
fall back to ``ev``.
"""

from __future__ import annotations

import contextlib
import datetime

import duckdb
import numpy as np
import pytest

from sportstradamus.helpers.archive import Archive
from sportstradamus.helpers.config import book_gate, stat_cv, stat_dist
from sportstradamus.helpers.distributions import SN_MAX_MEAN_FACTOR, get_ev, get_odds
from sportstradamus.scripts.migrate_archive_shapefree import _migrate_cell

_TS = datetime.datetime(2026, 5, 8, 12, 0, 0)
# One cell per production family; both are real configured cells (the Cardoso market is
# WNBA/AST). Skipped gracefully if a future config drops one.
_CELLS = [("WNBA", "AST"), ("NBA", "FG3M")]


@pytest.fixture
def archive(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.duckdb"
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(db_path))
    if Archive._instance is not None:
        with contextlib.suppress(Exception):
            Archive._instance._connection.close()
        Archive._instance._initialized = False
    a = Archive()
    yield a
    with contextlib.suppress(Exception):
        a._connection.close()
    Archive._instance._initialized = False


def _shape(league, market):
    dist = stat_dist[league][market]
    return dist, stat_cv[league].get(market, 1), book_gate(league, market, dist)


def test_schema_has_shapefree_columns(archive):
    cols = archive._table_columns("odds")
    assert {"under_prob", "line"} <= cols


def test_auto_migrate_adds_columns_preserving_ev(tmp_path, monkeypatch):
    """Opening a pre-WS1 (7-column) DB heals it: columns added, ev untouched, under NULL."""
    db = tmp_path / "old.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE odds (league TEXT, market TEXT, game_date DATE, entity TEXT, "
        "book TEXT, ev DOUBLE, observed_at TIMESTAMP); "
        "CREATE TABLE lines (league TEXT, market TEXT, game_date DATE, entity TEXT, "
        "line DOUBLE, observed_at TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at) "
        "VALUES ('WNBA', 'AST', DATE '2026-05-08', 'P', 'pinnacle', 2.4, ?)",
        [_TS],
    )
    con.close()

    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(db))
    if Archive._instance is not None:
        with contextlib.suppress(Exception):
            Archive._instance._connection.close()
        Archive._instance._initialized = False
    a = Archive()
    try:
        assert {"under_prob", "line"} <= a._table_columns("odds")
        row = a._connection.execute("SELECT ev, under_prob, line FROM odds").fetchone()
        assert row == (2.4, None, None)
    finally:
        a._connection.close()
        Archive._instance._initialized = False


@pytest.mark.parametrize("league,market", _CELLS)
def test_add_dfs_write_round_trips(archive, league, market):
    """``add_dfs`` stores ``(under_prob, line)`` that re-encode to the stored ``ev``."""
    dist, cv, gate = _shape(league, market)
    offer = {
        "League": league, "Market": market, "Player": "Test Player",
        "Date": "2026-05-08", "Line": 3.5, "Boost": 1.0, "Boost_Over": 1.0, "Boost_Under": 1.0,
    }
    archive.add_dfs([offer], "Underdog", {})
    archive.write()
    ev, under, line = archive._connection.execute(
        "SELECT ev, under_prob, line FROM odds WHERE entity='Test Player'"
    ).fetchone()
    assert under is not None and line == 3.5
    assert get_ev(line, under, cv, dist=dist, gate=gate) == pytest.approx(ev, abs=1e-6)


def test_merge_player_books_quotes_optional(archive):
    """Per-book ``book_quotes`` populate ``(under_prob, line)``; omitting them leaves NULL."""
    archive.merge_player_books(
        "WNBA", "AST", "2026-05-08", "Quoted", {"fanduel": 2.6},
        book_quotes={"fanduel": (3.5, 0.55)},
    )
    archive.merge_player_books("WNBA", "AST", "2026-05-08", "Bare", {"fanduel": 2.6})
    archive.write()
    quoted = archive._connection.execute(
        "SELECT under_prob, line FROM odds WHERE entity='Quoted'"
    ).fetchone()
    bare = archive._connection.execute(
        "SELECT under_prob, line FROM odds WHERE entity='Bare'"
    ).fetchone()
    assert quoted == (0.55, 3.5)
    assert bare == (None, None)


@pytest.mark.parametrize("league,market", _CELLS)
def test_migration_backfills_and_preserves_ev(archive, league, market):
    """The backfill recovers the true under-prob, leaves ``ev`` untouched, and drops only
    un-invertible (cap-exceeding) rows to NULL."""
    dist, cv, gate = _shape(league, market)
    line = 3.5
    known_unders = [0.45, 0.6, 0.75]
    evs = [get_ev(line, u, cv, dist=dist, gate=gate) for u in known_unders]
    over_cap_ev = SN_MAX_MEAN_FACTOR * line + 2.0  # no faithful devig -> must stay NULL
    rows = [(f"P{i}", ev) for i, ev in enumerate(evs)] + [("Pcap", over_cap_ev)]
    for ent, ev in rows:
        archive._connection.execute(
            "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at) "
            "VALUES (?, ?, DATE '2026-05-08', ?, 'pinnacle', ?, ?)",
            [league, market, ent, float(ev), _TS],
        )
        archive._connection.execute(
            "INSERT INTO lines (league, market, game_date, entity, line, observed_at) "
            "VALUES (?, ?, DATE '2026-05-08', ?, ?, ?)",
            [league, market, ent, line, _TS],
        )

    n, err, dropped = _migrate_cell(archive._connection, league, market)
    assert n == len(known_unders) and dropped == 1 and err <= 1e-6

    for (ent, ev), u0 in zip(rows, known_unders, strict=False):  # cap row has no under to check
        stored_ev, u, ln = archive._connection.execute(
            "SELECT ev, under_prob, line FROM odds WHERE entity=?", [ent]
        ).fetchone()
        assert stored_ev == pytest.approx(ev)  # ev never rewritten
        assert u == pytest.approx(u0, abs=1e-6) and ln == line
    cap_row = archive._connection.execute(
        "SELECT under_prob FROM odds WHERE entity='Pcap'"
    ).fetchone()
    assert cap_row == (None,)

    # Idempotent in effect: a second pass writes nothing (the cap row stays NULL and is
    # harmlessly re-examined, never re-filled).
    n2, err2, _ = _migrate_cell(archive._connection, league, market)
    assert n2 == 0 and err2 == 0.0


def test_migration_recovers_floored_cv_cell(archive, monkeypatch):
    """A SkewNormal cell whose live cv was floored (the MLB doubles/hits, NHL goals/assists class
    at ``cv==0.02``) recovers the true quote. Crucially this pins the floored-cv *distrust*: even
    though one mean sits at the window edge and DOES round-trip at the degenerate floor (the MLB
    doubles partial-success that would otherwise migrate ~0% of the cell at the spike), the floored
    cv is never accepted — the decode falls through to the std-recovered cv (``std / median ev``)
    and recovers every row. ``std == cv*mean`` by construction, so the recovery is exact.
    """
    from sportstradamus.scripts import migrate_archive_shapefree as mig

    league, market = "WNBA", "AST"
    dist = stat_dist[league][market]
    assert dist == "SkewNormal"
    gate = book_gate(league, market, dist)
    line = 1.5
    cv_true = 0.6  # the real, wide cv the ev was encoded at, before the floor
    # 0.40 -> mean ~1.8 sits near the window edge and round-trips at the floored cv=0.02 (the
    # doubles partial-success); 0.50/0.65 collapse to the spike. Without the floored-cv distrust
    # the backfill would accept the 0.02 decode and migrate just the edge row at a wrong ~0.5.
    known_unders = [0.40, 0.50, 0.65]
    evs = [get_ev(line, u, cv_true, dist=dist, gate=gate) for u in known_unders]

    # Live config after the floor: cv clamped to 0.02, std still sane. Pick std so
    # std/median(ev) == cv_true exactly, mirroring the real cells (cv == std/mean).
    monkeypatch.setitem(mig.stat_cv[league], market, 0.02)
    monkeypatch.setitem(mig.stat_std.setdefault(league, {}), market, cv_true * float(np.median(evs)))

    for i, ev in enumerate(evs):
        archive._connection.execute(
            "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at) "
            "VALUES (?, ?, DATE '2026-05-08', ?, 'pinnacle', ?, ?)",
            [league, market, f"F{i}", float(ev), _TS],
        )
        archive._connection.execute(
            "INSERT INTO lines (league, market, game_date, entity, line, observed_at) "
            "VALUES (?, ?, DATE '2026-05-08', ?, ?, ?)",
            [league, market, f"F{i}", line, _TS],
        )

    n, err, dropped = _migrate_cell(archive._connection, league, market)
    assert n == len(known_unders) and dropped == 0 and err <= 1e-6

    for i, u0 in enumerate(known_unders):
        stored_ev, u, ln = archive._connection.execute(
            "SELECT ev, under_prob, line FROM odds WHERE entity=?", [f"F{i}"]
        ).fetchone()
        assert u == pytest.approx(u0, abs=1e-6) and ln == line
        assert stored_ev == pytest.approx(evs[i])  # ev never rewritten
