"""WS1 shape-free storage + WS2 read re-encode for the ``odds`` table.

**Storage (WS1):** the table carries the book's ``(under_prob, line)`` quote alongside the
legacy ``ev``, and every write/backfill stays an exact inverse of ``get_ev``. Those pins are
calibration-*invariant*: they assert the round trip ``get_ev(line, stored_under, shape) ==
stored_ev`` rather than any frozen EV, and that un-invertible rows (``ev`` beyond
``SN_MAX_MEAN_FACTOR*line``) and team markets stay NULL so readers fall back to ``ev``.

**Read re-encode (WS2):** reads rebuild ``ev`` from the stored ``(under_prob, line)`` at the
per-cell fitted SkewNormal shape, **gated on a fitted ``book_shape``** — unfitted cells (every
cell today) read the stored ``ev`` bit-identically, so the wiring is behavior-preserving until
a retrain populates the coefficients.
"""

from __future__ import annotations

import contextlib
import datetime

import duckdb
import numpy as np
import pytest

from sportstradamus.helpers import config
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


def test_confer_prop_write_keeps_shapefree_quote(archive):
    """Durability guard: the live confer prop-write path must keep the shape-free scheme.

    Runs a synthetic Odds-API event through the real parser (``_archive_event_props`` ->
    ``merge_player_books``) into a real archive, flushes, and asserts every player-prop row
    persists a ``(under_prob, line)`` quote that round-trips to its stored ``ev``. The
    migration is only durable if new odds keep landing in this shape — were the write path to
    regress to ev-only, migrated cells would silently erode. One row per (player, book): each
    book stores its OWN line (LeBron is priced at 25.5 on DK, 26.5 on FD). Team markets stay
    ev-only by design (the migration skips ``_TEAM_ONLY_MARKETS``) and are not asserted here.
    """
    from sportstradamus import moneylines

    props = {"NBA": {"player_points": "PTS"}}
    game = {
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"description": "LeBron James", "name": "Over", "point": 25.5, "price": 1.91},
                            {"description": "LeBron James", "name": "Under", "point": 25.5, "price": 1.95},
                            {"description": "Anthony Davis", "name": "Over", "point": 18.5, "price": 1.83},
                            {"description": "Anthony Davis", "name": "Under", "point": 18.5, "price": 2.05},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"description": "LeBron James", "name": "Over", "point": 26.5, "price": 2.0},
                            {"description": "LeBron James", "name": "Under", "point": 26.5, "price": 1.87},
                        ],
                    }
                ],
            },
        ],
    }

    moneylines._archive_event_props(archive, game, "NBA", props, "2026-06-04")
    archive.write()

    dist, cv, gate = _shape("NBA", "PTS")
    rows = archive._connection.execute(
        "SELECT entity, book, ev, under_prob, line FROM odds WHERE league='NBA' AND market='PTS'"
    ).fetchall()

    assert len(rows) == 3  # Davis@DK, LeBron@DK, LeBron@FD
    for ent, book, ev, under_prob, line in rows:
        assert under_prob is not None and line is not None, f"{ent}/{book} wrote ev-only"
        assert 0.0 <= under_prob <= 1.0 and line > 0
        assert get_ev(line, under_prob, cv, dist=dist, gate=gate) == pytest.approx(ev, abs=1e-6)


# --- WS2 read re-encode -------------------------------------------------------

_PLANTED_SHAPE = {"a": 1.3, "b": 1.1, "skew_c": 0.6, "skew_d": -0.1, "n_bins": 9}


def _insert_book_row(archive, league, market, entity, book, ev, under_prob, line):
    archive._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, "
        "under_prob, line) VALUES (?, ?, DATE '2026-05-08', ?, ?, ?, ?, ?, ?)",
        [league, market, entity, book, float(ev), _TS, float(under_prob), float(line)],
    )


def _expected_reencode(league, market, under, line):
    sigma, skew = config.book_skewnormal_shape(league, market, line)
    return get_ev(
        line, under, dist="SkewNormal", sigma=float(sigma), skew_alpha=float(skew),
        gate=book_gate(league, market, "SkewNormal"),
    )


def test_read_reencode_noop_for_unfitted_cell(archive):
    """No fitted book_shape -> get_ev returns the stored ev bit-identically."""
    league, market = "WNBA", "AST"
    assert stat_dist[league][market] == "SkewNormal"
    assert config.stat_meta[league][market].get("book_shape") is None
    _insert_book_row(archive, league, market, "P", "pinnacle", 2.4, 0.55, 1.5)
    assert archive.get_ev(league, market, "2026-05-08", "P") == 2.4


def test_read_reencode_uses_fitted_shape(archive, monkeypatch):
    """A fitted book_shape -> get_ev re-inverts (under_prob, line) at sigma(line)/skew(line),
    ignoring the stored ev (here deliberately wrong)."""
    league, market = "WNBA", "AST"
    under, line = 0.62, 1.5
    monkeypatch.setitem(config.stat_meta[league][market], "book_shape", _PLANTED_SHAPE)
    _insert_book_row(archive, league, market, "P", "pinnacle", 5.0, under, line)

    ev = archive.get_ev(league, market, "2026-05-08", "P")

    assert ev == pytest.approx(_expected_reencode(league, market, under, line), abs=1e-9)
    sigma, skew = config.book_skewnormal_shape(league, market, line)
    assert get_odds(line, ev, "SkewNormal", sigma=float(sigma), skew_alpha=float(skew)) == (
        pytest.approx(under, abs=1e-7)
    )


def test_to_pandas_reencode_gated(archive, monkeypatch):
    """to_pandas re-encodes per-book ev for a fitted cell, leaves an unfitted cell identical."""
    league, market = "WNBA", "AST"
    under, line = 0.62, 1.5
    _insert_book_row(archive, league, market, "P", "pinnacle", 5.0, under, line)
    archive._connection.execute(
        "INSERT INTO lines (league, market, game_date, entity, line, observed_at) "
        "VALUES (?, ?, DATE '2026-05-08', 'P', ?, ?)",
        [league, market, line, _TS],
    )

    df0 = archive.to_pandas(league, market)
    assert float(df0.loc[("2026-05-08", "P"), "pinnacle"]) == 5.0

    monkeypatch.setitem(config.stat_meta[league][market], "book_shape", _PLANTED_SHAPE)
    df1 = archive.to_pandas(league, market)
    assert float(df1.loc[("2026-05-08", "P"), "pinnacle"]) == pytest.approx(
        _expected_reencode(league, market, under, line), abs=1e-9
    )
