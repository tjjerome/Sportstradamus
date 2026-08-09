"""WS1 shape-free storage + WS2 read re-encode for the ``odds`` table.

**Storage (WS1):** the table carries the book's ``(under_prob, line)`` quote alongside the
legacy ``ev``, and every write/backfill stays an exact inverse of ``get_ev``. Those pins are
calibration-*invariant*: they assert the round trip ``get_ev(line, stored_under, shape) ==
stored_ev`` rather than any frozen EV, and that un-invertible rows (``ev`` beyond
``SN_MAX_MEAN_FACTOR*line``) and team markets stay NULL so readers fall back to ``ev``.

**Reads (WS2):** ``get_ev`` / ``to_pandas`` return the stored (symmetric-devigged) ``ev``
regardless of a fitted ``book_shape`` — the settling verdict routed the shaped book to the
book-only prediction leg, not the archive. The stored ``(under_prob, line)`` columns instead
back :meth:`Archive.get_composite_under_prob`, the book-weighted consensus de-vigged
under-probability the WS2 2d training feature reads shape-free.
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
from sportstradamus.helpers.training_quotes import resolve_training_quote
from sportstradamus.scripts.migrate_archive_shapefree import _migrate_cell

_TS = datetime.datetime(2026, 5, 8, 12, 0, 0)


# The fixture under-probs reach down to 0.45; a count cell whose structural-zero
# floor sits at or above that (home-runs-class zero rates) is un-invertible by
# construction, so the discovered cell must carry a smaller gate.
_MAX_COUNT_GATE = 0.4


def _first_cell(family, max_gate=None):
    for lg in sorted(stat_dist):
        for mkt in sorted(stat_dist[lg]):
            if stat_dist[lg][mkt] != family:
                continue
            if max_gate is not None and (book_gate(lg, mkt, family) or 0) >= max_gate:
                continue
            return lg, mkt
    return None


# One live cell per decode family class the shape-free scheme round-trips
# (continuous SkewNormal + gated count), discovered from the live config so a
# strategy sweep that reships a pinned cell under a new family can't rot these
# tests. Skipped gracefully if a future config drops a family entirely.
_SN_CELL = _first_cell("SkewNormal")
_COUNT_CELL = _first_cell("ZINB", max_gate=_MAX_COUNT_GATE)
_CELLS = [c for c in (_SN_CELL, _COUNT_CELL) if c]


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


def test_latest_book_rows_have_stable_book_order(archive):
    """Weighted reductions must not inherit DuckDB's unspecified row order."""
    for book, ev in (("zeta", 2.2), ("alpha", 1.8), ("middle", 2.0)):
        archive._connection.execute(
            "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at) "
            "VALUES ('WNBA', 'AST', DATE '2026-05-08', 'P', ?, ?, ?)",
            [book, ev, _TS],
        )

    rows = archive._book_rows("WNBA", "AST", "2026-05-08", "P")

    assert rows == [("alpha", 1.8), ("middle", 2.0), ("zeta", 2.2)]


def test_training_book_quotes_keep_latest_row_fields_coherent(archive):
    later = _TS + datetime.timedelta(hours=1)
    archive._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, "
        "under_prob, line) VALUES "
        "('WNBA', 'AST', DATE '2026-05-08', 'P', 'zeta', 2.0, ?, 0.40, 1.5), "
        "('WNBA', 'AST', DATE '2026-05-08', 'P', 'zeta', 3.0, ?, 0.60, 2.5), "
        "('WNBA', 'AST', DATE '2026-05-08', 'P', 'alpha', 2.4, ?, 0.55, 1.5)",
        [_TS, later, later],
    )

    rows = archive.get_training_book_quotes("WNBA", "AST", "2026-05-08", "P")

    assert [row.book for row in rows] == ["alpha", "zeta"]
    assert (rows[1].ev, rows[1].under_probability, rows[1].line, rows[1].observed_at) == (
        3.0,
        0.60,
        2.5,
        later,
    )


def test_training_quote_batch_matches_scalar_rows_and_lines(archive):
    later = _TS + datetime.timedelta(hours=1)
    archive._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, "
        "under_prob, line) VALUES "
        "('WNBA', 'AST', DATE '2026-05-08', 'P', 'zeta', 2.0, ?, 0.40, 1.5), "
        "('WNBA', 'AST', DATE '2026-05-08', 'P', 'zeta', 3.0, ?, 0.60, 2.5), "
        "('WNBA', 'AST', DATE '2026-05-08', 'Q', 'alpha', 2.4, ?, 0.55, 1.5)",
        [_TS, later, later],
    )
    archive._connection.execute(
        "INSERT INTO lines (league, market, game_date, entity, line, observed_at) VALUES "
        "('WNBA', 'AST', DATE '2026-05-08', 'P', 2.5, ?), "
        "('WNBA', 'AST', DATE '2026-05-08', 'Q', 1.5, ?)",
        [later, later],
    )

    batch = archive.get_training_quote_inputs("WNBA", "AST", "2026-05-08", ["Q", "P", "Missing"])

    for entity in ("P", "Q", "Missing"):
        rows, line = batch[entity]
        assert rows == archive.get_training_book_quotes("WNBA", "AST", "2026-05-08", entity)
        assert line == archive.get_line("WNBA", "AST", "2026-05-08", entity)


def test_unpriced_pickem_line_resolves_as_the_platforms_own_symmetric_quote(archive):
    """DFS priced rows begin only in 2026, so pre-2026 pick'em entries archived a bare
    line. That line is the platform's own 50/50 quote, not an absent book — but only
    where nothing priced the entry, and only where the market names the platform."""
    archive._connection.execute(
        "INSERT INTO lines (league, market, game_date, entity, line, observed_at) VALUES "
        "('NFL', 'fantasy points underdog', DATE '2026-05-08', 'P', 11.5, ?), "
        "('NFL', 'fantasy points underdog', DATE '2026-05-08', 'Priced', 9.5, ?), "
        "('NFL', 'receiving yards', DATE '2026-05-08', 'P', 46.5, ?)",
        [_TS, _TS, _TS],
    )
    archive._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, "
        "under_prob, line) VALUES "
        "('NFL', 'fantasy points underdog', DATE '2026-05-08', 'Priced', 'Underdog', "
        "10.2, ?, 0.42, 9.5)",
        [_TS],
    )

    def resolve(market, entity):
        rows, legacy = archive.get_training_quote_inputs("NFL", market, "2026-05-08", [entity])[
            entity
        ]
        return rows, resolve_training_quote(
            rows, legacy_line=legacy, fallback_line=5.0, fallback_ev=None, dist="Gamma", cv=1.0
        )

    rows, quote = resolve("fantasy points underdog", "P")
    assert [row.book for row in rows] == ["Underdog"]
    assert (quote.line, quote.over_probability, quote.source) == (11.5, 0.5, "book_direct")
    assert (quote.authenticity, quote.observed_at) == ("authentic", _TS)

    # A real quote outranks the stand-in — a boost prices away from 50/50 and must survive.
    # Its mean is re-derived at the resolving cell's cv/dist, so it need not match the 10.2
    # the boost encoded under whatever config was live then.
    _, priced = resolve("fantasy points underdog", "Priced")
    assert (priced.over_probability, priced.ev) == (
        pytest.approx(0.58),
        pytest.approx(17.4268, abs=1e-4),
    )

    # A sportsbook market's bare line is unattributable: `lines` has no book column.
    rows, unattributed = resolve("receiving yards", "P")
    assert rows == []
    assert (unattributed.source, unattributed.authenticity) == ("neutral_fallback", "synthetic")


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
        "League": league,
        "Market": market,
        "Player": "Test Player",
        "Date": "2026-05-08",
        "Line": 3.5,
        "Boost": 1.0,
        "Boost_Over": 1.0,
        "Boost_Under": 1.0,
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
        "WNBA",
        "AST",
        "2026-05-08",
        "Quoted",
        {"fanduel": 2.6},
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


@pytest.mark.skipif(_SN_CELL is None, reason="no SkewNormal cell in live config")
def test_migration_recovers_floored_cv_cell(archive, monkeypatch):
    """A SkewNormal cell whose live cv was floored (the MLB doubles/hits, NHL goals/assists class
    at ``cv==0.02``) recovers the true quote. Crucially this pins the floored-cv *distrust*: even
    though one mean sits at the window edge and DOES round-trip at the degenerate floor (the MLB
    doubles partial-success that would otherwise migrate ~0% of the cell at the spike), the floored
    cv is never accepted — the decode falls through to the std-recovered cv (``std / median ev``)
    and recovers every row. ``std == cv*mean`` by construction, so the recovery is exact.
    """
    from sportstradamus.scripts import migrate_archive_shapefree as mig

    league, market = _SN_CELL
    dist = "SkewNormal"
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
    monkeypatch.setitem(
        mig.stat_std.setdefault(league, {}), market, cv_true * float(np.median(evs))
    )

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
                            {
                                "description": "LeBron James",
                                "name": "Over",
                                "point": 25.5,
                                "price": 1.91,
                            },
                            {
                                "description": "LeBron James",
                                "name": "Under",
                                "point": 25.5,
                                "price": 1.95,
                            },
                            {
                                "description": "Anthony Davis",
                                "name": "Over",
                                "point": 18.5,
                                "price": 1.83,
                            },
                            {
                                "description": "Anthony Davis",
                                "name": "Under",
                                "point": 18.5,
                                "price": 2.05,
                            },
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
                            {
                                "description": "LeBron James",
                                "name": "Over",
                                "point": 26.5,
                                "price": 2.0,
                            },
                            {
                                "description": "LeBron James",
                                "name": "Under",
                                "point": 26.5,
                                "price": 1.87,
                            },
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


# --- WS2 read: NO shape re-encode (collapse-to-cv per the settling verdict) ---
# The settling experiment routed the shaped book to the book-only prediction leg
# (model_prob._book_over_prob), not the archive. get_ev / to_pandas return the stored
# (symmetric-devigged) ev regardless of a fitted book_shape; the WS1 (under_prob, line)
# columns stay for the 2d prob-space feature, not a mean re-encode.

_PLANTED_SHAPE = {"a": 1.3, "b": 1.1, "skew_c": 0.6, "skew_d": -0.1, "n_bins": 9}


def _insert_book_row(archive, league, market, entity, book, ev, under_prob, line):
    archive._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, "
        "under_prob, line) VALUES (?, ?, DATE '2026-05-08', ?, ?, ?, ?, ?, ?)",
        [league, market, entity, book, float(ev), _TS, float(under_prob), float(line)],
    )


@pytest.mark.skipif(_SN_CELL is None, reason="no SkewNormal cell in live config")
def test_get_ev_ignores_book_shape(archive, monkeypatch):
    """get_ev returns the stored ev whether or not a book_shape is fitted (no read re-encode)."""
    league, market = _SN_CELL
    _insert_book_row(archive, league, market, "P", "pinnacle", 2.4, 0.55, 1.5)
    assert archive.get_ev(league, market, "2026-05-08", "P") == 2.4

    monkeypatch.setitem(config.stat_meta[league][market], "book_shape", _PLANTED_SHAPE)
    assert archive.get_ev(league, market, "2026-05-08", "P") == 2.4


def test_to_pandas_ignores_book_shape(archive, monkeypatch):
    """to_pandas returns the stored ev whether or not a book_shape is fitted."""
    league, market = "WNBA", "AST"
    _insert_book_row(archive, league, market, "P", "pinnacle", 5.0, 0.62, 1.5)

    df0 = archive.to_pandas(league, market)
    assert float(df0.loc[("2026-05-08", "P"), "pinnacle"]) == 5.0

    monkeypatch.setitem(config.stat_meta[league][market], "book_shape", _PLANTED_SHAPE)
    df1 = archive.to_pandas(league, market)
    assert float(df1.loc[("2026-05-08", "P"), "pinnacle"]) == 5.0


def test_composite_under_prob_weights_books(archive):
    """get_composite_under_prob returns the book-weighted mean of the stored under_prob."""
    league, market = "WNBA", "AST"
    _insert_book_row(archive, league, market, "P", "pinnacle", 2.4, 0.55, 1.5)
    _insert_book_row(archive, league, market, "P", "fanduel", 2.6, 0.65, 1.5)

    weights = config.book_weights.get(league, {}).get(market, {})
    wp, wf = weights.get("pinnacle", 1), weights.get("fanduel", 1)
    expected = (wp * 0.55 + wf * 0.65) / (wp + wf)

    got = archive.get_composite_under_prob(league, market, "2026-05-08", "P")
    assert got == pytest.approx(expected)


def test_composite_under_prob_nan_when_absent(archive):
    """No book quote -> NaN, so the caller falls back to the symmetric get_odds feature."""
    assert np.isnan(archive.get_composite_under_prob("WNBA", "AST", "2026-05-08", "ghost"))


def test_team_market_readers_split_defaulting_from_null_on_a_miss(archive):
    """An unquoted game reads NaN through ``get_team_market``, defaulted through the pair.

    Both semantics are load-bearing and must not converge. ``get_moneyline`` /
    ``get_total`` default because the gamelog builds the same feature columns with the
    same miss values — the model would otherwise see different inputs in training and
    serving. ``get_team_market`` stays NaN so snapshot writers can tell an unquoted
    game from a real pick'em; collapsing that distinction once hid a whole league
    going unfetched behind a plausible-looking 0.5.
    """
    assert np.isnan(archive.get_team_market("MLB", "Moneyline", "2026-05-08", "ATH"))
    assert np.isnan(archive.get_team_market("MLB", "Totals", "2026-05-08", "ATH"))

    assert archive.get_moneyline("MLB", "2026-05-08", "ATH") == 0.5
    assert archive.get_total("MLB", "2026-05-08", "ATH") == archive.default_totals["MLB"]

    _insert_book_row(archive, "MLB", "Moneyline", "ATH", "pinnacle", 0.62, 0.5, 0.0)
    assert archive.get_moneyline("MLB", "2026-05-08", "ATH") == pytest.approx(0.62)
    assert archive.get_team_market("MLB", "Moneyline", "2026-05-08", "ATH") == pytest.approx(0.62)
