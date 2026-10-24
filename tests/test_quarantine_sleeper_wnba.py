"""Unit tests for ``scripts.quarantine_sleeper_wnba`` — the Sleeper WNBA odds repair.

Sleeper's discounted WNBA lines were archived as fair 50/50 quotes from Aug 3,
2026 until the ingestion fix. The quarantine nulls ``ev``/``under_prob`` on
exactly ``league='WNBA' AND book='Sleeper' AND game_date >= 2026-08-03 AND
observed_at <= --until`` while keeping each row's ``line`` and the ``lines``
table. These pin that predicate on every boundary (other book, other league,
pre-window game date, post-``--until`` observation), the dry-run/apply split,
the safety backup, and the --until requirement — on a throwaway DuckDB.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
from click.testing import CliRunner

from sportstradamus.scripts.quarantine_sleeper_wnba import main, quarantine_sleeper_wnba

_ODDS_DDL = (
    "CREATE TABLE odds (league VARCHAR, market VARCHAR, game_date DATE, "
    "entity VARCHAR, book VARCHAR, ev DOUBLE, observed_at TIMESTAMP, "
    "under_prob DOUBLE, line DOUBLE)"
)
_LINES_DDL = (
    "CREATE TABLE lines (league VARCHAR, market VARCHAR, game_date DATE, "
    "entity VARCHAR, line DOUBLE, observed_at TIMESTAMP)"
)

# The ingestion-fix "deploy time" the apply run is bounded by.
_UNTIL = datetime(2026, 8, 27, 12, 0)


def _make_archive(path) -> None:
    con = duckdb.connect(str(path))
    con.execute(_ODDS_DDL)
    con.execute(_LINES_DDL)
    con.execute(
        "INSERT INTO lines VALUES ('WNBA','PRA','2026-08-10','Poisoned A',33.5,'2026-08-10 12:00:00')"
    )
    con.execute(
        "INSERT INTO odds VALUES "
        # In-window Sleeper WNBA — ev/under_prob nulled, line kept.
        "('WNBA','PRA','2026-08-10','Poisoned A','Sleeper',33.5,'2026-08-10 12:00:00',0.5,33.5),"
        "('WNBA','PTS','2026-08-12','Poisoned B','Sleeper',18.5,'2026-08-12 12:00:00',0.5,18.5),"
        # game_date exactly on the poison start is in scope (>= is inclusive).
        "('WNBA','PRA','2026-08-03','Poisoned C','Sleeper',29.5,'2026-08-03 09:00:00',0.5,29.5),"
        # Same slot, real book — untouched.
        "('WNBA','PRA','2026-08-10','Poisoned A','fanduel',34.1,'2026-08-10 12:00:00',0.55,33.5),"
        # Sleeper, other league — untouched.
        "('MLB','hits','2026-08-10','Some Batter','Sleeper',1.2,'2026-08-10 12:00:00',0.5,0.5),"
        # Sleeper WNBA before the product change — untouched.
        "('WNBA','PRA','2026-07-15','Pre Window','Sleeper',30.0,'2026-07-15 12:00:00',0.5,29.5),"
        # Sleeper WNBA observed after --until (fixed ingest) — untouched.
        "('WNBA','PRA','2026-08-28','Post Fix','Sleeper',31.0,'2026-08-28 12:00:00',0.48,30.5)"
    )
    con.execute("CHECKPOINT")
    con.close()


def _all_rows(path):
    con = duckdb.connect(str(path), read_only=True)
    odds = con.execute("SELECT * FROM odds ORDER BY entity, book").fetchall()
    lines = con.execute("SELECT * FROM lines").fetchall()
    con.close()
    return odds, lines


def test_dry_run_counts_but_writes_nothing(tmp_path):
    db = tmp_path / "archive.duckdb"
    _make_archive(db)
    before = _all_rows(db)

    report = quarantine_sleeper_wnba(db, until=_UNTIL, apply=False)

    assert report["affected"] == 3
    assert report["total_sleeper_wnba"] == 5
    assert report["per_market"] == [("PRA", 2), ("PTS", 1)]
    assert report["updated"] == 0
    assert report["backup"] is None
    assert _all_rows(db) == before
    assert list(tmp_path.glob("*.bak-*")) == []


def test_apply_nulls_ev_and_under_prob_but_keeps_the_line(tmp_path):
    """The line is the record of what Sleeper offered; only the fabricated fair-quote
    encoding (ev ~= line, under_prob = 0.5) is poison, so only those two are nulled."""
    db = tmp_path / "archive.duckdb"
    _make_archive(db)
    _, lines_before = _all_rows(db)

    report = quarantine_sleeper_wnba(db, until=_UNTIL, apply=True)

    assert report["affected"] == report["updated"] == 3
    assert report["per_market"] == [("PRA", 2), ("PTS", 1)]
    assert report["backup"] is not None and report["backup"].is_file()

    con = duckdb.connect(str(db), read_only=True)
    quarantined = con.execute(
        "SELECT entity, ev, under_prob, line FROM odds "
        "WHERE book = 'Sleeper' AND league = 'WNBA' AND entity LIKE 'Poisoned%' ORDER BY entity"
    ).fetchall()
    untouched = con.execute(
        "SELECT entity, book, ev, under_prob, line FROM odds "
        "WHERE NOT (book = 'Sleeper' AND league = 'WNBA' AND entity LIKE 'Poisoned%') "
        "ORDER BY entity"
    ).fetchall()
    con.close()

    assert quarantined == [
        ("Poisoned A", None, None, 33.5),
        ("Poisoned B", None, None, 18.5),
        ("Poisoned C", None, None, 29.5),
    ]
    assert untouched == [
        ("Poisoned A", "fanduel", 34.1, 0.55, 33.5),
        ("Post Fix", "Sleeper", 31.0, 0.48, 30.5),
        ("Pre Window", "Sleeper", 30.0, 0.5, 29.5),
        ("Some Batter", "Sleeper", 1.2, 0.5, 0.5),
    ]
    assert _all_rows(db)[1] == lines_before


def test_dry_run_reports_the_same_counts_apply_would(tmp_path):
    db = tmp_path / "archive.duckdb"
    _make_archive(db)

    dry = quarantine_sleeper_wnba(db, until=_UNTIL, apply=False)
    applied = quarantine_sleeper_wnba(db, until=_UNTIL, apply=True)

    assert dry["affected"] == applied["affected"] == applied["updated"]
    assert dry["per_market"] == applied["per_market"]


def test_apply_no_backup_skips_copy(tmp_path):
    db = tmp_path / "archive.duckdb"
    _make_archive(db)

    report = quarantine_sleeper_wnba(db, until=_UNTIL, apply=True, backup=False)

    assert report["backup"] is None
    assert list(tmp_path.glob("*.bak-*")) == []


def test_cli_apply_requires_until():
    result = CliRunner().invoke(main, ["--apply"])
    assert result.exit_code == 2
    assert "--until" in result.output


def test_cli_dry_run_defaults_until_to_now(tmp_path, monkeypatch):
    monkeypatch.setenv("SPORTSTRADAMUS_LOCK_DIR", str(tmp_path))
    db = tmp_path / "archive.duckdb"
    _make_archive(db)
    before = _all_rows(db)

    result = CliRunner().invoke(main, ["--archive", str(db)])

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    # until defaulted to now, so the row the fixed ingest wrote on Aug 28 is in
    # scope too — the dry run is for sizing, the bounded --until is for applying.
    assert "affected: 4 of 5" in result.output
    assert _all_rows(db) == before
