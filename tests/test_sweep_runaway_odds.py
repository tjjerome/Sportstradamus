"""Unit tests for ``scripts.sweep_runaway_odds`` — the global blown-row archive sweep.

The sweep deletes every magnitude-blown ``ev`` row (``> 2000`` absolute, or
``> 5×`` the latest line) in one pass, across all cells. These pin both predicate
arms, the dry-run/apply split, and the safety backup, on a throwaway DuckDB.
"""

from __future__ import annotations

import duckdb

from sportstradamus.scripts.sweep_runaway_odds import sweep_runaway_odds


def _make_archive(path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE odds (league VARCHAR, market VARCHAR, game_date DATE, "
        "entity VARCHAR, book VARCHAR, ev DOUBLE, observed_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE lines (league VARCHAR, market VARCHAR, game_date DATE, "
        "entity VARCHAR, line DOUBLE, observed_at TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO lines VALUES "
        "('NBA','STL','2025-12-23','Clean Guy',1.5,'2025-12-23 00:00:00'),"
        "('NBA','STL','2025-12-23','Blown Guy',1.5,'2025-12-23 00:00:00')"
    )
    con.execute(
        "INSERT INTO odds VALUES "
        "('NBA','STL','2025-12-23','Clean Guy','dk',2.0,'2025-12-23 01:00:00'),"  # 2.0 < 5*1.5
        "('NBA','STL','2025-12-23','Blown Guy','dk',9.0,'2025-12-23 01:00:00'),"  # 9.0 > 5*1.5
        "('TENNIS','Aces','2025-12-23','No Line','ud',5000.0,'2025-12-23 01:00:00')"  # > 2000, no line
    )
    con.execute("CHECKPOINT")
    con.close()


def test_sweep_dry_run_counts_blown_but_writes_nothing(tmp_path):
    db = tmp_path / "archive.duckdb"
    _make_archive(db)

    report = sweep_runaway_odds(db, apply=False)

    assert report["total_rows"] == 3
    assert report["blown"] == 2  # line-relative (Blown Guy) + absolute (No Line)
    assert report["backup"] is None
    # nothing deleted, no backup file written
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 3
    con.close()
    assert list(tmp_path.glob("*.bak-*")) == []


def test_sweep_apply_deletes_blown_keeps_clean_and_backs_up(tmp_path):
    db = tmp_path / "archive.duckdb"
    _make_archive(db)

    report = sweep_runaway_odds(db, apply=True)

    assert report["blown"] == 2
    assert report["remaining"] == 0
    assert report["backup"] is not None and report["backup"].is_file()
    con = duckdb.connect(str(db), read_only=True)
    survivors = con.execute("SELECT entity FROM odds ORDER BY entity").fetchall()
    con.close()
    assert survivors == [("Clean Guy",)]


def test_sweep_apply_no_backup_skips_copy(tmp_path):
    db = tmp_path / "archive.duckdb"
    _make_archive(db)

    report = sweep_runaway_odds(db, apply=True, backup=False)

    assert report["backup"] is None
    assert list(tmp_path.glob("*.bak-*")) == []


def _empty_archive(path):
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE odds (league VARCHAR, market VARCHAR, game_date DATE, "
        "entity VARCHAR, book VARCHAR, ev DOUBLE, observed_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE lines (league VARCHAR, market VARCHAR, game_date DATE, "
        "entity VARCHAR, line DOUBLE, observed_at TIMESTAMP)"
    )
    return con


def test_sweep_spares_multibook_higher_line_clamp(tmp_path):
    """A book that priced a higher line writes a correctly-clamped ev; a later,
    lower line from another book must not make it look blown. The predicate keys
    on MAX(line) over the slot, not the latest line, so 5×own-line clamps survive."""
    db = tmp_path / "archive.duckdb"
    con = _empty_archive(db)
    # Two line rows for one slot: an early 2.5 (some books), a later 1.5 (latest).
    con.execute(
        "INSERT INTO lines VALUES "
        "('WNBA','FG3M','2025-12-23','Sharp Shooter',2.5,'2025-12-23 01:00:00'),"
        "('WNBA','FG3M','2025-12-23','Sharp Shooter',1.5,'2025-12-23 02:00:00')"
    )
    con.execute(
        "INSERT INTO odds VALUES "
        "('WNBA','FG3M','2025-12-23','Sharp Shooter','dk',12.5,'2025-12-23 01:00:00'),"  # 5*2.5 clamp
        "('WNBA','FG3M','2025-12-23','Sharp Shooter','sl',7.5,'2025-12-23 02:00:00')"  # 5*1.5 clamp
    )
    con.execute("CHECKPOINT")
    con.close()

    report = sweep_runaway_odds(db, apply=False)
    assert (
        report["blown"] == 0
    )  # both within 5×MAX(line)=12.5; latest-line join would wrongly flag the 12.5


def test_sweep_spares_negative_line_team_market(tmp_path):
    """Team markets quote negative lines (run line, spread). 5×(negative line) is
    a meaningless ceiling, so the line arm must apply only to positive lines."""
    db = tmp_path / "archive.duckdb"
    con = _empty_archive(db)
    con.execute(
        "INSERT INTO lines VALUES "
        "('MLB','1st Inning Run Line','2025-12-23','LA Dodgers',-1.5,'2025-12-23 01:00:00')"
    )
    con.execute(
        "INSERT INTO odds VALUES "
        "('MLB','1st Inning Run Line','2025-12-23','LA Dodgers','dk',-2.0,'2025-12-23 01:00:00')"
    )
    con.execute("CHECKPOINT")
    con.close()

    report = sweep_runaway_odds(db, apply=False)
    assert (
        report["blown"] == 0
    )  # no positive line for the slot -> line arm off, ev well under the absolute ceiling


def test_sweep_flags_sub_2000_runaway(tmp_path):
    """Pre-fix residue between 5×line and the absolute 2000 ceiling is real garbage
    the fixed clamp could never write; the line arm (not absolute-only) must catch it."""
    db = tmp_path / "archive.duckdb"
    con = _empty_archive(db)
    con.execute(
        "INSERT INTO lines VALUES "
        "('NBA','STL','2025-12-23','Residue Guy',0.5,'2025-12-23 01:00:00')"
    )
    con.execute(
        "INSERT INTO odds VALUES "
        "('NBA','STL','2025-12-23','Residue Guy','dk',600.0,'2025-12-23 01:00:00')"  # > 5*0.5, < 2000
    )
    con.execute("CHECKPOINT")
    con.close()

    report = sweep_runaway_odds(db, apply=False)
    assert report["blown"] == 1
