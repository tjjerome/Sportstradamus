"""Contracts for the read-only training-matrix integrity auditor."""

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from sportstradamus.helpers.training_quotes import PROVENANCE_COLUMNS
from sportstradamus.training.matrix_audit import (
    audit_matrix,
    incomplete_provenance_rows,
    main,
)


def _write_matrix(path: Path, **overrides) -> None:
    columns = {
        "Player": ["A", "B"],
        "Date": ["2025-01-01", "2025-01-01"],
        "Result": [1.0, 2.0],
        "Line": [1.5, 1.5],
        "Odds": [0.45, 0.55],
        "EV": [1.4, 1.6],
        "Archived": [True, False],
        "Odds_synthetic": [False, True],
        "QuoteSource": ["book_direct", "book_ev_inversion"],
        "QuoteAuthenticity": ["authentic", "derived"],
        "QuoteSyntheticReason": [None, "ev_inversion"],
        "QuoteObservedAt": ["2024-12-31T12:00:00", None],
        "QuoteBookCount": [2, 0],
        "Player position": [1, 2],
    }
    columns.update(overrides)
    pd.DataFrame(columns).to_parquet(path, index=False)


def test_auditor_reports_identity_lattice_positions_and_quote_classes(tmp_path):
    path = tmp_path / "NBA_PTS.parquet"
    _write_matrix(path)

    report = audit_matrix(path)

    assert report["violations"] == []
    assert report["target_lattice"] == {
        "unique_values": 2,
        "minimum_step": 1.0,
        "integer": True,
    }
    assert report["positions"] == {1: 1, 2: 1}
    assert report["quote_counts"] == {
        "authentic": 1,
        "synthetic": 0,
        "derived": 1,
        "bookless": 1,
        "archive_coverage": 0.5,
    }
    assert report["quote_classification"] == "partial"


def test_auditor_flags_duplicate_invalid_and_missing_provenance(tmp_path):
    path = tmp_path / "bad.parquet"
    _write_matrix(
        path,
        Player=["A", "A"],
        Date=["2025-01-01", "2025-01-01"],
        Odds=[1.2, 0.5],
        EV=[float("nan"), 1.0],
    )
    frame = pd.read_parquet(path).drop(columns=["QuoteSource"])
    frame.to_parquet(path, index=False)

    report = audit_matrix(path)

    assert report["duplicate_identity_rows"] == 1
    assert report["invalid_archived_rows"] == 1
    assert set(report["violations"]) == {
        "duplicate Player/Date identity",
        "nonfinite core value",
        "Odds outside [0, 1]",
        "invalid archived quote",
        "missing quote provenance",
    }


def test_incomplete_provenance_counts_only_the_half_populated_block(tmp_path):
    path = tmp_path / "NBA_PTS.parquet"
    _write_matrix(path)
    populated = pd.read_parquet(path)

    assert incomplete_provenance_rows(populated) == 0
    assert incomplete_provenance_rows(populated.drop(columns=list(PROVENANCE_COLUMNS))) == 0

    half = populated.copy()
    half.loc[0, "QuoteAuthenticity"] = None
    assert incomplete_provenance_rows(half) == 1


def test_auditor_flags_a_half_populated_provenance_block(tmp_path):
    path = tmp_path / "MLB_runs-allowed.parquet"
    _write_matrix(path, QuoteBookCount=[None, 0])

    report = audit_matrix(path)

    assert "null required quote provenance" in report["violations"]


def test_auditor_cli_exits_nonzero_on_violation(tmp_path):
    path = tmp_path / "bad.parquet"
    _write_matrix(path, Odds=[1.2, 0.5])

    result = CliRunner().invoke(main, [str(path)])

    assert result.exit_code != 0
    assert "failed integrity audit" in result.output


def test_auditor_rejects_non_goalies_in_goalie_matrix(tmp_path):
    path = tmp_path / "NHL_saves.parquet"
    _write_matrix(path, **{"Player position": [4, 1]})

    report = audit_matrix(path)

    assert report["invalid_nhl_position_rows"] == 1
    assert "ineligible NHL market position" in report["violations"]
