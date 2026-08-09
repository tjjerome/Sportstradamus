"""Unit tests for the WS-1 Stage-0 era backfill CLI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from click.testing import CliRunner

from sportstradamus.scripts.backfill_history_eras import backfill_history_eras


def _row(player, date, cv, *, league="NBA", market="PTS", platform="Underdog"):
    return {
        "Player": player,
        "League": league,
        "Market": market,
        "Date": date,
        "Platform": platform,
        "Dist": "SkewNormal",
        "CV": cv,
        "Temperature": 1.5,
        "Disp Cal": 1.0,
        "Step": 1.0,
    }


def _history_with_two_fingerprints() -> pd.DataFrame:
    # era.0: an all-NaN pre-schema block; era.1/era.2: two distinct CV eras.
    return pd.DataFrame(
        [
            _row("A", "2026-02-01", np.nan),
            _row("B", "2026-02-02", np.nan),
            _row("C", "2026-03-01", 0.5100),
            _row("D", "2026-03-02", 0.5100),
            _row("E", "2026-04-01", 0.5200),
        ]
    )


def _run(monkeypatch, tmp_path, history):
    monkeypatch.setattr(
        "sportstradamus.scripts.backfill_history_eras.read_history", lambda: history
    )
    out = tmp_path / "history_eras.parquet"
    result = CliRunner().invoke(backfill_history_eras, ["--output", str(out)])
    assert result.exit_code == 0, result.output
    return result, pd.read_parquet(out)


def test_distinct_fingerprints_become_ordered_eras(monkeypatch, tmp_path):
    _, eras = _run(monkeypatch, tmp_path, _history_with_two_fingerprints())

    by_player = eras.set_index("Player")["era_id"].to_dict()
    # NaN block collapses to one era, ordered first.
    assert by_player["A"] == by_player["B"] == "era.0"
    # Each distinct CV is its own era, numbered by first-seen date.
    assert by_player["C"] == by_player["D"] == "era.1"
    assert by_player["E"] == "era.2"
    assert eras["era_first_date"].loc[eras["Player"] == "E"].iloc[0] == "2026-04-01"


def test_join_key_is_unique(monkeypatch, tmp_path):
    # Same (League, Market, Date, Player) across two platforms — one era row out.
    history = pd.DataFrame(
        [
            _row("A", "2026-03-01", 0.51, platform="Underdog"),
            _row("A", "2026-03-01", 0.51, platform="Sleeper"),
        ]
    )
    _, eras = _run(monkeypatch, tmp_path, history)
    assert not eras.duplicated(subset=["League", "Market", "Date", "Player"]).any()
    assert len(eras) == 1


def test_same_boundary_day_stays_two_eras(monkeypatch, tmp_path):
    # Two fingerprints first seen on the same day must not merge.
    history = pd.DataFrame(
        [
            _row("A", "2026-05-08", 0.5146),
            _row("B", "2026-05-08", 0.5147),
        ]
    )
    _, eras = _run(monkeypatch, tmp_path, history)
    assert eras["era_id"].nunique() == 2


def test_flip_date_annotates_recent_eras(monkeypatch, tmp_path):
    # A late-2026 era should pick up a real stat_meta.json flip date; the very
    # first (Feb) era predates the config history and stays unannotated.
    history = pd.DataFrame(
        [
            _row("A", "2026-02-01", 0.5100),
            _row("B", "2026-07-01", 0.5200),
        ]
    )
    _, eras = _run(monkeypatch, tmp_path, history)
    late = eras.loc[eras["Player"] == "B", "label_flip_date"].iloc[0]
    assert late is not None and late <= "2026-07-01"


def test_empty_history_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setattr("sportstradamus.scripts.backfill_history_eras.read_history", pd.DataFrame)
    out = tmp_path / "out.parquet"
    result = CliRunner().invoke(backfill_history_eras, ["--output", str(out)])
    assert result.exit_code == 0
    assert "empty" in result.output.lower()
