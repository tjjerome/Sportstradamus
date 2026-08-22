"""Golden pins for meditate's warn-only ship gate.

A served cell whose fresh offline gates fail is *reported*, never pruned — the
cull (flip ``shipped: "withheld"``, or ``generate-ship-config --prune``) is the
human's decision. Replaces the retired serve-iff-ship invariant test.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from sportstradamus.training import cli as training_cli


def _stats(rows: list[dict]) -> pd.DataFrame:
    base = {
        "ship": True,
        "g4_pit_ks": 0.04,
        "g4_pit_ks_max": 0.05,
        "brier_skill_score": 0.01,
        **{f"g{i}_pass": True for i in range(1, 7)},
    }
    frame = pd.DataFrame([{**base, **row} for row in rows])
    return frame.astype({col: "boolean" for col in frame.columns if col.endswith("_pass")})


@pytest.fixture
def _warn_env(monkeypatch, tmp_path):
    def run(rows, active_markets, ship_config, loaded_leagues):
        stats_path = tmp_path / "model_stats.parquet"
        _stats(rows).to_parquet(stats_path, engine="pyarrow", index=False)
        monkeypatch.setattr(training_cli, "MODEL_STATS_PATH", stats_path)
        pruned = []
        monkeypatch.setattr(
            training_cli, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt))
        )
        training_cli._warn_ship_gate(active_markets, ship_config, loaded_leagues, MagicMock())
        return pruned

    return run


def test_failing_served_cell_warns_with_its_gates_and_never_prunes(_warn_env, capsys):
    rows = [
        {
            "league": "NBA",
            "market": "PTS",
            "ship": False,
            "g4_pass": False,
            "g6_pass": pd.NA,  # scorecard never wrote it: NA reads as not passing
            "g4_pit_ks": 0.081,
            "brier_skill_score": -0.002,
        },
        {"league": "NBA", "market": "AST"},  # shipping cell stays out of the table
    ]
    pruned = _warn_env(
        rows,
        {"NBA": ["PTS", "AST"]},
        {"NBA": {"PTS": "ratio_meanyr", "AST": "ratio_meanyr"}},
        {"NBA"},
    )
    out = capsys.readouterr().out
    assert "SHIP-GATE WARNINGS" in out
    assert "still serving" in out
    assert "NBA PTS" in out
    assert "g4 g6" in out
    assert "0.081" in out
    assert "NBA AST" not in out
    assert 'set "shipped": "withheld"' in out
    assert pruned == []


def test_withheld_and_unloaded_cells_are_excluded(_warn_env, capsys):
    rows = [
        {"league": "NBA", "market": "PTS", "ship": False, "g4_pass": False},
        {"league": "NFL", "market": "passing yards", "ship": False, "g4_pass": False},
    ]
    pruned = _warn_env(
        rows,
        {"NBA": ["PTS"], "NFL": ["passing yards"]},
        # NBA PTS withheld on this branch; NFL served but its league was not loaded this run.
        {"NBA": {"PTS": training_cli.WITHHELD}, "NFL": {"passing yards": "ratio_meanyr"}},
        {"NBA"},
    )
    assert "SHIP-GATE WARNINGS" not in capsys.readouterr().out
    assert pruned == []


def test_all_ship_prints_nothing(_warn_env, capsys):
    pruned = _warn_env(
        [{"league": "NBA", "market": "PTS"}],
        {"NBA": ["PTS"]},
        {"NBA": {"PTS": "ratio_meanyr"}},
        {"NBA"},
    )
    assert capsys.readouterr().out == ""
    assert pruned == []
