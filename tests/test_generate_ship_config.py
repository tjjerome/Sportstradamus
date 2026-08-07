"""Unit tests for the gate-driven generate-ship-config CLI + helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.scripts.generate_ship_config import (
    main,
    promote_for_main,
)
from sportstradamus.training.ship_config import (
    TARGET_NORM_NONE,
    WITHHELD,
    load_ship_config,
)


def _write(path, mapping):
    Path(path).write_text(json.dumps(mapping))


def _invoke(args):
    return CliRunner().invoke(main, args)


def _meta_cell(dist: str, shipped: str, target_normalization: str) -> dict:
    return {"dist": dist, "shipped": shipped, "target_normalization": target_normalization}


def test_promote_for_main_promotes_graduated_devel_cells():
    meta = {
        "NBA": {
            "PTS": _meta_cell("SkewNormal", "devel", "ratio_meanyr"),
            "REB": _meta_cell("SkewNormal", "devel", "ratio_meanyr"),
        }
    }
    out, changes = promote_for_main(meta, {("NBA", "PTS")})
    assert out["NBA"]["PTS"]["shipped"] == "main"
    assert out["NBA"]["REB"]["shipped"] == "devel"
    assert ("NBA", "PTS", "devel", "main") in changes
    assert len(changes) == 1


def test_promote_for_main_demotes_failed_main_cells():
    meta = {
        "NBA": {
            "PTS": _meta_cell("SkewNormal", "main", "ratio_meanyr"),
        }
    }
    out, changes = promote_for_main(meta, set())
    assert out["NBA"]["PTS"]["shipped"] == "devel"
    assert ("NBA", "PTS", "main", "devel") in changes


def test_promote_for_main_leaves_withheld_alone():
    meta = {
        "NBA": {
            "PTS": _meta_cell("SkewNormal", "withheld", TARGET_NORM_NONE),
        }
    }
    # Even if a "withheld" cell appears in graduated (stale state), do not
    # silently flip it to main — the human must opt in via shipping it on
    # devel first.
    out, changes = promote_for_main(meta, {("NBA", "PTS")})
    assert out["NBA"]["PTS"]["shipped"] == "withheld"
    assert changes == []


def test_load_ship_config_devel_includes_devel_and_main(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {
            "NBA": {
                "PTS": _meta_cell("SkewNormal", "main", "ratio_meanyr"),
                "REB": _meta_cell("SkewNormal", "devel", "ratio_meanyr"),
                "AST": _meta_cell("SkewNormal", "withheld", TARGET_NORM_NONE),
            }
        },
    )
    cfg = load_ship_config(branch="devel", path=meta_path)
    assert cfg["NBA"]["PTS"] == "ratio_meanyr"
    assert cfg["NBA"]["REB"] == "ratio_meanyr"
    assert cfg["NBA"]["AST"] == WITHHELD


def test_load_ship_config_main_only_includes_main(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {
            "NBA": {
                "PTS": _meta_cell("SkewNormal", "main", "ratio_meanyr"),
                "REB": _meta_cell("SkewNormal", "devel", "ratio_meanyr"),
            }
        },
    )
    cfg = load_ship_config(branch="main", path=meta_path)
    assert cfg["NBA"]["PTS"] == "ratio_meanyr"
    assert cfg["NBA"]["REB"] == WITHHELD


def test_load_ship_config_rejects_invalid_strategy(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {"NBA": {"PTS": _meta_cell("SkewNormal", "devel", "bogus_slug")}},
    )
    with pytest.raises(ValueError):
        load_ship_config(branch="devel", path=meta_path)


def test_load_ship_config_rejects_skew_with_strategy_none(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {"NBA": {"PTS": _meta_cell("SkewNormal", "devel", TARGET_NORM_NONE)}},
    )
    with pytest.raises(ValueError):
        load_ship_config(branch="devel", path=meta_path)


def test_load_ship_config_rejects_count_with_real_strategy(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {"NBA": {"FG3M": _meta_cell("ZINB", "devel", "ratio_meanyr")}},
    )
    with pytest.raises(ValueError):
        load_ship_config(branch="devel", path=meta_path)


def test_cli_devel_validates_and_summarizes(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {
            "NBA": {
                "PTS": _meta_cell("SkewNormal", "devel", "ratio_meanyr"),
                "REB": _meta_cell("SkewNormal", "withheld", TARGET_NORM_NONE),
            }
        },
    )
    # The devel branch enforces serve-iff-ship against model_stats; without an
    # explicit fixture the CLI reads the real production parquet.
    ms = tmp_path / "ms.parquet"
    pd.DataFrame([{"league": "NBA", "market": "PTS", "ship": True}]).to_parquet(
        ms, engine="pyarrow", index=False
    )
    before = meta_path.read_text()
    result = _invoke(["--branch", "devel", "--meta", str(meta_path), "--model-stats", str(ms)])
    assert result.exit_code == 0, result.output
    assert "active=1" in result.output
    assert "withheld=1" in result.output
    assert meta_path.read_text() == before


def test_cli_reports_free_passers_without_mutating(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {"NBA": {"REB": _meta_cell("SkewNormal", "withheld", TARGET_NORM_NONE)}},
    )
    ms = tmp_path / "ms.parquet"
    pd.DataFrame([{"league": "NBA", "market": "REB", "ship": True}]).to_parquet(
        ms, engine="pyarrow", index=False
    )
    before = meta_path.read_text()
    result = _invoke(["--branch", "devel", "--meta", str(meta_path), "--model-stats", str(ms)])
    assert result.exit_code == 0, result.output
    assert "FREE-PASSER" in result.output
    assert "NBA/REB" in result.output
    assert meta_path.read_text() == before


def test_cli_main_promotes_graduated_cells(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {"NBA": {"PTS": _meta_cell("SkewNormal", "devel", "ratio_meanyr")}},
    )
    ms = tmp_path / "ms.parquet"
    lm = tmp_path / "lm.parquet"
    pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "PTS",
                "distribution": "SkewNormal",
                "row_kind": "model",
                "metric_row": "calibrated",
                "ship": True,
                "brier_skill_score": 0.12,
                "predicted_over_rate": 0.5,
                "empirical_over_rate": 0.5,
                "kelly_shrinkage": 0.1,
            }
        ]
    ).to_parquet(ms, engine="pyarrow", index=False)
    pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "PTS",
                "computed_at": pd.Timestamp("2026-05-20"),
                "window_days": np.int16(30),
                "n_settled": np.int64(300),
                "book_bss": 0.05,
                "empirical_over_rate": 0.5,
                "predicted_over_rate": 0.5,
                "top_decile_mae": 2.0,
                "profit_sim_yield": 0.03,
            }
        ]
    ).to_parquet(lm, engine="pyarrow", index=False)

    result = _invoke(
        [
            "--branch",
            "main",
            "--meta",
            str(meta_path),
            "--model-stats",
            str(ms),
            "--live-metrics",
            str(lm),
        ]
    )
    assert result.exit_code == 0, result.output
    written = json.loads(meta_path.read_text())
    assert written["NBA"]["PTS"]["shipped"] == "main"


def test_cli_main_dry_run_does_not_write(tmp_path):
    meta_path = tmp_path / "stat_meta.json"
    initial = {"NBA": {"PTS": _meta_cell("SkewNormal", "devel", "ratio_meanyr")}}
    _write(meta_path, initial)
    ms = tmp_path / "ms.parquet"
    lm = tmp_path / "lm.parquet"
    pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "PTS",
                "distribution": "SkewNormal",
                "row_kind": "model",
                "metric_row": "calibrated",
                "ship": True,
                "brier_skill_score": 0.12,
                "predicted_over_rate": 0.5,
                "empirical_over_rate": 0.5,
                "kelly_shrinkage": 0.1,
            }
        ]
    ).to_parquet(ms, engine="pyarrow", index=False)
    pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "PTS",
                "computed_at": pd.Timestamp("2026-05-20"),
                "window_days": np.int16(30),
                "n_settled": np.int64(300),
                "book_bss": 0.05,
                "empirical_over_rate": 0.5,
                "predicted_over_rate": 0.5,
                "top_decile_mae": 2.0,
                "profit_sim_yield": 0.03,
            }
        ]
    ).to_parquet(lm, engine="pyarrow", index=False)

    result = _invoke(
        [
            "--branch",
            "main",
            "--meta",
            str(meta_path),
            "--model-stats",
            str(ms),
            "--live-metrics",
            str(lm),
            "--dry-run",
        ]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(meta_path.read_text()) == initial


def test_cli_prune_deletes_only_withheld_pickles(tmp_path, monkeypatch):
    import sportstradamus.helpers.io as io_mod

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(io_mod, "MODELS_DIR", models)
    (models / "NBA_PTS.mdl").write_text("x")  # active -> kept
    (models / "NBA_REB.mdl").write_text("x")  # withheld -> pruned

    meta_path = tmp_path / "stat_meta.json"
    _write(
        meta_path,
        {
            "NBA": {
                "PTS": _meta_cell("SkewNormal", "devel", "ratio_meanyr"),
                "REB": _meta_cell("SkewNormal", "withheld", TARGET_NORM_NONE),
            }
        },
    )
    # Serve-iff-ship fixture: keep the served cell passing so the CLI exits 0.
    ms = tmp_path / "ms.parquet"
    pd.DataFrame([{"league": "NBA", "market": "PTS", "ship": True}]).to_parquet(
        ms, engine="pyarrow", index=False
    )
    result = _invoke(
        [
            "--branch",
            "devel",
            "--meta",
            str(meta_path),
            "--model-stats",
            str(ms),
            "--prune",
        ]
    )
    assert result.exit_code == 0, result.output
    assert (models / "NBA_PTS.mdl").exists()
    assert not (models / "NBA_REB.mdl").exists()
