"""Unit tests for the gate-driven ship_config generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sportstradamus.scripts.generate_ship_config import (
    active_cells,
    build_ship_config,
    load_decisions,
)
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.ship_config import WITHHELD, load_ship_config

_N_ALL_MARKETS = sum(len(markets) for markets in ALL_MARKETS.values())  # 96


def _write(path, mapping):
    Path(path).write_text(json.dumps(mapping))


def test_load_decisions_rejects_withheld(tmp_path):
    p = tmp_path / "d.json"
    _write(p, {"NBA": {"PTS": WITHHELD}})
    with pytest.raises(ValueError):
        load_decisions(p)


def test_load_decisions_accepts_real_slugs(tmp_path):
    p = tmp_path / "d.json"
    _write(p, {"NBA": {"PTS": "ratio_meanyr"}})
    assert load_decisions(p) == {"NBA": {"PTS": "ratio_meanyr"}}


def test_build_ship_config_is_exhaustive_and_withholds_the_rest():
    cfg = build_ship_config({"NBA": {"PTS": "ratio_meanyr"}}, {("NBA", "PTS")})
    assert sum(len(markets) for markets in cfg.values()) == _N_ALL_MARKETS
    assert cfg["NBA"]["PTS"] == "ratio_meanyr"
    assert cfg["NBA"]["REB"] == WITHHELD
    assert cfg["MLB"]["hits"] == WITHHELD


def test_build_ship_config_rejects_cell_not_in_all_markets():
    with pytest.raises(ValueError):
        build_ship_config({"NBA": {"NOT_A_MARKET": "ratio_meanyr"}}, set())


def test_build_ship_config_is_deterministic():
    decisions = {"NBA": {"PTS": "ratio_meanyr"}}
    first = build_ship_config(decisions, {("NBA", "PTS")})
    second = build_ship_config(decisions, {("NBA", "PTS")})
    # Same inputs -> byte-identical serialized output (sorted keys).
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_ship_config_output_passes_load_ship_config(tmp_path):
    cfg = build_ship_config({"NBA": {"PTS": "ratio_meanyr"}}, {("NBA", "PTS")})
    p = tmp_path / "ship_config.json"
    p.write_text(json.dumps(cfg, indent=4, sort_keys=True))
    assert load_ship_config(p) == cfg


def test_active_cells_devel_is_all_decisions(tmp_path):
    decisions = {"NBA": {"PTS": "ratio_meanyr", "REB": "ratio_meanyr"}}
    got = active_cells("devel", decisions, tmp_path / "ms", tmp_path / "lm")
    assert got == {("NBA", "PTS"), ("NBA", "REB")}


def test_active_cells_main_no_data_is_empty(tmp_path):
    decisions = {"NBA": {"PTS": "ratio_meanyr"}}
    got = active_cells("main", decisions, tmp_path / "ms", tmp_path / "lm")
    assert got == set()


def test_active_cells_main_intersects_graduated(tmp_path):
    decisions = {"NBA": {"PTS": "ratio_meanyr"}}
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
    assert active_cells("main", decisions, ms, lm) == {("NBA", "PTS")}
