"""Pins the one-league merge behind ``scripts/sync_to_prod.sh --league``.

``book_weights.json``, ``stat_calibration.json`` and ``model_stats.parquet`` each hold every
league. The merge must hand production back its own entries for every league but the one being
sent, or a one-league send carries the other leagues' values along with it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_league_only_serving.py"
_SPEC = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
merge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge)

_PROD_JSON = {
    "MLB": {"hits": {"draftkings": 0.5, "fanduel": 0.5}},
    "NFL": {"yards": {"draftkings": 0.4}, "passing yards": {"fanduel": 1.0}},
    "NHL": {"shots": {"fanduel": 1.0}},
}
_HERE_JSON = {
    "NFL": {"yards": {"draftkings": 0.7, "fanduel": 0.3}},
    "MLB": {"hits": {"draftkings": 0.9}},
}
_PROD_STATS = pd.DataFrame(
    {
        "league": ["MLB", "NFL", "NFL", "NHL"],
        "market": ["hits", "yards", "passing yards", "shots"],
        "model_weight": [0.5, 0.53, 0.9, None],
        "ship": [True, False, False, True],
    }
)
_HERE_STATS = pd.DataFrame(
    {
        "league": ["NFL", "MLB", "NFL"],
        "market": ["yards", "hits", "targets"],
        "model_weight": [1.0, 0.1, None],
        "ship": [True, False, True],
    }
)


@pytest.fixture
def boxes(tmp_path: Path) -> tuple[Path, Path]:
    """The training box's data folder and a folder of production's copies."""
    here, prod = tmp_path / "here", tmp_path / "prod"
    (here / "config").mkdir(parents=True)
    (here / "training").mkdir()
    prod.mkdir()
    for name in ("book_weights.json", "stat_calibration.json"):
        (here / "config" / name).write_text(json.dumps(_HERE_JSON))
        (prod / name).write_text(json.dumps(_PROD_JSON))
    _HERE_STATS.to_parquet(here / "training" / "model_stats.parquet", index=False)
    _PROD_STATS.to_parquet(prod / "model_stats.parquet", index=False)
    return here, prod


def test_merge_json_replaces_one_league_and_keeps_production_for_the_rest(boxes, tmp_path):
    here, prod = boxes
    merged_path = tmp_path / "merged.json"

    proof_passed = merge._merge_json(
        "NFL",
        here / "config" / "book_weights.json",
        prod / "book_weights.json",
        merged_path,
        trailing_newline=True,
    )

    expected = {"MLB": _PROD_JSON["MLB"], "NFL": _HERE_JSON["NFL"], "NHL": _PROD_JSON["NHL"]}
    assert merged_path.read_text() == json.dumps(expected, indent=4) + "\n"
    assert proof_passed


def test_merge_stats_replaces_one_league_rows_and_keeps_production_for_the_rest(boxes, tmp_path):
    here, prod = boxes

    proof_passed = merge._merge_stats(
        "NFL", here / "training" / "model_stats.parquet", prod / "model_stats.parquet", tmp_path
    )

    expected = pd.DataFrame(
        {
            "league": ["MLB", "NFL", "NFL", "NHL"],
            "market": ["hits", "yards", "targets", "shots"],
            "model_weight": [0.5, 1.0, None, None],
            "ship": [True, True, True, True],
        }
    )
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "model_stats.parquet"), expected)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "model_stats.csv"), expected)
    assert proof_passed


def test_merge_stats_takes_columns_the_training_box_added(boxes, tmp_path):
    """A retrain adds report columns before production's table has them."""
    here, prod = boxes
    retrained = _HERE_STATS.copy()
    retrained.insert(2, "book_citl_offline", [0.02, -0.01, None])
    retrained["g1_book_thin"] = pd.array([False, True, True], dtype="boolean")
    retrained.to_parquet(here / "training" / "model_stats.parquet", index=False)

    proof_passed = merge._merge_stats(
        "NFL", here / "training" / "model_stats.parquet", prod / "model_stats.parquet", tmp_path
    )

    expected = pd.DataFrame(
        {
            "league": ["MLB", "NFL", "NFL", "NHL"],
            "market": ["hits", "yards", "targets", "shots"],
            "book_citl_offline": [None, 0.02, None, None],
            "model_weight": [0.5, 1.0, None, None],
            "ship": [True, True, True, True],
            "g1_book_thin": pd.array([None, False, True, None], dtype="boolean"),
        }
    ).astype({"book_citl_offline": float})
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "model_stats.parquet"), expected)
    assert proof_passed


def test_merge_stats_refuses_a_shared_column_typed_differently(boxes, tmp_path):
    here, prod = boxes
    _HERE_STATS.assign(ship=[1, 0, 1]).to_parquet(
        here / "training" / "model_stats.parquet", index=False
    )

    assert not merge._merge_stats(
        "NFL", here / "training" / "model_stats.parquet", prod / "model_stats.parquet", tmp_path
    )
    assert not (tmp_path / "model_stats.parquet").exists()


def test_cli_exits_nonzero_when_production_has_a_column_the_training_box_lacks(boxes, tmp_path):
    """``sync_to_prod.sh`` sends only on exit 0, so an unmerged table must not exit 0."""
    here, prod = boxes
    _PROD_STATS.assign(retired_column=0.0).to_parquet(prod / "model_stats.parquet", index=False)
    out_dir = tmp_path / "merged"

    result = CliRunner().invoke(
        merge.main,
        [
            "--league",
            "NFL",
            "--here-dir",
            str(here),
            "--prod-dir",
            str(prod),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert not (out_dir / "model_stats.parquet").exists()
