"""Golden pins for the g4-only calibrated-HPO retry orchestration in ``training.cli``.

The retry predicate decides when a cell that failed ship under loss-selection
gets its one-shot ``hpo_selection="calibrated"`` retrain, and the persist helper
pins the knob into stat_meta so the weekly warm cron keeps the selection that
shipped. Both are pure enough to pin exhaustively here; the count closure itself
is covered by ``tests/integration/test_calibrated_count_closure.py``.
"""

import json

import pandas as pd
import pytest

from sportstradamus.training import cli, model_strategy_confirm
from sportstradamus.training.cli import (
    _g4_only_retry_wanted,
    _gate_passed,
    _persist_calibrated_hpo,
)

_ALL_PASS = {f"g{i}_pass": True for i in range(1, 7)}


def _row(**overrides) -> pd.Series:
    base = {"distribution": "NegBin", "ship": False, **_ALL_PASS, "g4_pass": False}
    base.update(overrides)
    return pd.Series(base)


def test_gate_passed_na_safe():
    assert _gate_passed(True) is True
    assert _gate_passed(False) is False
    assert _gate_passed(pd.NA) is False
    assert _gate_passed(float("nan")) is False
    assert _gate_passed(None) is False


@pytest.mark.parametrize(
    ("row", "hpo_selection", "zinb_mode", "wanted"),
    [
        (_row(), "loss", "joint", True),
        (_row(distribution="ZINB"), "loss", "joint", True),
        (_row(distribution="DPO"), "loss", "joint", True),
        # Cell already pinned (or flag-forced) to calibrated — the first run WAS calibrated.
        (_row(), "calibrated", "joint", False),
        # Hurdle has no Optuna search: nothing for calibrated selection to pick from.
        (_row(distribution="ZINB"), "loss", "hurdle", False),
        # A stale inert zinb_mode on a non-ZINB cell must not block the retry.
        (_row(distribution="NegBin"), "loss", "hurdle", True),
        (_row(ship=True, g4_pass=True), "loss", "joint", False),
        # g4 passed — the ship failure is someone else's (here g5): not our lever.
        (_row(g4_pass=True, g5_pass=False), "loss", "joint", False),
        (_row(g2_pass=False), "loss", "joint", False),
        (None, "loss", "joint", False),
        # Scorecard never ran: every gate NA reads as not-passing, so g1 blocks.
        (_row(**{f"g{i}_pass": pd.NA for i in range(1, 7)}), "loss", "joint", False),
        (_row(g1_pass=pd.NA), "loss", "joint", False),
    ],
)
def test_g4_only_retry_predicate(row, hpo_selection, zinb_mode, wanted):
    assert _g4_only_retry_wanted(row, hpo_selection, zinb_mode) is wanted


def test_persist_calibrated_hpo_disk_and_memory(tmp_path, monkeypatch):
    meta_path = tmp_path / "stat_meta.json"
    meta = {"NBA": {"FTM": {"dist": "ZINB", "shipped": "devel"}}}
    meta_path.write_text(json.dumps(meta, indent=4) + "\n")
    monkeypatch.setattr(cli, "STAT_META_PATH", meta_path)

    _persist_calibrated_hpo(meta, "NBA", "FTM")

    assert meta["NBA"]["FTM"]["hpo_selection"] == "calibrated"
    text = meta_path.read_text()
    assert text.endswith("\n")
    on_disk = json.loads(text)
    assert on_disk["NBA"]["FTM"]["hpo_selection"] == "calibrated"
    # Committed stat_meta format: 4-space indent, so the review diff stays one line.
    assert '    "NBA"' in text


def test_confirm_sync_cell_picks_up_subprocess_pin(tmp_path, monkeypatch):
    meta_path = tmp_path / "stat_meta.json"
    on_disk = {"NBA": {"FTM": {"dist": "ZINB", "shipped": "devel", "hpo_selection": "calibrated"}}}
    meta_path.write_text(json.dumps(on_disk, indent=4) + "\n")
    monkeypatch.setattr(model_strategy_confirm, "_STAT_META", meta_path)

    stale = {"NBA": {"FTM": {"dist": "ZINB", "shipped": "devel"}}}
    model_strategy_confirm._sync_cell_from_disk(stale, "NBA", "FTM")

    assert stale["NBA"]["FTM"]["hpo_selection"] == "calibrated"
