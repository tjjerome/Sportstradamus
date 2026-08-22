"""Golden pins for the g4-only calibrated-HPO retry owned by the confirm walk.

The retry predicate decides when a nominee that failed ship under loss-selection
gets its one-shot ``hpo_selection="calibrated"`` retrain, and the pin helper
persists the knob (stat_meta + the nominee's edits) so the weekly meditate keeps
the selection that shipped. Both are pure enough to pin exhaustively here; the
count closure itself is covered by
``tests/integration/test_calibrated_count_closure.py``.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from sportstradamus.training.model_strategy import confirm
from sportstradamus.training.model_strategy.confirm import (
    _g4_only_retry_wanted,
    _gate_passed,
    _pin_calibrated,
    _retry_calibrated_wanted,
)

_ALL_PASS = {f"g{i}_pass": True for i in range(1, 7)}


# Base row is a sole-g4 near miss: excess 0.005 sits inside the 0.010 retry band,
# and the passing g1's CI bound sits under the 0.005 non-inferiority margin.
def _row(**overrides) -> pd.Series:
    base = {
        "distribution": "NegBin",
        "ship": False,
        **_ALL_PASS,
        "g4_pass": False,
        "g4_pit_ks": 0.055,
        "g4_pit_ks_max": 0.05,
        "g1_brier_diff_ci_hi": 0.003,
    }
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
        # Far miss (excess 0.049, the structural band): margin decides, not gate identity.
        (_row(g4_pit_ks=0.099), "loss", "joint", False),
        # Double near miss (NHL goalsAgainst, brief R3): g4 excess 0.0059, g1 over its bar by 0.0009.
        (
            _row(
                g4_pit_ks=0.0704,
                g4_pit_ks_max=0.0645,
                g1_pass=False,
                g1_brier_diff_ci_hi=0.0059,
            ),
            "loss",
            "joint",
            True,
        ),
        # A g1 miss past the 0.002 noise band blocks even with g4 near.
        (_row(g1_pass=False, g1_brier_diff_ci_hi=0.0246), "loss", "joint", False),
        # g6 is explicitly excluded — deferred pending the Experiment C measurement (brief R3).
        (_row(g6_pass=False), "loss", "joint", False),
        (_row(g5_pass=False), "loss", "joint", False),
        # Cell already pinned (or flag-forced) to calibrated — the first run WAS calibrated.
        (_row(), "calibrated", "joint", False),
        # Hurdle has no Optuna search: nothing for calibrated selection to pick from.
        (_row(distribution="ZINB"), "loss", "hurdle", False),
        # A stale inert zinb_mode on a non-ZINB cell must not block the retry.
        (_row(distribution="NegBin"), "loss", "hurdle", True),
        # Mixture has no calibrated trial-selection closure.
        (_row(distribution="Mixture"), "loss", "joint", False),
        (_row(ship=True, g4_pass=True), "loss", "joint", False),
        # g4 passed — the ship failure is someone else's (here g5): not our lever.
        (_row(g4_pass=True, g5_pass=False), "loss", "joint", False),
        (_row(g2_pass=False), "loss", "joint", False),
        (None, "loss", "joint", False),
        # Scorecard never ran: every gate NA reads as not-passing, so g2-g6 block.
        (_row(**{f"g{i}_pass": pd.NA for i in range(1, 7)}), "loss", "joint", False),
        # NA g1 with no measured CI (scorecard never wrote one): no band to be within, so blocks.
        (_row(g1_pass=pd.NA, g1_brier_diff_ci_hi=float("nan")), "loss", "joint", False),
        # Vacuous g1 (no book): auto-passed with NaN values must not block a sole-g4 near miss.
        (_row(g1_brier_diff_ci_hi=float("nan")), "loss", "joint", True),
    ],
)
def test_g4_only_retry_predicate(row, hpo_selection, zinb_mode, wanted):
    assert _g4_only_retry_wanted(row, hpo_selection, zinb_mode) is wanted


def _candidate() -> dict:
    return {"league": "NBA", "market": "FTM", "strategy_slug": "zinb", "edits": {"dist": "ZINB"}}


def test_pin_calibrated_meta_edits_and_disk(tmp_path, monkeypatch):
    meta_path = tmp_path / "stat_meta.json"
    meta = {"NBA": {"FTM": {"dist": "ZINB", "shipped": "devel"}}}
    meta_path.write_text(json.dumps(meta, indent=4) + "\n")
    monkeypatch.setattr(confirm, "_STAT_META", meta_path)
    cand = _candidate()

    _pin_calibrated(meta, cand)

    assert meta["NBA"]["FTM"]["hpo_selection"] == "calibrated"
    assert cand["edits"]["hpo_selection"] == "calibrated"
    text = meta_path.read_text()
    assert text.endswith("\n")
    on_disk = json.loads(text)
    assert on_disk["NBA"]["FTM"]["hpo_selection"] == "calibrated"
    # Committed stat_meta format: 4-space indent, so the review diff stays one line.
    assert '    "NBA"' in text


def test_retry_wanted_structural_never_reads_model_stats(monkeypatch):
    monkeypatch.setattr(
        confirm, "get_strategy", lambda slug: SimpleNamespace(is_structural=True)
    )
    monkeypatch.setattr(
        confirm, "_cell_row", lambda *a: pytest.fail("structural must short-circuit")
    )
    assert _retry_calibrated_wanted(_candidate(), {}) is False


def test_retry_wanted_unreadable_model_stats(monkeypatch):
    monkeypatch.setattr(
        confirm, "get_strategy", lambda slug: SimpleNamespace(is_structural=False)
    )

    def _raise(*args):
        raise OSError("no parquet")

    monkeypatch.setattr(confirm, "_cell_row", _raise)
    assert _retry_calibrated_wanted(_candidate(), {}) is False


def test_retry_wanted_delegates_to_predicate(monkeypatch):
    monkeypatch.setattr(
        confirm, "get_strategy", lambda slug: SimpleNamespace(is_structural=False)
    )
    monkeypatch.setattr(confirm, "_cell_row", lambda *a: _row())
    assert _retry_calibrated_wanted(_candidate(), {}) is True
    # The cell's own persisted knobs decide, exactly as the subprocess resolved them.
    assert _retry_calibrated_wanted(_candidate(), {"hpo_selection": "calibrated"}) is False
    monkeypatch.setattr(confirm, "_cell_row", lambda *a: _row(distribution="ZINB"))
    assert _retry_calibrated_wanted(_candidate(), {"zinb_mode": "hurdle"}) is False
