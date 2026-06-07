"""Unit tests for the Operation Ship 75 per-trial search primitives (``training.model_strategy_search``).

The calibration-mode scorer is the free post-hoc axis of the search: off one trained deterministic
dump it gates every ``{none, dispersion, skew-joint, skew-sequential}`` mode without a retrain.
Here the heavy dump load + scorecard gate are monkeypatched so the test pins only the plumbing:
the dump pickle's auto-fit ``(dispersion_cal, skew_cal)`` reaches the blended recovery, and each
returned ship-row carries that mode's own fitted calibration (not the pickle's).
"""

import pandas as pd

from sportstradamus.training import model_strategy_search


def _canned_row(*, ship, g4_pit_ks, dispersion_cal, skew_cal):
    """A scorecard ship-row where Gate 4 binds the slack (g1/g2/g3/g5 pass with full headroom)."""
    return {
        "ship": ship,
        "g1_brier_diff_ci_hi": 0.0,
        "g2_star_z": 0.0,
        "g3_bench_z": 0.0,
        "g4_pit_ks": g4_pit_ks,
        "g4_pit_ks_max": 0.05,
        "g5_ece_debiased": 0.0,
        "n_rows": 1500,
        "dispersion_cal": dispersion_cal,
        "skew_cal": skew_cal,
    }


def test_score_calibration_modes_threads_pickle_cal_and_maps_rows(monkeypatch):
    captured = {}

    def fake_gate_rows(df, pred_col, **kwargs):
        captured.update(kwargs)
        return {
            "none": _canned_row(ship=False, g4_pit_ks=0.08, dispersion_cal=1.0, skew_cal=0.0),
            "skew-joint": _canned_row(ship=True, g4_pit_ks=0.03, dispersion_cal=1.27, skew_cal=2.3),
        }

    monkeypatch.setattr(model_strategy_search, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(
        model_strategy_search.pd,
        "read_pickle",
        lambda path: {"dispersion_cal": 1.4, "skew_cal": 0.5},
    )
    monkeypatch.setattr(model_strategy_search, "gate_rows_by_calibration_mode", fake_gate_rows)

    rows = model_strategy_search.score_calibration_modes("WNBA", "AST", "centered_additive_mean10")

    # The pickle's auto-fit baseline is what the blended recovery inverts against.
    assert captured["dispersion_cal"] == 1.4
    assert captured["skew_cal"] == 0.5
    assert captured["strategy"] == "centered_additive_mean10"
    assert captured["decode_strategy"] == "centered_additive_mean10"

    by_mode = {r["calibration"]: r for r in rows}
    assert set(by_mode) == {"none", "skew-joint"}
    # Each row carries its OWN mode's fitted (c, s), not the pickle's baseline.
    assert by_mode["skew-joint"]["dispersion_cal"] == 1.27
    assert by_mode["skew-joint"]["skew_cal"] == 2.3
    assert by_mode["skew-joint"]["ships"] is True
    assert by_mode["none"]["ships"] is False
    # Gate 4 binds: slack = (g4_max - g4) / g4_max.
    assert by_mode["skew-joint"]["slack"] == (0.05 - 0.03) / 0.05
    assert all(r["normalization"] == "centered_additive_mean10" for r in rows)
    assert by_mode["skew-joint"]["n"] == 1500
