"""Unit tests for the Operation Ship 75 per-trial search primitive (``training.model_strategy_search``).

``_score_normalization`` is the honest per-corner scorer: it loads one trained deterministic dump and
runs the production :func:`scorecard.gate_row` on its served (validation-fit-calibrated) predictive —
no test re-fit. Here the heavy dump load + scorecard gate are monkeypatched so the test pins only the
plumbing: the dump reaches ``gate_row`` under the trial's normalization, the pickle's auto-fit
calibration is surfaced for context, and the gate verdict maps to the board's slack / ships / Gate-4
fields.
"""

import pandas as pd

from sportstradamus.training import model_strategy_search


def _canned_row(*, ship, g4_pit_ks):
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
    }


def test_score_normalization_runs_production_gate_and_maps_row(monkeypatch):
    captured = {}

    def fake_gate_row(df, pred_col, **kwargs):
        captured.update(kwargs)
        return _canned_row(ship=True, g4_pit_ks=0.03)

    monkeypatch.setattr(model_strategy_search, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(
        model_strategy_search.pd,
        "read_pickle",
        lambda path: {"dispersion_cal": 1.27, "skew_cal": 2.3},
    )
    monkeypatch.setattr(model_strategy_search, "gate_row", fake_gate_row)
    monkeypatch.setattr(model_strategy_search, "apply_thresholds", lambda row: row)

    row = model_strategy_search._score_normalization("WNBA", "AST", "centered_additive_mean10")

    # The honest gate runs under the trial's normalization — no test re-fit of calibration.
    assert captured["strategy"] == "centered_additive_mean10"
    assert captured["decode_strategy"] == "centered_additive_mean10"
    assert row["normalization"] == "centered_additive_mean10"
    assert row["ships"] is True
    # Gate 4 binds: slack = (g4_max - g4) / g4_max.
    assert row["slack"] == (0.05 - 0.03) / 0.05
    # The dump already bakes the pipeline's val-fit calibration into the served predictive; the
    # pickle's (dispersion_cal, skew_cal) are surfaced for context only.
    assert row["dispersion_cal"] == 1.27
    assert row["skew_cal"] == 2.3
    assert row["n"] == 1500
