"""Unit tests for the Operation Ship 75 strategy sweep (``training.model_strategy_sweep``).

Two layers are covered here. The per-corner *primitive* ``_score_normalization`` is the honest
scorer: it loads one trained deterministic dump and runs the production :func:`scorecard.gate_row`
on its served (validation-fit-calibrated) predictive — no test re-fit; the heavy dump load +
scorecard gate are monkeypatched so the test pins only the plumbing. The *orchestration* (retrain-
grid enumeration, board assembly, verdict formatting) is exercised with the heavy per-corner
``meditate`` train+score monkeypatched out, so no model trains: the Optuna GridSampler study visits
every retrain corner (normalization × dist-loss × blend-loss) once, scores each by the honest gate,
and ranks the board by ship slack.
"""

import subprocess

import pandas as pd

from sportstradamus.training import calibration, model_strategy_sweep


def _canned_row(*, ship, g4_pit_ks):
    """A scorecard ship-row where Gate 4 binds the slack (g1/g2/g3/g5 pass with full headroom)."""
    return {
        "ship": ship,
        "g1_brier_diff_ci_hi": -0.01,
        "g1_brier_skill_score": 0.04,
        "g1_pass": True,
        "g2_star_z": 0.1,
        "g2_pass": True,
        "g3_bench_z": 0.2,
        "g3_pass": True,
        "g4_pit_ks": g4_pit_ks,
        "g4_pit_ks_max": 0.05,
        "g4_pass": g4_pit_ks < 0.05,
        "g5_ece_debiased": 0.03,
        "g5_pass": True,
        "central50_coverage": 0.49,
        "n_rows": 1500,
    }


def _fake_run_and_score(league, market, *, normalization, dist_training_loss, blending_loss_fn):
    """One honest row per retrain corner; slack favors centered-mean10 + nll + crps blend."""
    norm_slack = {
        "ratio_meanyr": 0.00,
        "centered_additive_mean10": 0.20,
        "centered_additive_eb_meanyr_k10": -0.10,
    }[normalization]
    loss_bonus = 0.05 if dist_training_loss == "nll" else 0.0
    blend_bonus = 0.02 if blending_loss_fn == "crps" else 0.0
    slack = norm_slack + loss_bonus + blend_bonus
    return [
        {
            "normalization": normalization,
            "dist_training_loss": dist_training_loss,
            "blending_loss_fn": blending_loss_fn,
            "slack": slack,
            "ships": slack > 0,
            "g1_pass": True,
            "g1_brier_diff_ci_hi": -0.01,
            "g1_brier_skill": 0.03,
            "g2_pass": True,
            "g2_star_z": 0.1,
            "g3_pass": True,
            "g3_bench_z": 0.2,
            "g4_pass": True,
            "g4_pit_ks": 0.04,
            "g4_pit_ks_max": 0.05,
            "g5_pass": True,
            "g5_ece_debiased": 0.03,
            "central50_coverage": 0.49,
            "dispersion_cal": 1.0,
            "skew_cal": 0.0,
            "n": 2000,
        }
    ]


def test_score_normalization_runs_production_gate_and_maps_row(monkeypatch):
    captured = {}

    def fake_gate_row(df, pred_col, **kwargs):
        captured.update(kwargs)
        return _canned_row(ship=True, g4_pit_ks=0.03)

    monkeypatch.setattr(model_strategy_sweep, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(
        model_strategy_sweep.pd,
        "read_pickle",
        lambda path: {"dispersion_cal": 1.27, "skew_cal": 2.3},
    )
    monkeypatch.setattr(model_strategy_sweep, "gate_row", fake_gate_row)
    monkeypatch.setattr(model_strategy_sweep, "apply_thresholds", lambda row: row)

    row = model_strategy_sweep._score_normalization("WNBA", "AST", "centered_additive_mean10")

    # The honest gate runs under the trial's normalization — no test re-fit of calibration.
    assert captured["strategy"] == "centered_additive_mean10"
    assert captured["decode_strategy"] == "centered_additive_mean10"
    assert row["normalization"] == "centered_additive_mean10"
    assert row["ships"] is True
    # Gate 4 binds: slack = (g4_max - g4) / g4_max.
    assert row["slack"] == (0.05 - 0.03) / 0.05
    # All five gates are surfaced (value + pass), not just g4 — so crps-vs-nll cost on g1/g5 is
    # visible on the board, not merely inferable from the min-gate slack.
    assert row["g1_pass"] is True and row["g1_brier_diff_ci_hi"] == -0.01
    assert row["g1_brier_skill"] == 0.04
    assert row["g2_pass"] is True and row["g2_star_z"] == 0.1
    assert row["g3_pass"] is True and row["g3_bench_z"] == 0.2
    assert row["g4_pass"] is True
    assert row["g5_pass"] is True and row["g5_ece_debiased"] == 0.03
    assert row["central50_coverage"] == 0.49
    # The dump already bakes the pipeline's val-fit calibration into the served predictive; the
    # pickle's (dispersion_cal, skew_cal) are surfaced for context only.
    assert row["dispersion_cal"] == 1.27
    assert row["skew_cal"] == 2.3
    assert row["n"] == 1500


def test_verdict_and_failed_gates():
    """The scannable corner verdict: SHIP when it ships, else KILL naming the failing gates."""
    assert model_strategy_sweep._verdict({"ships": True}) == "SHIP"
    killed = {"ships": False, "g1_pass": True, "g4_pass": False, "g5_pass": False}
    assert model_strategy_sweep._failed_gates(killed) == ["g4", "g5"]
    assert model_strategy_sweep._verdict(killed) == "KILL: g4 g5"


def test_search_space_is_all_retrain_kind_spec_stage_shape():
    """SEARCH_SPACE mirrors pipeline's hp_search_space ``[kind, spec]`` idiom, plus a ``stage``
    tag. Every axis is a *retrain* axis: the sweep is a fixed-HP replica of the production
    pipeline (same val-fit calibration, same gate), so there is no separate post-hoc calibration
    axis — calibration is auto-fit on validation inside each deterministic train.
    """
    space = model_strategy_sweep.SEARCH_SPACE
    assert set(space) == {"normalization", "dist_training_loss", "blending_loss_fn"}
    for kind, spec, stage in space.values():
        assert kind == "categorical"
        assert isinstance(spec, list) and spec
        assert stage == "retrain"
    assert space["normalization"] == [
        "categorical",
        list(model_strategy_sweep._DECODABLE_SN_NORMS),
        "retrain",
    ]
    assert space["dist_training_loss"] == ["categorical", ["crps", "nll"], "retrain"]
    assert space["blending_loss_fn"] == [
        "categorical",
        sorted(calibration.BLENDING_SLUGS),
        "retrain",
    ]


def test_retrain_grid_keeps_only_retrain_axes():
    grid = model_strategy_sweep._retrain_grid(model_strategy_sweep.SEARCH_SPACE)
    assert set(grid) == {"normalization", "dist_training_loss", "blending_loss_fn"}
    assert grid["dist_training_loss"] == ["crps", "nll"]


def test_run_and_score_scores_with_honest_gate_row(monkeypatch):
    """One deterministic train, then the HONEST val-fit scorer (``_score_normalization`` → the
    production ``gate_row``), not a test-refit calibration sweep. Returns one row carrying the
    corner's loss axes.
    """
    monkeypatch.setattr(model_strategy_sweep, "_run_deterministic_meditate", lambda *a, **k: None)
    captured = {}

    def fake_score(league, market, norm):
        captured["args"] = (league, market, norm)
        return {
            "normalization": norm,
            "slack": 0.1,
            "ships": True,
            "g4_pit_ks": 0.04,
            "g4_pit_ks_max": 0.05,
            "dispersion_cal": 1.2,
            "skew_cal": 0.5,
            "n": 2000,
        }

    monkeypatch.setattr(model_strategy_sweep, "_score_normalization", fake_score)
    rows = model_strategy_sweep._run_and_score(
        "WNBA",
        "AST",
        normalization="ratio_meanyr",
        dist_training_loss="nll",
        blending_loss_fn="crps",
    )
    assert captured["args"] == ("WNBA", "AST", "ratio_meanyr")
    assert len(rows) == 1
    assert rows[0]["dist_training_loss"] == "nll"
    assert rows[0]["blending_loss_fn"] == "crps"
    assert rows[0]["slack"] == 0.1


def test_search_cell_enumerates_retrain_grid_and_ranks_by_slack(monkeypatch):
    monkeypatch.setattr(model_strategy_sweep, "_run_and_score", _fake_run_and_score)
    board = model_strategy_sweep.search_cell("WNBA", "AST")

    space = model_strategy_sweep.SEARCH_SPACE
    n_norms = len(space["normalization"][1])
    n_losses = len(space["dist_training_loss"][1])
    n_blend = len(space["blending_loss_fn"][1])
    # One honest row per retrain corner (norm × dist-loss × blend-loss).
    assert len(board) == n_norms * n_losses * n_blend
    corner_cols = ["normalization", "dist_training_loss", "blending_loss_fn"]
    assert set(zip(*(board[c] for c in corner_cols), strict=True)) == {
        (n, loss, blend)
        for n in space["normalization"][1]
        for loss in space["dist_training_loss"][1]
        for blend in space["blending_loss_fn"][1]
    }
    # Best corner: centered-mean10 + nll + crps blend (0.20 + 0.05 + 0.02).
    assert board.iloc[0]["normalization"] == "centered_additive_mean10"
    assert board.iloc[0]["dist_training_loss"] == "nll"
    assert board.iloc[0]["blending_loss_fn"] == "crps"
    assert board["slack"].is_monotonic_decreasing
    assert "calibration" not in board.columns


def test_search_cell_carries_league_market_columns(monkeypatch):
    monkeypatch.setattr(model_strategy_sweep, "_run_and_score", _fake_run_and_score)
    board = model_strategy_sweep.search_cell("NBA", "FGA")
    assert (board["league"] == "NBA").all()
    assert (board["market"] == "FGA").all()


def test_search_cell_records_all_five_gates(monkeypatch):
    """The board surfaces every gate (value + pass), not just g4 — so a normalization/loss
    choice's cost on g1/g5 is readable off the board, not merely inferable from min-gate slack.
    """
    monkeypatch.setattr(model_strategy_sweep, "_run_and_score", _fake_run_and_score)
    board = model_strategy_sweep.search_cell("WNBA", "AST")
    for col in (
        "g1_pass",
        "g1_brier_diff_ci_hi",
        "g2_pass",
        "g2_star_z",
        "g3_pass",
        "g3_bench_z",
        "g4_pass",
        "g4_pit_ks",
        "g5_pass",
        "g5_ece_debiased",
    ):
        assert col in board.columns


def test_run_deterministic_meditate_forwards_loss_axes_and_captures_log(monkeypatch, tmp_path):
    """The primitive forwards non-auto training/blend losses as their flags, omits on auto, and
    captures meditate's output to a per-corner log file instead of streaming it.
    """
    monkeypatch.setattr(model_strategy_sweep, "_DETERMINISTIC_LOG_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        model_strategy_sweep.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    model_strategy_sweep._run_deterministic_meditate(
        "WNBA", "AST", "ratio_meanyr", dist_training_loss="nll", blending_loss_fn="crps"
    )
    last = calls[-1]
    assert last[last.index("--dist-training-loss") + 1] == "nll"
    assert last[last.index("--blending-loss-fn") + 1] == "crps"
    # The log file for this corner was opened under the (monkeypatched) research log root.
    assert model_strategy_sweep._log_path("WNBA", "AST", "ratio_meanyr", "nll", "crps").exists()

    model_strategy_sweep._run_deterministic_meditate("WNBA", "AST", "ratio_meanyr")
    assert "--dist-training-loss" not in calls[-1]
    assert "--blending-loss-fn" not in calls[-1]


def test_cli_runs_a_single_cell(monkeypatch, tmp_path):
    from click.testing import CliRunner

    monkeypatch.setattr(model_strategy_sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(
        model_strategy_sweep.main, ["--league", "WNBA", "--market", "AST", "--out", out]
    )
    assert result.exit_code == 0, result.output
    assert "centered_additive_mean10" in result.output


def test_cli_single_cell_upserts_into_existing_board(monkeypatch, tmp_path):
    """A single-cell run merges into the board file — replacing that cell's prior rows, keeping
    the others — so it refreshes the living board instead of clobbering it.
    """
    from click.testing import CliRunner

    monkeypatch.setattr(model_strategy_sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    runner = CliRunner()

    def run(league, market):
        return runner.invoke(
            model_strategy_sweep.main, ["--league", league, "--market", market, "--out", out]
        )

    assert run("WNBA", "AST").exit_code == 0
    assert run("NBA", "FGA").exit_code == 0
    board = pd.read_csv(out)
    assert set(zip(board["league"], board["market"], strict=True)) == {
        ("WNBA", "AST"),
        ("NBA", "FGA"),
    }
    n_two_cells = len(board)
    assert run("WNBA", "AST").exit_code == 0  # re-run replaces the cell's rows, not appends
    assert len(pd.read_csv(out)) == n_two_cells
