"""Unit tests for the Operation Ship 75 search driver (``training.model_strategy_driver``).

The orchestration (retrain-grid enumeration, board assembly) is exercised with the heavy
per-corner ``meditate`` train+score monkeypatched out, so no model trains. The real per-corner
primitives are covered separately (``training.model_strategy_search`` + ``training.scorecard``,
and end-to-end by a deterministic meditate run); here we pin that the Optuna GridSampler study
visits every retrain corner (normalization × dist-loss × blend-loss) once, scores each by the
*honest* production gate (the deterministic dump's own val-fit calibration, read through
``gate_row`` — no test re-fit), and ranks the board by ship slack.
"""

import subprocess

import pandas as pd

from sportstradamus.training import calibration, model_strategy_driver, model_strategy_search


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
            "dispersion_cal": 1.0,
            "skew_cal": 0.0,
            "g4_pit_ks": 0.04,
            "g4_pit_ks_max": 0.05,
            "n": 2000,
        }
    ]


def test_search_space_is_all_retrain_kind_spec_stage_shape():
    """SEARCH_SPACE mirrors pipeline's hp_search_space ``[kind, spec]`` idiom, plus a ``stage``
    tag. Every axis is a *retrain* axis: the driver is a fixed-HP replica of the production
    pipeline (same val-fit calibration, same gate), so there is no separate post-hoc calibration
    axis — calibration is auto-fit on validation inside each deterministic train.
    """
    space = model_strategy_driver.SEARCH_SPACE
    assert set(space) == {"normalization", "dist_training_loss", "blending_loss_fn"}
    for kind, spec, stage in space.values():
        assert kind == "categorical"
        assert isinstance(spec, list) and spec
        assert stage == "retrain"
    assert space["normalization"] == [
        "categorical",
        list(model_strategy_search._DECODABLE_SN_NORMS),
        "retrain",
    ]
    assert space["dist_training_loss"] == ["categorical", ["crps", "nll"], "retrain"]
    assert space["blending_loss_fn"] == ["categorical", sorted(calibration.BLENDING_SLUGS), "retrain"]


def test_retrain_grid_keeps_only_retrain_axes():
    grid = model_strategy_driver._retrain_grid(model_strategy_driver.SEARCH_SPACE)
    assert set(grid) == {"normalization", "dist_training_loss", "blending_loss_fn"}
    assert grid["dist_training_loss"] == ["crps", "nll"]


def test_run_and_score_scores_with_honest_gate_row(monkeypatch):
    """One deterministic train, then the HONEST val-fit scorer (``_score_normalization`` → the
    production ``gate_row``), not a test-refit calibration sweep. Returns one row carrying the
    corner's loss axes.
    """
    monkeypatch.setattr(model_strategy_driver, "_run_deterministic_meditate", lambda *a, **k: None)
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

    monkeypatch.setattr(model_strategy_driver, "_score_normalization", fake_score)
    rows = model_strategy_driver._run_and_score(
        "WNBA", "AST", normalization="ratio_meanyr", dist_training_loss="nll", blending_loss_fn="crps"
    )
    assert captured["args"] == ("WNBA", "AST", "ratio_meanyr")
    assert len(rows) == 1
    assert rows[0]["dist_training_loss"] == "nll"
    assert rows[0]["blending_loss_fn"] == "crps"
    assert rows[0]["slack"] == 0.1


def test_search_cell_enumerates_retrain_grid_and_ranks_by_slack(monkeypatch):
    monkeypatch.setattr(model_strategy_driver, "_run_and_score", _fake_run_and_score)
    board = model_strategy_driver.search_cell("WNBA", "AST")

    space = model_strategy_driver.SEARCH_SPACE
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
    monkeypatch.setattr(model_strategy_driver, "_run_and_score", _fake_run_and_score)
    board = model_strategy_driver.search_cell("NBA", "FGA")
    assert (board["league"] == "NBA").all()
    assert (board["market"] == "FGA").all()


def test_run_deterministic_meditate_forwards_loss_axes(monkeypatch):
    """The search primitive forwards non-auto training/blend losses as their flags, omits on auto."""
    calls = []
    monkeypatch.setattr(
        model_strategy_search.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    model_strategy_search._run_deterministic_meditate(
        "WNBA", "AST", "ratio_meanyr", dist_training_loss="nll", blending_loss_fn="crps"
    )
    last = calls[-1]
    assert last[last.index("--dist-training-loss") + 1] == "nll"
    assert last[last.index("--blending-loss-fn") + 1] == "crps"

    model_strategy_search._run_deterministic_meditate("WNBA", "AST", "ratio_meanyr")
    assert "--dist-training-loss" not in calls[-1]
    assert "--blending-loss-fn" not in calls[-1]


def test_cli_runs_a_single_cell(monkeypatch):
    from click.testing import CliRunner

    monkeypatch.setattr(model_strategy_driver, "_run_and_score", _fake_run_and_score)
    result = CliRunner().invoke(model_strategy_driver.main, ["--league", "WNBA", "--market", "AST"])
    assert result.exit_code == 0, result.output
    assert "centered_additive_mean10" in result.output
