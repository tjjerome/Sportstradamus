"""Unit tests for the Operation Ship 75 search driver (``training.model_strategy_driver``).

The orchestration (retrain-grid enumeration, free post-hoc calibration sweep, board assembly) is
exercised with the heavy per-corner ``meditate`` train+score monkeypatched out, so no model trains.
The real per-corner primitives are covered separately (``training.model_strategy_search`` +
``training.scorecard``, and end-to-end by a deterministic meditate run); here we pin that the
Optuna GridSampler study visits every *retrain* corner once, that each corner contributes one row
per post-hoc (blending × calibration) combination, and that the board ranks by ship slack.
"""

import subprocess

import pandas as pd
import pytest

from sportstradamus.training import model_strategy_driver, model_strategy_search


def _fake_run_and_score(league, market, *, normalization, dist_training_loss, blending_choices):
    """Canned per-corner rows: one row per (blending × calibration mode); slack favors
    centered-mean10 + nll training-loss + skew-joint calibration, deterministic per corner.
    """
    norm_slack = {
        "ratio_meanyr": 0.00,
        "centered_additive_mean10": 0.20,
        "centered_additive_eb_meanyr_k10": -0.10,
    }[normalization]
    loss_bonus = 0.05 if dist_training_loss == "nll" else 0.0
    rows = []
    for blending in blending_choices:
        for mode in model_strategy_driver.CALIBRATION_MODES:
            slack = norm_slack + loss_bonus + (0.03 if mode == "skew-joint" else 0.0)
            rows.append(
                {
                    "normalization": normalization,
                    "dist_training_loss": dist_training_loss,
                    "blending_loss_fn": blending,
                    "calibration": mode,
                    "slack": slack,
                    "ships": slack > 0,
                    "dispersion_cal": 1.0,
                    "skew_cal": 0.0,
                    "g4_pit_ks": 0.04,
                    "g4_pit_ks_max": 0.05,
                    "n": 2000,
                }
            )
    return rows


def test_search_space_is_kind_spec_stage_shape():
    """SEARCH_SPACE mirrors pipeline's hp_search_space ``[kind, spec]`` idiom, plus a ``stage``
    tag splitting retrain axes (the GridSampler grid) from the free post-hoc sweep.
    """
    space = model_strategy_driver.SEARCH_SPACE
    assert set(space) == {"normalization", "dist_training_loss", "blending_loss_fn", "calibration"}
    for kind, spec, stage in space.values():
        assert kind == "categorical"
        assert isinstance(spec, list) and spec
        assert stage in {"retrain", "posthoc"}
    assert space["normalization"] == ["categorical", list(model_strategy_search._DECODABLE_SN_NORMS), "retrain"]
    assert space["dist_training_loss"] == ["categorical", ["crps", "nll"], "retrain"]
    assert space["blending_loss_fn"][2] == "posthoc"
    assert space["calibration"] == ["categorical", list(model_strategy_driver.CALIBRATION_MODES), "posthoc"]


def test_retrain_grid_keeps_only_retrain_axes():
    grid = model_strategy_driver._retrain_grid(model_strategy_driver.SEARCH_SPACE)
    assert set(grid) == {"normalization", "dist_training_loss"}
    assert grid["dist_training_loss"] == ["crps", "nll"]


def test_search_cell_enumerates_retrain_grid_times_posthoc_and_ranks_by_slack(monkeypatch):
    monkeypatch.setattr(model_strategy_driver, "_run_and_score", _fake_run_and_score)
    board = model_strategy_driver.search_cell("WNBA", "AST")

    space = model_strategy_driver.SEARCH_SPACE
    n_norms = len(space["normalization"][1])
    n_losses = len(space["dist_training_loss"][1])
    n_blend = len(space["blending_loss_fn"][1])
    n_cal = len(space["calibration"][1])
    # Retrain grid (norm × dist-loss) trained once each, fanned out over the free post-hoc sweep.
    assert len(board) == n_norms * n_losses * n_blend * n_cal
    # Every (retrain corner × calibration mode) appears exactly once.
    assert set(zip(board["normalization"], board["dist_training_loss"], board["calibration"], strict=True)) == {
        (n, loss, cal)
        for n in space["normalization"][1]
        for loss in space["dist_training_loss"][1]
        for cal in space["calibration"][1]
    }
    # Best corner: centered-mean10 + nll training-loss + skew-joint calibration (0.20 + 0.05 + 0.03).
    assert board.iloc[0]["normalization"] == "centered_additive_mean10"
    assert board.iloc[0]["dist_training_loss"] == "nll"
    assert board.iloc[0]["calibration"] == "skew-joint"
    assert board["slack"].is_monotonic_decreasing


def test_search_cell_carries_league_market_columns(monkeypatch):
    monkeypatch.setattr(model_strategy_driver, "_run_and_score", _fake_run_and_score)
    board = model_strategy_driver.search_cell("NBA", "FGA")
    assert (board["league"] == "NBA").all()
    assert (board["market"] == "FGA").all()


def test_run_and_score_fails_loud_on_unbuilt_blending():
    """A non-nll blend objective fails loud (and before any meditate) until the post-hoc crps refit lands."""
    with pytest.raises(NotImplementedError, match="crps-blending"):
        model_strategy_driver._run_and_score(
            "WNBA",
            "AST",
            normalization="ratio_meanyr",
            dist_training_loss="nll",
            blending_choices=("crps",),
        )


def test_run_deterministic_meditate_forwards_dist_training_loss(monkeypatch):
    """The search primitive forwards a non-auto training loss as ``--dist-training-loss`` and omits it on auto."""
    calls = []
    monkeypatch.setattr(
        model_strategy_search.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    model_strategy_search._run_deterministic_meditate("WNBA", "AST", "ratio_meanyr", dist_training_loss="nll")
    assert "--dist-training-loss" in calls[-1]
    assert calls[-1][calls[-1].index("--dist-training-loss") + 1] == "nll"

    model_strategy_search._run_deterministic_meditate("WNBA", "AST", "ratio_meanyr")
    assert "--dist-training-loss" not in calls[-1]


def test_cli_runs_a_single_cell(monkeypatch):
    from click.testing import CliRunner

    monkeypatch.setattr(model_strategy_driver, "_run_and_score", _fake_run_and_score)
    result = CliRunner().invoke(model_strategy_driver.main, ["--league", "WNBA", "--market", "AST"])
    assert result.exit_code == 0, result.output
    assert "centered_additive_mean10" in result.output
