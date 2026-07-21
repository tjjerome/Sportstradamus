"""Golden routing guards for the final deterministic rushing-yards candidate."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.training import cli as training_cli
from sportstradamus.training import pipeline as pipe
from sportstradamus.training.cli import (
    _resolve_structural_strategy,
    _validate_structural_controls,
    meditate,
)
from sportstradamus.training.model_strategy_registry import (
    AUTO_STRUCTURAL_STRATEGY,
    BASE_STRUCTURAL_STRATEGY,
    get_strategy,
)
from sportstradamus.training.structural_context import build_rushing_expert_context
from sportstradamus.training.structural_strategies import (
    AUTO,
    NONE,
    RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
    RUSHING_POSITIONS,
    resolve_experiment_selection,
    validate_experiment_selection,
    validate_receiving_recipe,
)


def _frame(index, positions) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Player position": positions,
            "MeanYr": np.linspace(10.0, 20.0, len(index)),
            "STDYr": 3.0,
            "ZeroYr": 0.1,
            "Feature": np.arange(len(index), dtype=float),
        },
        index=index,
    )


def _context_splits() -> dict:
    train_index = pd.Index([9, 2, 8, 3, 7])
    validation_index = pd.Index([19, 12, 18, 13, 17])
    test_index = pd.Index([29, 22, 28, 23, 27])
    return {
        "X_train": _frame(train_index, [1, 3, 8, 1, 3]),
        "X_validation": _frame(validation_index, [3, 1, 8, 3, 1]),
        "X_test": _frame(test_index, [8, 3, 1, 1, 3]),
        "players_train": pd.Series([f"t-{i}" for i in range(5)], index=train_index),
        "players_validation": pd.Series([f"v-{i}" for i in range(5)], index=validation_index),
    }


def _tiny_support_floor() -> dict[str, int]:
    return {
        "train_rows": 1,
        "validation_rows": 1,
        "test_rows": 1,
        "train_players": 1,
        "validation_players": 1,
    }


def _valid_recipe(**overrides) -> dict:
    recipe = {
        "experiment": RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
        "league": "NFL",
        "market": "rushing yards",
        "distribution": "SkewNormal",
        "target_normalization": "ratio_meanyr",
        "posthoc": "none",
        "dist_training_loss": "crps",
        "blending": "nll",
        "hpo_selection": "loss",
        "sn_param": "direct",
        "stabilization": "None",
    }
    return recipe | overrides


def test_rushing_selector_and_fixed_recipe_are_exactly_scoped():
    validate_experiment_selection(
        RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
        league="NFL",
        market_selection="rushing yards",
    )
    validate_receiving_recipe(**_valid_recipe())
    with pytest.raises(ValueError, match="registered only"):
        validate_experiment_selection(
            RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
            league="NFL",
            market_selection="receiving yards",
        )
    with pytest.raises(ValueError, match="sn_param"):
        validate_receiving_recipe(**_valid_recipe(sn_param="centered"))


def test_cli_rejects_rushing_selector_without_deterministic_before_data_load():
    result = CliRunner().invoke(
        meditate,
        [
            "--league",
            "NFL",
            "--market",
            "rushing yards",
            "--structural-strategy",
            RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
        ],
    )
    assert result.exit_code == 2
    assert "requires --deterministic" in result.output


@pytest.mark.parametrize("path_flag", ["--matrix-only", "--rebuild-correlations"])
def test_cli_rejects_structural_selector_on_nontraining_paths(path_flag):
    result = CliRunner().invoke(
        meditate,
        [
            "--deterministic",
            "--league",
            "NFL",
            "--market",
            "rushing yards",
            path_flag,
            "--structural-strategy",
            RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
        ],
    )
    assert result.exit_code == 2
    assert "requires model training" in result.output


def test_structural_selector_auto_uses_cell_meta_and_none_disables_it():
    cell = {"structural_strategy": RUSHING_AFFINE_GROUPCDF_BOOK_POOL}
    assert (
        _resolve_structural_strategy(
            AUTO_STRUCTURAL_STRATEGY,
            cell,
            league="NFL",
            market="rushing yards",
        )
        == RUSHING_AFFINE_GROUPCDF_BOOK_POOL
    )
    assert (
        _resolve_structural_strategy(
            BASE_STRUCTURAL_STRATEGY,
            cell,
            league="NFL",
            market="rushing yards",
        )
        == BASE_STRUCTURAL_STRATEGY
    )
    with pytest.raises(ValueError, match="unknown model strategy"):
        _resolve_structural_strategy(
            AUTO_STRUCTURAL_STRATEGY,
            {"structural_strategy": "not-registered"},
            league="NFL",
            market="rushing yards",
        )
    with pytest.raises(ValueError, match="not a structural strategy"):
        _resolve_structural_strategy(
            AUTO_STRUCTURAL_STRATEGY,
            {"structural_strategy": "SkewNormal"},
            league="NFL",
            market="rushing yards",
        )


def test_cli_passes_auto_resolved_structural_strategy_to_train_market(monkeypatch):
    fake_stats = MagicMock()
    fake_stats.trim_gamelog.return_value = None
    cell_meta = {
        "dist": "SkewNormal",
        "structural_strategy": RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
        "target_normalization": "ratio_meanyr",
        "posthoc": "none",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending": "nll",
        "hpo_selection": "loss",
    }
    train_market = MagicMock()
    monkeypatch.setattr(training_cli, "StatsNFL", lambda: fake_stats)
    monkeypatch.setattr(
        training_cli,
        "load_stat_meta",
        lambda _path: {"NFL": {"rushing yards": cell_meta}},
    )
    monkeypatch.setattr(training_cli, "train_market", train_market)

    result = CliRunner().invoke(
        meditate,
        ["--deterministic", "--league", "NFL", "--market", "rushing yards"],
    )

    assert result.exit_code == 0, result.output
    assert train_market.call_args.kwargs["structural_strategy"] == (
        RUSHING_AFFINE_GROUPCDF_BOOK_POOL
    )


def test_structural_strategy_skips_calibrated_g4_retry(monkeypatch):
    model_stats_row = MagicMock()
    monkeypatch.setattr(training_cli, "_model_stats_row", model_stats_row)

    training_cli._retry_calibrated_if_g4_only(
        "NFL",
        "rushing yards",
        MagicMock(),
        MagicMock(),
        None,
        {
            "deterministic": False,
            "matrix_only": False,
            "structural_strategy": RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
        },
        {},
        MagicMock(),
    )

    model_stats_row.assert_not_called()


def test_structural_selector_validates_registry_distribution_and_fixed_controls():
    spec = get_strategy(RUSHING_AFFINE_GROUPCDF_BOOK_POOL)
    settings = {**spec.fixed_controls, **spec.fixed_persist}
    _validate_structural_controls(
        spec.slug,
        league="NFL",
        market="rushing yards",
        distribution="SkewNormal",
        settings=settings,
        deterministic=True,
    )
    with pytest.raises(ValueError, match="distribution 'ZINB' is not applicable"):
        _validate_structural_controls(
            spec.slug,
            league="NFL",
            market="rushing yards",
            distribution="ZINB",
            settings=settings | {"dist": "ZINB"},
            deterministic=True,
        )
    with pytest.raises(ValueError, match="sn_param='centered'"):
        _validate_structural_controls(
            spec.slug,
            league="NFL",
            market="rushing yards",
            distribution="SkewNormal",
            settings=settings | {"sn_param": "centered"},
            deterministic=True,
        )


def test_experiment_auto_honors_persisted_method_and_explicit_none_disables_it():
    cell = {"nfl_yards_experiment": RUSHING_AFFINE_GROUPCDF_BOOK_POOL}
    assert resolve_experiment_selection(AUTO, cell) == RUSHING_AFFINE_GROUPCDF_BOOK_POOL
    assert resolve_experiment_selection(NONE, cell) == NONE
    assert resolve_experiment_selection(AUTO, {}) == NONE


def test_rushing_context_routes_qb_rb_and_allows_unknown_pooled_fallback():
    context = build_rushing_expert_context(_context_splits(), support_floor=_tiny_support_floor())
    assert context["status"] == "active"
    assert context["routes"]["train"].tolist() == [
        "QB",
        "RB",
        "pooled_fallback",
        "QB",
        "RB",
    ]
    assert context["support"]["1"] == {
        "train_rows": 2,
        "validation_rows": 2,
        "test_rows": 2,
        "train_players": 2,
        "validation_players": 2,
    }
    assert context["support"]["3"] == context["support"]["1"]


def test_rushing_support_failure_kills_and_routes_everything_pooled():
    context = build_rushing_expert_context(
        _context_splits(),
        support_floor=_tiny_support_floor() | {"train_rows": 3},
    )
    assert context["status"] == "killed_fallback"
    assert "QB.train_rows=2<3" in context["kill_reason"]
    assert all(routes.eq("pooled_fallback").all() for routes in context["routes"].values())


def test_experts_fit_qb_then_rb_with_pooled_columns_labels_and_params(monkeypatch):
    splits = _context_splits()
    splits["y_train_labels"] = np.array([90.0, 20.0, 80.0, 30.0, 70.0])
    context = build_rushing_expert_context(splits, support_floor=_tiny_support_floor())
    opt_params = {"opt_rounds": 30}
    calls = []

    def _fit(dist, dist_obj, X_train, y_train, params, **kwargs):
        calls.append((dist, dist_obj, X_train.copy(), y_train.copy(), params, kwargs))
        code = int(pd.to_numeric(X_train["Player position"]).iloc[0])
        return f"expert-{code}"

    monkeypatch.setattr(pipe, "_step_fit_model", _fit)
    experts, fitted_context = pipe._step_fit_rushing_experts(
        context,
        dist="SkewNormal",
        dist_obj="same-dist-object",
        splits=splits,
        opt_params=opt_params,
        use_hurdle=False,
        normalize=True,
        offset_mode=False,
        shape_ceiling=None,
        deterministic=True,
        sn_param="direct",
    )

    assert list(experts) == [1, 3]
    assert experts == {1: "expert-1", 3: "expert-3"}
    assert fitted_context["status"] == "active"
    assert [int(call[2]["Player position"].iloc[0]) for call in calls] == [1, 3]
    assert all(list(call[2].columns) == list(splits["X_train"].columns) for call in calls)
    assert all(call[4] is opt_params for call in calls)
    np.testing.assert_array_equal(calls[0][3], [90.0, 30.0])
    np.testing.assert_array_equal(calls[1][3], [20.0, 70.0])


def _prediction_splits() -> dict:
    frames = {
        "train": _frame(pd.Index([5, 1, 4]), [3, 8, 1]),
        "validation": _frame(pd.Index([15, 11, 14]), [1, 3, 8]),
        "test": _frame(pd.Index([25, 21, 24]), [8, 1, 3]),
    }
    splits = {}
    for split, frame in frames.items():
        splits[f"X_{split}"] = frame
        splits[f"y_{split}"] = pd.DataFrame(
            {"Result": np.arange(len(frame), dtype=float)}, index=frame.index
        )
        splits[f"B_{split}"] = pd.DataFrame(
            {"Line": 10.0, "Odds": 0.5, "EV": 10.0}, index=frame.index
        )
    splits["y_train_labels"] = splits["y_train"]["Result"].to_numpy()
    return splits


def _prediction_context(splits: dict) -> dict:
    context_splits = splits | {
        "players_train": pd.Series(
            [f"t-{i}" for i in range(len(splits["X_train"]))],
            index=splits["X_train"].index,
        ),
        "players_validation": pd.Series(
            [f"v-{i}" for i in range(len(splits["X_validation"]))],
            index=splits["X_validation"].index,
        ),
    }
    return build_rushing_expert_context(
        context_splits,
        support_floor=dict.fromkeys(_tiny_support_floor(), 0),
    )


def test_prediction_dispatch_overwrites_by_index_and_keeps_unknown_pooled(monkeypatch):
    splits = _prediction_splits()
    context = _prediction_context(splits)
    base = {"pooled": 0.0, "QB": 100.0, "RB": 200.0}

    def _predict(model, _dist, frame, **_kwargs):
        return pd.DataFrame(
            {
                "loc": base[model] + frame.index.to_numpy(dtype=float),
                "scale": np.full(len(frame), base[model] + 1.0),
                "skewness": np.zeros(len(frame)),
            },
            index=frame.index,
        )

    monkeypatch.setattr(pipe, "predict_lss_params", _predict)
    result = pipe._step_predict_splits(
        "pooled",
        "SkewNormal",
        splits,
        normalize=True,
        offset_mode=False,
        sn_param="direct",
        expert_models={1: "QB", 3: "RB"},
        structural_context=context,
    )

    test_params = result["prob_params"]
    assert test_params.loc[21, "loc"] == 121.0
    assert test_params.loc[24, "loc"] == 224.0
    assert test_params.loc[25, "loc"] == 25.0
    assert result["structural_context"]["status"] == "active"


def test_nonfinite_expert_prediction_kills_and_atomically_restores_all_pooled(monkeypatch):
    splits = _prediction_splits()
    context = _prediction_context(splits)

    def _predict(model, _dist, frame, **_kwargs):
        values = frame.index.to_numpy(dtype=float)
        if model == "RB" and 24 in frame.index:
            values = values.copy()
            values[frame.index.get_loc(24)] = np.nan
        return pd.DataFrame({"loc": values, "scale": 1.0, "skewness": 0.0}, index=frame.index)

    monkeypatch.setattr(pipe, "predict_lss_params", _predict)
    result = pipe._step_predict_splits(
        "pooled",
        "SkewNormal",
        splits,
        normalize=True,
        offset_mode=False,
        sn_param="direct",
        expert_models={1: "QB", 3: "RB"},
        structural_context=context,
    )

    assert result["structural_context"]["status"] == "killed_fallback"
    assert "invalid RB expert parameters on test" in result["structural_context"]["kill_reason"]
    for split, result_key in (
        ("train", "prob_params_train"),
        ("validation", "prob_params_validation"),
        ("test", "prob_params"),
    ):
        params = result[result_key]
        np.testing.assert_allclose(params["loc"], params.index.to_numpy(dtype=float))
        assert result["structural_context"]["routes"][split].eq("pooled_fallback").all()


def test_rushing_candidate_has_separate_namespace_and_default_is_unchanged():
    assert pipe._deterministic_dump_subdir("SkewNormal", "joint", "ratio_meanyr") == (
        "ratio_meanyr"
    )
    assert (
        pipe._deterministic_dump_subdir(
            "SkewNormal",
            "joint",
            "ratio_meanyr",
            RUSHING_AFFINE_GROUPCDF_BOOK_POOL,
        )
        == f"ratio_meanyr__{RUSHING_AFFINE_GROUPCDF_BOOK_POOL}"
    )
    assert list(RUSHING_POSITIONS) == [1, 3]
