from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training import pipeline as pipe


class _CountStats:
    log_strings = {"player": "Player"}
    gamelog = pd.DataFrame(
        {
            "Player": np.repeat(["qb-a", "qb-b"], 12),
            "passing tds": np.tile([1.0, 2.0, 1.0], 8),
        }
    )


def _count_splits() -> dict:
    return {
        "X_train": pd.DataFrame({"MeanYr": [1.0, 2.0, 1.5, 2.5]}),
        "y_train_labels": np.array([0.0, 1.0, 2.0, 1.0]),
    }


def test_isolated_distribution_fit_does_not_write_calibration_config(monkeypatch):
    save_zi = MagicMock()
    save_cv = MagicMock()
    monkeypatch.setattr(pipe, "save_zi_config", save_zi)
    monkeypatch.setattr(pipe, "save_cv_std_config", save_cv)

    pipe._step_select_distribution(
        _count_splits(),
        _CountStats(),
        "passing tds",
        "NFL",
        "none",
        "joint",
        deterministic=False,
        isolated=True,
        dist_override="DPO",
    )

    save_zi.assert_not_called()
    save_cv.assert_not_called()


def test_isolated_skewnormal_fit_does_not_write_book_shape_config(monkeypatch):
    fit_book_shape = MagicMock()
    save_book_shape = MagicMock()
    monkeypatch.setattr(pipe.calibration, "fit_book_shape", fit_book_shape)
    monkeypatch.setattr(pipe, "save_book_shape_config", save_book_shape)

    pipe._step_persist_book_shape(
        pd.DataFrame({"Result": [1.0], "Line": [1.5]}),
        "NFL",
        "passing yards",
        "SkewNormal",
        deterministic=False,
        isolated=True,
    )

    fit_book_shape.assert_not_called()
    save_book_shape.assert_not_called()


def test_pinned_params_skip_the_search_and_take_the_current_monotone(monkeypatch):
    def search(*_args, **_kwargs):
        raise AssertionError("pinned params must not run the search")

    monkeypatch.setattr(pipe, "run_hyper_opt", search)
    pinned = {"params": {"num_leaves": 9, "opt_rounds": 7, "monotone_constraints": [0]}}

    params, monotone = pipe._step_select_hyperparams(
        pd.DataFrame({"a": [1.0], "MeanYr": [2.0]}),
        "SkewNormal",
        None,
        {"num_leaves": 41, "opt_rounds": 50},
        None,
        use_hurdle=False,
        deterministic=False,
        pinned=pinned,
    )

    assert params == {"num_leaves": 9, "opt_rounds": 7, "monotone_constraints": [0, 1]}
    assert monotone == [0, 1]


def test_pinned_pickle_supplies_cv_and_shape_ceiling(monkeypatch):
    monkeypatch.setattr(pipe, "stat_cv", {"NFL": {}})
    monkeypatch.setattr(pipe, "stat_zi", {"NFL": {}})
    pinned = {"distribution": "NegBin", "cv": 0.123, "shape_ceiling": 4.5}

    dist_info = pipe._step_select_distribution(
        _count_splits(),
        _CountStats(),
        "passing tds",
        "NFL",
        "none",
        "joint",
        deterministic=False,
        isolated=True,
        dist_override="NegBin",
        pinned=pinned,
    )

    assert (dist_info["cv"], dist_info["shape_ceiling"]) == (0.123, 4.5)
    assert pipe.stat_cv["NFL"]["passing tds"] == 0.123


def test_pinned_pickle_from_another_family_is_refused(monkeypatch):
    monkeypatch.setattr(pipe, "stat_zi", {"NFL": {}})

    with pytest.raises(ValueError, match="another family"):
        pipe._step_select_distribution(
            _count_splits(),
            _CountStats(),
            "passing tds",
            "NFL",
            "none",
            "joint",
            deterministic=False,
            isolated=True,
            dist_override="NegBin",
            pinned={"distribution": "SkewNormal", "cv": 0.1, "shape_ceiling": None},
        )
