from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from sportstradamus.training import pipeline as pipe


class _CountStats:
    log_strings = {"player": "Player"}
    gamelog = pd.DataFrame(
        {
            "Player": np.repeat(["qb-a", "qb-b"], 12),
            "passing tds": np.tile([1.0, 2.0, 1.0], 8),
        }
    )


def test_isolated_distribution_fit_does_not_write_calibration_config(monkeypatch):
    save_zi = MagicMock()
    save_cv = MagicMock()
    monkeypatch.setattr(pipe, "save_zi_config", save_zi)
    monkeypatch.setattr(pipe, "save_cv_std_config", save_cv)
    splits = {
        "X_train": pd.DataFrame({"MeanYr": [1.0, 2.0, 1.5, 2.5]}),
        "y_train_labels": np.array([0.0, 1.0, 2.0, 1.0]),
    }

    pipe._step_select_distribution(
        splits,
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
