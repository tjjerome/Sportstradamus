"""The hurdle warm-start path must resize the pickle's monotone vector to the current columns.

``_step_select_hyperparams``'s hurdle branch hands its params straight to LightGBM, which aborts
the fit unless ``monotone_constraints`` has exactly one entry per training column. A pickle's
vector is sized to the feature set it was trained on, so any feature-set change since then used to
kill the retrain with a ``num_total_features() == monotone_constraints.size()`` check failure.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.training.pipeline import _step_select_hyperparams


def _frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({name: [1.0, 2.0, 3.0] for name in columns})


def _hurdle_params(X_train, opt_params_in):
    return _step_select_hyperparams(
        X_train,
        "ZINB",
        None,
        opt_params_in,
        None,
        use_hurdle=True,
        deterministic=False,
    )


def test_stale_warm_start_vector_is_resized_and_other_params_kept():
    X_train = _frame(["MeanYr", "a", "b"])
    warm = {
        "opt_rounds": 200,
        "num_leaves": 41,
        "monotone_constraints": [0] * 419,
    }

    opt_params, monotone = _hurdle_params(X_train, warm)

    assert opt_params["monotone_constraints"] == monotone
    assert len(opt_params["monotone_constraints"]) == len(X_train.columns)
    assert opt_params["num_leaves"] == 41
    assert opt_params["opt_rounds"] == 200
    assert warm["monotone_constraints"] == [0] * 419


def test_cold_hurdle_start_sizes_the_vector_to_the_current_columns():
    X_train = _frame(["MeanYr", "a", "b", "c"])

    opt_params, monotone = _hurdle_params(X_train, None)

    assert opt_params["monotone_constraints"] == monotone
    assert len(monotone) == len(X_train.columns)
    assert opt_params["opt_rounds"] == 200


def test_meanyr_constraint_is_positional_and_count_families_get_none():
    X_train = _frame(["a", "MeanYr", "b"])

    _, hurdle_monotone = _hurdle_params(X_train, None)
    _, skew_monotone = _step_select_hyperparams(
        X_train,
        "SkewNormal",
        None,
        None,
        None,
        use_hurdle=True,
        deterministic=False,
    )

    assert hurdle_monotone == [0, 0, 0]
    assert skew_monotone == [0, 1, 0]
