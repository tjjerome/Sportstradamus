"""Plain squared-error baseline fit beside each cell's distributional model.

docs/gbdt_feature_study.md §4 found the served mean (the distributional model's own mean)
losing to a plain LightGBM squared-error regression on the same inputs by 1.5 to 7 % MSE in
most NBA and NFL cells. ``train_market`` fits that plain model on every run, so the gap is a
standing per-cell number in ``model_stats``, and the plain mean rides the test-set dump as
``Plain``. A cell whose stat_meta sets ``mean_anchor: "plain"`` also serves it: its pickle
carries the booster and ``baselines.decode_model_params`` re-centres the distributional
model's mean on it.
"""

from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from sportstradamus.helpers.training_quotes import PRICED_AUTHENTICITY

# The study's settings (its x6_pover.py ``plain_preds``): small trees on half the columns with
# 60 rows a leaf, for NBA and NFL cells that train on only 14 to 48 rows per input column.
_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 5.0,
}
# A ceiling, not a choice: early stopping ends the fit long before it (NBA PRA stops near 600).
_MAX_ROUNDS = 3000
# Older share of the train part, by date, the plain model fits on; the newest rest picks its
# round count, so the stopping rows sit closest in time to the test rows.
_FIT_SHARE = 0.85
# Rounds without a lower squared error on the stopping rows before the fit stops.
_STOPPING_ROUNDS = 100
# One fixed seed: one reproducible fit, not the study's three-seed average.
_SEED = 0


class PlainBaseline(NamedTuple):
    """The plain model and its read on one cell's test rows.

    ``booster`` is the fitted squared-error model, the one an anchored cell's pickle carries
    as ``mean_anchor``. ``predictions`` is each test row's plain mean, in ``X_test`` order.
    """

    booster: lgb.Booster
    predictions: np.ndarray


def fit_plain_baseline(
    splits: dict, dates: pd.Series, determinism: dict[str, bool | int]
) -> PlainBaseline:
    """Fit the plain regression on the train part and predict the test rows.

    The plain model sees the distributional model's own feature columns and its whole train
    part, the oldest 70 % of rows by date. That includes the zero rows a gated SkewNormal fits
    without, because the served mean it stands beside carries the gate. It fits the older
    85 % of the train part, stops early on the newest 15 % and predicts the test rows. It only
    reads ``splits``, so nothing the distributional model, its calibration, the blend or the
    gates compute can move it.

    Args:
        splits: ``train_market``'s splits after ``_step_predict_splits``. Reads ``X`` (every
            row's features, categoricals already cast) and the ``y_train`` and ``X_test`` row
            sets.
        dates: Every matrix row's game date, indexed like the matrix.
        determinism: LightGBM's bit-reproducibility params on a ``--deterministic`` run,
            else empty.

    Returns:
        The booster and each test row's plain mean (see :class:`PlainBaseline`).
    """
    features = splits["X"]
    y_train = splits["y_train"]["Result"]
    by_date = dates.loc[y_train.index].sort_values(kind="stable").index
    n_fit = int(len(by_date) * _FIT_SHARE)
    fit_rows, stop_rows = by_date[:n_fit], by_date[n_fit:]
    model = lgb.LGBMRegressor(
        **_PARAMS, **determinism, n_estimators=_MAX_ROUNDS, random_state=_SEED, verbose=-1
    )
    model.fit(
        features.loc[fit_rows],
        y_train.loc[fit_rows],
        eval_set=[(features.loc[stop_rows], y_train.loc[stop_rows])],
        callbacks=[lgb.early_stopping(_STOPPING_ROUNDS, verbose=False)],
    )
    return PlainBaseline(model.booster_, model.predict(features.loc[splits["X_test"].index]))


def plain_diagnostics(
    splits: dict, plain: PlainBaseline, served_mean: np.ndarray
) -> dict[str, float]:
    """Score the plain mean against the served mean on the test rows.

    Args:
        splits: As :func:`fit_plain_baseline` reads them, plus ``y_test`` and
            ``quote_authenticity_test``.
        plain: The fitted plain model and its test-row predictions.
        served_mean: The model's own predictive mean of each test row, in ``X_test`` order:
            the dump's ``EV`` with the model's zero gate reapplied where its family has one.

    Returns:
        ``plain_mse_ratio`` (plain MSE over the served mean's on every test row; below 1 the
        plain model wins), ``plain_mse_ratio_priced`` (the same on pick'em and authentic
        rows, NaN without any) and ``plain_rounds`` (the early-stopped round count).
    """
    test_rows = splits["X_test"].index
    outcome = splits["y_test"].loc[test_rows, "Result"].to_numpy(dtype=float)
    plain_error = (plain.predictions - outcome) ** 2
    served_error = (served_mean - outcome) ** 2
    authenticity = splits["quote_authenticity_test"]
    priced = (
        np.zeros(len(test_rows), dtype=bool)
        if authenticity is None
        else authenticity.reindex(test_rows).isin(PRICED_AUTHENTICITY).to_numpy()
    )
    priced_ratio = (
        plain_error[priced].mean() / served_error[priced].mean() if priced.any() else np.nan
    )
    return {
        "plain_mse_ratio": float(plain_error.mean() / served_error.mean()),
        "plain_mse_ratio_priced": float(priced_ratio),
        "plain_rounds": plain.booster.best_iteration,
    }
