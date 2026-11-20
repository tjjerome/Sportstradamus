"""Pins for the plain squared-error baseline ``train_market`` fits beside every cell.

``training.plain_baseline`` keeps docs/gbdt_feature_study.md §4 a standing per-cell number: a
plain LightGBM regression on the distributional model's own columns and train part, stopped
early on the train part's newest 15 %, scored against the served mean on the test rows. These
pins hold what a reader of ``plain_mse_ratio`` relies on: the ratio falls below 1 when the
served mean is the worse mean, the stopping rows are the newest by date whatever the label
order, the priced ratio reads pick'em and authentic rows aligned by label, and the fit leaves
``splits`` as it found it.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from sportstradamus.training import plain_baseline

N_ROWS = 3000
# Rows sharing one game date, so the date sort has ties to keep in order.
ROWS_PER_DATE = 20
# The train part's share, as train_market's temporal split takes it.
TRAIN_SHARE = 0.7
QUOTE_CLASSES = ("authentic", "pickem", "derived", "synthetic")


def _cell(classes=QUOTE_CLASSES):
    """A synthetic cell shaped like ``train_market``'s splits after ``_step_predict_splits``.

    The splits are sorted by label and the labels are shuffled against date order, so only a
    date sort finds the newest rows. ``Home`` is a pandas categorical, as the pipeline casts it.
    Returns the splits, every row's date, the served mean (the truth plus more noise than the
    outcome carries) and each test row's quote class in ``X_test`` order.
    """
    rng = np.random.default_rng(7)
    labels = pd.Index(rng.permutation(N_ROWS) + 100)
    dates = pd.Series(
        pd.Timestamp("2025-10-01") + pd.to_timedelta(np.arange(N_ROWS) // ROWS_PER_DATE, unit="D"),
        index=labels,
    )
    features = pd.DataFrame(
        {
            "MeanYr": rng.gamma(4.0, 2.0, N_ROWS),
            "Home": pd.Categorical(rng.integers(0, 2, N_ROWS)),
            "Noise": rng.normal(size=N_ROWS),
        },
        index=labels,
    )
    truth = features["MeanYr"] + 2.0 * features["Home"].astype(int)
    outcome = pd.DataFrame({"Result": truth + rng.normal(0.0, 1.0, N_ROWS)}, index=labels)
    n_train = int(N_ROWS * TRAIN_SHARE)
    train, test = labels[:n_train].sort_values(), labels[n_train:].sort_values()
    quote_class = rng.choice(classes, len(test))
    splits = {
        "X": features,
        "X_test": features.loc[test],
        "y_train": outcome.loc[train],
        "y_test": outcome.loc[test],
        # Reversed, so only a read aligned by label lands each class on its own row.
        "quote_authenticity_test": pd.Series(quote_class, index=test).iloc[::-1],
    }
    served = truth.loc[test].to_numpy() + rng.normal(0.0, 2.0, len(test))
    return splits, dates, served, quote_class


def test_plain_model_beats_a_noisier_served_mean_and_reads_priced_rows_by_label():
    splits, dates, served, quote_class = _cell()
    before = {key: value.copy() for key, value in splits.items()}

    plain = plain_baseline.fit_plain_baseline(splits, dates, served)

    outcome = splits["y_test"]["Result"].to_numpy()
    plain_error = (plain.predictions - outcome) ** 2
    served_error = (served - outcome) ** 2
    priced = np.isin(quote_class, ("authentic", "pickem"))
    diagnostics = plain.diagnostics
    assert diagnostics["plain_mse_ratio"] == pytest.approx(plain_error.mean() / served_error.mean())
    assert diagnostics["plain_mse_ratio"] < 1
    assert diagnostics["plain_mse_ratio_priced"] == pytest.approx(
        plain_error[priced].mean() / served_error[priced].mean()
    )
    assert 0 < diagnostics["plain_rounds"] < plain_baseline._MAX_ROUNDS
    for key, frame in before.items():
        assert frame.equals(splits[key]), f"fit_plain_baseline changed splits[{key!r}]"


@pytest.mark.parametrize("unpriced", [("derived", "synthetic"), None], ids=["derived", "legacy"])
def test_priced_ratio_is_nan_without_a_priced_row(unpriced):
    splits, dates, served, _ = _cell(unpriced or QUOTE_CLASSES)
    if unpriced is None:
        # A legacy matrix carries no QuoteAuthenticity column at all.
        splits["quote_authenticity_test"] = None

    diagnostics = plain_baseline.fit_plain_baseline(splits, dates, served).diagnostics

    assert np.isnan(diagnostics["plain_mse_ratio_priced"])
    assert np.isfinite(diagnostics["plain_mse_ratio"])


def test_early_stopping_reads_the_newest_rows_of_the_train_part(monkeypatch):
    seen = {}

    class RecordingRegressor(lgb.LGBMRegressor):
        def fit(self, X, y, **kwargs):
            seen["fit"] = X.index
            seen["stop"] = kwargs["eval_set"][0][0].index
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(plain_baseline.lgb, "LGBMRegressor", RecordingRegressor)
    splits, dates, served, _ = _cell()

    plain_baseline.fit_plain_baseline(splits, dates, served)

    train = splits["y_train"].index
    fit_rows, stop_rows = seen["fit"], seen["stop"]
    assert len(fit_rows) == int(len(train) * plain_baseline._FIT_SHARE)
    assert fit_rows.append(stop_rows).sort_values().equals(train)
    assert dates.loc[stop_rows].min() >= dates.loc[fit_rows].max()
    assert not fit_rows.append(stop_rows).isin(splits["X_test"].index).any()
