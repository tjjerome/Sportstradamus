"""Unit pins for the model-EV runaway guard.

Winsorizes a runaway model predictive mean toward the player's own realized scale
before the book-fusion pool, and drops offers with no informative own-scale history.
Research verdict: /tmp/researcher_model_ev_runaway.md (Mekelburg & Strauss 2024).
"""
from __future__ import annotations

import pandas as pd

from sportstradamus.prediction.model_prob import (
    _drop_no_history_offers,
    _own_scale,
    _sanitize_model_ev,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_own_scale_is_row_max_of_three_columns():
    assert _own_scale(_df([{"MeanYr": 1.0, "Mean10": 3.0, "STDYr": 2.0}]))[0] == 3.0


def test_own_scale_none_when_columns_absent():
    assert _own_scale(_df([{"Foo": 1.0}])) is None


def test_clamps_runaway_skewnormal_mean_and_rescales_sigma_to_hold_cv():
    # own_scale=2 -> cap=20; EV=50 is runaway, Sigma=10 (CV=0.2).
    df = _df([{"MeanYr": 2.0, "Mean10": 2.0, "STDYr": 1.0, "Projection": 50.0, "Model Sigma": 10.0}])
    _sanitize_model_ev(df, "SkewNormal")
    assert df.loc[0, "Projection"] == 20.0
    # Sigma shrinks by the same 20/50 factor so CV stays 0.2.
    assert df.loc[0, "Model Sigma"] == 4.0


def test_leaves_healthy_rows_untouched():
    df = _df([{"MeanYr": 2.0, "Mean10": 2.0, "STDYr": 1.0, "Projection": 15.0, "Model Sigma": 3.0}])
    _sanitize_model_ev(df, "SkewNormal")
    assert df.loc[0, "Projection"] == 15.0
    assert df.loc[0, "Model Sigma"] == 3.0


def test_count_family_clamps_mean_only():
    # ZINB has no Model Sigma; own_scale=1 -> cap=10; EV=2671 -> 10.
    df = _df([{"MeanYr": 1.0, "Mean10": 1.0, "STDYr": 0.5, "Projection": 2671.0}])
    _sanitize_model_ev(df, "ZINB")
    assert df.loc[0, "Projection"] == 10.0


def test_cap_floor_protects_low_volume_rows():
    # own_scale=0.2 clips to floor 0.5 -> cap=5; EV=4 stays, EV=6 clamps to 5.
    df = _df(
        [
            {"MeanYr": 0.2, "Mean10": 0.0, "STDYr": 0.0, "Projection": 4.0},
            {"MeanYr": 0.2, "Mean10": 0.0, "STDYr": 0.0, "Projection": 6.0},
        ]
    )
    _sanitize_model_ev(df, "ZINB")
    assert df.loc[0, "Projection"] == 4.0
    assert df.loc[1, "Projection"] == 5.0


def test_noop_without_own_scale_columns():
    df = _df([{"Projection": 9999.0}])
    _sanitize_model_ev(df, "ZINB")  # book-fallback frame: must not raise
    assert df.loc[0, "Projection"] == 9999.0


def test_drop_no_history_drops_zero_scale_keeps_history_and_combos():
    df = _df(
        [
            {"Player": "Has History", "MeanYr": 2.0, "Mean10": 2.0, "STDYr": 1.0},
            {"Player": "Debutant", "MeanYr": 0.0, "Mean10": 0.0, "STDYr": 0.0},
            {"Player": "A vs. B", "MeanYr": 0.0, "Mean10": 0.0, "STDYr": 0.0},
        ]
    )
    assert set(_drop_no_history_offers(df)["Player"]) == {"Has History", "A vs. B"}


def test_drop_no_history_noop_without_own_scale_columns():
    assert len(_drop_no_history_offers(_df([{"Player": "x", "Foo": 1.0}]))) == 1
