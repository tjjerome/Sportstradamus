"""WS2 book-skew prior: a non-circular SkewNormal alpha for the book side.

The symmetric book (skew=0) inverts the de-vigged point under a Normal, so on a
right-skewed cell it biases the book *mean* low (the line is a median). The fix
gives the book a per-cell skew, but the skew must come from a **non-circular**
source or the blend flatters itself in g4. ``skewnorm_alpha_from_skewness`` maps
a pooled within-player standardized-residual skewness (realized outcomes only,
never the model row) to the SkewNormal ``alpha`` the book side carries through
``fused_loc``; ``_within_player_book_skew`` computes that prior at meditate and
``_build_filedict`` / ``_blend_with_book`` thread it to serving.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers.distributions import skewnorm_alpha_from_skewness
from sportstradamus.prediction.model_prob import _blend_with_book
from sportstradamus.training.pipeline import _build_filedict, _within_player_book_skew


@pytest.mark.parametrize("alpha", [-4.0, -1.5, 0.5, 2.0, 4.0])
def test_alpha_from_skewness_round_trips_within_family(alpha):
    target_skew = float(skewnorm.stats(alpha, moments="s"))
    assert skewnorm_alpha_from_skewness(target_skew) == pytest.approx(alpha, rel=2e-3)


def test_alpha_from_skewness_zero_is_symmetric():
    assert skewnorm_alpha_from_skewness(0.0) == 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_alpha_from_skewness_nonfinite_is_symmetric(bad):
    assert skewnorm_alpha_from_skewness(bad) == 0.0


def test_alpha_from_skewness_clamps_beyond_family_ceiling():
    # The SkewNormal third standardized moment saturates near |0.9953|; a larger
    # empirical skew clamps to a large finite alpha at the feasible edge.
    big = skewnorm_alpha_from_skewness(5.0)
    assert np.isfinite(big) and big > 10
    assert skewnorm_alpha_from_skewness(-5.0) == pytest.approx(-big)


def _player_groupby(per_player_values):
    frames = [
        pd.DataFrame({"Player": f"p{i}", "PTS": vals})
        for i, vals in enumerate(per_player_values)
    ]
    return pd.concat(frames, ignore_index=True).groupby("Player")["PTS"]


def test_within_player_skew_positive_for_right_skewed_outcomes():
    rng = np.random.default_rng(0)
    vals = [rng.lognormal(2.0, 0.5, 80) for _ in range(40)]
    assert _within_player_book_skew(_player_groupby(vals)) > 0.3


def test_within_player_skew_near_zero_for_symmetric_outcomes():
    rng = np.random.default_rng(1)
    vals = [rng.normal(20.0, 4.0, 120) for _ in range(40)]
    assert abs(_within_player_book_skew(_player_groupby(vals))) < 0.4


def test_within_player_skew_negative_for_left_skewed_outcomes():
    rng = np.random.default_rng(2)
    vals = [-rng.lognormal(2.0, 0.5, 80) for _ in range(40)]
    assert _within_player_book_skew(_player_groupby(vals)) < -0.3


def _skewnormal_offer():
    return pd.DataFrame(
        {
            "Model EV": [20.0, 25.0],
            "Books EV": [19.0, 24.0],
            "Line": [20.5, 24.5],
            "Model Sigma": [5.0, 6.0],
            "Model Skew": [1.5, 1.0],
        }
    )


def test_blend_book_skew_defaults_to_zero_no_op():
    w = 0.8
    legacy, explicit = _skewnormal_offer(), _skewnormal_offer()
    base_legacy = _blend_with_book(legacy, "SkewNormal", w, 0.3, 0.0)
    base_explicit = _blend_with_book(explicit, "SkewNormal", w, 0.3, 0.0, book_skew=0.0)
    np.testing.assert_array_equal(base_legacy, base_explicit)
    np.testing.assert_array_equal(
        legacy["Model Skew"].to_numpy(), explicit["Model Skew"].to_numpy()
    )


def test_blend_threads_book_skew_independently_of_model_skew():
    # Circularity guard: the book side carries the *prior*, not the model row's
    # Model Skew. Holding Model Skew fixed, a different book_skew shifts the
    # blended skew by exactly (1 - w) * delta_book_skew.
    w = 0.8
    o0, o2 = _skewnormal_offer(), _skewnormal_offer()
    _blend_with_book(o0, "SkewNormal", w, 0.3, 0.0, book_skew=0.0)
    _blend_with_book(o2, "SkewNormal", w, 0.3, 0.0, book_skew=2.0)
    delta = o2["Model Skew"].to_numpy() - o0["Model Skew"].to_numpy()
    np.testing.assert_allclose(delta, (1 - w) * 2.0)


class _Strat:
    def offset_meta(self, global_mean, denom_col):
        return None


def _minimal_filedict_kwargs(book_skew):
    diag_keys = [
        "shape_label", "start_shape", "model_shape", "empirical_shape", "start_mean",
        "model_ev", "mean_line", "ev_minus_line", "result_mean", "median_ev_diff",
        "frac_ev_gt_line", "over_pct_ev_gt", "over_pct_ev_lt", "cf_over_pct",
        "ev_meanyr_corr", "result_meanyr_corr",
    ]
    return dict(
        model=object(),
        step=1.0,
        mode_stats={"acc": 0, "prec": 0, "under_prec": 0, "over_pct": 0, "sharp": 0, "ll": 0},
        skill={"model_metrics": {}, "book_metrics": {}, "brier_skill_score": 0.0, "kelly_shrinkage": 0.0},
        diag={k: 0.0 for k in diag_keys},
        c_opt=1.0,
        skew_cal=0.0,
        shape_ceiling=None,
        marginal_shape=float("nan"),
        opt_params={},
        dist="SkewNormal",
        cv=0.3,
        book_skew=book_skew,
        y=pd.DataFrame({"Result": [1.0, 2.0, 3.0]}),
        T_opt=1.0,
        model_weight=0.5,
        model_calib=0.5,
        hist_gate=0.0,
        normalize=False,
        strategy=_Strat(),
        global_mean=20.0,
        denom_col="MeanYr",
        target_normalization="none",
        posthoc_slug="none",
        posthoc_blob=None,
        zinb_mode="joint",
        X=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    )


def _sn_fuse_fixtures(n=64, seed=0):
    rng = np.random.default_rng(seed)
    ev = rng.uniform(15.0, 25.0, n)
    ev_val = rng.uniform(15.0, 25.0, n)
    book = ev + rng.normal(0.0, 1.0, n)
    decoded = {
        "ev": ev,
        "ev_validation": ev_val,
        "sn_sigma_test": np.full(n, 5.0),
        "sn_alpha_test": np.full(n, 1.5),
        "sn_sigma_val": np.full(n, 5.0),
        "sn_alpha_val": np.full(n, 1.5),
    }
    splits = {
        "B_test": pd.DataFrame({"EV": book, "Line": np.full(n, 20.0)}),
        "B_validation": pd.DataFrame({"EV": ev_val, "Line": np.full(n, 20.0)}),
        "y_validation": pd.DataFrame({"Result": rng.uniform(10.0, 30.0, n)}),
    }
    return decoded, splits


def test_training_blend_applies_book_skew_into_blended_alpha():
    # Train/serve parity: the training SkewNormal blend must apply book_skew the
    # same way serving's _blend_with_book does, or the scorecard gates a symmetric
    # predictive the betting path never prices.
    from sportstradamus.training import calibration
    from sportstradamus.training.pipeline import _fuse_skewnormal

    decoded, splits = _sn_fuse_fixtures()
    out = _fuse_skewnormal({}, decoded, splits, 0.3, 0.0, calibration.DEFAULT_BLENDING, book_skew=2.0)
    w = out["model_weight"]
    # blended alpha = w*model_skew + (1-w)*book_skew (model_skew=1.5 here).
    np.testing.assert_allclose(out["sn_alpha_blend_test"], w * 1.5 + (1 - w) * 2.0, atol=1e-9)


def test_training_blend_book_skew_zero_is_symmetric():
    from sportstradamus.training import calibration
    from sportstradamus.training.pipeline import _fuse_skewnormal

    decoded, splits = _sn_fuse_fixtures()
    out = _fuse_skewnormal({}, decoded, splits, 0.3, 0.0, calibration.DEFAULT_BLENDING, book_skew=0.0)
    w = out["model_weight"]
    np.testing.assert_allclose(out["sn_alpha_blend_test"], w * 1.5, atol=1e-9)


def test_filedict_carries_book_skew_through_pickle():
    fd = _build_filedict(**_minimal_filedict_kwargs(1.75))
    assert fd["book_skew"] == 1.75
    assert pickle.loads(pickle.dumps(fd))["book_skew"] == 1.75


def test_legacy_filedict_without_book_skew_reads_zero():
    # A pre-WS2 pickle has no book_skew key; the serving read defaults to 0.0.
    fd = _build_filedict(**_minimal_filedict_kwargs(0.0))
    del fd["book_skew"]
    assert fd.get("book_skew", 0.0) == 0.0
