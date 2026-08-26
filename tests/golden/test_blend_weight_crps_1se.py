"""Behavior of the crps_1se blend-weight fitter: floor on no-edge cells, argmin on real edge.

All fixtures are synthetic, seeded, and I/O-free; the fitter's own bootstrap seed is a
module constant, so every value here is deterministic.
"""

import numpy as np
import pytest

from sportstradamus.training import calibration

# 30 clusters x 12 rows mirrors an NFL-sized panel — few enough clusters that a
# cluster-level noise "edge" has a wide paired SE, the regime the 1-SE rule targets.
_N_PLAYERS = 30
_ROWS_PER_PLAYER = 12


def _skewnormal_case(seed, model_noise_sd, edge_sd):
    rng = np.random.default_rng(seed)
    n = _N_PLAYERS * _ROWS_PER_PLAYER
    players = np.repeat(np.arange(_N_PLAYERS), _ROWS_PER_PLAYER)
    book_ev = rng.uniform(15.0, 30.0, n)
    if edge_sd:
        true_mean = book_ev + rng.normal(0.0, edge_sd, n)
        result = rng.normal(true_mean, 5.0)
        model_ev = true_mean + rng.normal(0.0, model_noise_sd, n)
    else:
        model_ev = book_ev + rng.normal(0.0, model_noise_sd, _N_PLAYERS)[players]
        result = rng.normal(book_ev, 5.0)
    kwargs = {"cv": 0.22, "model_sigma": np.full(n, 5.0), "model_skew_alpha": np.zeros(n)}
    return model_ev, book_ev, result, players, kwargs


def _grid_argmin(model_ev, book_ev, result, kwargs):
    loss = calibration._crps_loss_vector(model_ev, book_ev, result, "SkewNormal", **kwargs)
    means = [np.mean(loss(w)) for w in calibration._ONE_SE_GRID]
    return float(calibration._ONE_SE_GRID[int(np.argmin(means))])


def test_no_edge_cell_floors_while_plain_crps_chases_noise():
    # Model = book + player-level pure noise: any apparent CRPS gain is estimation error.
    model_ev, book_ev, result, players, kwargs = _skewnormal_case(7, 2.5, 0.0)
    w_1se = calibration.fit_model_weight_crps_1se(
        model_ev, book_ev, result, "SkewNormal", clusters=players, **kwargs
    )
    w_crps = calibration.fit_model_weight_crps(model_ev, book_ev, result, "SkewNormal", **kwargs)
    assert w_1se == pytest.approx(calibration._MODEL_WEIGHT_MIN)
    assert w_crps > 0.15


def test_no_edge_collapse_guards_return_plain_argmin():
    model_ev, book_ev, result, players, kwargs = _skewnormal_case(7, 2.5, 0.0)
    expected = _grid_argmin(model_ev, book_ev, result, kwargs)
    w_none = calibration.fit_model_weight_crps_1se(
        model_ev, book_ev, result, "SkewNormal", clusters=None, **kwargs
    )
    w_few = calibration.fit_model_weight_crps_1se(
        model_ev, book_ev, result, "SkewNormal", clusters=players % 5, **kwargs
    )
    assert w_none == pytest.approx(expected)
    assert w_few == pytest.approx(expected)
    assert expected > calibration._MODEL_WEIGHT_MIN


def test_real_edge_cell_stays_near_the_crps_argmin():
    # Model tracks a real row-level deviation the book misses: the 1-SE band is tight.
    model_ev, book_ev, result, players, kwargs = _skewnormal_case(3, 4.0, 6.0)
    w_1se = calibration.fit_model_weight_crps_1se(
        model_ev, book_ev, result, "SkewNormal", clusters=players, **kwargs
    )
    w_crps = calibration.fit_model_weight_crps(model_ev, book_ev, result, "SkewNormal", **kwargs)
    assert w_1se >= w_crps - 0.075
    assert w_1se >= 0.6


def test_dpo_crps_1se_branch_shrinks_and_guards():
    rng = np.random.default_rng(2)
    n = _N_PLAYERS * 10
    players = np.repeat(np.arange(_N_PLAYERS), 10)
    book_ev = rng.uniform(3.0, 8.0, n)
    result = rng.poisson(book_ev).astype(float)
    model_ev = book_ev * np.exp(rng.normal(0.0, 0.25, _N_PLAYERS)[players])
    phi = np.ones(n)
    w_crps = calibration.fit_dpo_weight(model_ev, book_ev, result, phi, 0.3, "crps")
    w_1se = calibration.fit_dpo_weight(
        model_ev, book_ev, result, phi, 0.3, "crps_1se", clusters=players
    )
    w_none = calibration.fit_dpo_weight(model_ev, book_ev, result, phi, 0.3, "crps_1se")
    w_one_cluster = calibration.fit_dpo_weight(
        model_ev, book_ev, result, phi, 0.3, "crps_1se", clusters=np.zeros(n)
    )
    assert w_1se < w_crps
    assert w_1se < w_none
    assert w_one_cluster == pytest.approx(w_none)
