"""Unit tests for the P0.5 deterministic-mode helpers (no training)."""
import numpy as np
import torch

from sportstradamus.training.pipeline import DETERMINISTIC_FIXED_PARAMS, seed_everything


def test_seed_everything_returns_lgb_determinism_kwargs():
    kwargs = seed_everything(1234)
    for k in ("seed", "bagging_seed", "feature_fraction_seed"):
        assert kwargs[k] == 1234
    assert kwargs["deterministic"] is True
    assert kwargs["force_row_wise"] is True


def test_seed_everything_pins_python_numpy_and_torch():
    import random

    seed_everything(7)
    a = (random.random(), np.random.rand(5), torch.randn(5))
    seed_everything(7)
    b = (random.random(), np.random.rand(5), torch.randn(5))
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_fixed_params_have_required_keys():
    p = DETERMINISTIC_FIXED_PARAMS
    # opt_rounds is consumed as num_boost_round by model.train(...)
    assert isinstance(p["opt_rounds"], int) and p["opt_rounds"] > 0
    # deliberately tiny so the gate runs in seconds (reproducibility, not quality)
    assert p["opt_rounds"] <= 50
    for k in ("num_leaves", "learning_rate", "min_child_samples"):
        assert k in p
