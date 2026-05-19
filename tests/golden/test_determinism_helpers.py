"""Unit tests for the P0.5 deterministic-mode helpers (no training)."""
import numpy as np
import torch

from sportstradamus.training.pipeline import seed_everything


def test_seed_everything_returns_lgb_determinism_kwargs():
    kwargs = seed_everything(1234)
    for k in ("seed", "bagging_seed", "feature_fraction_seed"):
        assert kwargs[k] == 1234
    assert kwargs["deterministic"] is True
    assert kwargs["force_row_wise"] is True


def test_seed_everything_pins_numpy_and_torch():
    seed_everything(7)
    a = (np.random.rand(5), torch.randn(5))
    seed_everything(7)
    b = (np.random.rand(5), torch.randn(5))
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
