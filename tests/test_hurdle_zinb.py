"""Tests for the HurdleZINB two-stage distributional model.

Tests follow TDD order: each test is written to fail before implementation,
then pass once the module is complete.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training.pipeline import DETERMINISTIC_SEED


def _data(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic ZINB-like data with 35% structural zeros.

    Columns beyond the two noise features are required by
    ``set_model_start_values``: ``MeanYr``, ``STDYr``, ``ZeroYr``.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "MeanYr": np.full(n, 1.5),
            "STDYr": np.full(n, 1.0),
            "ZeroYr": np.full(n, 0.35),
        }
    )
    is_zero = rng.random(n) < 0.35
    y = np.where(is_zero, 0, rng.poisson(2.0, n) + 1).astype(float)
    return X, y


def test_hurdle_predict_returns_zinb_param_columns() -> None:
    """predict() must return exactly ['total_count', 'probs', 'gate'] with correct length."""
    from sportstradamus.hurdle import HurdleZINB

    X, y = _data()
    h = HurdleZINB()
    h.fit(X, y, hp={}, rounds=30)
    pp = h.predict(X)

    assert list(pp.columns) == ["total_count", "probs", "gate"]
    assert len(pp) == len(X)


def test_hurdle_gate_recovers_true_zero_rate() -> None:
    """Derived-π gate must reconstruct P(Y=0) ≈ 0.35.

    Checks the identity: P(Y=0) = π + (1−π)·NB(0).
    The mean reconstructed zero rate should be within 0.10 of the true 0.35.
    Also asserts the gate mean itself is substantially higher than the ~0.18
    that a joint ZINB would produce (the bug this module fixes).
    """
    from sportstradamus.hurdle import HurdleZINB

    X, y = _data()
    h = HurdleZINB()
    h.fit(X, y, hp={}, rounds=60)
    pp = h.predict(X)

    gate = pp["gate"].to_numpy()
    probs = pp["probs"].to_numpy()
    total_count = pp["total_count"].to_numpy()

    # NB(0) in PyTorch convention: (1 - p)^r
    nb0 = (1 - probs) ** total_count
    # ZINB identity reconstruction of P(Y=0)
    reconstructed_pzero = gate + (1 - gate) * nb0

    assert abs(reconstructed_pzero.mean() - 0.35) < 0.10, (
        f"Reconstructed P(Y=0)={reconstructed_pzero.mean():.3f} deviates >0.10 from 0.35"
    )
    # Gate should be well above the ~0.18 the joint ZINB produces.
    assert gate.mean() > 0.20, f"gate.mean()={gate.mean():.3f} unexpectedly low"


def test_hurdle_pickle_roundtrip() -> None:
    """Pickle → unpickle must produce identical predict() output."""
    from sportstradamus.hurdle import HurdleZINB

    X, y = _data()
    h = HurdleZINB()
    h.fit(X, y, hp={}, rounds=30)
    before = h.predict(X)

    blob = pickle.dumps(h)
    h2 = pickle.loads(blob)
    after = h2.predict(X)

    pd.testing.assert_frame_equal(before, after)


def test_hurdle_deterministic_under_seed() -> None:
    """Two independent HurdleZINB().fit(..., seed=DETERMINISTIC_SEED) must produce identical predictions."""
    from sportstradamus.hurdle import HurdleZINB

    X, y = _data()
    hp = {
        "num_leaves": 15,
        "learning_rate": 0.1,
        "min_child_samples": 50,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
    }
    h1 = HurdleZINB()
    h1.fit(X, y, hp=hp, rounds=20, seed=DETERMINISTIC_SEED)

    h2 = HurdleZINB()
    h2.fit(X, y, hp=hp, rounds=20, seed=DETERMINISTIC_SEED)

    pd.testing.assert_frame_equal(h1.predict(X), h2.predict(X), check_exact=True)


def test_hurdle_set_model_start_values_delegates() -> None:
    """set_model_start_values() must assign nb.start_values with shape (n, 2)."""
    from sportstradamus.hurdle import HurdleZINB

    X, y = _data()
    h = HurdleZINB()
    h.fit(X, y, hp={}, rounds=30)
    h.set_model_start_values(X)

    assert h.nb.start_values is not None
    assert h.nb.start_values.shape == (
        len(X),
        2,
    ), f"Expected start_values shape ({len(X)}, 2), got {h.nb.start_values.shape}"
