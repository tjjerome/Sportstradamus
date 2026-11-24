"""HurdleZINB caps its count stage's mean at the largest positive count it trained on.

Stage 2 fits the positive rows only but predicts every row, and on a few low scorers far from
those rows both NHL goals refits read r = 100 (the shape ceiling) and p ≈ .83: a count mean
r·p/(1 − p) near 490 goals. ``predict`` lowers p to bring that mean down to the ceiling before it
derives NB(0) and the gate, so P(Y = 0) stays the classifier's q. A pickle fit before the cap
carries no ceiling and must predict exactly as it did.

The stages are stubbed with the dtypes the real ones return (LightGBM's float64 q, LightGBMLSS's
float32 parameters), so the pins read the decode alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.hurdle import HurdleZINB

# Rows 0 and 1 run away as the NHL goals refits did; rows 2 and 3 sit far under any ceiling.
_R = np.array([100.0, 100.0, 5.0, 100.0], dtype=np.float32)
_P = np.array([0.83, 0.83, 0.2, 0.01], dtype=np.float32)
_Q = np.array([0.9, 0.95, 0.6, 0.7])
_RUNAWAY = np.array([True, True, False, False])
_X = pd.DataFrame(
    {"MeanYr": [0.2, 0.3, 1.5, 1.0], "STDYr": [0.4, 0.5, 1.2, 1.0], "ZeroYr": [0.85, 0.8, 0.4, 0.5]}
)
# The decode works in the count stage's float32, so its mean and its P(Y = 0) = q hold to that.
_FLOAT32_RTOL = 1e-6


class _Classifier:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _Q


class _CountStage:
    def predict(self, X: pd.DataFrame, pred_type: str) -> pd.DataFrame:
        return pd.DataFrame({"total_count": _R, "probs": _P})


def _decoded_before_the_cap() -> pd.DataFrame:
    nb0 = (1 - _P) ** _R
    gate = np.clip((_Q - nb0) / (1 - nb0), 0.0, 1.0)
    return pd.DataFrame({"total_count": _R, "probs": _P, "gate": gate}, index=_X.index)


def test_runaway_count_mean_is_capped_at_the_largest_training_count():
    y = np.tile([0.0, 0.0, 1.0, 2.0, 4.0], 40)
    f0 = np.random.default_rng(0).normal(size=len(y))
    X_train = pd.DataFrame({"f0": f0, "MeanYr": 1.4, "STDYr": 1.3, "ZeroYr": 0.4})
    hurdle = HurdleZINB().fit(X_train, y, hp={}, rounds=1)
    hurdle.clf, hurdle.nb = _Classifier(), _CountStage()

    params = hurdle.predict(_X)

    r, p, gate = params["total_count"], params["probs"], params["gate"]
    np.testing.assert_allclose((r * p / (1 - p))[_RUNAWAY], y.max(), rtol=_FLOAT32_RTOL)
    np.testing.assert_array_equal(r, _R)
    np.testing.assert_allclose(gate + (1 - gate) * (1 - p) ** r, _Q, rtol=_FLOAT32_RTOL)
    pd.testing.assert_frame_equal(
        params.loc[~_RUNAWAY], _decoded_before_the_cap().loc[~_RUNAWAY], check_exact=True
    )


def test_a_pickle_fit_before_the_cap_predicts_as_it_did():
    # Unpickling restores __dict__ without __init__, and a pre-cap fit pickled these three.
    old = HurdleZINB.__new__(HurdleZINB)
    old.__dict__.update(clf=_Classifier(), nb=_CountStage(), _shape_ceiling=100.0)

    pd.testing.assert_frame_equal(old.predict(_X), _decoded_before_the_cap(), check_exact=True)
