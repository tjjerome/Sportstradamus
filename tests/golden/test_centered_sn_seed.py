"""Golden pins for the centered-parametrization SkewNormal start values.

``set_model_start_values(..., sn_param="centered")`` seeds CenteredSkewNormal's
raw heads as ``(mean, log sd, atanh(gamma1/bound))``. The direct branch's alpha
seed is 0, so its loc/scale seeds are already the predictive mean/sd — the
centered mean/sd seeds must match them bit-for-bit — and the gamma1 response
bound duplicated in ``helpers.distributions`` (torch-free import graph, same
precedent as the DP constants) must track ``skew_normal_centered._GAMMA1_MAX``.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers.distributions import (
    _SN_CENTERED_GAMMA1_BOUND,
    _SN_CENTERED_GAMMA1_SEED,
    set_model_start_values,
)

_X = pd.DataFrame(
    {
        "MeanYr": [5.0, 12.0, 0.5, 25.0],
        "STDYr": [3.0, 4.0, 1.0, 6.0],
        "ZeroYr": [0.1, 0.0, 0.6, 0.05],
    }
)


class _Model:
    pass


def _sn_start_values(**kwargs):
    model = _Model()
    set_model_start_values(model, "SkewNormal", _X, **kwargs)
    return model.start_values


def test_centered_seed_shape_and_response_round_trip():
    sv = _sn_start_values(sn_param="centered")
    assert sv.shape == (len(_X), 3)
    # Response fns: mean → identity, sd → exp, gamma1 → bound·tanh.
    np.testing.assert_allclose(sv[:, 0], _X["MeanYr"])
    np.testing.assert_allclose(np.exp(sv[:, 1]), _X["STDYr"])
    np.testing.assert_allclose(
        _SN_CENTERED_GAMMA1_BOUND * np.tanh(sv[:, 2]), _SN_CENTERED_GAMMA1_SEED
    )


@pytest.mark.parametrize(
    "branch",
    [{}, {"offset_mode": True}, {"normalized": True}],
    ids=["raw", "offset", "normalized"],
)
def test_centered_mean_sd_seeds_equal_direct_loc_scale(branch):
    direct = _sn_start_values(sn_param="direct", **branch)
    centered = _sn_start_values(sn_param="centered", **branch)
    assert np.all(direct[:, 2] == 0.0)
    np.testing.assert_array_equal(centered[:, :2], direct[:, :2])


def test_centered_seed_maps_to_near_symmetric_direct_params():
    pytest.importorskip("torch")  # skew_normal_centered imports torch at module top
    from sportstradamus.skew_normal_centered import _centered_to_direct

    sv = _sn_start_values(sn_param="centered")
    mean = sv[:, 0]
    sd = np.exp(sv[:, 1])
    gamma1 = _SN_CENTERED_GAMMA1_BOUND * np.tanh(sv[:, 2])
    loc, scale, alpha = _centered_to_direct(mean, sd, gamma1, np)
    # The mapped direct params describe the same predictive mean/sd the direct
    # seed encodes (its alpha seeds at 0, so mean == loc and sd == scale there).
    m, v = skewnorm.stats(alpha, loc=loc, scale=scale, moments="mv")
    np.testing.assert_allclose(m, mean, rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(v), sd, rtol=1e-6)
    # gamma1=0.05 maps to alpha≈0.66 — the CP→DP inversion is cube-root-steep
    # near zero (the singularity motivating the centered heads), so an
    # off-zero seed cannot land alpha≈0; pin it positive like the seed and
    # sub-unit, far from the ~28.4 alpha at the railed gamma1 bound.
    assert np.all(alpha > 0.0) and np.all(alpha < 1.0)


def test_gamma1_bound_cross_pin():
    pytest.importorskip("torch")
    from sportstradamus import skew_normal_centered

    assert _SN_CENTERED_GAMMA1_BOUND == skew_normal_centered._GAMMA1_MAX


def test_sn_param_defaults_to_direct():
    np.testing.assert_array_equal(_sn_start_values(), _sn_start_values(sn_param="direct"))


def test_non_skewnormal_families_ignore_sn_param():
    negbin_default, negbin_centered = _Model(), _Model()
    set_model_start_values(negbin_default, "NegBin", _X)
    set_model_start_values(negbin_centered, "NegBin", _X, sn_param="centered")
    np.testing.assert_array_equal(negbin_default.start_values, negbin_centered.start_values)
