"""Pins the mean/SD -> family-shape inversion the volume serving path depends on.

``_build_prob_params`` short-circuits a market that is also a volume stat: it reads the
budget-rescaled ``proj {market} mean`` / ``proj {market} std`` out of ``playerProfile``
instead of predicting fitted distribution parameters. ``fused_loc`` nevertheless blends the
family's own shape, so the branch has to recover that shape from the two moments it has.

Getting this wrong is not loud in every family. ``fused_loc`` clips DPO's ``phi``, so a
missing shape raises ``TypeError: '>=' not supported between instances of 'NoneType' and
'float'``; the other families run ``np.asarray(None, dtype=float)`` first, which yields
``nan`` and poisons the blend silently. Both are covered below.
"""

import numpy as np
import pytest

from sportstradamus.helpers import DecodedParams, predictive_std, shape_from_moments
from sportstradamus.helpers.distributions import _DP_PHI_CEILING, _DP_PHI_FLOOR

# Every std here exceeds sqrt(mean), keeping the pair inside NegBin's over-dispersed
# support so one shared fixture round-trips for all six families.
_MEAN = np.array([3.0, 10.0, 28.5, 62.0])
_STD = np.array([2.0, 4.0, 8.25, 12.0])

# The DecodedParams field each family names its shape with, and the offer-frame column
# the serving path carries it in.
_FAMILIES = [
    ("NegBin", "r", "Model R"),
    ("ZINB", "r", "Model R"),
    ("DPO", "phi", "Model Phi"),
    ("Gamma", "alpha", "Model Alpha"),
    ("ZAGamma", "alpha", "Model Alpha"),
    ("SkewNormal", "sigma", "Model Sigma"),
]


@pytest.mark.parametrize(("dist", "field", "column"), _FAMILIES)
def test_shape_round_trips_through_predictive_std(dist, field, column):
    recovered = shape_from_moments(dist, _MEAN, _STD)
    assert recovered is not None
    assert recovered[0] == column

    back = predictive_std(dist, DecodedParams(ev=_MEAN, **{field: recovered[1]}))
    np.testing.assert_allclose(back, _STD, rtol=1e-9)


def test_dpo_phi_is_clipped_into_the_kernel_bounds():
    # fused_loc clips phi to the same bounds; an out-of-range value would be silently
    # moved there anyway, so pin it at the source where the log-pool reads it.
    tiny_spread = shape_from_moments("DPO", np.array([100.0]), np.array([0.01]))
    huge_spread = shape_from_moments("DPO", np.array([1.0]), np.array([500.0]))

    assert tiny_spread[1][0] == _DP_PHI_CEILING
    assert huge_spread[1][0] == _DP_PHI_FLOOR


def test_underdispersed_negbin_falls_back_to_the_poisson_limit():
    # var < mean is outside NegBin's support, so the round-trip cannot hold. The floor
    # keeps r large-but-finite, which is the Poisson limit (var -> mean) rather than a
    # division by zero reaching the blend as inf/nan.
    mean = np.array([10.0])
    _, r = shape_from_moments("NegBin", mean, np.array([1.0]))
    assert np.isfinite(r).all()
    np.testing.assert_allclose(
        predictive_std("NegBin", DecodedParams(ev=mean, r=r)), np.sqrt(mean), rtol=1e-6
    )


def test_unknown_family_declines_rather_than_guessing():
    assert shape_from_moments("Poisson", _MEAN, _STD) is None
    assert shape_from_moments("Mixture", _MEAN, _STD) is None
