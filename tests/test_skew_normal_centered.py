"""Unit pins for the centered-parametrization SkewNormal.

The centered class must be indistinguishable from a direct SkewNormal at
mapped parameters (same density, same samples) while exposing (mean, sd,
gamma1) heads to the booster and re-emitting direct (loc, scale, alpha)
from ``predict_dist`` — the zero-serving-delta contract.
"""

import math

import numpy as np
import pytest
import torch

from sportstradamus.skew_normal import SkewNormalTorch
from sportstradamus.skew_normal_centered import (
    _GAMMA1_MAX,
    CenteredSkewNormal,
    CenteredSkewNormalTorch,
    _centered_to_direct,
    _gamma1_response,
)

# Max attainable skew-normal Pearson skewness sqrt(2)(4-pi)/(pi-2)^1.5.
_SN_GAMMA1_ATTAINABLE_MAX = 0.9952717


def _forward_gamma1(alpha):
    """Pearson skewness of a SkewNormal with shape ``alpha`` (numpy)."""
    delta = alpha / np.sqrt(1.0 + alpha**2)
    m = delta * math.sqrt(2.0 / math.pi)
    return (4.0 - math.pi) / 2.0 * m**3 / (1.0 - m**2) ** 1.5


GAMMA1_GRID = [-0.9, -0.5, -0.05, 0.05, 0.3, 0.7, 0.99]


@pytest.mark.parametrize("g1", GAMMA1_GRID)
def test_inversion_round_trip_moments(g1):
    mean, sd = 7.5, 2.0
    dist = CenteredSkewNormalTorch(mean, sd, g1)
    assert dist.mean.item() == pytest.approx(mean, abs=5e-3)
    assert dist.variance.sqrt().item() == pytest.approx(sd, abs=5e-3)
    assert _forward_gamma1(dist.alpha.numpy()) == pytest.approx(g1, abs=2e-3)


def test_gamma1_zero_is_exact_gaussian():
    dist = CenteredSkewNormalTorch(3.0, 1.5, 0.0)
    assert dist.alpha.item() == 0.0
    assert dist.loc.item() == pytest.approx(3.0)
    assert dist.scale.item() == pytest.approx(1.5)


def test_log_prob_matches_direct_at_mapped_params():
    mean = np.array([2.0, 5.0])
    sd = np.array([1.0, 2.5])
    g1 = np.array([0.6, -0.4])
    loc, scale, alpha = _centered_to_direct(mean, sd, g1, np)
    centered = CenteredSkewNormalTorch(mean, sd, g1)
    direct = SkewNormalTorch(loc, scale, alpha)
    x = torch.tensor([[1.0, 4.0], [2.5, 6.0], [0.0, 9.0]])
    np.testing.assert_allclose(centered.log_prob(x).numpy(), direct.log_prob(x).numpy(), atol=1e-5)


def test_numpy_and_torch_mappings_agree():
    mean = np.linspace(-2.0, 8.0, 7)
    sd = np.linspace(0.5, 3.0, 7)
    g1 = np.linspace(-0.99, 0.99, 7)
    np_out = _centered_to_direct(mean, sd, g1, np)
    torch_out = _centered_to_direct(*(torch.tensor(v) for v in (mean, sd, g1)), torch)
    for a, b in zip(np_out, torch_out, strict=True):
        np.testing.assert_allclose(a, b.numpy(), rtol=1e-6)


def test_gamma1_response_bounded_inside_attainable_max():
    raw = torch.tensor([-50.0, -3.0, 0.0, 3.0, 50.0])
    g1 = _gamma1_response(raw)
    assert torch.all(g1.abs() <= _GAMMA1_MAX)
    assert _GAMMA1_MAX < _SN_GAMMA1_ATTAINABLE_MAX
    assert g1[2].item() == 0.0
    # Mapped alpha stays finite even railed against the response bound.
    _, _, alpha = _centered_to_direct(
        torch.tensor([0.0]), torch.tensor([1.0]), g1.max().reshape(1), torch
    )
    assert torch.isfinite(alpha).all()


def test_autograd_through_inversion():
    for g1_val in (-0.9, -0.1, 0.2, 0.95):
        mean = torch.tensor([1.0], requires_grad=True)
        sd = torch.tensor([1.2], requires_grad=True)
        g1 = torch.tensor([g1_val], requires_grad=True)
        dist = CenteredSkewNormalTorch(mean, sd, g1)
        loss = -dist.log_prob(torch.tensor([1.4])).sum()
        loss.backward()
        for p in (mean, sd, g1):
            assert torch.isfinite(p.grad).all()


def test_wrapper_contract_and_predict_reemits_direct_params():
    lgb = pytest.importorskip("lightgbm")
    from lightgbmlss.model import LightGBMLSS

    dist = CenteredSkewNormal()
    assert list(dist.param_dict.keys()) == ["mean", "sd", "gamma1"]
    assert dist.distribution is CenteredSkewNormalTorch
    assert dist.distribution_arg_names == ["mean", "sd", "gamma1"]

    rng = np.random.default_rng(5)
    n = 400
    x0 = rng.normal(size=n)
    X = np.column_stack([x0, rng.normal(size=n)])
    # SkewNormal-realizable target (gamma1 ~ 0.67 at alpha=3) so the skew head
    # has an interior optimum instead of railing against the response bound.
    from scipy.stats import skew, skewnorm

    y = 4.0 + 2.0 * x0 + skewnorm.rvs(3.0, scale=1.5, size=n, random_state=7)

    model = LightGBMLSS(CenteredSkewNormal())
    # Production seeds the heads via set_model_start_values; mirror that recipe
    # (mean, log sd, atanh(gamma1/bound)) — the auto start-value fallback seeds
    # all heads at 0.5, which stalls the mean head under stabilization.
    g1_seed = float(np.clip(skew(y), -0.5, 0.5))
    model.start_values = np.array([y.mean(), math.log(y.std()), math.atanh(g1_seed / _GAMMA1_MAX)])
    torch.manual_seed(0)
    model.train(
        {"num_leaves": 7, "learning_rate": 0.1, "verbose": -1, "seed": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=100,
    )
    # Regression guard for the boosting stall: unstabilized centered heads
    # NaN'd out within ~4 rounds before the L2 default landed. Healthy runs
    # terminate benignly anywhere above ~40 rounds depending on CPU model
    # (float wobble in the histogram sums), so the bar only needs to clear
    # the catastrophe; the finiteness/mean-recovery asserts below carry the
    # model-quality check.
    assert model.booster.current_iteration() >= 20

    pp = model.predict(X[:25], pred_type="parameters")
    # Zero-serving-delta contract: downstream consumers see the direct frame.
    assert list(pp.columns) == ["loc", "scale", "alpha"]
    assert np.isfinite(pp.to_numpy()).all()
    assert (pp["scale"] > 0).all()

    # The re-emitted direct params must describe the same distribution the
    # centered heads trained: mean recovered from (loc, scale, alpha) sits
    # near the target mean.
    delta = pp["alpha"] / np.sqrt(1.0 + pp["alpha"] ** 2)
    mean_back = pp["loc"] + pp["scale"] * delta * math.sqrt(2.0 / math.pi)
    assert abs(mean_back.mean() - y.mean()) < 1.0


def test_crps_loss_path_has_rsample():
    dist = CenteredSkewNormalTorch(2.0, 1.0, 0.5)
    torch.manual_seed(2)
    draws = dist.rsample((5000,))
    assert draws.shape[0] == 5000
    assert draws.mean().item() == pytest.approx(2.0, abs=0.1)
