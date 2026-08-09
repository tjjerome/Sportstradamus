"""Unit tests for the Double Poisson distribution (Efron 1986, exact-normalized).

Pins the §6.6 build requirements: truncation adequacy of the exact normalizer,
Poisson recovery at phi=1, autograd-clean gradients, both-directions dispersion,
and the LightGBMLSS wrapper contract.
"""

import numpy as np
import pytest
import torch

from sportstradamus.double_poisson import DoublePoisson, DoublePoissonTorch

PARAM_GRID = [(0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (0.5, 4.0), (2.0, 0.5), (2.0, 2.0), (6.0, 1.0)]


@pytest.mark.parametrize("mu,phi", PARAM_GRID)
def test_pmf_sums_to_one(mu, phi):
    dist = DoublePoissonTorch(torch.tensor([mu]), torch.tensor([phi]))
    grid = torch.arange(4096, dtype=torch.float32)
    total = torch.exp(dist.log_prob(grid.unsqueeze(-1))).sum().item()
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("mu", [0.5, 2.0, 6.0])
def test_phi_one_recovers_poisson(mu):
    dist = DoublePoissonTorch(torch.tensor([mu]), torch.tensor([1.0]))
    ks = torch.arange(30, dtype=torch.float32)
    got = dist.log_prob(ks.unsqueeze(-1)).squeeze(-1)
    want = torch.distributions.Poisson(torch.tensor(mu)).log_prob(ks)
    assert torch.allclose(got, want, atol=1e-4)


def test_truncation_round_trip():
    # Doubling the support grid must not move log_prob at the heaviest-tail
    # corner the cohort reaches (high mean, over-dispersed).
    mu, phi = torch.tensor([6.0]), torch.tensor([0.5])
    dist = DoublePoissonTorch(mu, phi)
    baseline = dist.log_prob(torch.tensor([[0.0], [3.0], [12.0]]))
    wide = dist._log_kernel(torch.tensor([[0.0], [3.0], [12.0]])) - torch.logsumexp(
        dist._log_kernel(
            torch.arange(2 * len(dist._support_grid()), dtype=torch.float32).unsqueeze(-1)
        ),
        dim=0,
    )
    assert torch.allclose(baseline, wide, atol=1e-5)


def test_gradients_finite_and_nonzero():
    raw_mu = torch.zeros(6, requires_grad=True)
    raw_phi = torch.zeros(6, requires_grad=True)
    mu = torch.nn.functional.softplus(raw_mu)
    phi = torch.nn.functional.softplus(raw_phi)
    y = torch.tensor([0.0, 1.0, 2.0, 0.0, 4.0, 1.0])
    nll = -DoublePoissonTorch(mu, phi).log_prob(y).sum()
    nll.backward()
    for grad in (raw_mu.grad, raw_phi.grad):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_both_dispersion_directions():
    mu = torch.tensor([2.0])
    under = DoublePoissonTorch(mu, torch.tensor([2.0]))
    over = DoublePoissonTorch(mu, torch.tensor([0.5]))
    vm_under = (under.variance / under.mean).item()
    vm_over = (over.variance / over.mean).item()
    assert vm_under < 0.75  # below the NegBin V/M >= 1 floor
    assert vm_over > 1.25
    assert under.mean.item() == pytest.approx(2.0, rel=0.15)
    assert np.isclose(vm_under, 1 / 2.0, rtol=0.25)


def test_zero_and_low_mean_edges():
    dist = DoublePoissonTorch(torch.tensor([0.5]), torch.tensor([4.0]))
    lp0 = dist.log_prob(torch.tensor([0.0]))
    assert torch.isfinite(lp0).all()
    # Under-dispersion pulls P(0) below the Poisson's (the zero-deflation fix).
    p0_poisson = torch.distributions.Poisson(torch.tensor(0.5)).log_prob(torch.tensor(0.0))
    assert lp0.item() < p0_poisson.item()


def test_wrapper_contract():
    dp = DoublePoisson()
    assert list(dp.param_dict.keys()) == ["mu", "phi"]
    assert dp.discrete is True
    assert dp.loss_fn == "nll"
    with pytest.raises(ValueError, match="nll"):
        DoublePoisson(loss_fn="crps")
    with pytest.raises(ValueError, match="stabilization"):
        DoublePoisson(stabilization="bogus")


def test_lightgbmlss_loss_path():
    dp = DoublePoisson()
    rng = np.random.default_rng(7)
    predt = rng.normal(0.5, 0.1, size=(50, 2)).astype(np.float64).flatten(order="F")
    target = torch.tensor(rng.poisson(1.5, size=50).astype(np.float32))
    _, loss = dp.get_params_loss(predt, target, start_values=[0.5, 0.5], requires_grad=True)
    assert torch.isfinite(loss)


def test_sample_shapes_and_moments():
    mu = torch.tensor([0.7, 2.0, 6.0])
    phi = torch.tensor([2.0, 1.0, 0.5])
    dist = DoublePoissonTorch(mu, phi)
    torch.manual_seed(3)
    draws = dist.sample((4000,))
    assert draws.shape == (4000, 3)
    assert (draws >= 0).all() and draws.dtype == mu.dtype
    assert torch.allclose(draws.mean(dim=0), dist.mean, rtol=0.1)
    assert dist.sample().shape == (3,)


def test_sample_deterministic_under_manual_seed():
    dist = DoublePoissonTorch(torch.tensor([1.5]), torch.tensor([1.2]))
    torch.manual_seed(11)
    a = dist.sample((256,))
    torch.manual_seed(11)
    b = dist.sample((256,))
    assert torch.equal(a, b)
