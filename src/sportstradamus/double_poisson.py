"""Custom Double Poisson distribution for use with LightGBMLSS.

Provides:
  - DoublePoissonTorch: a PyTorch Distribution subclass implementing Efron's
    (1986) double Poisson with an exact (truncated-logsumexp) normalizer.
  - DoublePoisson: a LightGBMLSS DistributionClass subclass that wraps
    DoublePoissonTorch and can be passed directly to LightGBMLSS() like any
    built-in distribution.

The family is mean-parametrized (mu trees move only the mean, to first order)
with both dispersion directions unbounded: Var ≈ mu/phi, so phi > 1 expresses
the under-dispersion / zero-deflation the NegBin variance floor cannot reach
and phi < 1 expresses over-dispersion. mu is only the *approximate* mean; the
serve/decode path recovers the exact mean by the same truncated series (see
``helpers.distributions``).

Usage:
    from sportstradamus.double_poisson import DoublePoisson
    from lightgbmlss.model import LightGBMLSS

    model = LightGBMLSS(dist=DoublePoisson())
"""

import torch
import torch.distributions as D
from lightgbmlss.distributions.distribution_utils import DistributionClass
from lightgbmlss.utils import softplus_fn
from torch.distributions import constraints

# Support-grid sizing for the exact normalizer: at least _DP_KMAX_MIN terms so
# low-mean cells still cover the tail, extend to mu + _DP_GRID_SIGMAS·sd for
# high-mean rows (kernel terms decay super-factorially past the mode), and cap
# at _DP_KMAX_CAP so a transient runaway mu during boosting cannot blow up the
# (rows × grid) tensor.
_DP_KMAX_MIN = 30
_DP_GRID_SIGMAS = 12.0
_DP_KMAX_CAP = 1024

# Hard parameter bounds applied inside the distribution (torch.clamp — gradient
# is zero when railed, which is visible in diagnostics and preferable to a
# silent numerical explosion). The cohort needs phi ≈ 0.5–3 (V/M 0.36–1.45 at
# the screened means); the bounds sit far outside that.
_DP_MU_FLOOR = 1e-6
_DP_PHI_FLOOR = 0.02
_DP_PHI_CEILING = 25.0


class DoublePoissonTorch(D.Distribution):
    r"""Double Poisson distribution (Efron 1986), exactly normalized.

    .. math::

        f(y; \mu, \phi) = \frac{1}{C(\mu, \phi)}\,
            \phi^{1/2} e^{-\phi\mu}
            \left(\frac{e^{-y} y^y}{y!}\right)
            \left(\frac{e\mu}{y}\right)^{\phi y}

    where :math:`C(\mu, \phi)` is the normalizing constant, computed here by
    exact ``logsumexp`` over a truncated support grid rather than Efron's
    closed-form approximation. :math:`E[Y] \approx \mu` and
    :math:`\mathrm{Var}[Y] \approx \mu/\phi`; :math:`\phi = 1` recovers the
    Poisson.

    Args:
        mu (float or Tensor):  Approximate-mean parameter :math:`\mu > 0`.
        phi (float or Tensor): Precision parameter :math:`\phi > 0`
            (under-dispersed above 1, over-dispersed below).
    """

    arg_constraints = {
        "mu": constraints.positive,
        "phi": constraints.positive,
    }
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, mu, phi, validate_args=None):
        mu, phi = torch.broadcast_tensors(
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(phi, dtype=torch.float32),
        )
        self.mu = torch.clamp(mu, min=_DP_MU_FLOOR)
        self.phi = torch.clamp(phi, min=_DP_PHI_FLOOR, max=_DP_PHI_CEILING)
        super().__init__(batch_shape=self.mu.shape, validate_args=validate_args)

    def _log_kernel(self, value):
        """Unnormalized log-density at ``value`` (broadcasts against mu/phi)."""
        # 0.5·log φ − φµ + (φ−1)y + (1−φ)·y·log y − log y! + φ·y·log µ,
        # with y·log y := 0 at y = 0 (xlogy).
        y_log_y = torch.special.xlogy(value, value)
        return (
            0.5 * torch.log(self.phi)
            - self.phi * self.mu
            + (self.phi - 1.0) * value
            + (1.0 - self.phi) * y_log_y
            - torch.lgamma(value + 1.0)
            + self.phi * value * torch.log(self.mu)
        )

    def _support_grid(self):
        """Integer support grid sized to cover the batch's widest predictive."""
        sd = torch.sqrt(self.mu / self.phi)
        kmax = int((self.mu + _DP_GRID_SIGMAS * sd).max().item()) + 1
        kmax = min(max(kmax, _DP_KMAX_MIN), _DP_KMAX_CAP)
        return torch.arange(kmax + 1, dtype=self.mu.dtype, device=self.mu.device)

    def _log_normalizer(self):
        grid = self._support_grid()
        mu = self.mu.unsqueeze(-1)
        phi = self.phi.unsqueeze(-1)
        y_log_y = torch.special.xlogy(grid, grid)
        log_kernel = (
            0.5 * torch.log(phi)
            - phi * mu
            + (phi - 1.0) * grid
            + (1.0 - phi) * y_log_y
            - torch.lgamma(grid + 1.0)
            + phi * grid * torch.log(mu)
        )
        return torch.logsumexp(log_kernel, dim=-1)

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.mu.dtype, device=self.mu.device)
        return self._log_kernel(value) - self._log_normalizer()

    def _grid_log_pmf(self):
        """Normalized log-PMF over the support grid, shape ``(K, *batch)``."""
        grid = self._support_grid()
        log_pmf = (
            self._log_kernel(grid.expand(*self.batch_shape, -1).movedim(-1, 0))
            - self._log_normalizer()
        )
        return grid, log_pmf

    def sample(self, sample_shape=torch.Size()):
        """Inverse-CDF sampling over the truncated support grid.

        Not reparameterized (no gradient path) — exists because
        ``LightGBMLSS.predict`` unconditionally draws samples via
        ``Distribution.sample`` even for ``pred_type="parameters"``.
        """
        with torch.no_grad():
            grid, log_pmf = self._grid_log_pmf()
            batch_numel = self.mu.numel()
            cdf = torch.cumsum(torch.exp(log_pmf), dim=0).reshape(-1, batch_numel)
            sample_shape = torch.Size(sample_shape)
            u = torch.rand(
                sample_shape.numel(), batch_numel,
                dtype=self.mu.dtype, device=self.mu.device,
            )
            # searchsorted wants the sorted axis last: (B, K) vs (B, n). Scale u
            # by the total truncated mass so u can never exceed cdf[-1].
            idx = torch.searchsorted(
                cdf.T.contiguous(), (u * cdf[-1]).T.contiguous()
            ).T
            out = idx.to(self.mu.dtype).clamp(max=grid[-1])
            return out.reshape(sample_shape + self.batch_shape)

    def _grid_moments(self):
        grid, log_pmf = self._grid_log_pmf()
        pmf = torch.exp(log_pmf)
        shaped_grid = grid.reshape((-1,) + (1,) * len(self.batch_shape))
        mean = (shaped_grid * pmf).sum(dim=0)
        second = (shaped_grid**2 * pmf).sum(dim=0)
        return mean, second - mean**2

    @property
    def mean(self):
        return self._grid_moments()[0]

    @property
    def variance(self):
        return self._grid_moments()[1]


class DoublePoisson(DistributionClass):
    """Double Poisson Distribution Class for LightGBMLSS.

    Distributional Parameters
    -------------------------
    mu : torch.Tensor
        Approximate-mean parameter (strictly positive).
    phi : torch.Tensor
        Precision parameter (strictly positive). phi > 1 is under-dispersed,
        phi < 1 over-dispersed, phi = 1 Poisson.

    Parameters
    -------------------------
    stabilization : str
        Stabilization method for the Gradient and Hessian.
        Options are ``"None"``, ``"MAD"``, ``"L2"``.
    loss_fn : str
        ``"nll"`` only: the double Poisson has no reparameterized sampler, so
        the CRPS loss path (which needs ``rsample``) is unavailable.
    initialize : bool
        Whether to initialize the distributional parameters with
        unconditional start values.
    """

    def __init__(
        self,
        stabilization: str = "None",
        loss_fn: str = "nll",
        initialize: bool = False,
    ):
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError(
                "Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'."
            )
        if loss_fn != "nll":
            raise ValueError(
                "DoublePoisson supports loss_fn='nll' only (no rsample for the crps path)."
            )
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        param_dict = {
            "mu": softplus_fn,  # positive
            "phi": softplus_fn,  # positive; hard-bounded inside DoublePoissonTorch
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        super().__init__(
            distribution=DoublePoissonTorch,
            univariate=True,
            discrete=True,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
            initialize=initialize,
        )
