"""
Custom SkewNormal distribution for use with LightGBMLSS.

Provides:
  - SkewNormalTorch: a PyTorch Distribution subclass implementing the skew-normal
    distribution with loc, scale, and alpha (skewness) parameters.
  - SkewNormal: a LightGBMLSS DistributionClass subclass that wraps SkewNormalTorch
    and can be passed directly to LightGBMLSS() like any built-in distribution.

Usage:
    from sportstradamus.skew_normal import SkewNormal
    from lightgbmlss.model import LightGBMLSS

    model = LightGBMLSS(dist=SkewNormal())
    # ... use model.train(), model.predict(), etc. exactly as normal
"""

import math
import torch
import torch.distributions as D
from torch.distributions import constraints
from lightgbmlss.distributions.distribution_utils import DistributionClass
from lightgbmlss.utils import identity_fn, exp_fn, softplus_fn


# ---------------------------------------------------------------------------
# PyTorch Distribution: SkewNormal
# ---------------------------------------------------------------------------

class SkewNormalTorch(D.Distribution):
    r"""
    Skew-Normal distribution.

    The skew-normal distribution is a continuous probability distribution that
    generalises the normal distribution to allow for non-zero skewness.

    .. math::

        f(x; \xi, \omega, \alpha) =
            \frac{2}{\omega}\,
            \phi\!\left(\frac{x - \xi}{\omega}\right)\,
            \Phi\!\left(\alpha\,\frac{x - \xi}{\omega}\right)

    where :math:`\phi` and :math:`\Phi` are the standard normal PDF and CDF,
    :math:`\xi` is the location, :math:`\omega > 0` is the scale, and
    :math:`\alpha \in \mathbb{R}` is the shape (skewness) parameter.

    When :math:`\alpha = 0` the distribution reduces to a Gaussian.

    Args:
        loc (float or Tensor):   Location parameter :math:`\xi`.
        scale (float or Tensor): Scale parameter :math:`\omega > 0`.
        alpha (float or Tensor): Skewness parameter :math:`\alpha`.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "alpha": constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, alpha, validate_args=None):
        self.loc, self.scale, self.alpha = torch.broadcast_tensors(
            torch.as_tensor(loc, dtype=torch.float32),
            torch.as_tensor(scale, dtype=torch.float32),
            torch.as_tensor(alpha, dtype=torch.float32),
        )
        batch_shape = self.loc.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    # ---- helpers ----------------------------------------------------------
    _standard_normal = D.Normal(0.0, 1.0)

    @staticmethod
    def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable log Φ(x) using erfc for the left tail."""
        # For large positive x, log(Phi(x)) ≈ 0.
        # For large negative x, use the erfc formulation to avoid log(0).
        return torch.where(
            x > -5.0,
            torch.log(torch.clamp(0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))), min=1e-30)),
            # log(erfc(-x/sqrt(2))/2)  – numerically stable for x << 0
            torch.log(torch.clamp(torch.erfc(-x / math.sqrt(2.0)), min=1e-30)) - math.log(2.0),
        )

    # ---- core methods -----------------------------------------------------
    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        # log(2) + log φ(z) + log Φ(α·z) - log(ω)
        log_pdf = (
            math.log(2.0)
            + self._standard_normal.log_prob(z)
            + self._log_ndtr(self.alpha * z)
            - torch.log(self.scale)
        )
        return log_pdf

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterised sample via the stochastic representation:

            X = ξ + ω · (δ·|Z₀| + √(1−δ²)·Z₁)

        where δ = α / √(1+α²), and Z₀, Z₁ are independent standard normals.

        This gives a differentiable sample path needed for autograd-based
        gradient/hessian computation in LightGBMLSS.
        """
        shape = self._extended_shape(sample_shape)
        delta = self.alpha / torch.sqrt(1.0 + self.alpha ** 2)

        z0 = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        z1 = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)

        # |Z₀| via soft-abs (retains gradient flow)
        abs_z0 = torch.abs(z0)

        x = self.loc + self.scale * (delta * abs_z0 + torch.sqrt(1.0 - delta ** 2) * z1)
        return x

    @property
    def mean(self):
        delta = self.alpha / torch.sqrt(1.0 + self.alpha ** 2)
        return self.loc + self.scale * delta * math.sqrt(2.0 / math.pi)

    @property
    def variance(self):
        delta = self.alpha / torch.sqrt(1.0 + self.alpha ** 2)
        return self.scale ** 2 * (1.0 - 2.0 * delta ** 2 / math.pi)


# ---------------------------------------------------------------------------
# LightGBMLSS Distribution wrapper: SkewNormal
# ---------------------------------------------------------------------------

class SkewNormal(DistributionClass):
    """
    Skew-Normal Distribution Class for LightGBMLSS.

    Distributional Parameters
    -------------------------
    loc : torch.Tensor
        Location parameter (unconstrained).
    scale : torch.Tensor
        Scale parameter (strictly positive).
    alpha : torch.Tensor
        Skewness parameter (unconstrained).  alpha=0 recovers the Gaussian.

    Parameters
    -------------------------
    stabilization : str
        Stabilization method for the Gradient and Hessian.
        Options are ``"None"``, ``"MAD"``, ``"L2"``.
    response_fn : str
        Response function for transforming the scale parameter to the
        positive reals.  Options are ``"exp"`` or ``"softplus"``.
    loss_fn : str
        Loss function. Options are ``"nll"`` (negative log-likelihood) or
        ``"crps"`` (continuous ranked probability score).  Note that if
        ``"crps"`` is used, the Hessian is set to 1, as the current CRPS
        version is not twice differentiable.
    initialize : bool
        Whether to initialize the distributional parameters with
        unconditional start values.
    """

    def __init__(
        self,
        stabilization: str = "None",
        response_fn: str = "exp",
        loss_fn: str = "nll",
        initialize: bool = False,
    ):
        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError(
                "Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'."
            )
        if loss_fn not in ["nll", "crps"]:
            raise ValueError(
                "Invalid loss function. Please choose from 'nll' or 'crps'."
            )
        if not isinstance(initialize, bool):
            raise ValueError(
                "Invalid initialize. Please choose from True or False."
            )

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'."
            )

        # Set the parameters specific to the distribution
        distribution = SkewNormalTorch
        param_dict = {
            "loc": identity_fn,    # unconstrained
            "scale": response_fn,  # positive
            "alpha": identity_fn,  # unconstrained (skewness)
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
            initialize=initialize,
        )
