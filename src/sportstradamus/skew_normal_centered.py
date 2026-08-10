"""Centered-parametrization SkewNormal for LightGBMLSS.

Provides:
  - CenteredSkewNormalTorch: SkewNormalTorch constructed from the centered
    parameters (mean, sd, gamma1) via Azzalini's closed-form inversion.
  - CenteredSkewNormal: a LightGBMLSS DistributionClass whose three boosted
    heads are the centered parameters, re-emitting direct (loc, scale, alpha)
    from ``predict_dist`` so every serving consumer sees exactly the frame a
    direct SkewNormal produces — zero serving delta.

Why: under the direct parametrization the Fisher information is singular at
alpha = 0 and the alpha score is collinear with loc, so per-parameter
boosting cancels the skew signal and the skew head rails. The centered
parametrization orthogonalizes the three scores (Arellano-Valle & Azzalini
2008; Azzalini & Capitanio 2014 §3.1.4), giving the gamma1 trees a clean
gradient. See /tmp/researcher_continuous_family.md §1a.

Usage:
    from sportstradamus.skew_normal_centered import CenteredSkewNormal
    from lightgbmlss.model import LightGBMLSS

    model = LightGBMLSS(dist=CenteredSkewNormal())
"""

import math

import numpy as np
import pandas as pd
import torch
from lightgbmlss.utils import identity_fn

from sportstradamus.skew_normal import SkewNormal, SkewNormalTorch

# Bound on |gamma1| applied by the response function. The attainable skew-normal
# max is ~0.995272, but at 0.9952 the alpha radicand is ~4e-5 — float32
# cancellation territory (alpha ~ 250, exploding hessians; observed killing
# boosting at ~4 rounds). 0.99 keeps the radicand at ~3.4e-3 (alpha capped
# ~28.4), float32-safe with bounded hessians.
_GAMMA1_MAX = 0.99

# Floor on the alpha radicand 1 - (pi/2 - 1) r^2. At the gamma1 bound its true
# value is ~3.4e-3; the floor only absorbs float noise, it never binds in exact
# arithmetic.
_ALPHA_RADICAND_FLOOR = 1e-6

# Formula constants of the Azzalini inversion, precomputed once.
_FOUR_MINUS_PI = 4.0 - math.pi
_SQRT_HALF_PI = math.sqrt(math.pi / 2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _gamma1_response(x):
    """Bounded response for the skewness head: (-_GAMMA1_MAX, _GAMMA1_MAX)."""
    return _GAMMA1_MAX * torch.tanh(x)


def _centered_to_direct(mean, sd, gamma1, xp):
    """Azzalini's closed-form CP -> DP map.

    ``xp`` is ``torch`` (inside the distribution — autograd-clean) or ``numpy``
    (the ``predict_dist`` re-emit); both expose the ops used here under the
    same names, so the two paths cannot drift.
    """
    r = xp.sign(gamma1) * (2.0 * xp.abs(gamma1) / _FOUR_MINUS_PI) ** (1.0 / 3.0)
    r_sq = r * r
    delta = _SQRT_HALF_PI * r / xp.sqrt(1.0 + r_sq)
    radicand = 1.0 - (math.pi / 2.0 - 1.0) * r_sq
    radicand = (
        np.clip(radicand, _ALPHA_RADICAND_FLOOR, None)
        if xp is np
        else radicand.clamp(min=_ALPHA_RADICAND_FLOOR)
    )
    alpha = _SQRT_HALF_PI * r / xp.sqrt(radicand)
    scale = sd / xp.sqrt(1.0 - 2.0 * delta * delta / math.pi)
    loc = mean - scale * delta * _SQRT_2_OVER_PI
    return loc, scale, alpha


class CenteredSkewNormalTorch(SkewNormalTorch):
    r"""Skew-Normal constructed from centered parameters (mean, sd, gamma1).

    The centered parameters — predictive mean, standard deviation, and Pearson
    skewness :math:`\gamma_1 \in (-0.9952, 0.9952)` — are mapped to the direct
    :math:`(\xi, \omega, \alpha)` by the closed-form inversion

    .. math::

        R = \operatorname{sign}(\gamma_1)\left(\frac{2|\gamma_1|}{4-\pi}\right)^{1/3},\quad
        \delta = \sqrt{\tfrac{\pi}{2}}\,\frac{R}{\sqrt{1+R^2}},\quad
        \alpha = \frac{\delta}{\sqrt{1-\delta^2}},

    with :math:`\omega = sd/\sqrt{1 - 2\delta^2/\pi}` and
    :math:`\xi = mean - \omega\delta\sqrt{2/\pi}`. All density/sampling math is
    inherited from :class:`SkewNormalTorch`, so both the nll and crps loss
    paths differentiate through the mapping. :math:`\gamma_1 = 0` maps exactly
    to :math:`\alpha = 0` (a Gaussian with the given mean and sd).
    """

    def __init__(self, mean, sd, gamma1, validate_args=None):
        mean, sd, gamma1 = (torch.as_tensor(v, dtype=torch.float32) for v in (mean, sd, gamma1))
        loc, scale, alpha = _centered_to_direct(mean, sd, gamma1, torch)
        super().__init__(loc, scale, alpha, validate_args=validate_args)


class CenteredSkewNormal(SkewNormal):
    """Centered-parametrization Skew-Normal Distribution Class for LightGBMLSS.

    Distributional Parameters
    -------------------------
    mean : torch.Tensor
        Predictive mean (unconstrained).
    sd : torch.Tensor
        Predictive standard deviation (strictly positive).
    gamma1 : torch.Tensor
        Pearson skewness, bounded to (-_GAMMA1_MAX, _GAMMA1_MAX) by a tanh
        response.

    Constructor arguments are identical to :class:`SkewNormal` (stabilization,
    response_fn for the sd head, loss_fn, initialize), except the
    stabilization default is ``"L2"``: near the gamma1 bound the mapped-alpha
    hessians are large, and unstabilized boosting NaNs out within a few rounds
    (the mgcv ``shash`` ridge-penalty precedent; R2 brief §1a gradient risk
    (iii)).
    """

    def __init__(
        self,
        stabilization: str = "L2",
        response_fn: str = "exp",
        loss_fn: str = "nll",
        initialize: bool = False,
    ):
        super().__init__(stabilization, response_fn, loss_fn, initialize)
        self.distribution = CenteredSkewNormalTorch
        self.param_dict = {
            "mean": identity_fn,
            "sd": self.param_dict["scale"],  # reuse the resolved response fn
            "gamma1": _gamma1_response,
        }
        self.distribution_arg_names = list(self.param_dict.keys())

    def predict_dist(
        self,
        booster,
        data,
        start_values,
        pred_type: str = "parameters",
        n_samples: int = 1000,
        quantiles: list | None = None,
        seed: int = 123,
    ) -> pd.DataFrame:
        """Predict, re-emitting direct (loc, scale, alpha) parameter columns.

        The pickle carries this class, so every ``dist == "SkewNormal"``
        consumer downstream (decode, blend, calibration, scorecard dumps)
        keeps seeing the direct-parameter frame it sees today.
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        pred = super().predict_dist(
            booster, data, start_values, pred_type, n_samples, quantiles, seed
        )
        if pred_type in ("parameters", "expectiles"):
            loc, scale, alpha = _centered_to_direct(
                pred["mean"].to_numpy(),
                pred["sd"].to_numpy(),
                pred["gamma1"].to_numpy(),
                np,
            )
            pred = pd.DataFrame({"loc": loc, "scale": scale, "alpha": alpha}, index=pred.index)
        return pred
