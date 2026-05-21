"""Unit tests for the zero-truncated NegBin log-likelihood (Stage B1.1).

``HurdleZINB`` Stage 2 fits a NegBin on the strictly-positive subset
``y > 0``. Training it with the vanilla full-support NB NLL is a
*misspecified* zero-truncated NB (ZTNB): the optimizer still sees mass at
``Y = 0`` and recovers a biased conditional-positive mean. ``_ZeroTruncatedNB``
wraps the torch NegBin so its ``log_prob`` is the ZTNB identity

    log p_ZTNB(y) = log p_NB(y) - log(1 - p_NB(0))

which is what these tests pin down, independently of any training run.

Convention check (load-bearing): torch ``NegativeBinomial(total_count=r,
probs=p)`` matches scipy ``nbinom(n=r, p=1-probs)`` — both give
``P(0) = (1-p)^r`` and ``mean = r*p/(1-p)``. The scipy reference below relies
on that mapping.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import nbinom

from sportstradamus.hurdle import _ZeroTruncatedNB

# Tolerance for matching the closed-form scipy ZTNB reference. torch and scipy
# both build the NB pmf from gamma functions, so agreement is tight.
_REL_TOL: float = 1e-6


def _scipy_ztnb_logpmf(y: np.ndarray, total_count: float, probs: float) -> np.ndarray:
    """Independent ZTNB log-likelihood reference via scipy.

    Uses the torch<->scipy convention map ``nbinom(n=total_count, p=1-probs)``.
    """
    scipy_p = 1.0 - probs
    log_pk = nbinom.logpmf(y, n=total_count, p=scipy_p)
    # log(1 - P(0)); log1p(-x) is the stable form of log(1 - x).
    log_one_minus_p0 = np.log1p(-nbinom.pmf(0, n=total_count, p=scipy_p))
    return log_pk - log_one_minus_p0


def test_ztnb_log_prob_matches_scipy_reference() -> None:
    total_count, probs = 4.0, 0.35
    y = np.array([1, 2, 3, 5, 8, 13], dtype=float)

    ztnb = _ZeroTruncatedNB(
        total_count=torch.tensor(total_count), probs=torch.tensor(probs)
    )
    got = ztnb.log_prob(torch.tensor(y)).detach().numpy()

    ref = _scipy_ztnb_logpmf(y, total_count, probs)
    assert np.allclose(got, ref, atol=_REL_TOL), f"\ngot={got}\nref={ref}"


def test_ztnb_log_prob_matches_scipy_per_row_params() -> None:
    """Per-row (total_count, probs) vectors — the shape LightGBMLSS passes."""
    total_count = np.array([2.0, 5.0, 9.0, 1.5], dtype=float)
    probs = np.array([0.2, 0.5, 0.7, 0.4], dtype=float)
    y = np.array([1, 4, 12, 2], dtype=float)

    ztnb = _ZeroTruncatedNB(
        total_count=torch.tensor(total_count).reshape(-1, 1),
        probs=torch.tensor(probs).reshape(-1, 1),
    )
    got = ztnb.log_prob(torch.tensor(y).reshape(-1, 1)).detach().numpy().ravel()

    ref = np.array(
        [
            _scipy_ztnb_logpmf(np.array([yi]), r, p)[0]
            for yi, r, p in zip(y, total_count, probs, strict=True)
        ]
    )
    assert np.allclose(got, ref, atol=_REL_TOL), f"\ngot={got}\nref={ref}"


def test_ztnb_log_prob_finite_under_degenerate_zero_mass() -> None:
    """When p(0) rounds to 1 the truncation term must stay finite (the floor).

    With ``probs`` ~ 0, ``(1-probs)^total_count`` rounds to 1.0 in float64, so
    ``1 - p(0) -> 0`` and an unguarded ``log(1 - p(0))`` is ``-inf`` (making the
    ZTNB log-prob ``+inf``). The ``_ZTNB_P0_FLOOR`` clamp keeps it finite while
    ``log p_NB(y)`` itself stays finite for these params.
    """
    ztnb = _ZeroTruncatedNB(
        total_count=torch.tensor(2.0), probs=torch.tensor(1e-20)
    )
    out = ztnb.log_prob(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isfinite(out).all(), f"non-finite ZTNB log-prob: {out}"


def test_ztnb_delegates_non_loss_attributes_to_nb() -> None:
    """sample()/mean delegate to the wrapped NB (used by LightGBMLSS internals)."""
    ztnb = _ZeroTruncatedNB(total_count=torch.tensor(4.0), probs=torch.tensor(0.35))
    torch.manual_seed(0)
    sample = ztnb.sample((5,))
    assert sample.shape == (5,)
    assert torch.isfinite(ztnb.mean).all()
