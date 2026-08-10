"""SkewNormal head clamps (research/briefs/researcher_hpo_objective_alignment.md R1).

The boosted skew-normal likelihood diverges systematically — one-sided leaves carry a
monotone likelihood in alpha and an unbounded ridge in the exp sigma head — so
``_continuous_dist_obj`` bounds the sigma-like head of both parametrizations in
robust-label-scale units (IQR/1.349, not std: std/robust reaches 46.9 on ratio_meanyr
cells) and the direct alpha head at +-30 (parity with centered's tanh-bounded gamma1).
``torch.clamp`` is the identity inside the band, so converged fits are bit-unaffected,
and the wrapped ``param_dict`` rides the pickle into serving.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from lightgbmlss.utils import identity_fn

from sportstradamus.training.hyperparams import _BoundedResponseFn
from sportstradamus.training.pipeline import _continuous_dist_obj

# Normalized-space labels whose outlier makes std (1.73) diverge from the robust
# scale (0.61) — pins that the clamp keys to IQR/1.349, not np.std.
_LABELS = np.array([0.2, 0.5, 0.8, 1.0, 1.1, 1.4, 2.0, 6.0])


def _robust_scale(labels: np.ndarray) -> float:
    q75, q25 = np.percentile(labels, [75, 25])
    return (q75 - q25) / 1.349


def test_direct_scale_and_alpha_heads_bounded():
    dist = _continuous_dist_obj("SkewNormal", "direct", "None", "auto", _LABELS)
    s = _robust_scale(_LABELS)
    scale_fn = dist.param_dict["scale"]
    assert isinstance(scale_fn, _BoundedResponseFn)
    assert scale_fn.ceiling == pytest.approx(10.0 * s)
    assert scale_fn.floor == pytest.approx(0.02 * s)
    alpha_fn = dist.param_dict["alpha"]
    assert isinstance(alpha_fn, _BoundedResponseFn)
    assert alpha_fn.ceiling == 30.0
    assert alpha_fn.floor == -30.0
    assert dist.param_dict["loc"] is identity_fn


def test_centered_sd_head_bounded_gamma1_and_mean_untouched():
    dist = _continuous_dist_obj("SkewNormal", "centered", "None", "auto", _LABELS)
    s = _robust_scale(_LABELS)
    sd_fn = dist.param_dict["sd"]
    assert isinstance(sd_fn, _BoundedResponseFn)
    assert sd_fn.ceiling == pytest.approx(10.0 * s)
    assert sd_fn.floor == pytest.approx(0.02 * s)
    # gamma1 already carries its own +-0.99 tanh bound; mean stays unconstrained.
    assert not isinstance(dist.param_dict["gamma1"], _BoundedResponseFn)
    assert dist.param_dict["mean"] is identity_fn


def test_wrapped_scale_response_clamps_only_outside_band():
    dist = _continuous_dist_obj("SkewNormal", "direct", "None", "auto", _LABELS)
    scale_fn = dist.param_dict["scale"]
    clamped = scale_fn(torch.tensor([-80.0, 80.0], dtype=torch.float64))
    assert clamped[0].item() == pytest.approx(scale_fn.floor)
    assert clamped[1].item() == pytest.approx(scale_fn.ceiling)
    inside = torch.log(
        torch.tensor([2.0 * scale_fn.floor, 0.5 * scale_fn.ceiling], dtype=torch.float64)
    )
    torch.testing.assert_close(scale_fn(inside), scale_fn.orig_fn(inside))


def test_mixture_scale_clamp_still_std_keyed():
    # Deliberately std-keyed (Mixture cells are additive; brief finding 5) — do not
    # migrate it to the robust scale.
    labels = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0])
    mix = _continuous_dist_obj("Mixture", "direct", "None", "auto", labels)
    fn = mix.param_dict["scale"]
    assert isinstance(fn, _BoundedResponseFn)
    assert fn.ceiling == pytest.approx(20.0 * float(np.std(labels)))
    assert fn.floor == pytest.approx(0.02 * float(np.std(labels)))
