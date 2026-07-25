"""Fit-stage dispatcher for the group-conditional CDF family.

``fit_group_conditional_cdf`` routes to the variant fit body a :class:`StrategyConfig`
selects. The two shipped variants keep their own bodies — a two-part boundary fit
(``_fit_two_part``) and an affine fit (``_fit_affine``) — because their nested-CV
loops are genuinely different shapes.
"""

from __future__ import annotations

from sportstradamus.training.group_conditional_cdf._config import (
    AFFINE_CONFIG,
    TWO_PART_CONFIG,
    StrategyConfig,
)
from sportstradamus.training.group_conditional_cdf._fit_affine import fit_affine
from sportstradamus.training.group_conditional_cdf._fit_two_part import fit_two_part


def fit_group_conditional_cdf(config: StrategyConfig, *args, **kwargs):
    """Dispatch to the variant fit body the config selects."""
    if config == TWO_PART_CONFIG:
        return fit_two_part(config, *args, **kwargs)
    if config == AFFINE_CONFIG:
        return fit_affine(config, *args, **kwargs)
    raise ValueError(f"unsupported group-conditional CDF strategy config: {config!r}")
