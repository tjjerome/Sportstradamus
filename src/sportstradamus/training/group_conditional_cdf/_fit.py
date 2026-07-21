"""Fit-stage dispatcher for the group-conditional CDF family.

``fit_group_conditional_cdf`` routes to the corner fit body a :class:`StrategyConfig`
selects. The two shipped corners keep their own bodies — a two-part boundary fit
(``_fit_receiving``) and an affine fit (``_fit_rushing``) — because their nested-CV
loops are genuinely different shapes.
"""

from __future__ import annotations

from sportstradamus.training.group_conditional_cdf._config import StrategyConfig
from sportstradamus.training.group_conditional_cdf._fit_receiving import fit_receiving
from sportstradamus.training.group_conditional_cdf._fit_rushing import fit_rushing


def fit_group_conditional_cdf(config: StrategyConfig, *args, **kwargs):
    """Dispatch to the corner fit body the config selects."""
    if config.boundary and not config.affine_marginal:
        return fit_receiving(config, *args, **kwargs)
    return fit_rushing(config, *args, **kwargs)
