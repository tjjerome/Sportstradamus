"""Stage-0 pins for confirmed contracts repaired in later ordered stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.training_quotes import resolve_training_quote
from sportstradamus.stats.nhl import StatsNHL
from sportstradamus.stats.nhl_position_policy import NHL_MARKET_POSITIONS
from sportstradamus.training.group_conditional_cdf import (
    AFFINE_SCHEMA_VERSION,
    AFFINE_STRATEGY_NAME,
    AffinePredictive,
    apply_affine_groupcdf,
)
from sportstradamus.training.markets import ALL_MARKETS


def test_one_coherent_training_quote_resolver_exists():
    assert callable(resolve_training_quote)


def test_affine_apply_accepts_explicit_authenticity():
    identity = {"kind": "group_empirical_cdf", "lam": 0.0, "x": [0.0, 1.0], "y": [0.0, 1.0]}
    blob = {
        "kind": AFFINE_STRATEGY_NAME,
        "schema_version": AFFINE_SCHEMA_VERSION,
        "line_probability_only": True,
        "affine": {
            "kind": "affine_marginal_mean",
            "intercept": 0.0,
            "slope": 1.0,
            "floor": 0.1,
        },
        "position_cdf": {"1": identity},
        "position_codes": [1],
        "fallback": {
            "kind": "model_only",
            "unseen_position": "raw_cdf_unpooled",
            "nonauthentic_quote": "candidate_unpooled",
        },
        "affine_bounds": {
            "kind": "train_scale_affine_bounds",
            "train_abs_p95": 10.0,
            "intercept": [-20.0, 20.0],
            "slope": [0.5, 1.5],
        },
        "fit_audit": {
            "expected_position_codes": [1],
            "outer_folds": [{} for _ in range(5)],
        },
        "temperature": 1.0,
        "probability_pool": {
            "kind": "structural_probability_pool",
            "schema_version": 1,
            "rho": 0.4,
            "raw_rho": 0.4,
        },
    }
    output = apply_affine_groupcdf(
        blob,
        AffinePredictive(
            np.array([10.0, 10.0]),
            np.array([3.0, 3.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        ),
        line=np.array([9.5, 9.5]),
        book_over=np.array([0.7, 0.7]),
        position=np.array([1, 1]),
        authentic=np.array([True, False]),
    )

    assert output.pooled_over[0] != output.candidate_over[0]
    assert output.pooled_over[1] == output.candidate_over[1]


@pytest.mark.parametrize(
    "market",
    ["shotsAgainst", "saves", "goalsAgainst", "goalie fantasy points underdog"],
)
def test_nhl_goalie_markets_retain_only_goalies(market):
    stats = StatsNHL()
    gamelog = pd.DataFrame({"position": ["C", "W", "D", "G"]})

    filtered = stats._market_position_filter(gamelog, market)

    assert filtered["position"].tolist() == ["G"]


def test_nhl_skater_market_excludes_goalies():
    stats = StatsNHL()
    gamelog = pd.DataFrame({"position": ["C", "W", "D", "G"]})

    filtered = stats._market_position_filter(gamelog, "shots")

    assert filtered["position"].tolist() == ["C", "W", "D"]


def test_nhl_position_policy_covers_the_complete_active_registry():
    assert set(NHL_MARKET_POSITIONS) == set(ALL_MARKETS["NHL"])
