# tests/integration/test_determinism_gate.py
"""P0.5 determinism gate: deterministic mode must be bit-reproducible.

Runs the pure fit/predict core twice on a fixed subsample of the real cached
NBA_FGA training matrix and asserts the test-split predicted distribution
parameters are exactly equal. The gate locally overrides DETERMINISTIC_FIXED_PARAMS
to enable real LightGBM stochasticity (feature/row subsampling); this way it
genuinely tests that seed_everything + the merged LightGBM seed kwargs pin
stochastic tree building. Production --deterministic mode uses the
stochastic-free defaults, so this gate is intentionally stricter than
production behavior — a true regression guard for the seeding mechanism.
No network (cached parquet + offline columns).
"""

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    fit_predict_hurdle_params,
    fit_predict_params,
)

# Fixed subsample size: large enough to exercise real tree building, small
# enough to keep the gate < ~30s. Deterministic head() after a stable sort.
GATE_N_ROWS = 4000


@pytest.mark.integration
def test_deterministic_mode_is_bit_reproducible():
    parquet = pkg_resources.files(data) / "training_data/NBA_FGA.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_FGA.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(GATE_N_ROWS).reset_index(drop=True)

    cols = StatsNBA().get_stat_columns("FGA")
    X = M[cols].copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = M[["Result"]]

    # Same temporal split shape train_market uses (no validation needed here).
    n_train = int(len(M) * 0.7)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]

    # FGA is SkewNormal + ratio-normalized (global_mean >= 2.0 branch).
    y_train_labels = np.ravel(y_train.to_numpy()).astype(float)
    meanyr_train = X_train["MeanYr"].clip(lower=0.5).to_numpy()
    y_train_labels = np.clip(y_train_labels / meanyr_train, 0.01, None)

    def run():
        dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")
        return fit_predict_params(
            dist_obj,
            "SkewNormal",
            X_train,
            y_train_labels,
            X_test,
            # Override DETERMINISTIC_FIXED_PARAMS to introduce real stochasticity
            # (feature/row subsampling). This way the gate ACTUALLY tests that the
            # seeding mechanism pins LightGBM's stochastic tree building, not that
            # no-randomness is trivially deterministic. seed_everything + LightGBM
            # seed kwargs (merged inside fit_lss_model) must yield bit-identical
            # output across two runs at the same seed; different seeds must differ.
            params=(
                {
                    **DETERMINISTIC_FIXED_PARAMS,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "monotone_constraints": [0] * X_train.shape[1],
                }
            ),
            normalized=True,
            shape_ceiling=100.0,
            seed=DETERMINISTIC_SEED,
        )

    p1 = run()
    p2 = run()
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)


def _assert_hurdle_params_bit_reproducible(parquet_name: str, stats, market: str) -> None:
    """Two seeded HurdleZINB fit/predicts on a cached parquet must match exactly.

    Shared body for the per-league hurdle determinism gates. HurdleZINB has two
    LightGBM boosters (binary clf + zero-truncated NegBin) and its internal
    ``fit`` calls ``seed_everything`` once. The compression_eval A/B for
    ``meditate --deterministic --zinb-mode hurdle`` is meaningful only if this
    two-run bit-identity holds on every covered league; without it the A/B
    verdict would be noise (the lesson from docs/CENTERED_TARGET_NEGATIVE_RESULT.md).

    Args:
        parquet_name: Cached training matrix filename under ``data/training_data``.
        stats: League ``Stats`` instance providing ``get_stat_columns``.
        market: Market stem to pull feature columns for.
    """
    parquet = pkg_resources.files(data) / f"training_data/{parquet_name}"
    if not parquet.is_file():
        pytest.skip(f"cached {parquet_name} not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(GATE_N_ROWS).reset_index(drop=True)

    cols = stats.get_stat_columns(market)
    # ``reindex`` over ``M[cols]`` matches production's resilience to stale
    # cached parquets (see ``_step_build_splits`` in ``training/pipeline.py``).
    # When a feature_filter addition lands before the cached training parquet
    # is regenerated, the missing columns fill with NaN; LightGBM handles
    # NaN deterministically so bit-reproducibility is preserved.
    X = M.reindex(columns=cols).copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = np.ravel(M[["Result"]].to_numpy()).astype(float)

    n_train = int(len(M) * 0.7)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:].reset_index(drop=True)
    y_train = y[:n_train]

    params = {
        **DETERMINISTIC_FIXED_PARAMS,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "monotone_constraints": [0] * X_train.shape[1],
    }

    def run() -> pd.DataFrame:
        return fit_predict_hurdle_params(
            X_train,
            y_train,
            X_test,
            params=params,
            shape_ceiling=50.0,
            seed=DETERMINISTIC_SEED,
        )

    pd.testing.assert_frame_equal(run(), run(), check_exact=True)


@pytest.mark.integration
def test_deterministic_mode_hurdle_is_bit_reproducible():
    """NBA FG3M — canonical ZINB market (33% zero rate, the overconfidence
    investigation's primary case study)."""
    _assert_hurdle_params_bit_reproducible("NBA_FG3M.parquet", StatsNBA(), "FG3M")


@pytest.mark.integration
def test_deterministic_mode_hurdle_is_bit_reproducible_wnba():
    """WNBA FG3M — ZTNB (Stage B1) is the first cross-league model change, so the
    hurdle bit-identity must be proven on WNBA before the 23-cell A/B is trusted."""
    _assert_hurdle_params_bit_reproducible("WNBA_FG3M.parquet", StatsWNBA(), "FG3M")


@pytest.mark.integration
def test_deterministic_mode_hurdle_is_bit_reproducible_nfl():
    """NFL interceptions — cross-league ZINB determinism guard for Stage B1's
    zero-truncated NegBin, on a low-mean NFL count market."""
    _assert_hurdle_params_bit_reproducible("NFL_interceptions.parquet", StatsNFL(), "interceptions")
