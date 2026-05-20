"""Unit tests for the target/baseline strategy registry and EB-prior helper.

Covers:

* ``compute_eb_prior`` correctness (formula, shrinkage, high-games trust).
* ``ratio_meanyr`` strategy reproduces the legacy ``y / MeanYr`` transform
  exactly — critical so the default ``meditate`` run stays byte-identical
  to current production.
* ``centered_additive_eb_meanyr_k10`` strategy lifts ``y → y − EB_prior``
  and inverts to ``EB_prior + loc`` at decode.
* End-to-end round-trip: ``forward`` followed by ``decode_loc +
  decode_scale·delta·√(2/π)`` recovers ``y`` on a noise-free synthetic
  input, locking the train/predict symmetry the prediction mirror relies
  on.
* Unknown-slug rejection on the registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training.baselines import (
    EB_SHRINKAGE_K,
    STRATEGY_SLUGS,
    compute_eb_prior,
    get_strategy,
)


def _synth_x(n: int = 8, rng_seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic feature frame with the columns the strategies touch."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame(
        {
            "MeanYr": rng.uniform(2.0, 25.0, size=n),
            "MeanYr_nonzero": rng.uniform(2.0, 25.0, size=n),
            "Mean10": rng.uniform(2.0, 25.0, size=n),
            "GamesPlayed": rng.integers(1, 80, size=n).astype(float),
            "STDYr": rng.uniform(0.5, 5.0, size=n),
            "ZeroYr": np.zeros(n),
        }
    )


def test_compute_eb_prior_exact_formula():
    """``(games·player_mean + k·global_mean) / (games + k)`` to machine precision."""
    out = compute_eb_prior(
        np.array([20.0]), np.array([10.0]), global_mean=5.0, k=10.0
    )
    # (10*20 + 10*5) / (10 + 10) = 12.5
    assert abs(out[0] - 12.5) < 1e-12


def test_compute_eb_prior_shrinks_low_games_toward_global():
    out = compute_eb_prior(
        np.array([30.0]), np.array([1.0]), global_mean=10.0, k=EB_SHRINKAGE_K
    )
    # 1 game, player mean 30, global 10, K=10 -> heavy shrinkage:
    # (1*30 + 10*10) / (1 + 10) ≈ 11.8 — closer to 10 than to 30.
    assert 10.0 < out[0] < 14.0


def test_compute_eb_prior_trusts_high_games():
    out = compute_eb_prior(
        np.array([30.0]), np.array([200.0]), global_mean=10.0, k=EB_SHRINKAGE_K
    )
    # 200 games dominates -> prior pulls toward player mean (30).
    assert abs(out[0] - 30.0) < 1.5


def test_compute_eb_prior_clips_negative_games():
    """Defensive: negative GamesPlayed (shouldn't happen) clips to zero."""
    out = compute_eb_prior(
        np.array([20.0]), np.array([-3.0]), global_mean=5.0, k=EB_SHRINKAGE_K
    )
    # games clipped to 0 -> prior == global_mean.
    assert abs(out[0] - 5.0) < 1e-12


def test_ratio_meanyr_forward_matches_legacy_pipeline_behavior():
    """``ratio_meanyr.forward`` must be bitwise-equivalent to pipeline.py:466-468."""
    X = _synth_x()
    y = np.array([12.0, 0.0, 30.0, 5.5, 18.0, 22.0, 0.5, 11.0])

    legacy_meanyr = X["MeanYr"].clip(lower=0.5).to_numpy()
    legacy_target = np.clip(y / legacy_meanyr, 0.01, None)

    out = get_strategy("ratio_meanyr").forward(y, X, global_mean=10.0)

    np.testing.assert_array_equal(out, legacy_target)


def test_ratio_meanyr_decode_round_trips_on_noise_free_input():
    """ratio decode: loc·MeanYr at decode recovers y on a noise-free fit."""
    X = _synth_x()
    y = np.array([12.0, 8.0, 30.0, 5.5, 18.0, 22.0, 4.0, 11.0])
    strat = get_strategy("ratio_meanyr")

    y_t = strat.forward(y, X, global_mean=10.0)
    # Noise-free "model": predict loc = y_t exactly, scale = 0, alpha = 0.
    decoded = strat.decode_loc(y_t, X, global_mean=10.0)

    np.testing.assert_allclose(decoded, y, atol=1e-9)


def test_centered_eb_forward_subtracts_eb_prior():
    X = _synth_x()
    y = np.array([12.0, 8.0, 30.0, 5.5, 18.0, 22.0, 4.0, 11.0])

    expected_eb = compute_eb_prior(
        X["MeanYr"].clip(lower=0.5).to_numpy(),
        X["GamesPlayed"].clip(lower=0).to_numpy(),
        global_mean=10.0,
        k=EB_SHRINKAGE_K,
    )
    expected = y - expected_eb

    out = get_strategy("centered_additive_eb_meanyr_k10").forward(y, X, global_mean=10.0)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_centered_eb_decode_adds_back_same_eb_prior():
    """Train/predict mirror: decode must reuse the same EB formula as forward."""
    X = _synth_x()
    y = np.array([12.0, 8.0, 30.0, 5.5, 18.0, 22.0, 4.0, 11.0])
    strat = get_strategy("centered_additive_eb_meanyr_k10")

    y_t = strat.forward(y, X, global_mean=10.0)
    # Noise-free "model" predicts the residual exactly: decode must recover y.
    decoded = strat.decode_loc(y_t, X, global_mean=10.0)

    np.testing.assert_allclose(decoded, y, atol=1e-9)


def test_centered_eb_scale_is_absolute_not_scaled():
    """Centered scale stays in absolute residual units; no ``× MeanYr``."""
    X = _synth_x()
    scale = np.full(len(X), 3.0)
    out = get_strategy("centered_additive_eb_meanyr_k10").decode_scale(scale, X)
    np.testing.assert_array_equal(out, scale)


def test_ratio_scale_is_multiplied_by_meanyr():
    """Ratio scale must multiply by the same MeanYr_clipped used in forward."""
    X = _synth_x()
    scale = np.full(len(X), 0.3)
    expected = scale * X["MeanYr"].clip(lower=0.5).to_numpy()
    out = get_strategy("ratio_meanyr").decode_scale(scale, X)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_offset_meta_centered_eb_has_required_keys():
    """Pickle-persisted meta must let model_prob.py recompute EB identically."""
    meta = get_strategy("centered_additive_eb_meanyr_k10").offset_meta(
        global_mean=8.69, denom_col="MeanYr"
    )
    assert meta is not None
    assert meta["method"] == "eb_additive"
    assert meta["k"] == EB_SHRINKAGE_K
    assert abs(meta["global_mean"] - 8.69) < 1e-12
    assert meta["prior_col"] == "MeanYr"
    assert meta["games_col"] == "GamesPlayed"


def test_offset_meta_ratio_is_none():
    """Ratio strategy needs no baseline mirror — denom column + MeanYr suffice."""
    assert get_strategy("ratio_meanyr").offset_meta(global_mean=8.69) is None


def test_start_mode_flag_per_strategy():
    """Plumbs into ``set_model_start_values``; mismatch breaks SkewNormal seeding."""
    assert get_strategy("ratio_meanyr").start_mode_flag == "normalized"
    assert get_strategy("centered_additive_eb_meanyr_k10").start_mode_flag == "offset"


def test_full_round_trip_with_skewnormal_mean_formula():
    """y == decode_loc(loc) + decode_scale(scale) · delta · √(2/π) on noise-free fit."""
    X = _synth_x(n=12, rng_seed=42)
    y = np.array([12.0, 8.0, 30.0, 5.5, 18.0, 22.0, 4.0, 11.0, 7.5, 14.0, 25.0, 3.0])
    alpha = np.zeros(len(y))  # symmetric (delta = 0); EV == loc + 0.
    scale = np.zeros(len(y))
    delta = alpha / np.sqrt(1 + alpha**2)

    for slug in STRATEGY_SLUGS:
        strat = get_strategy(slug)
        y_t = strat.forward(y, X, global_mean=10.0)
        ev = strat.decode_loc(y_t, X, global_mean=10.0) + strat.decode_scale(
            scale, X
        ) * delta * np.sqrt(2 / np.pi)
        np.testing.assert_allclose(ev, y, atol=1e-9, err_msg=f"round-trip failed for {slug}")


def test_centered_mean10_forward_subtracts_mean10():
    X = _synth_x()
    X["Mean10"] = np.array([10.0, 8.0, 6.0, 4.0, 12.0, 14.0, 5.0, 9.0])
    y = np.array([12.0, 8.0, 30.0, 5.5, 18.0, 22.0, 4.0, 11.0])
    expected = y - X["Mean10"].clip(lower=0.5).to_numpy()
    out = get_strategy("centered_additive_mean10").forward(y, X, global_mean=10.0)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_centered_mean10_decode_round_trip():
    """Train/predict mirror: decode must add back the same Mean10 baseline."""
    X = _synth_x()
    X["Mean10"] = np.array([10.0, 8.0, 6.0, 4.0, 12.0, 14.0, 5.0, 9.0])
    y = np.array([12.0, 8.0, 30.0, 5.5, 18.0, 22.0, 4.0, 11.0])
    strat = get_strategy("centered_additive_mean10")
    y_t = strat.forward(y, X, global_mean=10.0)
    decoded = strat.decode_loc(y_t, X, global_mean=10.0)
    np.testing.assert_allclose(decoded, y, atol=1e-9)


def test_centered_mean10_falls_back_to_denom_when_mean10_missing():
    """Sparse-history players (Mean10 = NaN) must use the denom_col fallback."""
    X = _synth_x(n=3)
    X["Mean10"] = np.array([np.nan, 9.0, np.nan])
    y = np.array([15.0, 11.0, 7.0])
    out = get_strategy("centered_additive_mean10").forward(y, X, global_mean=10.0)
    # Row 0 and 2 fall back to MeanYr; row 1 uses Mean10 directly.
    fallback = X["MeanYr"].clip(lower=0.5).to_numpy()
    expected = y.copy()
    expected[0] = y[0] - fallback[0]
    expected[1] = y[1] - 9.0
    expected[2] = y[2] - fallback[2]
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_centered_mean10_offset_meta_records_mean10_baseline():
    meta = get_strategy("centered_additive_mean10").offset_meta(
        global_mean=8.69, denom_col="MeanYr"
    )
    assert meta is not None
    assert meta["method"] == "mean10_additive"
    assert meta["prior_col"] == "Mean10"
    assert meta["prior_fallback_col"] == "MeanYr"


def test_centered_mean10_scale_is_absolute():
    X = _synth_x()
    X["Mean10"] = np.full(len(X), 9.0)
    scale = np.full(len(X), 2.5)
    out = get_strategy("centered_additive_mean10").decode_scale(scale, X)
    np.testing.assert_array_equal(out, scale)


def test_get_strategy_unknown_slug_raises():
    with pytest.raises(ValueError, match="Unknown target strategy"):
        get_strategy("not_a_real_strategy")


def test_strategy_slugs_publishes_registered_options():
    """CLI Click choice list will read STRATEGY_SLUGS — must include all."""
    assert "ratio_meanyr" in STRATEGY_SLUGS
    assert "centered_additive_eb_meanyr_k10" in STRATEGY_SLUGS
    assert "centered_additive_mean10" in STRATEGY_SLUGS


# ---------------------------------------------------------------------------
# set_model_start_values offset_mode (Task 2)
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal stand-in for LightGBMLSS so we can capture ``start_values``."""

    def __init__(self):
        self.start_values = None


def _start_values_x(n: int = 5) -> pd.DataFrame:
    """Synthetic feature frame with the columns set_model_start_values reads."""
    return pd.DataFrame(
        {
            "MeanYr": np.array([12.0, 8.0, 4.0, 20.0, 1.5]),
            "STDYr": np.array([3.0, 2.5, 1.5, 5.0, 0.8]),
            "ZeroYr": np.array([0.0, 0.05, 0.1, 0.0, 0.2]),
        }
    )


def test_set_start_values_skewnormal_offset_mode_shape_n_by_3():
    """Guards the (n, 3) broadcast — a scalar 0 here was a prior regression."""
    from sportstradamus.helpers import set_model_start_values

    X = _start_values_x(n=5)
    m = _StubModel()
    set_model_start_values(m, "SkewNormal", X, offset_mode=True)

    assert m.start_values.shape == (5, 3)


def test_set_start_values_skewnormal_offset_mode_loc_zero_per_row():
    """Centered residual: loc starts at 0 PER ROW, not scalar 0."""
    from sportstradamus.helpers import set_model_start_values

    X = _start_values_x(n=5)
    m = _StubModel()
    set_model_start_values(m, "SkewNormal", X, offset_mode=True)
    sv = m.start_values

    # Col 0 = loc (identity response). All zeros, length n.
    np.testing.assert_array_equal(sv[:, 0], np.zeros(5))
    # Col 1 = log(scale). exp() must recover STDYr (within the 1e-6 clip).
    np.testing.assert_allclose(np.exp(sv[:, 1]), X["STDYr"].to_numpy(), rtol=1e-9)
    # Col 2 = alpha (start symmetric).
    np.testing.assert_array_equal(sv[:, 2], np.zeros(5))


def test_set_start_values_skewnormal_normalized_unchanged():
    """Regression: normalized=True path still emits loc=ones (current production)."""
    from sportstradamus.helpers import set_model_start_values

    X = _start_values_x(n=5)
    m = _StubModel()
    set_model_start_values(m, "SkewNormal", X, normalized=True)
    sv = m.start_values

    assert sv.shape == (5, 3)
    np.testing.assert_array_equal(sv[:, 0], np.ones(5))


def test_set_start_values_skewnormal_raw_unchanged():
    """Regression: default (no normalize, no offset) seeds loc=MeanYr."""
    from sportstradamus.helpers import set_model_start_values

    X = _start_values_x(n=5)
    m = _StubModel()
    set_model_start_values(m, "SkewNormal", X)
    sv = m.start_values

    assert sv.shape == (5, 3)
    np.testing.assert_array_equal(sv[:, 0], X["MeanYr"].to_numpy())


def test_set_start_values_offset_mode_ignored_for_non_skewnormal():
    """offset_mode is SkewNormal-only — passing it for NegBin must not affect shape."""
    from sportstradamus.helpers import set_model_start_values

    X = _start_values_x(n=5)
    m_offset = _StubModel()
    m_default = _StubModel()
    set_model_start_values(m_offset, "NegBin", X, offset_mode=True)
    set_model_start_values(m_default, "NegBin", X)

    # NegBin returns 2-col (r, logit(probs)); offset_mode kwarg is a no-op
    # for non-SkewNormal — the two start-value arrays must be identical.
    assert m_offset.start_values.shape == m_default.start_values.shape
    np.testing.assert_array_equal(m_offset.start_values, m_default.start_values)


def test_meditate_accepts_target_strategy_flag():
    """CLI surface: --target-strategy must accept registered slugs."""
    from click.testing import CliRunner

    from sportstradamus.training.cli import meditate

    runner = CliRunner()
    result = runner.invoke(meditate, ["--help"])
    assert result.exit_code == 0
    assert "--target-strategy" in result.output
    assert "ratio_meanyr" in result.output
    assert "centered_additive_eb_meanyr_k10" in result.output


def test_meditate_rejects_unknown_target_strategy():
    """Click validates against STRATEGY_SLUGS; unknown slug -> non-zero exit."""
    from click.testing import CliRunner

    from sportstradamus.training.cli import meditate

    runner = CliRunner()
    result = runner.invoke(meditate, ["--target-strategy", "not_a_real_strategy"])
    assert result.exit_code != 0
    assert "not_a_real_strategy" in result.output or "Invalid" in result.output
