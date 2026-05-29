"""Regression tests for SHAP multi-output reduction in training/shap.py.

LightGBMLSS fits a multi-output booster for any distribution with more than
one parameter (NegBin/Gamma both 2, ZINB/ZAGamma/SkewNormal 3, Mixture2G 5).
SHAP's ``TreeExplainer`` returns one of three shapes for those:

* 2-D ``(samples, features)`` for single-output (passthrough),
* ``list`` of ``(samples, features)`` per output (older SHAP <0.45),
* 3-D ``(samples, features, outputs)`` for multi-output (newer SHAP).

`_collapse_multi_output_shap` collapses all three into a 2-D ``(samples,
features)`` array of absolute SHAP values so the downstream
``np.mean(..., axis=0)`` yields one scalar per feature. If SHAP changes its
return-shape convention again — the failure mode that produced the
``TypeError: Could not convert [array(...)]`` crash in ``_refresh_all_aggregates``
during NFL ``attempts`` — these tests will catch it before the live pipeline.
"""

import numpy as np

from sportstradamus.training.shap import _collapse_multi_output_shap


def test_list_and_3d_reduce_identically():
    """Older list-of-2D and newer 3-D ndarray inputs must produce the same |SHAP| matrix."""
    rng = np.random.default_rng(seed=42)
    n_samples, n_features, n_outputs = 8, 5, 3
    per_output = [rng.standard_normal((n_samples, n_features)) for _ in range(n_outputs)]
    arr_3d = np.stack(per_output, axis=-1)

    reduced_list = _collapse_multi_output_shap(per_output)
    reduced_3d = _collapse_multi_output_shap(arr_3d)

    np.testing.assert_allclose(reduced_list, reduced_3d)
    assert reduced_list.shape == (n_samples, n_features)


def test_2d_single_output_returns_absolute_values():
    """For single-output models (2-D signed SHAP), the helper returns the elementwise abs."""
    rng = np.random.default_rng(seed=42)
    arr_2d = rng.standard_normal((8, 5))

    out = _collapse_multi_output_shap(arr_2d)

    np.testing.assert_array_equal(out, np.abs(arr_2d))
    assert out.shape == (8, 5)


def test_downstream_mean_yields_one_scalar_per_feature():
    """After collapse + mean(axis=0) the per-feature importance must be 1-D scalars."""
    rng = np.random.default_rng(seed=42)
    n_samples, n_features, n_outputs = 8, 5, 3
    arr_3d = rng.standard_normal((n_samples, n_features, n_outputs))

    collapsed = _collapse_multi_output_shap(arr_3d)
    vals = np.mean(collapsed, axis=0)

    assert vals.shape == (n_features,)
    assert vals.dtype.kind == "f"
    # No object-dtype cells — the failure mode in `_refresh_all_aggregates`
    # was that vals carried numpy arrays as elements, producing object dtype.
    assert not any(isinstance(v, np.ndarray) for v in vals)
