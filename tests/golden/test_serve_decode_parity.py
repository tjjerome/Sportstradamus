"""Serve must reproduce the training decode *decisions* from the pickle alone.

These gates pin the invariant a serve-side decode drift broke on 2026-06-24:
``centered_additive_mean10`` SkewNormal cells served ~2x over-projected because the
prediction path decided the SkewNormal seeding mode (``offset_mode``) from a hardcoded
``offset_meta["method"] == "eb_additive"`` string — so the newer ``mean10_additive``
strategy seeded with the raw mean instead of the centered-residual zero, doubling the
projected mean. A second drift let serve pick ``denom_col`` from the gitignored, per-box
``stat_zi`` instead of the value training persisted in the pickle.

Both gates are parametrized over the live ``TARGET_NORMALIZATION_SLUGS`` registry, so a
strategy added in the future that serve forgets to handle fails here rather than in
production. This file is the fast-gate backstop for the serve decode path, which
previously had zero ``tests/golden/`` coverage.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from sportstradamus.prediction.model_prob import _resolve_denom_col, _serve_offset_mode
from sportstradamus.training.baselines import (
    TARGET_NORMALIZATION_SLUGS,
    get_target_normalization,
)

# The function ``model_prob`` (re-exported in prediction/__init__) shadows the submodule
# in the package namespace, so fetch the real module object for monkeypatching.
_MP = sys.modules["sportstradamus.prediction.model_prob"]

# Intended serve seeding per strategy, declared explicitly (NOT derived from the
# registry) so adding a strategy forces a human to state and verify its seeding —
# the missing-entry case is exactly how the mean10 2x bug shipped.
_EXPECTED_SERVE_OFFSET = {
    "ratio_meanyr": False,
    "ratio_projvol": False,
    "centered_additive_eb_meanyr_k10": True,
    "centered_additive_mean10": True,
}


@pytest.mark.parametrize("slug", TARGET_NORMALIZATION_SLUGS)
def test_serve_offset_mode_per_strategy(slug):
    assert slug in _EXPECTED_SERVE_OFFSET, (
        f"New target_normalization {slug!r} has no expected serve offset_mode — declare it "
        "here and confirm model_prob seeds it correctly. A missing entry is how "
        "centered_additive_mean10 shipped a 2x serve-decode bug."
    )
    expected = _EXPECTED_SERVE_OFFSET[slug]
    assert _serve_offset_mode("SkewNormal", slug) is expected
    # offset_mode is meaningless off the SkewNormal seed — never offset other dists.
    assert _serve_offset_mode("ZINB", slug) is False
    # Parity anchor: serve's decision must equal training's start_mode_flag rule.
    assert expected == (get_target_normalization(slug).start_mode_flag == "offset")


@pytest.mark.parametrize("slug", TARGET_NORMALIZATION_SLUGS)
def test_serve_denom_col_reads_pickle_not_stale_stat_zi(slug):
    # Training chose MeanYr_nonzero (high zero-rate market); the serving box's runtime
    # stat_zi is stale/empty so the passed hist_gate is 0. Serve must still decode with
    # MeanYr_nonzero by reading the decision training persisted in the pickle.
    offset_meta = get_target_normalization(slug).offset_meta(
        global_mean=5.0, denom_col="MeanYr_nonzero"
    )
    denom = _resolve_denom_col(offset_meta, hist_gate=0.0, columns=("MeanYr", "MeanYr_nonzero"))
    assert denom == "MeanYr_nonzero"


def test_resolve_denom_col_legacy_pickle_falls_back_to_hist_gate():
    # Pre-persist pickle (offset_meta is None): recompute from the runtime hist_gate,
    # byte-identical to the legacy behavior.
    cols = ("MeanYr", "MeanYr_nonzero")
    assert _resolve_denom_col(None, hist_gate=0.30, columns=cols) == "MeanYr_nonzero"
    assert _resolve_denom_col(None, hist_gate=0.0, columns=cols) == "MeanYr"
    # No MeanYr_nonzero column ⇒ MeanYr even at a high gate.
    assert _resolve_denom_col(None, hist_gate=0.30, columns=("MeanYr",)) == "MeanYr"


class _FakeModel:
    is_hurdle = False

    def predict(self, X, pred_type):
        return pd.DataFrame({"loc": np.zeros(len(X))}, index=X.index)


class _FakeStats:
    volume_stats: set = set()


@pytest.mark.parametrize(
    "slug,expected",
    [("centered_additive_mean10", True), ("centered_additive_eb_meanyr_k10", True), ("ratio_meanyr", False)],
)
def test_build_prob_params_seeds_offset_mode_from_strategy(monkeypatch, slug, expected):
    """Call-site pin: ``_build_prob_params`` seeds ``set_model_start_values`` with the
    offset_mode the *strategy* dictates. A revert to the old hardcoded
    ``offset_meta["method"] == "eb_additive"`` check seeds ``mean10_additive`` False and
    fails here — this is the regression that shipped the live 2x.
    """
    captured = {}
    monkeypatch.setattr(
        _MP,
        "set_model_start_values",
        lambda model, dist, X, normalized, offset_mode: captured.update(offset_mode=offset_mode),
    )
    _MP._build_prob_params(
        {"model": _FakeModel()},
        "PTS",
        _FakeStats(),
        pd.DataFrame({"Home": [0, 1]}),
        "SkewNormal",
        False,
        slug,
    )
    assert captured["offset_mode"] is expected
