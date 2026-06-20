"""Train/serve parity gate: a corrupt book EV can't desync the live blend.

The 2026-06-05 FG3M incident was a *train/serve* divergence with a subtle root:
the offline test-set decode produced sane means (~1.4 threes), but the live
``model_prob`` blend pooled the model against a bookmaker EV of ~7800 — a
``get_ev`` zero-inflation runaway that ``add_dfs`` had written to the archive —
and published 6-8 threes. Offline never saw it because the historical archive
holds real two-sided prices, not DFS-boost runaways.

This gate locks the invariant that broke: the live blend
(``model_prob._blend_with_book``) driven by the *production* model artifact must
stay within the offline-decoded model's support even when the archived book EV is
corrupt. It is the end-to-end backstop behind the two unit gates
(``tests/golden/test_get_ev_robustness.py`` — generation;
``tests/golden/test_book_ev_blend_guard.py`` — the guard in isolation).

Marked @pytest.mark.integration; skips cleanly when the cached FG3M parquet or
the trained pickle is absent.
"""

from __future__ import annotations

import importlib
import importlib.resources as pkg_resources
import pickle

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.helpers.distributions import (
    decode_predictive_mean,
    set_model_start_values,
)
from sportstradamus.stats import StatsNBA

# A blown archive book EV in the thousands (the shape add_dfs wrote on 2026-06-05).
_CORRUPT_BOOK_EV = 7802.0
# The live blend must not exceed this multiple of the offline-decoded model mean.
# fused_loc is a log-opinion pool, so a clean book within range moves the mean by
# well under 2x; anything past it means the corrupt book leaked into the pool.
_MAX_BLEND_INFLATION = 2.0
_GATE_N_ROWS = 4000


def _load_fg3m_case():
    parquet = pkg_resources.files(data) / "training_data/NBA_FG3M.parquet"
    mdl_path = pkg_resources.files(data) / "models/NBA_FG3M.mdl"
    if not parquet.is_file() or not mdl_path.is_file():
        pytest.skip("cached NBA_FG3M parquet/pickle not present in this environment")

    M = pd.read_parquet(parquet).sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)
    with open(mdl_path, "rb") as fh:
        fd = pickle.load(fh)
    if fd["distribution"] != "ZINB":
        pytest.skip("NBA_FG3M pickle is not ZINB in this environment")

    X = M[fd["expected_columns"]].fillna(0).infer_objects(copy=False).copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    # Predict on a small recent slice (cheap, deterministic). Reset start_values
    # to the batch size exactly as model_prob does before predict.
    X = X.tail(200).reset_index(drop=True)
    set_model_start_values(
        fd["model"],
        "ZINB",
        X,
        normalized=bool(fd.get("normalized", False)),
        shape_ceiling=fd.get("shape_ceiling"),
    )
    pp = fd["model"].predict(X, pred_type="parameters")
    decoded = decode_predictive_mean(pp, "ZINB")
    line = M["Line"].tail(200).reset_index(drop=True).to_numpy()
    return decoded, line


def _blend(model_ev, decoded, line, books_ev):
    mp = importlib.import_module("sportstradamus.prediction.model_prob")
    offer_df = pd.DataFrame(
        {
            "Projection": model_ev,
            "Market Projection": books_ev,
            "Line": line,
            "Model R": np.asarray(decoded.r, dtype=float),
            "Model Gate": np.asarray(decoded.gate, dtype=float),
        }
    )
    base_mean = mp._blend_with_book(offer_df, "ZINB", model_weight=0.85, cv=0.566, hist_gate=0.337)
    return offer_df["Projection"].to_numpy(), np.asarray(base_mean, dtype=float)


@pytest.mark.integration
def test_corrupt_book_ev_cannot_desync_live_blend_from_offline_decode():
    decoded, line = _load_fg3m_case()
    model_ev = np.asarray(decoded.ev, dtype=float)

    # The offline reference must itself be in support (sanity anchor).
    assert np.isfinite(model_ev).all()
    assert (model_ev <= 5 * np.clip(line, 0.5, None)).all(), "offline decode already out of support"

    blended, _ = _blend(model_ev, decoded, line, np.full_like(model_ev, _CORRUPT_BOOK_EV))
    assert np.isfinite(blended).all()
    # The corrupt book must not pull the live blend out of the offline model's
    # support. Pre-fix this inflated ~3x (1.4 -> 6-8 threes).
    assert (blended <= _MAX_BLEND_INFLATION * model_ev).all(), (
        f"corrupt book EV inflated the live blend beyond the offline decode: "
        f"max ratio {float(np.max(blended / np.clip(model_ev, 1e-9, None))):.2f}"
    )


@pytest.mark.integration
def test_clean_book_ev_blend_matches_offline_decode_in_band():
    # Parity on agreement: when the book mean equals the model mean, fused_loc is a
    # log-opinion pool, so the pooled *base* mean must come back equal to the model
    # base (independent of the gate). The published Model EV then only applies the
    # re-pooled gate (1 - gate) * base, so it stays in (0, base]. This isolates the
    # pool from the legitimate gate transform that decode_predictive_mean leaves off.
    from sportstradamus.prediction.model_prob import _sanitize_book_ev

    decoded, line = _load_fg3m_case()
    model_ev = np.asarray(decoded.ev, dtype=float)
    # A book that agrees with the model leaves the guard a no-op.
    model_sd = np.ones_like(model_ev)
    np.testing.assert_array_equal(
        _sanitize_book_ev(model_ev.copy(), line, model_ev, model_sd), model_ev
    )

    blended, base_mean = _blend(model_ev, decoded, line, model_ev.copy())
    np.testing.assert_allclose(
        base_mean, model_ev, rtol=1e-6, err_msg="pool drifted off the decode"
    )
    assert np.isfinite(blended).all()
    assert (blended > 0).all()
    assert (blended <= base_mean + 1e-9).all(), "gate-applied mean must not exceed the base"
