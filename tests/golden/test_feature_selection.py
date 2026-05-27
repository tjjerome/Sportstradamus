"""Behavioral tests for ``feature_selection.filter_market_features``.

Locks in the 2026-05-26 composite-scoring fixes:

  * `REDUNDANCY_WEIGHT = 0.15` (down from 0.40) — lower collinearity
    penalty so tree models can use correlated features at separate splits.
  * `SHAP_FLOOR_K = 30` — veto-on-drop for top-K SHAP features per cell.
    Heuristics cannot override what the trained model is demonstrably using.
  * Per-feature `has_model` resolution in ``_score_features`` — candidates
    with SHAP entries from a prior training round get W_MODEL weights, not
    the all-or-nothing W_NO_MODEL fallback that zeroed the SHAP term.

The invariant the SHAP floor enforces: for every cell with a trained
model, no base feature whose ``|SHAP|`` rank is at or above K (modulo
Locked features and features not in the candidate pool) is missing from
the post-rebuild Filtered list.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sportstradamus import feature_selection as fs


@pytest.fixture()
def synthetic_filter_inputs(tmp_path, monkeypatch) -> dict:
    """Build a small synthetic market with a trained-model SHAP CSV.

    Pool layout:
      - 25 ``current`` features (already in the cell's Filtered list).
      - 10 ``candidate`` features in the training data but not in the
        filter; first 3 of these are deliberately given top-K |SHAP|
        entries to test the floor's re-add behaviour.
      - 5 LOCKED features (in Common/Always) that the function strips
        from current and excludes from the SHAP floor.
    """
    rng = np.random.default_rng(0)
    n_rows = 400
    current_feats = [f"Player cur{i:02d}" for i in range(25)]
    candidate_feats = [f"Player cand{i:02d}" for i in range(10)]
    locked_feats = [f"Player lock{i:02d}" for i in range(5)]
    all_feats = current_feats + candidate_feats + locked_feats

    target = rng.normal(size=n_rows)
    df_cols: dict[str, np.ndarray] = {
        "Date": pd.date_range("2024-01-01", periods=n_rows).astype(str)
    }
    df_cols["Result"] = target
    for i, feat in enumerate(all_feats):
        signal = rng.normal(size=n_rows) + (target * (0.5 if i < 10 else 0.05))
        df_cols[feat] = signal
    train_df = pd.DataFrame(df_cols)

    market = "synthetic-market"
    league = "TEST"
    mkey = fs.market_key(league, market)

    training_dir = tmp_path / "training_data"
    training_dir.mkdir()
    train_df.to_parquet(training_dir / f"{league}_{market}.parquet")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{league}_{market}.mdl").write_text("stub model pickle")

    def fake_training_path(_league: str, _market: str):
        return training_dir / f"{league}_{market}.parquet"

    def fake_model_path(_league: str, _market: str):
        return models_dir / f"{league}_{market}.mdl"

    monkeypatch.setattr(fs, "training_path", fake_training_path)
    monkeypatch.setattr(fs, "model_path", fake_model_path)

    # SHAP CSV: top of column for this cell holds a few candidate base
    # names (test the floor's re-add path) plus most current feats.
    shap_rows: dict[str, float] = {}
    for i, feat in enumerate(candidate_feats[:3]):
        shap_rows[feat] = 50.0 - i  # top of the rankings
    for i, feat in enumerate(current_feats):
        shap_rows[feat] = 10.0 - 0.1 * i
    shap_rows["Player othercell_only"] = 0.5
    shap_df = pd.DataFrame({mkey: shap_rows})

    corr_df = pd.DataFrame()

    feature_filter = {
        league: {
            "Common": locked_feats[:3],
            "Always": {"_default": locked_feats[3:], market.replace("-", " "): []},
            "Filtered": {market.replace("-", " "): current_feats},
        }
    }

    return dict(
        league=league,
        market=market,
        feature_filter=feature_filter,
        shap_df=shap_df,
        corr_df=corr_df,
        mkey=mkey,
        current_feats=current_feats,
        candidate_feats=candidate_feats,
        locked_feats=locked_feats,
    )


def test_redundancy_weight_pinned_at_0_15():
    """The 2026-05-26 tuning pins REDUNDANCY_WEIGHT at 0.15."""
    assert pytest.approx(0.15) == fs.REDUNDANCY_WEIGHT


def test_shap_floor_k_pinned_at_30():
    """SHAP_FLOOR_K is the documented top-K SHAP veto threshold."""
    assert fs.SHAP_FLOOR_K == 30


def test_feat_has_shap_entry_detects_variants():
    """``feat_has_shap_entry`` finds SHAP via any of base, short, growth."""
    shap_df = pd.DataFrame(
        {"X": {"Player foo": np.nan, "Player foo short": 1.5, "Player bar": 2.0}}
    )
    assert fs.feat_has_shap_entry("Player foo", "X", shap_df), "should match via 'short'"
    assert fs.feat_has_shap_entry("Player bar", "X", shap_df), "should match base"
    assert not fs.feat_has_shap_entry("Player baz", "X", shap_df)
    assert not fs.feat_has_shap_entry("Player foo", "MISSING_KEY", shap_df)


def test_shap_floor_protects_top_k_features(synthetic_filter_inputs):
    """Floor invariant: every top-K SHAP base name lands in the kept set."""
    inputs = synthetic_filter_inputs
    new_filtered, diag = fs.filter_market_features(
        league=inputs["league"],
        market=inputs["market"],
        feature_filter=inputs["feature_filter"],
        shap_df=inputs["shap_df"],
        corr_df=inputs["corr_df"],
        shap_floor_k=10,
    )
    new_set = set(new_filtered)
    for floored in diag["shap_floored"]:
        assert floored in new_set, f"SHAP-floored feature {floored!r} got dropped"

    # Specific check: candidate cand00..cand02 sit at top of |SHAP| ranking
    # despite being absent from the prior filter. The floor must re-add them.
    for cand in inputs["candidate_feats"][:3]:
        assert cand in new_set, f"top-SHAP candidate {cand!r} was not promoted"


def test_shap_floor_excludes_locked(synthetic_filter_inputs):
    """Locked features (Common+Always) are never re-added via the floor."""
    inputs = synthetic_filter_inputs
    # Inject one locked feature into the top of the SHAP ranking
    shap_df = inputs["shap_df"].copy()
    locked_top = inputs["locked_feats"][0]
    shap_df.loc[locked_top, inputs["mkey"]] = 999.0

    new_filtered, diag = fs.filter_market_features(
        league=inputs["league"],
        market=inputs["market"],
        feature_filter=inputs["feature_filter"],
        shap_df=shap_df,
        corr_df=inputs["corr_df"],
        shap_floor_k=10,
    )
    assert locked_top not in diag["shap_floored"], (
        f"locked feature {locked_top!r} should not be in SHAP-floored set"
    )
    assert locked_top not in set(new_filtered), (
        f"locked feature {locked_top!r} should not be in Filtered list"
    )


def test_candidate_with_prior_shap_uses_w_model(synthetic_filter_inputs):
    """Candidates with a SHAP entry get W_MODEL weights via per-feature has_model.

    Regression test for the 2026-05-26 bug fix: previously
    ``filter_market_features`` called ``_score_features`` for candidates
    with ``has_model=False``, zeroing the SHAP weight and gating only on
    corr/MI/stability. After the fix, a candidate whose SHAP entry exists
    in ``shap_df`` should report ``has_model: True`` in its breakdown.
    """
    inputs = synthetic_filter_inputs
    _, diag = fs.filter_market_features(
        league=inputs["league"],
        market=inputs["market"],
        feature_filter=inputs["feature_filter"],
        shap_df=inputs["shap_df"],
        corr_df=inputs["corr_df"],
    )
    top_shap_candidate = inputs["candidate_feats"][0]
    cand_breakdown = diag["candidate_scores"].get(top_shap_candidate, {})
    assert cand_breakdown.get("has_model") is True, (
        f"candidate {top_shap_candidate!r} has SHAP in CSV but breakdown reports "
        f"has_model={cand_breakdown.get('has_model')!r}"
    )

    candidate_without_shap = inputs["candidate_feats"][-1]
    nh_breakdown = diag["candidate_scores"].get(candidate_without_shap, {})
    assert nh_breakdown.get("has_model") is False, (
        f"candidate {candidate_without_shap!r} has no SHAP entry but breakdown "
        f"reports has_model={nh_breakdown.get('has_model')!r}"
    )


def test_filter_market_features_is_deterministic_across_hash_seeds(synthetic_filter_inputs, tmp_path):
    """Cross-process determinism: identical inputs must produce identical outputs.

    Regression test for the 2026-05-27 NaN-roulette bug. NaN in composite
    scores collapsed the KEEP_CAP sort to set-iteration order, which depends
    on ``PYTHONHASHSEED`` and re-randomizes per process. The NaN-tolerant
    patches (``_nanmax``, ``normalize_scores`` NaN coerce, KEEP_CAP cap_rank
    NaN guard + name tiebreak) plus the ``_filter_finite_features`` screen
    on input features eliminate every NaN source. The two outputs from two
    invocations against the same inputs must match exactly.
    """
    inputs = synthetic_filter_inputs
    # NaN-pollute: inject a constant-column feature into the training
    # parquet (gets screened by _filter_finite_features) and into the
    # candidate pool (would otherwise drive NaN into score normalization).
    ff = inputs["feature_filter"].copy()
    cell = inputs["market"].replace("-", " ")
    current_with_constant = list(ff[inputs["league"]]["Filtered"][cell])
    train_path = fs.training_path(inputs["league"], inputs["market"])
    df = pd.read_parquet(train_path)
    df["Player constant_zero"] = 0.0
    df.to_parquet(train_path)
    current_with_constant.append("Player constant_zero")
    ff[inputs["league"]]["Filtered"][cell] = current_with_constant

    # Persist inputs to disk so the subprocess can reload them
    shap_pkl = tmp_path / "_shap.pkl"
    corr_pkl = tmp_path / "_corr.pkl"
    ff_pkl = tmp_path / "_ff.pkl"
    stub_mdl = tmp_path / "stub.mdl"
    inputs["shap_df"].to_pickle(shap_pkl)
    inputs["corr_df"].to_pickle(corr_pkl)
    pd.to_pickle(ff, ff_pkl)
    stub_mdl.write_text("x")

    src_root = str(fs.__file__).rsplit("/", 2)[0]
    script = (
        "import hashlib, json, sys\n"
        "import pandas as pd\n"
        f"sys.path.insert(0, {src_root!r})\n"
        "from sportstradamus import feature_selection as fs\n"
        f"shap_df = pd.read_pickle({str(shap_pkl)!r})\n"
        f"corr_df = pd.read_pickle({str(corr_pkl)!r})\n"
        f"ff = pd.read_pickle({str(ff_pkl)!r})\n"
        f"fs.training_path = lambda *a, **k: {str(train_path)!r}\n"
        f"fs.model_path = lambda *a, **k: {str(stub_mdl)!r}\n"
        f"new, _ = fs.filter_market_features({inputs['league']!r}, "
        f"{inputs['market']!r}, ff, shap_df, corr_df)\n"
        "print(hashlib.md5(json.dumps(sorted(new)).encode()).hexdigest())\n"
    )

    env_base = os.environ.copy()
    out1 = subprocess.run(
        [sys.executable, "-c", script],
        env={**env_base, "PYTHONHASHSEED": "1"},
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    out2 = subprocess.run(
        [sys.executable, "-c", script],
        env={**env_base, "PYTHONHASHSEED": "9999"},
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out1 == out2, (
        f"filter_market_features non-deterministic across hash seeds: "
        f"PYTHONHASHSEED=1 -> {out1!r}, PYTHONHASHSEED=9999 -> {out2!r}"
    )


def test_keep_cap_protects_shap_floored_features(synthetic_filter_inputs):
    """KEEP_CAP=60 still caps total, but SHAP-floored features get cap-survival boost.

    Scenario: cap is artificially small (5) and current pool has 25 entries.
    Without the floor's cap-boost, the 3 top-SHAP candidates could be
    dropped by the cap. With the boost they survive even though the cap
    is tight.
    """
    inputs = synthetic_filter_inputs
    new_filtered, _ = fs.filter_market_features(
        league=inputs["league"],
        market=inputs["market"],
        feature_filter=inputs["feature_filter"],
        shap_df=inputs["shap_df"],
        corr_df=inputs["corr_df"],
        shap_floor_k=3,
        keep_cap=5,
    )
    assert len(new_filtered) <= 5
    new_set = set(new_filtered)
    # All 3 top-SHAP candidates should survive the cap
    for cand in inputs["candidate_feats"][:3]:
        assert cand in new_set, (
            f"SHAP-floored candidate {cand!r} got cut by KEEP_CAP — cap-survival "
            f"boost in Layer 4 is not working"
        )
