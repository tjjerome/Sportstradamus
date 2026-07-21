"""Pins for model-version attribution (WS-1 Stage 0).

Train-time :func:`sportstradamus.training.pipeline._build_filedict` appends a
``trained_at`` / ``model_version`` pair to the model pickle; serve-time
:func:`sportstradamus.prediction.model_prob.resolve_model_version` reads it back
and synthesizes a stable ``legacy.<sha>`` for the pre-stamp pickles. These pins
guard the two invariants the flow depends on: the stamp is *purely additive*
(no existing pickle key changes, so no served probability moves), and a fresh
filedict carries a well-formed version string.
"""

from __future__ import annotations

import importlib
import pickle

import pandas as pd

from sportstradamus.training.model_strategy_artifacts import MODEL_STRATEGY_MODEL_KEY
from sportstradamus.training.model_strategy_registry import corner_fingerprint, get_strategy
from sportstradamus.training.pipeline import _build_filedict, _model_version
from sportstradamus.training.structural_strategies import (
    TWO_PART_STRATEGY,
)

# ``sportstradamus.prediction`` re-exports ``model_prob`` the function, shadowing
# the submodule under attribute access — import the module explicitly.
mp = importlib.import_module("sportstradamus.prediction.model_prob")


class _StubModel:
    is_hurdle = False


class _StubStrategy:
    def offset_meta(self, global_mean, denom_col):
        return {"global_mean": global_mean, "denom_col": denom_col}


def _minimal_filedict_kwargs() -> dict:
    """The smallest kwarg set ``_build_filedict`` reads to assemble the dict."""
    mode_stats = dict.fromkeys(("acc", "prec", "under_prec", "over_pct", "sharp", "ll"), 0.0)
    skill = {
        "model_metrics": {"brier": 0.2},
        "book_metrics": {"brier": 0.25},
        "brier_skill_score": 0.2,
        "kelly_shrinkage": 0.2,
    }
    diag_keys = (
        "shape_label",
        "start_shape",
        "model_shape",
        "empirical_shape",
        "start_mean",
        "model_ev",
        "mean_line",
        "ev_minus_line",
        "result_mean",
        "median_ev_diff",
        "frac_ev_gt_line",
        "over_pct_ev_gt",
        "over_pct_ev_lt",
        "cf_over_pct",
        "ev_meanyr_corr",
        "result_meanyr_corr",
    )
    diag = dict.fromkeys(diag_keys, 0.0)
    return {
        "model": _StubModel(),
        "step": 0.5,
        "mode_stats": mode_stats,
        "skill": skill,
        "diag": diag,
        "c_opt": 1.0,
        "skew_cal": 0.0,
        "shape_ceiling": 5.0,
        "marginal_shape": 3.0,
        "opt_params": {"n_estimators": 100, "learning_rate": 0.05},
        "dist": "SkewNormal",
        "cv": 0.9,
        "y": pd.DataFrame({"Result": [1.0, 2.0, 3.0, 4.0]}),
        "T_opt": 1.5,
        "model_weight": 0.4,
        "model_calib": 1.0,
        "hist_gate": 0.0,
        "normalize": True,
        "strategy": _StubStrategy(),
        "global_mean": 10.5,
        "denom_col": "MeanYr",
        "target_normalization": "ratio_meanyr",
        "posthoc_slug": "none",
        "posthoc_blob": None,
        "pit_recal_blob": None,
        "zinb_mode": "auto",
        "sn_param": "direct",
        "X": pd.DataFrame({"f0": [1], "f1": [2]}),
        "league": "NBA",
        "market": "PTS",
        "matrix_hash": "a" * 64,
        "selected_controls": {
            "dist": "SkewNormal",
            "normalization": "ratio_meanyr",
            "dist_training_loss": "crps",
            "sn_param": "direct",
            "blending_loss_fn": "crps",
        },
    }


def test_model_version_wellformed_and_deterministic():
    ts = "2026-07-08T12:34:56+00:00"
    params = {"n_estimators": 100, "learning_rate": 0.05}
    v = _model_version(ts, params, "SkewNormal", 0.9, 0.5, "ratio_meanyr")

    date_slug, norm_slug, sha8 = v.split(".")
    assert date_slug == "20260708"
    assert norm_slug == "ratio_meanyr"
    assert len(sha8) == 8 and all(c in "0123456789abcdef" for c in sha8)

    # sha8 is order-independent in the params dict and independent of the
    # clock time within the same UTC day.
    reordered = {"learning_rate": 0.05, "n_estimators": 100}
    v2 = _model_version(
        "2026-07-08T23:59:59+00:00", reordered, "SkewNormal", 0.9, 0.5, "ratio_meanyr"
    )
    assert v2 == v

    # An unset normalization slugs to "none".
    v_none = _model_version(ts, params, "ZINB", 0.5, 1.0, "")
    assert v_none.split(".")[1] == "none"

    # A different identity axis (distribution) moves the hash.
    v_zinb = _model_version(ts, params, "ZINB", 0.9, 0.5, "ratio_meanyr")
    assert v_zinb.split(".")[2] != sha8


def test_build_filedict_carries_wellformed_version():
    fd = _build_filedict(**_minimal_filedict_kwargs())

    assert "trained_at" in fd and "model_version" in fd
    # trained_at is an ISO-8601 UTC timestamp; model_version is derived from it.
    assert fd["trained_at"].startswith("20") and "T" in fd["trained_at"]
    date_slug, norm_slug, sha8 = fd["model_version"].split(".")
    assert date_slug == fd["trained_at"][:10].replace("-", "")
    assert norm_slug == "ratio_meanyr"
    assert len(sha8) == 8


def test_build_filedict_carries_cell_bound_model_strategy_identity():
    fd = _build_filedict(**_minimal_filedict_kwargs())

    identity = fd[MODEL_STRATEGY_MODEL_KEY]
    assert identity["strategy_slug"] == "SkewNormal"
    assert identity["structural_strategy"] == "none"
    assert identity["league"] == "NBA"
    assert identity["market"] == "PTS"
    assert identity["matrix_hash"] == "a" * 64
    assert identity["controls_json"] == (
        '{"blending_loss_fn":"crps","dist":"SkewNormal","dist_training_loss":"crps",'
        '"normalization":"ratio_meanyr","sn_param":"direct"}'
    )


def test_model_strategy_corner_fingerprint_changes_with_each_selected_control():
    baseline_kwargs = _minimal_filedict_kwargs()
    baseline = _build_filedict(**baseline_kwargs)[MODEL_STRATEGY_MODEL_KEY]
    controls = baseline_kwargs["selected_controls"]
    spec = get_strategy("SkewNormal")
    assert baseline["corner_fingerprint"] == corner_fingerprint(
        spec, controls, baseline_kwargs["matrix_hash"]
    )
    alternatives = {
        "dist": "Mixture",
        "normalization": "centered_additive_mean10",
        "dist_training_loss": "nll",
        "sn_param": "centered",
        "blending_loss_fn": "nll",
    }

    for control, alternative in alternatives.items():
        changed_controls = {**controls, control: alternative}
        changed = corner_fingerprint(spec, changed_controls, baseline_kwargs["matrix_hash"])
        assert changed != baseline["corner_fingerprint"], control


def test_build_filedict_keeps_nfl_payload_as_structural_adapter_state():
    payload = {
        "schema_version": 3,
        "slug": TWO_PART_STRATEGY,
        "status": "active",
        "validation_audit": {"split_fingerprint_sha256": "split-123"},
    }
    kwargs = _minimal_filedict_kwargs()
    kwargs.update(
        {
            "league": "NFL",
            "market": "receiving yards",
            "structural_strategy": TWO_PART_STRATEGY,
            "algorithm_payload": payload,
            "selected_controls": {
                "dist": "SkewNormal",
                "normalization": "ratio_meanyr",
                "dist_training_loss": "crps",
                "sn_param": "direct",
                "blending_loss_fn": "nll",
                "hpo_selection": "loss",
                "stabilization": "None",
            },
        }
    )

    fd = _build_filedict(**kwargs)

    assert fd["nfl_yards_experiment"] is payload
    identity = fd[MODEL_STRATEGY_MODEL_KEY]
    assert identity["strategy_slug"] == TWO_PART_STRATEGY
    assert identity["structural_strategy"] == TWO_PART_STRATEGY
    assert identity["split_fingerprint"] == "split-123"


def test_stamp_is_purely_additive():
    """Every pre-existing pickle key is byte-identical with or without the stamp.

    Guarantees the version can never perturb a served probability — the two new
    keys are the *only* difference between a stamped filedict and the same dict
    with them stripped (the legacy shape).
    """
    fd = _build_filedict(**_minimal_filedict_kwargs())
    legacy_shape = {k: v for k, v in fd.items() if k not in ("trained_at", "model_version")}

    assert set(fd) - set(legacy_shape) == {"trained_at", "model_version"}
    # Serialized bytes of every shared key match exactly.
    for key, value in legacy_shape.items():
        assert pickle.dumps(value) == pickle.dumps(fd[key])


def test_resolve_model_version_prefers_stamp(tmp_path):
    pkl = tmp_path / "stamped.pkl"
    filedict = {"model_version": "20260708.ratio_meanyr.deadbeef", "cv": 0.9}
    pkl.write_bytes(pickle.dumps(filedict))

    assert mp.resolve_model_version(str(pkl), filedict) == "20260708.ratio_meanyr.deadbeef"


def test_resolve_model_version_synthesizes_stable_legacy(tmp_path):
    pkl = tmp_path / "legacy.pkl"
    legacy = {"cv": 0.9, "weight": 0.4}  # no model_version key
    pkl.write_bytes(pickle.dumps(legacy))

    v1 = mp.resolve_model_version(str(pkl), legacy)
    v2 = mp.resolve_model_version(str(pkl), legacy)

    assert v1.startswith("legacy.")
    assert len(v1) == len("legacy.") + 10
    assert v1 == v2  # memoized, reproducible

    # A different legacy pickle hashes to a different id.
    other = tmp_path / "legacy2.pkl"
    other.write_bytes(pickle.dumps({"cv": 0.5}))
    assert mp.resolve_model_version(str(other), {"cv": 0.5}) != v1
