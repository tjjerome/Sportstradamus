"""Pins for the prior-model store ``meditate`` keeps for the version-age test.

A retrain overwrites ``data/models/{LEAGUE}_{MARKET}.mdl``; ``training.prior_models`` first
copies the outgoing file to ``models/prior/`` under the ``Model Version`` it stamped on its
legs. These pins hold what the offline re-serve depends on: one prior file per version
stamp (through a confirm walk's retries too), nothing kept for a first save or a sandbox
output, a prune that counts from the day a model was superseded, and a ``prior/`` directory
the served-model scan never reads.
"""

from __future__ import annotations

import importlib
import importlib.resources as pkg_resources
import os
import pickle
import shutil
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.io import resolve_model_version
from sportstradamus.training import pipeline as pipe
from sportstradamus.training import prior_models
from sportstradamus.training.model_strategy import MODEL_STRATEGY_MODEL_KEY

# Both packages re-export a function under their submodule's name (``model_prob``,
# ``report``), which shadows the submodule on attribute access.
report_module = importlib.import_module("sportstradamus.training.report")

OUTGOING = "20260917.none.aaaaaaaa"
INCOMING = "20260924.none.bbbbbbbb"


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    """``tmp_path/models`` as the canonical models dir, for the save site and the store alike."""
    # pipeline and report resolve the package data dir through this at call time.
    monkeypatch.setattr(pkg_resources, "files", lambda _package: tmp_path)
    monkeypatch.setattr(prior_models, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()
    return tmp_path / "models"


def _save_model(model_version: str, *, deterministic=False, artifact_output=None) -> None:
    """Run the real save site once for the NBA PTS cell, stamping ``model_version``."""
    pipe._step_persist_artifacts(
        filedict={"model_version": model_version, MODEL_STRATEGY_MODEL_KEY: {}},
        splits={
            "X_test": pd.DataFrame({"MeanYr": [3.0, 8.0]}),
            "y_test": pd.DataFrame({"Result": [1.0, 10.0]}),
            "B_test": pd.DataFrame({"Line": [1.5, 8.5], "Odds": [0.4, 0.6], "EV": [2.0, 7.0]}),
            "dates_test": ["2026-01-01", "2026-01-02"],
        },
        prob_params=pd.DataFrame({"concentration": [2.0, 3.0]}),
        decoded={"ev": np.array([2.0, 7.0])},
        weighted_mean=np.array([2.2, 7.4]),
        y_proba_filt=np.full((2, 2), 0.5),
        y_proba_raw=np.full((2, 2), 0.5),
        dist="Gamma",
        hist_gate=0.0,
        filename="NBA_PTS",
        deterministic=deterministic,
        target_normalization="none",
        zinb_mode="joint",
        sn_scale_test=None,
        sn_skew_test=None,
        mix_test=None,
        r_test=None,
        gate_blend_test=None,
        phi_test=None,
        global_mean=5.0,
        denom_col="MeanYr",
        structural_rows=dict.fromkeys(pipe.STRUCTURAL_ROW_FIELDS),
        artifact_output=artifact_output,
    )


def test_first_save_keeps_nothing(models_dir):
    _save_model(OUTGOING)

    assert (models_dir / "NBA_PTS.mdl").is_file()
    assert not (models_dir / "prior").exists()


def test_overwrite_keeps_the_outgoing_model_under_its_version_stamp(models_dir):
    _save_model(OUTGOING)
    _save_model(INCOMING)

    kept = list((models_dir / "prior").iterdir())
    assert [path.name for path in kept] == [f"NBA_PTS__{OUTGOING}.mdl"]
    assert pickle.loads(kept[0].read_bytes())["model_version"] == OUTGOING
    served = pickle.loads((models_dir / "NBA_PTS.mdl").read_bytes())
    assert served["model_version"] == INCOMING


@pytest.mark.parametrize(
    "filedict",
    [{"model_version": "20260917.ratio_meanyr.0afa433e"}, {"cv": 0.9}],
    ids=["stamped", "pre-stamp"],
)
def test_prior_file_is_named_for_the_model_version_stamped_on_served_legs(models_dir, filedict):
    served = models_dir / "NFL_rushing-tds.mdl"
    served.write_bytes(pickle.dumps(filedict))

    prior_models.keep_prior_model(served)

    version = resolve_model_version(str(served), filedict)
    kept = [path.name for path in (models_dir / "prior").iterdir()]
    assert kept == [f"NFL_rushing-tds__{version}.mdl"]


def test_prune_counts_from_supersession_and_drops_only_expired_files(models_dir):
    def backdate(path, days):
        then = time.time() - timedelta(days=days).total_seconds()
        os.utime(path, (then, then))

    expired = prior_models.PRIOR_MODEL_RETENTION_DAYS + 1
    served = models_dir / "NBA_PTS.mdl"
    served.write_bytes(pickle.dumps({"model_version": OUTGOING}))
    # Trained long ago, superseded only now: the kept copy must still get its full window.
    backdate(served, expired)
    prior_dir = models_dir / "prior"
    prior_dir.mkdir()
    for name, days in {"NBA_REB__expired.mdl": expired, "NBA_AST__recent.mdl": expired - 2}.items():
        (prior_dir / name).write_bytes(b"prior")
        backdate(prior_dir / name, days)

    prior_models.keep_prior_model(served)

    assert sorted(path.name for path in prior_dir.iterdir()) == [
        "NBA_AST__recent.mdl",
        f"NBA_PTS__{OUTGOING}.mdl",
    ]


def test_sandbox_overwrites_keep_nothing(models_dir, monkeypatch, tmp_path):
    # The strictest deterministic layout: its output nested inside the models dir.
    monkeypatch.setattr(pipe, "_DETERMINISTIC_MODEL_ROOT", models_dir / "deterministic")
    for sandbox in ({"artifact_output": tmp_path / "artifacts"}, {"deterministic": True}):
        for model_version in (OUTGOING, INCOMING):
            _save_model(model_version, **sandbox)

    # One model per sandbox, and no copy of either.
    assert len(list(tmp_path.rglob("*.mdl"))) == 2
    assert not list(tmp_path.rglob("prior"))


def test_confirm_walk_retries_keep_one_file_per_version_stamp(models_dir, tmp_path):
    served = models_dir / "NBA_PTS.mdl"
    _save_model(OUTGOING)
    incumbent_backup = tmp_path / "incumbent.mdl"
    shutil.copy2(served, incumbent_backup)
    for nominee in ("20260924.none.cccccccc", INCOMING):
        _save_model(nominee)
        # A held nominee: the walk copies the incumbent back without going through the save site.
        shutil.copy2(incumbent_backup, served)

    kept = [path.name for path in (models_dir / "prior").iterdir()]
    assert kept == [f"NBA_PTS__{OUTGOING}.mdl"]


def test_report_reads_served_models_only(models_dir, monkeypatch):
    served = models_dir / "NBA_PTS.mdl"
    served.write_bytes(pickle.dumps({"model_version": OUTGOING, "cv": 0.5}))
    prior_models.keep_prior_model(served)
    incoming = {"model_version": INCOMING, "cv": 0.9}
    served.write_bytes(pickle.dumps(incoming))
    reported = {}
    for loader in ("load_shipped_config", "load_zi_config"):
        monkeypatch.setattr(report_module, loader, dict)
    for writer in ("save_cv_std_config", "save_zi_config"):
        monkeypatch.setattr(report_module, writer, lambda *_args: None)
    monkeypatch.setattr(
        report_module,
        "write_model_stats",
        lambda league_models, *_args: reported.update(league_models),
    )

    report_module.report()

    assert reported == {"NBA": {"PTS": incoming}}
