"""A diverged calibration fit must never scale the served distribution.

Two fail-closed serving guards in ``prediction/model_prob.py``. A
``dispersion_cal`` pinned at its optimizer bound (the 2026-08 WNBA
overconfident-unders vector) means the joint ``(c, s)`` fit diverged, so
``_serving_dispersion_cal`` serves the unscaled shape instead. And a pickle
whose strategy identity no longer validates costs its own cell only — it is
a known-unservable cell rather than a crash, so it warns and serves nothing
instead of reaching ``_match_league_offers``' per-market bulkhead as an
unexpected failure.
"""

import dataclasses
import importlib
import logging
import pickle

import pandas as pd
import pytest

from sportstradamus.prediction.model_prob import _serving_dispersion_cal
from sportstradamus.training.model_strategy import (
    MODEL_STRATEGY_MODEL_KEY,
    build_artifact_identity,
    get_strategy,
    strategy_controls,
)

# The package __init__ re-exports the model_prob *function*, shadowing the
# submodule on any attribute-style import; go through sys.modules instead.
model_prob_module = importlib.import_module("sportstradamus.prediction.model_prob")


@pytest.mark.parametrize(
    "dispersion_cal,skew_cal",
    [
        (0.1, 0.4),  # hard at the fit floor
        (0.1004, -0.2),  # inside the float-wobble tolerance around the floor
        (9.99, 0.0),  # inside the tolerance around the cap
        (10.0, 1.1),  # hard at the cap
    ],
)
def test_diverged_dispersion_serves_unscaled(dispersion_cal, skew_cal, caplog):
    # The skew shift resets with the scale: (c, s) come from the same joint fit.
    with caplog.at_level(logging.WARNING, logger="log"):
        assert _serving_dispersion_cal(dispersion_cal, skew_cal, "WNBA_PTS") == (1.0, 0.0)
    assert "pinned at its fit bound" in caplog.text
    assert "WNBA_PTS" in caplog.text


@pytest.mark.parametrize(
    "dispersion_cal,skew_cal",
    [
        (0.5, 0.3),  # ordinary joint fit
        (1.0, 0.5),  # skew-only cell: c neutral, s live
        (0.1006, 0.0),  # just above the floor tolerance
        (9.94, 0.0),  # just below the cap tolerance
    ],
)
def test_healthy_dispersion_passes_through(dispersion_cal, skew_cal, caplog):
    with caplog.at_level(logging.WARNING, logger="log"):
        result = _serving_dispersion_cal(dispersion_cal, skew_cal, "WNBA_PTS")
    assert result == (dispersion_cal, skew_cal)
    assert "pinned at its fit bound" not in caplog.text


def test_stale_strategy_identity_skips_the_cell_not_the_slate(monkeypatch, tmp_path, caplog):
    spec = get_strategy("SkewNormal")
    identity = build_artifact_identity(
        spec.slug,
        "WNBA",
        "PRA",
        strategy_controls(spec)[0],
        matrix_hash="matrix-sha",
    )
    pickle_path = tmp_path / "WNBA_PRA.mdl"
    with open(pickle_path, "wb") as outfile:
        pickle.dump(
            {
                "cv": 1.0,
                "weight": 0.5,
                "step": 0.5,
                "distribution": "SkewNormal",
                "model_version": "test",
                MODEL_STRATEGY_MODEL_KEY: dataclasses.asdict(identity)
                | {"signature": "rotated-away"},
            },
            outfile,
            -1,
        )
    monkeypatch.setattr(model_prob_module, "model_pickle_path", lambda league, market: pickle_path)

    with caplog.at_level(logging.WARNING, logger="log"):
        served = model_prob_module.model_prob(
            [{"Player": "Test Player", "Date": "2026-08-15"}],
            "WNBA",
            "PRA",
            "Underdog",
            None,
            pd.DataFrame(),
        )
    assert served == []
    assert "not served" in caplog.text
    assert "WNBA_PRA" in caplog.text
