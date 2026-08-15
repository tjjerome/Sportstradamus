"""A diverged calibration fit must never scale the served distribution.

Two fail-closed serving guards in ``prediction/model_prob.py``. A
``dispersion_cal`` pinned at its optimizer bound (the 2026-08 WNBA
overconfident-unders vector) means the joint ``(c, s)`` fit diverged, so
``_serving_dispersion_cal`` serves the unscaled shape instead. And a cell
whose ``stat_meta`` says ``withheld`` must serve nothing even when a missed
``meditate`` left a stale pickle on disk (two weeks of WNBA PRA, 2026-08).
"""

import importlib
import logging

import pandas as pd
import pytest

from sportstradamus.prediction.model_prob import _serving_dispersion_cal
from sportstradamus.training.ship_config import WITHHELD

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


def _withhold_wnba_pra(monkeypatch, pickle_path):
    monkeypatch.setattr(model_prob_module, "stat_meta", {"WNBA": {"PRA": {"shipped": WITHHELD}}})
    monkeypatch.setattr(model_prob_module, "model_pickle_path", lambda league, market: pickle_path)


def _serve_wnba_pra():
    return model_prob_module.model_prob(
        [{"Player": "Test Player", "Date": "2026-08-15"}],
        "WNBA",
        "PRA",
        "Underdog",
        None,
        pd.DataFrame(),
    )


def test_withheld_cell_with_stale_pickle_serves_nothing(monkeypatch, tmp_path, caplog):
    pickle_path = tmp_path / "WNBA_PRA.mdl"
    # Unloadable bytes: reaching pickle.load would raise, proving the withheld
    # check precedes the load.
    pickle_path.write_bytes(b"not a pickle")
    _withhold_wnba_pra(monkeypatch, pickle_path)
    with caplog.at_level(logging.WARNING, logger="log"):
        assert _serve_wnba_pra() == []
    assert "withheld but pickle on disk" in caplog.text


def test_withheld_cell_without_pickle_is_silent(monkeypatch, tmp_path, caplog):
    _withhold_wnba_pra(monkeypatch, tmp_path / "WNBA_PRA.mdl")
    with caplog.at_level(logging.WARNING, logger="log"):
        assert _serve_wnba_pra() == []
    assert "withheld but pickle on disk" not in caplog.text
