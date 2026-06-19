"""Characterization pin for ``StatsNFL.get_volume_stats``.

CC-11 method with ZERO end-to-end coverage. It walks ``volume_stats``
(``attempts`` → ``carries`` → ``targets``): collapses the ``attempts`` SkewNormal
params to mean/std and drops the raw loc/scale/alpha, then for ``carries``/
``targets`` runs a per-team precision-weighted deficit rescale against the
plays/pass-attempt budget.

This pins the resulting ``playerProfile`` so the decomposition (extracting the
``attempts`` finalize branch and the per-team rescale body) is behavior-preserving.
``load_volume_model_params`` is mocked to a no-op returning True; the proj columns
are pre-seeded. ``P2``'s carries use ``alpha=1`` to exercise the nonzero-delta
``loc != mean`` path; the others use ``alpha=0``.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from sportstradamus.stats.nfl import StatsNFL


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.setattr(StatsNFL, "load_volume_model_params", lambda self, *a, **k: True)
    s = StatsNFL()
    s.volume_stats = ["attempts", "carries", "targets"]
    s.playerProfile = pd.DataFrame(
        {
            "team": ["KC", "KC"],
            "proj attempts loc": [30.0, 0.0],
            "proj attempts scale": [5.0, 0.0],
            "proj attempts alpha": [0.5, 0.0],
            "proj carries loc": [3.0, 10.0],
            "proj carries scale": [1.0, 2.0],
            "proj carries alpha": [0.0, 1.0],
            "proj targets loc": [2.0, 6.0],
            "proj targets scale": [1.0, 2.0],
            "proj targets alpha": [0.0, 0.0],
        },
        index=["P1", "P2"],
    )
    s.teamProfile = pd.DataFrame({"plays_per_game": [65.0], "pass_rate": [0.6]}, index=["KC"])
    s.get_volume_stats({}, _dt.date(2024, 9, 8))
    return s.playerProfile


def test_attempts_collapsed_and_raw_params_dropped(profile):
    assert "proj attempts loc" not in profile.columns
    assert "proj attempts scale" not in profile.columns
    assert "proj attempts alpha" not in profile.columns
    assert profile["proj attempts mean"].tolist() == pytest.approx([31.784124, 0.0], rel=1e-5)
    assert profile["proj attempts std"].tolist() == pytest.approx([5.0, 0.0])


def test_carries_rescaled(profile):
    assert profile["proj carries mean"].tolist() == pytest.approx([5.617499, 21.598377], rel=1e-5)
    assert profile["proj carries std"].tolist() == pytest.approx([1.8725, 3.881675], rel=1e-5)
    assert profile["proj carries loc"].tolist() == pytest.approx([5.617499, 19.408376], rel=1e-5)
    assert profile["proj carries scale"].tolist() == pytest.approx([1.8725, 3.881675], rel=1e-5)


def test_targets_rescaled(profile):
    assert profile["proj targets mean"].tolist() == pytest.approx([5.156825, 18.627299], rel=1e-5)
    assert profile["proj targets std"].tolist() == pytest.approx([2.578412, 6.2091], rel=1e-5)
    assert profile["proj targets loc"].tolist() == pytest.approx([5.156825, 18.627299], rel=1e-5)
