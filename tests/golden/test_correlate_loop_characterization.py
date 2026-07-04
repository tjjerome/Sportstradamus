"""Characterization pin for the per-team correlation loop inside ``correlate``.

``correlate`` (CC 19) is decomposed into an orchestrator + phase helpers. The
existing ``test_correlate.py`` drives the orchestration end-to-end with an EMPTY
gamelog, so the per-team loop body (the part extracted into ``_correlate_teams``)
never runs with data. This pins that loop: it mocks ``_build_team_game_records``
(the gamelog->residualize->profile pipeline the refactor does not touch) to feed
known per-(team, game) records, then asserts the exact ``corr_same_team`` /
``corr_opposing`` R-series and the metadata observation counts that
``correlate("NBA", stub, force=True)`` writes.

Every R is shrunk by overlap/MIN_OVERLAP_FOR_FULL_WEIGHT = 6/30 = 0.2 (only 6
games per team); the diagonal self-correlations (1.0 -> 0.2 after shrink) and the
same-team/opposing split on the ``_OPP_`` prefix are part of the pinned behavior.
Writes are redirected to a per-test tmp_path (see the ``outputs`` fixture) so this
never touches real package data or races other tests under pytest-xdist.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json

import pandas as pd
import pytest

from sportstradamus import data as _data

corr_mod = importlib.import_module("sportstradamus.training.correlate")


class _Stub:
    def __init__(self):
        self.season_start = _dt.date(2026, 1, 1)
        self.gamelog = pd.DataFrame()
        self.log_strings = {"date": "GAME_DATE"}
        self.playerProfile = pd.DataFrame()

    def load(self):
        pass

    def update(self):
        pass

    def profile_market(self, *a, **k):
        pass


def _records(league, log, latest_date):
    today = _dt.datetime.today()
    rows = []
    for i, x in enumerate([1, 2, 3, 4, 5, 6]):
        d = today - _dt.timedelta(days=10 + i)
        rows.append(
            {"TEAM": "AAA", "DATE": d, "x": float(x), "y": float(2 * x + 1), "_OPP_x": float(7 - x)}
        )
        rows.append(
            {"TEAM": "BBB", "DATE": d, "x": float(7 - x), "y": float(x * x), "_OPP_x": float(x)}
        )
    return rows


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    """Redirect correlate()'s package-data reads/writes to an isolated tmp_path.

    Mirrors ``test_correlate.py``'s ``_preserve_nba_correlate_outputs``: the old
    snapshot/restore of the real ``leagues/nba`` dir missed the raw warm-start
    cache and still raced against other tests writing the same real path
    concurrently under pytest-xdist. Redirecting removes both problems.
    """
    real_files = corr_mod.pkg_resources.files

    def _fake_files(pkg):
        return tmp_path if pkg is _data else real_files(pkg)

    monkeypatch.setattr(corr_mod.pkg_resources, "files", _fake_files)
    monkeypatch.setattr(corr_mod, "_build_team_game_records", _records)

    corr_mod.correlate("NBA", _Stub(), force=True)
    league_dir = tmp_path / "leagues" / "nba"
    same = pd.read_parquet(league_dir / "corr_same_team.parquet")["R"].to_dict()
    opp = pd.read_parquet(league_dir / "corr_opposing.parquet")["R"].to_dict()
    meta = json.loads((league_dir / "corr_metadata.json").read_text())
    return {"same": same, "opp": opp, "meta": meta}


def test_same_team_blocks(outputs):
    expected = {
        ("AAA", "x", "x"): 0.2,
        ("AAA", "x", "y"): 0.2,
        ("AAA", "y", "x"): 0.2,
        ("AAA", "y", "y"): 0.2,
        ("BBB", "x", "x"): 0.2,
        ("BBB", "x", "y"): -0.2,
        ("BBB", "y", "x"): -0.2,
        ("BBB", "y", "y"): 0.2,
    }
    assert set(outputs["same"]) == set(expected)
    for k, v in expected.items():
        assert outputs["same"][k] == pytest.approx(v, abs=1e-9)


def test_opposing_blocks(outputs):
    expected = {
        ("AAA", "x", "x"): -0.2,
        ("AAA", "y", "x"): -0.2,
        ("BBB", "x", "x"): -0.2,
        ("BBB", "y", "x"): 0.2,
    }
    assert set(outputs["opp"]) == set(expected)
    for k, v in expected.items():
        assert outputs["opp"][k] == pytest.approx(v, abs=1e-9)


def test_metadata_observations(outputs):
    assert outputs["meta"]["per_team_observations"] == {"AAA": 6, "BBB": 6}
    assert outputs["meta"]["total_team_game_observations"] == 12
