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
Writes to the real package ``leagues/nba`` dir, restored by the fixture.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import importlib.resources as pkg_resources
import json
from pathlib import Path

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
def outputs(monkeypatch):
    league_dir = Path(str(pkg_resources.files(_data) / "leagues" / "nba"))
    raw = Path(str(pkg_resources.files(_data) / "training_data" / "NBA_corr.parquet"))
    targets = ["corr_metadata.json", "corr_same_team.parquet", "corr_opposing.parquet"]
    saved = {
        n: (league_dir / n).read_bytes() if (league_dir / n).is_file() else None for n in targets
    }
    saved_raw = raw.read_bytes() if raw.is_file() else None
    monkeypatch.setattr(corr_mod, "_build_team_game_records", _records)
    try:
        corr_mod.correlate("NBA", _Stub(), force=True)
        same = pd.read_parquet(league_dir / "corr_same_team.parquet")["R"].to_dict()
        opp = pd.read_parquet(league_dir / "corr_opposing.parquet")["R"].to_dict()
        meta = json.loads((league_dir / "corr_metadata.json").read_text())
        yield {"same": same, "opp": opp, "meta": meta}
    finally:
        for n, b in saved.items():
            p = league_dir / n
            if b is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(b)
        if saved_raw is None:
            raw.unlink(missing_ok=True)
        else:
            raw.write_bytes(saved_raw)


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
