"""WS2 Step D: ``save_book_shape_config`` persists the per-cell ``book_shape`` coeffs into
``stat_calibration.json`` (gitignored, runtime-recomputed), mirroring ``save_zi_config`` —
one field at a time so a writer updating ``book_shape`` never clobbers ``cv``/``std``/``zi``.
"""

from __future__ import annotations

import json

from sportstradamus.training import config as tcfg

_COEFFS = {"a": 1.3, "b": 1.1, "skew_c": 0.6, "skew_d": -0.1, "n_bins": 9}


def test_save_book_shape_config_round_trips_and_preserves(tmp_path, monkeypatch):
    cal_path = tmp_path / "stat_calibration.json"
    cal_path.write_text(json.dumps({"WNBA": {"AST": {"cv": 0.5, "std": 1.0, "zi": 0.1}}}))
    monkeypatch.setattr(tcfg, "_STAT_CAL_PATH", cal_path)

    tcfg.save_book_shape_config({"WNBA": {"AST": _COEFFS}})

    cell = json.loads(cal_path.read_text())["WNBA"]["AST"]
    assert cell["book_shape"] == _COEFFS
    assert cell["cv"] == 0.5
    assert cell["zi"] == 0.1


def test_save_book_shape_config_stores_none_for_unfittable(tmp_path, monkeypatch):
    cal_path = tmp_path / "stat_calibration.json"
    monkeypatch.setattr(tcfg, "_STAT_CAL_PATH", cal_path)

    tcfg.save_book_shape_config({"WNBA": {"AST": None}})

    assert json.loads(cal_path.read_text())["WNBA"]["AST"]["book_shape"] is None
