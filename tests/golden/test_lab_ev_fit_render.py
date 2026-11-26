"""AppTest for the Lab's EV Fit page: the fit's weights and every play's record, as written.

Lab pages register hyphenated ``url_path``s, so ``switch_page`` alone never lands on one;
the test sets ``AppTest._page_hash`` to the page's own hash after it (see
``test_dashboard_render_smoke.py``'s module docstring).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

_APP = Path("src/sportstradamus/dashboard/app.py").resolve()

_FIT = [
    {
        "league": "NHL",
        "kind": "underdog_even",
        "book_class": "A",
        "n_legs": 802,
        "n_dates": 40,
        "first_date": "2026-09-01",
        "last_date": "2026-11-25",
        "prior_legs": 4000,
        "c": 0.012,
        "a1": np.nan,
        "w_book": 0.31,
        "w_model": 0.08,
        "w_fb": np.nan,
        "se_c": 0.02,
        "se_a1": np.nan,
        "se_w_book": 0.05,
        "se_w_model": 0.03,
        "se_w_fb": np.nan,
        "cov": "[]",
    }
]
_RECORD = [
    {"play": "model_edge", "league": "NHL", "kind": "underdog_even", "n": 900, "roi": -0.081},
    {"play": "ev_pick", "league": "NHL", "kind": "underdog_even", "n": 269, "roi": 0.027},
]


def _lab(monkeypatch, tmp_path, *, fit: list[dict], record: list[dict]) -> AppTest:
    data_module = importlib.import_module("sportstradamus.dashboard.data")
    for name, rows in (("LIVE_FIT_PATH", fit), ("PLAY_RECORD_PATH", record)):
        path = tmp_path / f"{name}.parquet"
        if rows:
            pd.DataFrame(rows).to_parquet(path)
        monkeypatch.setattr(data_module, name, path)
    st.cache_data.clear()

    at = AppTest.from_file(str(_APP), default_timeout=30)
    at.run()
    at.switch_page("surfaces/lab_ev_fit.py")
    at._page_hash = calc_hash("lab-ev-fit")
    at.run()
    assert not at.exception
    return at


def test_lab_shows_each_weight_beside_its_standard_error_and_the_whole_record(
    monkeypatch, tmp_path
):
    at = _lab(monkeypatch, tmp_path, fit=_FIT, record=_RECORD)
    weights, record = (frame.value for frame in at.dataframe)
    assert weights.columns.tolist() == [
        "League",
        "Pick kind",
        "Class",
        "Legs",
        "Intercept",
        "Intercept SE",
        "App price",
        "App price SE",
        "Books",
        "Books SE",
        "Model",
        "Model SE",
        "Book stand-in",
        "Book stand-in SE",
    ]
    assert weights.loc[0, ["Books", "Books SE"]].tolist() == [0.31, 0.05]
    assert record[["play", "n"]].values.tolist() == [["model_edge", 900], ["ev_pick", 269]]


def test_lab_before_the_nightly_job_runs_says_so(monkeypatch, tmp_path):
    at = _lab(monkeypatch, tmp_path, fit=[], record=[])
    assert not at.dataframe
    assert len(at.info) == 2
