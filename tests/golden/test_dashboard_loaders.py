"""Snapshot loaders backfill the columns newer surfaces read unguarded.

A dashboard restarted on the pull that brought a new column serves the previous
``prophecize`` snapshot until the next run, so the loader, not the surface, owns
the default.
"""

from __future__ import annotations

import importlib

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.legs import record_line


def _loaders(monkeypatch, tmp_path, *, offers_rows, stories_rows):
    data_module = importlib.import_module("sportstradamus.dashboard.data")
    offers = tmp_path / "current_offers.parquet"
    stories = tmp_path / "current_game_stories.parquet"
    pd.DataFrame(offers_rows).to_parquet(offers)
    pd.DataFrame(stories_rows).to_parquet(stories)
    monkeypatch.setattr(data_module, "CURRENT_OFFERS_PATH", offers)
    monkeypatch.setattr(data_module, "CURRENT_GAME_STORIES_PATH", stories)
    st.cache_data.clear()
    return data_module


def test_offers_loader_backfills_star_on_a_pre_star_snapshot(monkeypatch, tmp_path):
    data_module = _loaders(
        monkeypatch,
        tmp_path,
        offers_rows=[{"Player": "J. Brunson", "Kelly": 0.03}],
        stories_rows=[{"story_id": "NYK/SAS#0"}],
    )
    offers = data_module.load_current_offers()
    assert offers["Star"].tolist() == [0.0]
    assert offers["Star"].dtype == "float64"


def test_offers_loader_reads_a_snapshot_from_before_the_ev_stamp_as_not_priced(
    monkeypatch, tmp_path
):
    data_module = _loaders(
        monkeypatch,
        tmp_path,
        offers_rows=[{"Player": "J. Brunson", "Kelly": 0.03, "Star": 2.4}],
        stories_rows=[{"story_id": "NYK/SAS#0"}],
    )
    offers = data_module.load_current_offers()
    assert offers["Pick"].tolist() == [False]
    assert offers["Pick"].dtype == bool
    assert offers[["Fit Prob", "EV"]].isna().all().all()


def test_offers_loader_keeps_the_fitted_numbers_a_snapshot_already_has(monkeypatch, tmp_path):
    data_module = _loaders(
        monkeypatch,
        tmp_path,
        offers_rows=[
            {"Player": "J. Brunson", "Fit Prob": 0.52, "Fit Prob Low": 0.47, "EV": 0.95},
            {"Player": "V. Wembanyama", "Fit Prob": None, "Fit Prob Low": None, "EV": None},
        ],
        stories_rows=[{"story_id": "NYK/SAS#0"}],
    )
    offers = data_module.load_current_offers()
    assert offers["EV"].tolist()[0] == 0.95
    assert offers["Fit Prob Low"].tolist()[0] == 0.47
    assert offers["Pick"].tolist() == [False, False]


def test_stories_loader_backfills_lead_columns(monkeypatch, tmp_path):
    data_module = _loaders(
        monkeypatch,
        tmp_path,
        offers_rows=[{"Player": "J. Brunson", "Star": 2.4}],
        stories_rows=[{"story_id": "NYK/SAS#0", "headline": "x"}],
    )
    stories = data_module.load_current_game_stories()
    assert stories["lead"].tolist() == [False]
    assert stories["lead_side"].tolist() == [""]
    assert data_module.load_current_offers()["Star"].tolist() == [2.4]


def test_stories_loader_reads_a_snapshot_from_before_the_fit_priced_stories_as_not_priced(
    monkeypatch, tmp_path
):
    data_module = _loaders(
        monkeypatch,
        tmp_path,
        offers_rows=[{"Player": "J. Brunson"}],
        stories_rows=[{"story_id": "NYK/SAS#0", "model_ev": 1.3, "lead": True}],
    )
    stories = data_module.load_current_game_stories()
    assert stories["ev"].isna().all()
    assert stories["model_ev"].tolist() == [1.3]
    assert stories["lead"].tolist() == [True]


def test_stories_loader_keeps_the_ev_a_snapshot_already_has(monkeypatch, tmp_path):
    data_module = _loaders(
        monkeypatch,
        tmp_path,
        offers_rows=[{"Player": "J. Brunson"}],
        stories_rows=[{"story_id": "NYK/SAS#0", "ev": 0.97, "model_ev": 1.3}],
    )
    assert data_module.load_current_game_stories()["ev"].tolist() == [0.97]


def test_parlays_loader_reads_a_snapshot_from_before_the_fit_priced_entries_as_not_priced(
    monkeypatch, tmp_path
):
    data_module = importlib.import_module("sportstradamus.dashboard.data")
    parlays = tmp_path / "current_parlays.parquet"
    pd.DataFrame(
        [
            {"Game": "NYK/SAS", "Model EV": 1.4, "Thesis": "old"},
            {"Game": "NYK/SAS", "Model EV": 1.1, "Thesis": "older"},
        ]
    ).to_parquet(parlays)
    monkeypatch.setattr(data_module, "CURRENT_PARLAYS_PATH", parlays)
    st.cache_data.clear()
    loaded = data_module.load_current_parlays()
    assert loaded["EV"].isna().all()
    assert loaded["Model EV"].tolist() == [1.4, 1.1]


def test_parlays_loader_keeps_the_ev_a_snapshot_already_has(monkeypatch, tmp_path):
    data_module = importlib.import_module("sportstradamus.dashboard.data")
    parlays = tmp_path / "current_parlays.parquet"
    pd.DataFrame([{"Game": "NYK/SAS", "EV": 0.93, "Model EV": 1.4}]).to_parquet(parlays)
    monkeypatch.setattr(data_module, "CURRENT_PARLAYS_PATH", parlays)
    st.cache_data.clear()
    assert data_module.load_current_parlays()["EV"].tolist() == [0.93]


def test_before_the_nightly_job_runs_every_league_and_kind_reads_no_picks_settled(
    monkeypatch, tmp_path
):
    data_module = importlib.import_module("sportstradamus.dashboard.data")
    monkeypatch.setattr(data_module, "PLAY_RECORD_PATH", tmp_path / "play_record.parquet")
    st.cache_data.clear()
    assert record_line(data_module.load_play_record(), "NHL", "underdog_even") == (
        "NHL even picks: 0 legs yet",
        None,
    )
