"""AppTest render of the Receipts page over the ledger-reconciliation fixture.

The page runs in place through ``app.py`` (``AppTest.from_file`` on its absolute path keeps
``st.Page``'s ``__file__``-relative paths resolving; see ``test_dashboard_render_smoke.py``),
every snapshot it reads pointed at a temp file. The sidebar's default range (90 days back
from the newest row) holds the same offers as the ledger's 90-day window, so the hero shows
the recommended cohort the reconciliation test sums; the flat -110 grade appears nowhere;
and the sport switch narrows the by-side panel to that league's keys.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import streamlit as st
from streamlit.testing.v1 import AppTest

from sportstradamus.realized import calibration_summary, cohort_summary, settled_offers, window
from tests.golden.test_receipts_reconciles_ledger import HISTORY, NOW

_APP = Path(__file__).resolve().parents[2] / "src/sportstradamus/dashboard/app.py"


def _receipts_page(monkeypatch, tmp_path) -> AppTest:
    history_path = tmp_path / "history.parquet"
    HISTORY.to_parquet(history_path)
    calibration_path = tmp_path / "calibration_summary.parquet"
    calibration_summary(settled_offers(HISTORY)).to_parquet(calibration_path)
    # Patch the module objects, never dotted strings: see test_dashboard_render_smoke.py's
    # fifth gap.
    io_module = importlib.import_module("sportstradamus.helpers.io")
    data_module = importlib.import_module("sportstradamus.dashboard.data")
    for module in (io_module, data_module):
        monkeypatch.setattr(module, "HISTORY_PATH", history_path)
        monkeypatch.setattr(module, "USER_SLIPS_PATH", tmp_path / "user_slips.parquet")
    monkeypatch.setattr(data_module, "CALIBRATION_SUMMARY_PATH", calibration_path)
    monkeypatch.setattr(data_module, "PROFIT_SIM_SUMMARY_PATH", tmp_path / "profit_sim.parquet")
    st.cache_data.clear()

    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()
    at.switch_page("surfaces/receipts.py")
    at.run()
    assert not at.exception
    return at


def _hero_text(at: AppTest) -> str:
    hero = next(m.body for m in at.markdown if "The model's recommendations" in m.body)
    return re.sub(r"<[^>]+>", "", hero)


def _by_side_grid_rows(at: AppTest) -> pd.DataFrame:
    grid = next(
        c for c in at.get("component_instance") if c.proto.id.endswith("receipts_by_side_grid")
    )
    rows = next(arg for arg in grid.proto.special_args if arg.key == "data")
    return pa.ipc.open_stream(rows.arrow_dataframe.data.data).read_all().to_pandas()


def test_hero_shows_the_reconciled_recommended_cohort(monkeypatch, tmp_path):
    at = _receipts_page(monkeypatch, tmp_path)
    offers = window(settled_offers(HISTORY), 90, NOW)
    recommended = cohort_summary(offers[offers["Recommended"]])
    posted = cohort_summary(offers)

    hero = _hero_text(at)
    assert (
        f"{recommended['units']:+,.0f} units across {recommended['n']:,} recommended legs" in hero
    )
    assert f"{recommended['roi']:+.1%}" in hero
    assert f"All posted sides in the window: {posted['n']:,} legs" in hero

    rendered = [
        *(m.body for m in at.markdown),
        *(h.body for h in at.get("html")),
        *(element.value for element in (*at.caption, *at.info, *at.warning, *at.subheader)),
        *(f"{m.label} {m.value} {m.delta}" for m in at.metric),
    ]
    assert not [text for text in rendered if "-110" in text or "−110" in text]


def test_sport_filter_narrows_the_by_side_panel(monkeypatch, tmp_path):
    at = _receipts_page(monkeypatch, tmp_path)
    at.segmented_control(key="_sport_widget").set_value("NBA").run()
    at.segmented_control(key="receipts_by_side_split").set_value("league").run()
    assert not at.exception
    assert set(_by_side_grid_rows(at)["League"]) == {"NBA"}
