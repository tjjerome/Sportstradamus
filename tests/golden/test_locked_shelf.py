"""AppTest pins for the sidebar's locked-slip shelf (``dashboard/components/locked_shelf.py``).

Renders the shelf alone in a minimal script over a ``user_slips`` parquet in ``tmp_path``.
A slip locked with the fit shows its EV beside its Model EV; one locked before the fit
existed shows its Model EV alone, under that name.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

_SCRIPT = """
from sportstradamus.dashboard.components.locked_shelf import render_locked_shelf
from sportstradamus.dashboard.components.slip_state import init_slip_state

init_slip_state()
render_locked_shelf()
"""

_LEG = {"player": "Jalen Brunson", "market": "PTS", "bet": "Over", "line": 25.5, "league": "NBA"}


def _slip(slip_id: str, **overrides) -> dict:
    return {
        "slip_id": slip_id,
        "saved_at": f"2026-11-2{slip_id[-1]}T12:00:00",
        "builder_type": "constellation",
        "platform": "Underdog",
        "legs": [_LEG, _LEG | {"player": "Jayson Tatum"}],
        "bet_size": 2,
        "play_type": "Power",
        "payout_multiplier": 3.5,
        "model_ev": 1.2,
        "stake": 12.0,
        "headline": "",
        "status": "pending",
    } | overrides


def _shelf(monkeypatch, tmp_path, slips: list[dict]) -> AppTest:
    path = tmp_path / "user_slips.parquet"
    pd.DataFrame(slips).to_parquet(path)
    monkeypatch.setattr("sportstradamus.helpers.io.USER_SLIPS_PATH", path)
    monkeypatch.setattr("sportstradamus.dashboard.data.USER_SLIPS_PATH", path)
    monkeypatch.setattr(
        "sportstradamus.dashboard.data.CURRENT_OFFERS_PATH", tmp_path / "no_offers.parquet"
    )
    st.cache_data.clear()
    at = AppTest.from_string(_SCRIPT, default_timeout=15)
    at.run()
    assert not at.exception
    return at


def test_shelf_shows_the_fitted_ev_beside_the_model_ev(monkeypatch, tmp_path):
    """The newer slip (s2, listed first) carries its ev; the older one predates the fit."""
    at = _shelf(monkeypatch, tmp_path, [_slip("s1"), _slip("s2", ev=0.97, stake=0.0)])
    numbers = [c.value for c in at.caption if c.value.startswith("3.50x")]
    assert numbers == [
        "3.50x · EV 0.97 · Model EV 1.20 · $0.00 · pending",
        "3.50x · Model EV 1.20 · $12.00 · pending",
    ]


def test_shelf_of_slips_locked_before_the_fit_shows_model_ev_alone(monkeypatch, tmp_path):
    at = _shelf(monkeypatch, tmp_path, [_slip("s1")])
    assert "3.50x · Model EV 1.20 · $12.00 · pending" in [c.value for c in at.caption]


def test_slip_detail_labels_the_fit_ev_and_the_claim_model_ev(monkeypatch, tmp_path):
    at = _shelf(monkeypatch, tmp_path, [_slip("s1"), _slip("s2", ev=0.97)])
    at.button(key="shelf_view_s2").click().run()
    assert not at.exception
    assert [(m.label, m.value) for m in at.metric] == [
        ("Payout", "3.50x"),
        ("EV", "0.97"),
        ("Model EV", "1.20"),
        ("Stake", "$12.00"),
    ]
