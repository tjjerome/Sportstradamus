"""First AppTest smoke checks for the Streamlit dashboard (P8 Phase A, spec §3.2/§3.4).

``AppTest`` copies the target script into a temp location before executing it, which
breaks ``__file__``-relative paths — ``app.py`` builds its ``st.Page`` paths via
``Path(__file__).parent / "surfaces"``, so a naive ``AppTest.from_file("app.py")``
resolves ``_surfaces`` against the temp copy's directory and every ``st.Page(...)``
raises "file could not be found." The fix is a runpy wrapper: the wrapper string
bakes in ``app.py``'s real absolute path as a literal, then ``runpy.run_path``
executes it with its OWN ``__file__`` intact, sidestepping the temp-copy problem.

That workaround is only needed for ``AppTest.from_string``. ``AppTest.from_file``
given an absolute, existing path runs the script in place (no temp copy), so
``switch_page`` — which resolves relative to ``AppTest``'s own script path — works
against it directly; the Board test below uses ``from_file`` for that reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

_APP = Path("src/sportstradamus/dashboard/app.py").resolve()
_WRAPPER = f"import runpy; runpy.run_path(r'{_APP}', run_name='__main__')"

# Mirrors prediction/persist.py's _OFFER_KEEP_COLS schema (real current_offers.parquet
# contract) — two rows split across two dates so the default Tonight lens (soonest
# Date) has something to narrow against.
_OFFER_ROWS = [
    {
        "League": "NBA",
        "Date": "2026-07-04",
        "Team": "NYK",
        "Opponent": "SAS",
        "Home": False,
        "Game": "NYK/SAS",
        "Player": "J. Brunson",
        "Market": "PTS",
        "Platform": "Underdog",
        "Bet": "Over",
        "Line": 25.5,
        "Consensus Line": 25.0,
        "Boost": 1.85,
        "Win Prob": 0.58,
        "Model EV": 1.07,
        "Market EV": 1.02,
    },
    {
        "League": "WNBA",
        "Date": "2026-07-05",
        "Team": "LVA",
        "Opponent": "IND",
        "Home": True,
        "Game": "LVA/IND",
        "Player": "A. Wilson",
        "Market": "PTS",
        "Platform": "Underdog",
        "Bet": "Over",
        "Line": 23.5,
        "Consensus Line": 23.0,
        "Boost": 1.90,
        "Win Prob": 0.61,
        "Model EV": 1.11,
        "Market EV": 1.03,
    },
]


def test_app_boots_and_tonight_renders():
    """Bare ``.run()`` with no navigation lands on Tonight (``app.py``'s default page)."""
    at = AppTest.from_string(_WRAPPER, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Tonight"


def test_app_sport_switch_rerenders_without_exception():
    """Interacting with the global sport segmented-control triggers a clean rerun.

    Exercises the ``st.session_state["sport"]`` handoff at the top of ``app.py``
    (the widget-key-to-plain-key copy every surface reads) through an actual
    widget interaction, not just the initial render.
    """
    at = AppTest.from_string(_WRAPPER, default_timeout=30)
    at.run()
    at.segmented_control(key="_sport_widget").set_value("NBA").run()
    assert not at.exception
    assert at.session_state["sport"] == "NBA"


def test_board_renders_condensed_grid(monkeypatch, tmp_path):
    """P8 Phase B smoke: Board navigates clean and the condensed grid has no exception.

    ``load_current_offers`` mtime-keys off ``CURRENT_OFFERS_PATH`` on disk — this
    points that at a synthetic schema-correct fixture instead of touching the real
    ``data/runtime/current_offers.parquet`` a live dashboard/cron would depend on.
    """
    fixture = tmp_path / "current_offers.parquet"
    pd.DataFrame(_OFFER_ROWS).to_parquet(fixture)
    monkeypatch.setattr("sportstradamus.dashboard.data.CURRENT_OFFERS_PATH", fixture)

    at = AppTest.from_file(str(_APP), default_timeout=30)
    at.run()
    at.switch_page("surfaces/board.py")
    at.run()
    assert not at.exception
    assert at.title[0].value == "Today's Predictions"
