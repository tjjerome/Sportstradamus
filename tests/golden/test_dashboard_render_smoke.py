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

``switch_page`` itself has a second, separate gap for any page whose registered
``url_path=`` differs from its filename slug: it computes the target page hash as
``calc_hash(page_icon_and_name(script_path))`` (a filename-derived name — the old
pages/ directory convention), but the real ``StreamlitPage._script_hash`` a running
``st.navigation()`` matches against is ``calc_hash(self._url_path)``. Board/Games/
Receipts/Tonight all register a ``url_path`` equal to their filename stem, so the
two computations coincidentally agree and ``switch_page`` "just works" for them.
The three Model Lab pages register hyphenated ``url_path``s (``lab-correlations``
vs the filename ``lab_correlations``), so the hashes never match and
``switch_page`` silently no-ops — the run after it looks identical to never having
switched at all (no exception, title stays on the previous page). The Lab
Correlations test below overrides ``AppTest._page_hash`` directly with the correct
``calc_hash(url_path)`` right after calling ``switch_page`` (which still does its
other job of validating the file exists and copying it into the tree) to route
around this; any future AppTest smoke test for lab_diagnostics.py/lab_training.py
needs the identical override.

A third gap, hit by the Lab Diagnostics test: patching a ``@st.cache_data``-backed
loader's path constant (e.g. ``HISTORY_PATH``) is not reliably enough on its own when
an earlier test in the same session already ran an AppTest against a page that reads
through the same cached loader — Streamlit's script runner does not always re-execute
the loader's module fresh on ``switch_page``, so a stale (possibly empty) cache entry
from an earlier test can outlive that test's own monkeypatch teardown and get served
to this one instead of a fresh read of the newly patched path. Call
``st.cache_data.clear()`` right before constructing the ``AppTest`` for any page whose
data comes through a cached loader whose path constant this test patches — it makes
the test's outcome independent of what ran earlier in the session instead of chasing
which specific prior test caused the staleness.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

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


# Mirrors prediction/parlay.py's row-construction contract: legs is a
# list[dict] of leg_schema.LEG_FIELDS-shaped records, and Corr Pairs/Boost
# Pairs are flat tuples over itertools.combinations(legs, 2) order.
_PARLAY_ROWS = [
    {
        "League": "NBA",
        "Platform": "Underdog",
        "Date": "2026-07-01",
        "Bet Size": 2,
        "Legs Resolved": 2,
        "Misses": 0,
        "Boost": 3.0,
        "P": 0.42,
        "Indep P": 0.35,
        "legs": [
            {"player": "J. Brunson", "market": "PTS", "league": "NBA"},
            {"player": "K. Towns", "market": "REB", "league": "NBA"},
        ],
        "Corr Pairs": (0.12,),
        "Boost Pairs": (1.1,),
    },
    {
        "League": "NBA",
        "Platform": "Underdog",
        "Date": "2026-07-02",
        "Bet Size": 2,
        "Legs Resolved": 2,
        "Misses": 1,
        "Boost": 3.0,
        "P": 0.55,
        "Indep P": 0.40,
        "legs": [
            {"player": "A. Wilson", "market": "AST", "league": "NBA"},
            {"player": "A. Wilson", "market": "TOV", "league": "NBA"},
        ],
        "Corr Pairs": (0.42,),
        "Boost Pairs": (1.5,),
    },
]


def test_lab_correlations_renders_heatmap_and_panels(monkeypatch, tmp_path):
    """P8 Task B3 smoke: Lab Correlations navigates clean with the new heatmap section.

    ``load_parlays`` reads through ``helpers.io.read_parlay_hist``, which reads
    ``io``'s own ``PARLAY_HIST_PATH`` binding directly rather than the name
    ``dashboard.data`` imports — both bindings need patching, or the real
    ~1.7M-row production ``parlay_hist.parquet`` loads instead of the fixture.
    """
    fixture = tmp_path / "parlay_hist.parquet"
    pd.DataFrame(_PARLAY_ROWS).to_parquet(fixture)
    monkeypatch.setattr("sportstradamus.helpers.io.PARLAY_HIST_PATH", fixture)
    monkeypatch.setattr("sportstradamus.dashboard.data.PARLAY_HIST_PATH", fixture)

    at = AppTest.from_file(str(_APP), default_timeout=30)
    at.run()
    at.switch_page("surfaces/lab_correlations.py")
    at._page_hash = calc_hash("lab-correlations")  # see module docstring
    at.run()
    assert not at.exception
    assert at.title[0].value == "Correlations & Parlays"


def _diag_row(league: str, market: str, i: int) -> dict:
    """One resolved-offer row for the Lab Diagnostics fixture below.

    Alternates Over/Under and win/loss by ``i`` so both the market table (n>=5) and
    the Murphy-decomposition panel (n>=20) have a real accuracy/calibration spread
    per cell instead of a degenerate all-hit or all-miss column.
    """
    over = i % 2 == 0
    line = 20.0
    actual = line + 3 if (over == (i % 4 == 0)) else line - 3
    return {
        "Player": f"Player {i}",
        "League": league,
        "Date": f"2026-06-{(i % 27) + 1:02d}",
        "Market": market,
        "Team": "NYK",
        "Projection": line + 1,
        "Market Projection": line,
        "Dist": "SkewNormal",
        "CV": 0.3,
        "Model Param": None,
        "Gate": None,
        "Temperature": None,
        "Disp Cal": None,
        "Step": None,
        "Line": line,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Over" if over else "Under",
        "Win Prob": 0.55 + (0.01 * (i % 5)),
        "Market Prob": 0.52,
        "Close Market Prob": None,
        "Market CLV": None,
        "Model CLV": None,
        "Alt Line": False,
        "Actual": actual,
    }


# 25 rows/cell so the market table (n>=5), the sharpness view, and the Murphy
# decomposition panel (n>=20) all render real content for both cells.
_DIAG_ROWS = [_diag_row("NBA", "PTS", i) for i in range(25)] + [
    _diag_row("NBA", "AST", i) for i in range(25)
]


def test_lab_diagnostics_renders_market_table_and_start_here_strip(monkeypatch, tmp_path):
    """P8 Task B4 smoke: Lab Diagnostics navigates clean with the worst-BSS-first table,
    the "start here" tiles, and the Family/Norm meta columns.

    ``load_history`` reads through ``helpers.io.read_history``, which reads ``io``'s own
    ``HISTORY_PATH`` binding directly rather than the name ``dashboard.data`` imports —
    both need patching, same two-binding shape as Lab Correlations' ``PARLAY_HIST_PATH``
    above. Also needs ``st.cache_data.clear()`` — see the module docstring's third gap.
    """
    fixture = tmp_path / "history.parquet"
    pd.DataFrame(_DIAG_ROWS).to_parquet(fixture)
    monkeypatch.setattr("sportstradamus.helpers.io.HISTORY_PATH", fixture)
    monkeypatch.setattr("sportstradamus.dashboard.data.HISTORY_PATH", fixture)
    st.cache_data.clear()

    at = AppTest.from_file(str(_APP), default_timeout=30)
    at.run()
    at.switch_page("surfaces/lab_diagnostics.py")
    at._page_hash = calc_hash("lab-diagnostics")  # see module docstring
    at.run()
    assert not at.exception
    assert at.title[0].value == "Market Diagnostics & Forecast Quality"
    assert any("Start here" in c.value for c in at.caption)
    tile_labels = {m.label for m in at.metric}
    assert {"NBA - PTS", "NBA - AST"} <= tile_labels
