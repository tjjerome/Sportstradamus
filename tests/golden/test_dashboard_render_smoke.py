"""First AppTest smoke checks for the Streamlit dashboard (P8 Phase A, spec §3.2/§3.4).

``AppTest`` copies the target script into a temp location before executing it, which
breaks ``__file__``-relative paths — ``app.py`` builds its ``st.Page`` paths via
``Path(__file__).parent / "surfaces"``, so a naive ``AppTest.from_file("app.py")``
resolves ``_surfaces`` against the temp copy's directory and every ``st.Page(...)``
raises "file could not be found." The fix is a runpy wrapper: the wrapper string
bakes in ``app.py``'s real absolute path as a literal, then ``runpy.run_path``
executes it with its OWN ``__file__`` intact, sidestepping the temp-copy problem.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP = Path("src/sportstradamus/dashboard/app.py").resolve()
_WRAPPER = f"import runpy; runpy.run_path(r'{_APP}', run_name='__main__')"


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
