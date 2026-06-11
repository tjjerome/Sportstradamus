"""Characterization pins for ``surfaces/slips.py::_render_parlay``.

The page module executes Streamlit page code at import (data loads, ``st.stop``),
so it cannot be imported in a unit test. Instead the
function-definition region (``def _render_parlay`` up to the top-level
``for game`` loop) is sliced out of the source and exec'd into a namespace with
a recording fake ``st`` and stub ``parse_leg`` / ``find_offer_idx``. Assertions
pin the recorded Streamlit-call sequence: the metric row (6 vs 5 columns on
``Indep P`` presence), per-leg buttons + click handling, and the optional
recommended-bet caption.
"""

import pathlib
import types

import pandas as pd
import pytest

_SURFACES = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "sportstradamus"
    / "dashboard"
    / "surfaces"
)
_PAGE = _SURFACES / "slips.py"
_LEG_COLS = [f"Leg {i}" for i in range(1, 7)]


class _Col:
    def __init__(self, rec):
        self._rec = rec

    def metric(self, *args, **kwargs):
        self._rec.append(("metric", args, kwargs))


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSt:
    def __init__(self):
        self.calls = []
        self.session_state = types.SimpleNamespace(detail_stack=[], corr_nav=False)
        self.button_returns = {}

    def container(self, **kwargs):
        self.calls.append(("container", kwargs))
        return _Ctx()

    def columns(self, n):
        self.calls.append(("columns", n))
        return [_Col(self.calls) for _ in range(n)]

    def button(self, label, key=None, **kwargs):
        self.calls.append(("button", label, key))
        return self.button_returns.get(key, False)

    def caption(self, msg):
        self.calls.append(("caption", msg))

    def toast(self, msg):
        self.calls.append(("toast", msg))

    def rerun(self):
        self.calls.append(("rerun",))


def _load_render(fake_st, *, offer_idx=None):
    src = _PAGE.read_text(encoding="utf-8")
    start = src.index("def _render_parlay")
    end = src.index("\ntab_parlays, tab_pickem = st.tabs")
    ns = {
        "pd": pd,
        "st": fake_st,
        "LEG_COLS": _LEG_COLS,
        "parse_leg": lambda leg: leg,
        "find_offer_idx": lambda parsed, offers, platform: offer_idx,
    }
    exec(compile(src[start:end], str(_PAGE), "exec"), ns)
    return ns["_render_parlay"]


def _metric_calls(calls):
    return [(a, k) for tag, a, k in ((c[0], c[1], c[2]) for c in calls if c[0] == "metric")]


@pytest.fixture
def row_indep():
    return pd.Series(
        {
            "Indep P": 0.40,
            "Model EV": 1.5,
            "Books EV": 1.2,
            "Boost": 2.0,
            "Bet Size": 3,
            "Fun": 0.5,
            "P": 0.45,
            "Rec Bet": float("nan"),
            "Platform": "Underdog",
            "Leg 1": "A. Player Over 1.5",
            "Leg 2": "B. Player Under 2.5",
        },
        name="R0",
    )


def test_metric_row_with_indep_and_two_legs(row_indep):
    st = _FakeSt()
    render = _load_render(st)
    render(row_indep, pd.DataFrame())

    assert st.calls[0] == ("container", {"border": True})
    assert ("columns", 6) in st.calls
    labels = [a[0] for a, _ in _metric_calls(st.calls)]
    assert labels == ["Model EV", "Books EV", "Boost", "Bet Size", "Fun", "Joint vs Indep"]
    by_label = {a[0]: a for a, _ in _metric_calls(st.calls)}
    assert by_label["Model EV"][1] == "1.50"
    assert by_label["Boost"][1] == "2.00x"
    assert by_label["Bet Size"][1] == 3
    assert by_label["Joint vs Indep"][1:] == ("0.450", "+0.050")
    buttons = [(c[1], c[2]) for c in st.calls if c[0] == "button"]
    assert buttons == [
        ("  • A. Player Over 1.5", "plyleg::R0::0"),
        ("  • B. Player Under 2.5", "plyleg::R0::1"),
    ]
    assert not any(c[0] == "caption" for c in st.calls)


def test_no_indep_renders_five_columns_and_rec_bet_caption():
    st = _FakeSt()
    render = _load_render(st)
    row = pd.Series(
        {
            "Indep P": float("nan"),
            "Model EV": 3.0,
            "Boost": 1.0,
            "Bet Size": 2,
            "Fun": 0.1,
            "Rec Bet": 0.8,
            "Platform": "Underdog",
            "Leg 1": "C. Player Over 9.5",
        },
        name="R1",
    )
    render(row, pd.DataFrame())

    assert ("columns", 5) in st.calls
    labels = [a[0] for a, _ in _metric_calls(st.calls)]
    assert labels == ["Model EV", "Books EV", "Boost", "Bet Size", "Fun"]
    assert ("caption", "Recommended bet: 0.80 units") in st.calls


def test_leg_click_found_pushes_detail_and_reruns(row_indep):
    st = _FakeSt()
    st.button_returns["plyleg::R0::0"] = True
    render = _load_render(st, offer_idx=7)
    render(row_indep, pd.DataFrame())

    assert st.session_state.detail_stack == [7]
    assert ("rerun",) in st.calls


def test_leg_click_not_found_toasts(row_indep):
    st = _FakeSt()
    st.button_returns["plyleg::R0::0"] = True
    render = _load_render(st, offer_idx=None)
    render(row_indep, pd.DataFrame())

    assert st.session_state.detail_stack == []
    assert ("toast", "Detail unavailable — line moved since this parlay was built.") in st.calls
