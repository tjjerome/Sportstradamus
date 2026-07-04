"""Characterization pins for ``prediction.persist._normalize_parlays``.

Pure DataFrame transform: pins the empty-input shape, the internal-artifact
drop list (``Bet ID`` / ``Fun`` / correlation scoring cols), the survival of
the P8 structured ``legs`` column (list[dict], not a drop-list allowlist), and
the Model-descending sort.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.prediction.persist import _PARLAY_DROP_COLS, _normalize_parlays


def test_empty_input_returns_empty_frame():
    out = _normalize_parlays(pd.DataFrame())
    assert out.empty


def test_none_input_returns_empty_frame():
    out = _normalize_parlays(None)
    assert out.empty


def test_drop_cols_removed_legs_and_display_cols_survive():
    df = pd.DataFrame(
        [
            {
                "Game": "AAA/BBB",
                "League": "NBA",
                "Model EV": 2.0,
                "Market EV": 1.1,
                "legs": [{"player": "A", "market": "PTS"}, {"player": "B", "market": "REB"}],
                "Bet ID": (0, 1),
                "Fun": 0.8,
                "Leg Probs": (0.6, 0.6),
                "Corr Pairs": (0.1,),
                "Boost Pairs": (1.0,),
                "Indep P": 1.5,
                "Indep PB": 1.2,
            }
        ]
    )

    out = _normalize_parlays(df)

    for col in _PARLAY_DROP_COLS:
        assert col not in out.columns
    assert "legs" in out.columns
    assert out.loc[0, "legs"] == [
        {"player": "A", "market": "PTS"},
        {"player": "B", "market": "REB"},
    ]
    # Indep P is kept — the independence-joint comparison point.
    assert out.loc[0, "Indep P"] == 1.5
    assert out.loc[0, "Game"] == "AAA/BBB"


def test_sorts_by_model_ev_descending():
    df = pd.DataFrame(
        [
            {"Model EV": 1.5, "legs": [{"player": "Lo"}]},
            {"Model EV": 3.0, "legs": [{"player": "Hi"}]},
            {"Model EV": 2.0, "legs": [{"player": "Mid"}]},
        ]
    )

    out = _normalize_parlays(df)

    assert out["Model EV"].tolist() == [3.0, 2.0, 1.5]
