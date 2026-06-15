"""Characterization pins for ``prediction.persist._normalize_offers``.

Pure DataFrame transform: pins the empty-input shape, the no-signal row drop,
the Model Param → per-distribution column split (NegBin/Gamma/SkewNormal),
the Gate / Model Skew backfill, the keep-column projection (internal artifacts
dropped), and the Model-descending sort.
"""

import numpy as np
import pandas as pd

from sportstradamus.prediction.persist import _OFFER_KEEP_COLS, _normalize_offers


def test_empty_input_returns_keep_column_frame():
    out = _normalize_offers(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == _OFFER_KEEP_COLS


def test_none_input_returns_keep_column_frame():
    out = _normalize_offers(None)
    assert out.empty
    assert list(out.columns) == _OFFER_KEEP_COLS


def test_full_slate_drop_split_backfill_keep_sort():
    df = pd.DataFrame(
        {
            "Boost": [0.0, 0.0, 1.0, 0.0],
            "Model EV": [0.0, 0.5, 0.3, 0.9],
            "Market EV": [0.0, 0.0, 0.0, 0.0],
            "Model Param": [1.0, 2.0, 3.0, 4.0],
            "Dist": ["NegBin", "NegBin", "Gamma", "SkewNormal"],
            "Player": ["Zero", "Neg", "Gam", "Skew"],
            "Temperature": [5, 5, 5, 5],
        }
    )

    out = _normalize_offers(df)

    # No-signal row ("Zero") dropped; remainder sorted by Model descending.
    assert out["Player"].tolist() == ["Skew", "Neg", "Gam"]
    assert out["Model EV"].tolist() == [0.9, 0.5, 0.3]

    # Keep-column projection (in _OFFER_KEEP_COLS order); internal artifacts gone.
    assert list(out.columns) == [
        "Player",
        "Boost",
        "Model EV",
        "Market EV",
        "Dist",
        "Gate",
        "Model R",
        "Model Alpha",
        "Model Sigma",
        "Model Skew",
    ]
    assert "Model Param" not in out.columns
    assert "Temperature" not in out.columns

    by_player = out.set_index("Player")
    assert by_player.loc["Neg", "Model R"] == 2.0
    assert np.isnan(by_player.loc["Neg", "Model Alpha"])
    assert np.isnan(by_player.loc["Neg", "Model Sigma"])
    assert by_player.loc["Gam", "Model Alpha"] == 3.0
    assert np.isnan(by_player.loc["Gam", "Model R"])
    assert by_player.loc["Skew", "Model Sigma"] == 4.0
    assert np.isnan(by_player.loc["Skew", "Model R"])

    assert by_player["Gate"].isna().all()
    assert by_player["Model Skew"].isna().all()


def test_game_k_why_retained_in_keep_order():
    """The P2 narrative/Kelly columns survive the projection in keep-col order."""
    df = pd.DataFrame(
        {
            "Player": ["P"],
            "Game": ["AAA/BBB"],
            "Kelly": [0.42],
            "Why": ["form + edge"],
            "Boost": [1.0],
            "Model EV": [0.5],
            "Market EV": [0.0],
            "Dist": ["Gamma"],
            "Model Param": [2.0],
        }
    )

    out = _normalize_offers(df)

    assert list(out.columns) == [
        "Game",
        "Player",
        "Boost",
        "Model EV",
        "Market EV",
        "Kelly",
        "Why",
        "Dist",
        "Gate",
        "Model R",
        "Model Alpha",
        "Model Sigma",
        "Model Skew",
    ]
    assert out.loc[0, "Game"] == "AAA/BBB"
    assert out.loc[0, "Kelly"] == 0.42
    assert out.loc[0, "Why"] == "form + edge"


def test_position_label_retained_when_present():
    """The P2 depth-chart label survives the projection (build_game_context reads it)."""
    df = pd.DataFrame(
        {
            "Player": ["P"],
            "Position": ["G1"],
            "Boost": [1.0],
            "Model EV": [0.5],
            "Market EV": [0.0],
            "Dist": ["Gamma"],
            "Model Param": [2.0],
        }
    )

    out = _normalize_offers(df)

    assert "Position" in out.columns
    assert out.loc[0, "Position"] == "G1"
