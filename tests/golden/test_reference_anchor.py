"""The reference anchor: a line off its reference serves the price there, moved by the model.

Temperature and the probability-stage post-hoc are fit at each row's own line, and at a far
alternate rung they flatten what the model knows. ``anchor_to_reference`` keeps the served
price at the reference line and moves the rest of the ladder by the raw model's logit.
``tail_pricing.serve_live`` anchors a priced rung where serving anchors an offer: at an
authentic test row's own line. A row at its reference line keeps its price bit for bit, even
a saturated one the anchor would clip. ``test_feature_log`` pins the serving path end to end
against the replay.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import anchor_to_reference, apply_temperature, get_odds
from sportstradamus.helpers.distributions import REFERENCE_ANCHOR_CLIP
from sportstradamus.helpers.training_quotes import AUTHENTIC
from sportstradamus.scripts.tail_pricing import serve, serve_live

# The package __init__ re-exports the model_prob *function*, shadowing the submodule.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

# A step isotonic post-hoc: above 0.52 it serves (all but) 1, below 0.48 (all but) 0.
_SATURATING_ISOTONIC = {"kind": "isotonic", "x": [0.0, 0.48, 0.52, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}
# A SkewNormal cell whose temperature and isotonic post-hoc both bind.
_KNOBS = {
    "distribution": "SkewNormal",
    "weight": 0.7,
    "cv": 0.5,
    "dispersion_cal": 1.2,
    "skew_cal": 0.3,
    "hist_gate": 0.0,
    "step": 1.0,
    "temperature": 4.0,
    "posthoc": "prob_recal_isotonic",
    "posthoc_blob": _SATURATING_ISOTONIC,
    "pit_recal_blob": None,
}
# An authentic and a pick'em test row (own lines 3.5 and 4.5), each priced at three rungs.
_RUNGS = pd.DataFrame(
    {
        "Test Line": [3.5] * 3 + [4.5] * 3,
        "Line": [1.5, 3.5, 5.5, 2.5, 4.5, 6.5],
        "QuoteAuthenticity": [AUTHENTIC] * 3 + ["pickem"] * 3,
        "EV": [5.0] * 3 + [4.0] * 3,
        "Book_EV": [4.6] * 3 + [3.5] * 3,
        "SN_Sigma_model": [2.0] * 3 + [1.8] * 3,
        "SN_Alpha_model": [0.5] * 3 + [-0.3] * 3,
    }
)


def _unanchored(rows: pd.DataFrame, knobs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``serve`` under the serving convention with no reference line: the chain before it."""
    book = rows["Book_EV"].where(rows["QuoteAuthenticity"].eq(AUTHENTIC), rows["EV"])
    return serve(rows, knobs, np.full(len(rows), knobs["weight"]), book.to_numpy(float))


def test_at_its_reference_a_row_serves_the_clipped_reference_price():
    # A raw 0 or 1 stays finite through the logit; a served 0 or 1 is clipped.
    raw = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 0.4])
    served = np.array([0.0, 0.003, 0.3, 0.7, 0.995, 1.0])

    anchored = anchor_to_reference(raw, raw, served)

    clipped = np.clip(served, REFERENCE_ANCHOR_CLIP, 1 - REFERENCE_ANCHOR_CLIP)
    np.testing.assert_allclose(anchored, clipped, rtol=1e-12)


def test_the_anchored_ladder_falls_with_the_line_and_keeps_the_models_spread():
    lines = np.arange(0.5, 7.0, 1.0)
    raw = 1 - get_odds(lines, 2.4, "NegBin", r=3.0)
    main = 2
    # A temperature at its cap of 10 all but flattens the ladder it is applied across.
    tempered = apply_temperature(raw, 10.0)

    anchored = anchor_to_reference(raw, raw[main], tempered[main])

    assert (np.diff(anchored) < 0).all()
    assert anchored[main] == pytest.approx(tempered[main])
    assert anchored[0] > tempered[0]
    assert anchored[-1] < tempered[-1]


def test_serve_live_leaves_own_line_rows_unanchored():
    rows = _RUNGS.drop(columns="Test Line")

    for got, want in zip(serve_live(rows, _KNOBS), _unanchored(rows, _KNOBS), strict=True):
        np.testing.assert_array_equal(got, want)


def test_serve_live_anchors_only_an_authentic_rung_off_its_test_line():
    anchored, mean, sd = serve_live(_RUNGS, _KNOBS)
    today, today_mean, today_sd = _unanchored(_RUNGS, _KNOBS)
    authentic = _RUNGS["QuoteAuthenticity"].eq(AUTHENTIC)
    moved = (authentic & _RUNGS["Line"].ne(_RUNGS["Test Line"])).to_numpy()

    # The main rung, served at the isotonic's 1, and every pick'em rung keep today's price;
    # no mean or SD moves.
    assert today[1] > 1 - REFERENCE_ANCHOR_CLIP
    np.testing.assert_array_equal(anchored[~moved], today[~moved])
    np.testing.assert_array_equal(mean, today_mean)
    np.testing.assert_array_equal(sd, today_sd)
    raw_knobs = _KNOBS | {"temperature": None, "posthoc": "none"}
    at_test_line = _RUNGS.assign(Line=_RUNGS["Test Line"])
    expected = anchor_to_reference(
        _unanchored(_RUNGS, raw_knobs)[0],
        _unanchored(at_test_line, raw_knobs)[0],
        _unanchored(at_test_line, _KNOBS)[0],
    )
    np.testing.assert_allclose(anchored[moved], expected[moved], rtol=1e-12)
    assert not np.isclose(anchored[moved], today[moved]).any()


def test_serving_keeps_a_saturated_price_at_the_quote_line():
    """At its quote line an offer keeps the isotonic's 1; off it, the ladder anchors at 0.99."""
    offers = pd.DataFrame(
        {
            "Line": [1.5, 4.5, 4.5],
            "Quote Line": [1.5, 1.5, np.nan],
            "Quote Authenticity": [AUTHENTIC, AUTHENTIC, None],
            "Model R": 3.0,
        }
    )
    cell = {
        "dist": "NegBin",
        "cv": 1.0,
        "step": 1,
        "base_mean": np.full(3, 4.0),
        "temperature": 3.0,
        "pit_recal_blob": None,
        "posthoc_slug": "prob_recal_isotonic",
        "posthoc_blob": _SATURATING_ISOTONIC,
    }

    served, _ = mp._served_over_and_push(offers, **cell)
    today, _ = mp._served_over_and_push(offers.assign(**{"Quote Line": np.nan}), **cell)

    assert today[0] > 1 - REFERENCE_ANCHOR_CLIP
    np.testing.assert_array_equal(served[[0, 2]], today[[0, 2]])
    raw = 1 - get_odds(offers["Line"].to_numpy(), 4.0, "NegBin", r=3.0)
    assert served[1] == pytest.approx(anchor_to_reference(raw[1], raw[0], today[0]), rel=1e-12)
    assert served[1] < 1 - REFERENCE_ANCHOR_CLIP
