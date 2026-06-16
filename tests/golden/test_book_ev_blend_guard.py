"""A corrupt bookmaker EV must not inflate the live blend out of support.

Defense-in-depth behind the ``get_ev`` zero-inflation fix (see
``tests/golden/test_get_ev_robustness.py``): even when a blown book EV is already
committed in the archive — as on the 2026-06-05 FG3M slate, where ``get_ev`` had
written ~7800 to the ``odds`` table — the live decode in
``prediction/model_prob.py`` must ride the line instead of letting the
log-opinion pool ``fused_loc`` chase the garbage.

``_sanitize_book_ev`` drops any book mean beyond ``_BOOK_EV_LINE_CAP`` times its
line (model and book project the *same* stat, so a >10x gap is bad data, not an
edge) and ``_blend_with_book`` applies it before pooling.
"""

import logging

import numpy as np
import pandas as pd

from sportstradamus.prediction.model_prob import (
    _BOOK_EV_LINE_CAP,
    _blend_with_book,
    _sanitize_book_ev,
)


def test_sanitize_drops_only_implausibly_high_book_ev():
    line = np.array([2.5, 1.5, 0.5, 0.3])
    # 7802 >> 10*2.5 (corrupt); 3.0 <= 10*1.5 (kept); 0.4 below line (kept,
    # one-sided guard); 4.0 vs a 0.3 line uses the 0.5 floor -> cap 5.0 (kept).
    books = np.array([7802.0, 3.0, 0.4, 4.0])
    out = _sanitize_book_ev(books, line)
    assert out[0] == 2.5, "blown book EV must collapse to the line"
    assert out[1] == 3.0
    assert out[2] == 0.4
    assert out[3] == 4.0


def test_sanitize_passes_through_when_all_sane():
    line = np.array([2.5, 1.5])
    books = np.array([3.1, 1.2])
    out = _sanitize_book_ev(books, line)
    np.testing.assert_array_equal(out, books)


def _zinb_offer_df(books_ev):
    return pd.DataFrame(
        {
            "Projection": [2.70, 1.40, 0.90],
            "Market Projection": books_ev,
            "Line": [2.5, 1.5, 0.5],
            "Model R": [30.0, 30.0, 30.0],
            "Model Gate": [0.10, 0.20, 0.30],
        }
    )


def test_corrupt_book_ev_does_not_inflate_blend():
    # The exact failure shape: archive book EV in the thousands. The blended
    # Model EV must stay in the threes-prop support, not the 7x inflation
    # (Champagnie 2.7 -> 7.45) the unguarded pool produced live.
    offer_df = _zinb_offer_df([7802.0, 7802.0, 4268.0])
    _blend_with_book(offer_df, "ZINB", model_weight=0.85, cv=0.566, hist_gate=0.337)
    blended = offer_df["Projection"].to_numpy()
    line = offer_df["Line"].to_numpy()
    assert np.isfinite(blended).all()
    assert (blended <= _BOOK_EV_LINE_CAP * line).all()
    # Neutralized book -> blend rides the model, never the runaway.
    assert (blended < 4.0).all(), f"corrupt book still inflated the blend: {blended}"


def test_warning_reports_offender_ratio_and_cell(caplog):
    # The benign-vs-corrupt signal lives in the worst ev/line ratio + cell, not the
    # array max: a ~5x DFS clamp placeholder must read apart from a >1000x runaway.
    line = np.array([0.5, 1.5])
    books = np.array([7.5, 1.0])  # 7.5 on a 0.5 line = 15x cap -> dropped; 1.0 kept
    with caplog.at_level(logging.WARNING, logger="log"):
        out = _sanitize_book_ev(books, line, cell="WNBA FG3M")
    assert out[0] == 0.5 and out[1] == 1.0
    assert "WNBA FG3M" in caplog.text
    assert "15x" in caplog.text  # worst ev/line = 7.5 / 0.5


def test_sane_book_ev_blend_unchanged_by_guard():
    # A legitimate book EV near the line must blend normally (the guard is inert).
    sane = _zinb_offer_df([2.6, 1.5, 0.6])
    guarded = sane.copy()
    _blend_with_book(guarded, "ZINB", model_weight=0.85, cv=0.566, hist_gate=0.337)
    blended = guarded["Projection"].to_numpy()
    # Blended mean sits in the plausible band between the model and the book.
    assert (blended > 0).all()
    assert (blended < 3.5).all()
