"""The book pull in the replay: ``tail_pricing.serve`` pulls what serving and training pull.

A cell's ``book_pull`` b serves ``b·M + (1 - b)·P`` on authentic rows, last in the
chain, and leaves pick'em rows alone. M is the book's over-probability: the training layer's
``1 - Odds`` at a test row's own line, serving's ``Market EV`` at a priced rung (``Book_EV``
decoded at the rung's line, with the cell's zero rate on ZINB only). Training pulls a test
row's ``P`` at its own line, so a pulled dump reconstructs only under the pull. A pull of
0 serves the unpulled prices bit for bit.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.training_quotes import AUTHENTIC
from sportstradamus.prediction.offer_records import book_over_prob
from sportstradamus.scripts.tail_pricing import (
    RECONSTRUCTION_TOL,
    CellExcludedError,
    _training_weight,
    load_cell,
    serve,
    serve_live,
)

# A ZINB cell whose temperature binds, with the names load_cell reads off its dump's path.
_KNOBS = {
    "distribution": "ZINB",
    "weight": 0.6,
    "cv": 0.8,
    "dispersion_cal": 1.1,
    "skew_cal": 0.0,
    "hist_gate": 0.2,
    "step": 1.0,
    "temperature": 1.3,
    "posthoc": "none",
    "posthoc_blob": None,
    "pit_recal_blob": None,
    "league": "MLB",
    "market": "stolen bases",
}
# Two authentic test rows and a pick'em one, at their own lines.
_OWN = pd.DataFrame(
    {
        "Line": [1.5, 2.5, 1.5],
        "QuoteAuthenticity": [AUTHENTIC, AUTHENTIC, "pickem"],
        "EV": [1.8, 2.6, 1.4],
        "Book_EV": [2.0, 2.3, 1.3],
        "Odds": [0.45, 0.6, 0.55],
        "R_model": [1.4, 2.2, 0.9],
        "Gate_model": [0.15, 0.05, 0.3],
    }
)
# Each test row priced a rung below and a rung above its own line.
_RUNGS = pd.concat(
    [_OWN.assign(**{"Test Line": _OWN["Line"], "Line": _OWN["Line"] + shift}) for shift in (-1, 1)],
    ignore_index=True,
)
_OWN_AUTHENTIC = _OWN["QuoteAuthenticity"].eq(AUTHENTIC).to_numpy()
_RUNG_AUTHENTIC = _RUNGS["QuoteAuthenticity"].eq(AUTHENTIC).to_numpy()


def _served_book(dist: str, gate: float | None) -> np.ndarray:
    """Serving's ``Market EV`` on the authentic rungs: ``Book_EV`` decoded at each rung's line."""
    board = _RUNGS.loc[_RUNG_AUTHENTIC, ["Line", "Book_EV"]]
    board = board.rename(columns={"Book_EV": "Market Projection"})
    return book_over_prob(
        board, dist, _KNOBS["cv"], _KNOBS["step"], gate, _KNOBS["league"], _KNOBS["market"]
    ).to_numpy()


@pytest.mark.parametrize("rows", [_OWN, _RUNGS], ids=["own line", "rungs"])
def test_zero_pull_serves_the_unpulled_prices_bit_for_bit(rows):
    none = serve_live(rows, _KNOBS | {"book_pull": 0.0})

    for got, want in zip(none, serve_live(rows, _KNOBS), strict=True):
        np.testing.assert_array_equal(got, want)


def test_full_pull_serves_the_books_price_on_authentic_rows_only():
    own = serve_live(_OWN, _KNOBS | {"book_pull": 1.0})[0]
    rungs = serve_live(_RUNGS, _KNOBS | {"book_pull": 1.0})[0]

    np.testing.assert_array_equal(own[_OWN_AUTHENTIC], 1 - _OWN["Odds"][_OWN_AUTHENTIC])
    np.testing.assert_array_equal(rungs[_RUNG_AUTHENTIC], _served_book("ZINB", _KNOBS["hist_gate"]))
    unpulled_own, unpulled_rungs = serve_live(_OWN, _KNOBS)[0], serve_live(_RUNGS, _KNOBS)[0]
    np.testing.assert_array_equal(own[~_OWN_AUTHENTIC], unpulled_own[~_OWN_AUTHENTIC])
    np.testing.assert_array_equal(rungs[~_RUNG_AUTHENTIC], unpulled_rungs[~_RUNG_AUTHENTIC])


def test_half_pull_moves_a_zinb_rung_toward_the_book_decoded_at_its_zero_rate():
    pull = 0.5
    pulled = serve_live(_RUNGS, _KNOBS | {"book_pull": pull})[0]
    unpulled = serve_live(_RUNGS, _KNOBS)[0]
    book = _served_book("ZINB", _KNOBS["hist_gate"])

    np.testing.assert_allclose(
        pulled[_RUNG_AUTHENTIC], pull * book + (1 - pull) * unpulled[_RUNG_AUTHENTIC], rtol=1e-12
    )
    np.testing.assert_array_equal(pulled[~_RUNG_AUTHENTIC], unpulled[~_RUNG_AUTHENTIC])
    # The zero rate moves every rung's decode, so a gate-free decode would not pass.
    assert not np.isclose(book, _served_book("ZINB", None)).any()


def test_a_negbin_rung_decodes_the_book_without_its_pickles_zero_rate():
    # NegBin pickles carry a zero rate (NFL tds .78) that serving never decodes the book with.
    knobs = _KNOBS | {"distribution": "NegBin", "hist_gate": 0.7, "book_pull": 1.0}

    served = serve_live(_RUNGS, knobs)[0]

    np.testing.assert_array_equal(served[_RUNG_AUTHENTIC], _served_book("NegBin", None))


def test_a_dump_training_pulled_reconstructs_only_under_the_pull(tmp_path):
    pull = 0.5
    weight = _training_weight(_OWN, _KNOBS)
    unpulled = serve(_OWN, _KNOBS, weight, _OWN["Book_EV"].to_numpy(float))[0]
    # Training pulls an authentic test row toward its own book price, 1 - Odds.
    pulled = np.where(_OWN_AUTHENTIC, pull * (1 - _OWN["Odds"]) + (1 - pull) * unpulled, unpulled)
    with (tmp_path / "MLB_stolen-bases.mdl").open("wb") as outfile:
        pickle.dump(_KNOBS | {"book_pull": pull, "model_version": "fixture"}, outfile)
    dump = _OWN.assign(
        Player="A", Date="2026-11-01", Result=1.0, P_standalone=0.5, StructuralStrategy="none"
    )
    path = tmp_path / "MLB_stolen-bases.csv"

    dump.assign(P=pulled).to_csv(path, index=False)
    _, knobs, error = load_cell(path, tmp_path)
    assert error <= RECONSTRUCTION_TOL
    assert knobs == _KNOBS | {"book_pull": pull, "model_version": "fixture", "refit_through": None}

    dump.assign(P=unpulled).to_csv(path, index=False)
    with pytest.raises(CellExcludedError, match="reconstruction"):
        load_cell(path, tmp_path)
