"""The book-odds encode/decode round trip must not manufacture an edge.

`get_ev` encodes a bookmaker (line, under-prob) to a mean stored in the archive;
`get_odds` decodes a mean back to a probability on the model path's book leg
(`_book_over_prob`; `book_fallback_prob` runs its own shape-consistent
invert+decode pair off `resolve_training_quote` cohorts). Encode and decode must
be inverses, and the zero-inflation gate must be applied symmetrically, or a
book-only leg shows a fake edge that buries the genuinely-modeled markets.
"""

import math

import pytest

from sportstradamus.helpers import stat_cv, stat_dist
from sportstradamus.helpers.distributions import get_ev, get_odds
from sportstradamus.prediction.model_prob import _book_cell_params

# First three SkewNormal cells in the live config: discovered rather than named so a
# strategy sweep that reships a cell under a new family retargets these pins instead
# of rotting them (the deliberately-named ZINB cells below document the opposite call).
_SN_CELLS = [
    (lg, mkt)
    for lg in sorted(stat_dist)
    for mkt in sorted(stat_dist[lg])
    if stat_dist[lg][mkt] == "SkewNormal"
][:3]


def _decode(line, ev, dist, cv, gate):
    step = 1.0 if dist in ("NegBin", "ZINB", "Poisson") else 0.5
    sigma = ev * cv if dist in ("SkewNormal", "Normal") else None
    return get_odds(line, ev, dist, cv=cv, step=step, gate=gate, sigma=sigma)


# (line, cv, dist, gate) chosen so the implied mean stays inside the plausibility
# cap (SN_MAX_MEAN_FACTOR * line) — the regime where the round trip must be exact.
_ROUND_TRIP_CASES = [
    (16.5, 0.544, "SkewNormal", None),
    (8.5, 0.3, "Normal", None),
    (6.5, 0.55, "SkewNormal", None),
    (3.5, 0.555, "SkewNormal", None),
    (24.5, 0.5, "SkewNormal", None),
    (7.5, 0.5, "NegBin", None),
    (10.5, 0.8, "Gamma", None),
    (7.5, 0.5, "ZINB", 0.15),
    (10.5, 0.8, "ZAGamma", 0.2),
]


@pytest.mark.parametrize("line,cv,dist,gate", _ROUND_TRIP_CASES)
@pytest.mark.parametrize("under", [0.3, 0.5, 0.7])
def test_get_ev_inverts_get_odds(line, cv, dist, gate, under):
    """`get_ev` is the numerical inverse of `get_odds` within the plausibility cap."""
    ev = get_ev(line, under, cv, dist=dist, gate=gate)
    assert _decode(line, ev, dist, cv, gate) == pytest.approx(under, abs=5e-3)


@pytest.mark.parametrize("league,market", _SN_CELLS)
def test_book_fallback_skewnormal_has_no_gate(league, market):
    """SkewNormal book-fallback must not re-add a zero-inflation gate the encode never removed."""
    dist, _cv, gate, _step = _book_cell_params(league, market)
    assert dist == "SkewNormal"
    assert not gate


@pytest.mark.parametrize(
    "league,market,line",
    [(lg, mkt, line) for (lg, mkt), line in zip(_SN_CELLS, (6.5, 16.5, 3.5), strict=False)],
)
def test_skewnormal_even_money_book_is_neutral(league, market, line):
    """An even-money SkewNormal book price decodes to ~0.5.

    The archive encodes SkewNormal book EV without a gate (`add_dfs` /
    `moneylines` gate ZINB/ZAGamma only), so the book-fallback decode must not
    apply one either; otherwise the gate mass inflates P(under).
    """
    dist, cv, decode_gate, step = _book_cell_params(league, market)
    ev = get_ev(line, 0.5, cv, dist=dist, gate=None)  # encode side: SkewNormal is ungated
    p_under = get_odds(line, ev, dist, cv=cv, step=step, sigma=ev * cv, gate=decode_gate)
    assert p_under == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("league,market,line", [("WNBA", "FTM", 0.5), ("NBA", "OREB", 0.5)])
def test_zinb_even_money_book_not_overconfident(league, market, line):
    """A high-zero-rate ZINB count at a 0.5 line must not manufacture a wildly
    overconfident Under from the round trip.

    The structural zero rate floors P(under) near the gate so it cannot reach
    exactly 0.5, but it must be far below the ~0.82 the broken round trip
    produced (`get_ev` returned the bare line, `get_odds` re-added the full gate).

    The cells are named rather than discovered, so a sweep that flips one to
    another count family (NBA FTM → NegBin, WNBA FG3M → DPO both did) fails the
    `dist` assertion below. Repoint at a ZINB cell whose zero rate is near 0.45:
    a correct round trip returns ~`max(0.5, zero_rate)`, so above ~0.6 the floor
    alone clears 0.65 and the bar stops testing the double-gate it was written for.
    """
    dist, cv, gate, step = _book_cell_params(league, market)
    assert dist == "ZINB"
    ev = get_ev(line, 0.5, cv, dist=dist, gate=gate)
    p_under = get_odds(line, ev, dist, cv=cv, step=step, gate=gate)
    assert p_under < 0.65


@pytest.mark.parametrize("line", [1.5, 2.5])
def test_dpo_even_money_book_not_overconfident(line):
    """An even-money DPO book price must round-trip back to ~0.5.

    WNBA FG3M left the named-ZINB guard above when 47805e38 reshipped it as
    DPO, dropping the cell's round-trip coverage; this pin restores it under
    the new family. DPO is ungated (`book_gate` covers ZINB/ZAGamma only), so
    unlike the ZINB floor there is no structural-zero mass and the decode must
    land on 0.5 itself. Named, not discovered: if the cell is ever re-familied
    again, the `dist` assertion fails loudly and this guard moves with it.
    """
    dist, cv, gate, step = _book_cell_params("WNBA", "FG3M")
    assert dist == "DPO"
    assert not gate
    ev = get_ev(line, 0.5, cv, dist=dist, gate=gate)
    p_under = get_odds(line, ev, dist, cv=cv, step=step, gate=gate)
    assert p_under == pytest.approx(0.5, abs=0.02)


# Realistic game-line points: NBA/NFL/MLB totals and a spread. The even-money
# guardrail must hold regardless of cv (a Normal's median equals its mean).
_GAME_LINES = [229.5, 44.5, 8.5, -6.5, 2.5]


@pytest.mark.parametrize("line", _GAME_LINES)
@pytest.mark.parametrize("cv", [0.3, 1.0, 2.0])
def test_normal_even_money_price_returns_the_line(line, cv):
    """Game-line guardrail: a no-vig (0.5) price under ``dist="Normal"`` implies
    a mean equal to the line.

    ``moneylines._GAME_LINE_DIST`` pins totals/spreads to ``"Normal"`` precisely
    so this holds. A Normal's median equals its mean, so the implied value of an
    even-money line is the line itself for any cv — no encode-side inflation.
    """
    assert get_ev(line, 0.5, cv, dist="Normal") == pytest.approx(line, abs=0.6)


@pytest.mark.parametrize("line", [229.5, 44.5, 8.5])
def test_gamma_even_money_inflates_line_so_normal_pin_matters(line):
    """Why game lines must pin ``"Normal"``, not ride ``get_ev``'s default.

    A 2026-03 flip of the default to ``"Gamma"`` silently inflated every archived
    NBA total: Gamma at cv=1 is exponential, whose mean is ``median / ln(2) ≈
    1.4427 × median``. This asserts the regression family really does diverge from
    the line at an even-money price (so the Normal pin is load-bearing, not
    cosmetic), while ``"Normal"`` stays on the line.
    """
    gamma_ev = get_ev(line, 0.5, 1.0, dist="Gamma")
    assert gamma_ev == pytest.approx(line / math.log(2), rel=0.05)
    assert gamma_ev > line * 1.4  # materially above the true line
    assert get_ev(line, 0.5, 1.0, dist="Normal") == pytest.approx(line, abs=0.6)


def test_stat_cv_present_for_reported_cells():
    """Guard: the reported cells carry calibration the round trip reads."""
    for league, market in [("NBA", "STL"), ("WNBA", "FG3M"), ("WNBA", "REB")]:
        assert stat_cv[league][market] > 0
