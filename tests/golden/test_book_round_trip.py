"""The book-odds encode/decode round trip must not manufacture an edge.

`get_ev` encodes a bookmaker (line, under-prob) to a mean stored in the archive;
`get_odds` decodes a mean back to a probability in `book_fallback_prob`. The two
must be inverses, and the zero-inflation gate must be applied symmetrically, or a
book-only leg shows a fake edge that buries the genuinely-modeled markets.
"""

import math

import pytest

from sportstradamus.helpers import stat_cv
from sportstradamus.helpers.distributions import get_ev, get_odds
from sportstradamus.prediction.model_prob import _book_cell_params


def _decode(line, ev, dist, cv, gate):
    step = 1.0 if dist in ("NegBin", "ZINB", "Poisson") else 0.5
    sigma = ev * cv if dist in ("SkewNormal", "Normal") else None
    return get_odds(line, ev, dist, cv=cv, step=step, gate=gate, sigma=sigma)


# (line, cv, dist, gate) chosen so the implied mean stays inside the plausibility
# cap (SN_MAX_MEAN_FACTOR * line) — the regime where the round trip must be exact.
# One representative per distribution family (gate handling differs by family).
_ROUND_TRIP_CASES = [
    (16.5, 0.544, "SkewNormal", None),
    (8.5, 0.3, "Normal", None),
    (7.5, 0.5, "NegBin", None),
    (10.5, 0.8, "Gamma", None),
    (7.5, 0.5, "ZINB", 0.15),
    (10.5, 0.8, "ZAGamma", 0.2),
]


@pytest.mark.parametrize("line,cv,dist,gate", _ROUND_TRIP_CASES)
@pytest.mark.parametrize("under", [0.3, 0.7])
def test_get_ev_inverts_get_odds(line, cv, dist, gate, under):
    """`get_ev` is the numerical inverse of `get_odds` within the plausibility cap."""
    ev = get_ev(line, under, cv, dist=dist, gate=gate)
    assert _decode(line, ev, dist, cv, gate) == pytest.approx(under, abs=5e-3)


@pytest.mark.parametrize("league,market", [("WNBA", "REB")])
def test_book_fallback_skewnormal_has_no_gate(league, market):
    """SkewNormal book-fallback must not re-add a zero-inflation gate the encode never removed."""
    dist, _cv, gate, _step = _book_cell_params(league, market)
    assert dist == "SkewNormal"
    assert not gate


@pytest.mark.parametrize("league,market,line", [("WNBA", "REB", 6.5)])
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


@pytest.mark.parametrize("league,market,line", [("NBA", "STL", 0.5)])
def test_zinb_even_money_book_not_overconfident(league, market, line):
    """A high-zero-rate ZINB count at a 0.5 line must not manufacture a wildly
    overconfident Under from the round trip.

    The structural zero rate floors P(under) near the gate so it cannot reach
    exactly 0.5, but it must be far below the ~0.82 the broken round trip
    produced (`get_ev` returned the bare line, `get_odds` re-added the full gate).
    """
    dist, cv, gate, step = _book_cell_params(league, market)
    assert dist == "ZINB"
    ev = get_ev(line, 0.5, cv, dist=dist, gate=gate)
    p_under = get_odds(line, ev, dist, cv=cv, step=step, gate=gate)
    assert p_under < 0.65


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
