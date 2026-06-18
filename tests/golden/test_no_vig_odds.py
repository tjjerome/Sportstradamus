"""Power (logarithmic) de-vig — WS3.

``no_vig_odds(method="power")`` corrects favourite-longshot bias on lopsided
quotes (Clarke, Kovalchik & Ingram 2017) by raising each raw implied probability
to a common exponent ``k`` solving ``o**k + u**k == 1``. It is gated to lopsided
quotes (``|p_over - 0.5| > 0.3``) and is identical to the proportional default on
the body, so the two methods agree at even money.
"""

from __future__ import annotations

import numpy as np
import pytest

from sportstradamus.helpers.distributions import no_vig_odds


def test_power_devig_matches_proportional_at_even_money():
    # Two-sided callers pass decimal odds (a negative American ``under`` trips the
    # one-sided branch); symmetric decimal -> 0.5 on both methods.
    assert no_vig_odds(1.909, 1.909, method="power") == pytest.approx([0.5, 0.5], abs=1e-9)


def test_power_devig_solves_common_exponent_on_lopsided_quote():
    over_dec, under_dec = 1.2, 5.0  # raw implied 0.8333 / 0.2 -> |p-0.5| > 0.3
    p_over, p_under = no_vig_odds(over_dec, under_dec, method="power")

    assert p_over + p_under == pytest.approx(1.0, abs=1e-9)
    o_raw, u_raw = 1 / over_dec, 1 / under_dec
    k_over = np.log(p_over) / np.log(o_raw)
    k_under = np.log(p_under) / np.log(u_raw)
    assert k_over == pytest.approx(k_under, abs=1e-6)  # one common power
    # differs from proportional on a lopsided quote
    assert p_over != pytest.approx(o_raw / (o_raw + u_raw), abs=1e-3)


def test_power_devig_leaves_the_body_proportional():
    over_dec, under_dec = 1.8, 2.1  # |p_over - 0.5| < 0.3 -> not flagged
    assert no_vig_odds(over_dec, under_dec, method="power") == pytest.approx(
        no_vig_odds(over_dec, under_dec), abs=1e-12
    )
