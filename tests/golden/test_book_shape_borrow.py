"""Shape-borrow book recovery — WS2 core.

The symmetric book (skew=0) inverts the de-vigged point under a Normal, so on a
right-skewed cell it understates the book *mean* (the line is a median, not a
mean). ``_book_loc_with_skew`` gives the book a per-cell skew while **holding the
de-vigged point** (the SkewNormal CDF at the line equals the symmetric Normal CDF
at the line), so the mean shifts but the priced probability does not. ``fused_loc``
threads it in and reduces to the current blend when ``book_skew == 0``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, skewnorm

from sportstradamus.helpers.distributions import _book_loc_with_skew, fused_loc


def test_book_loc_with_skew_preserves_devig_point():
    ev_b, sigma, line, a = 20.0, 6.0, 22.0, 3.0
    loc = _book_loc_with_skew(ev_b, sigma, line, a)
    # The skewed book reproduces the symmetric book's probability AT THE LINE.
    assert skewnorm.cdf(line, a, loc, sigma) == pytest.approx(
        norm.cdf(line, ev_b, sigma), abs=1e-9
    )


def test_book_loc_with_skew_is_symmetric_mean_at_zero_skew():
    assert _book_loc_with_skew(20.0, 6.0, 22.0, 0.0) == pytest.approx(20.0, abs=1e-9)


def test_right_skew_raises_book_mean_above_the_median_line():
    ev_b = line = 20.0  # even-money: the line is the median
    sigma, a = 6.0, 3.0
    loc = _book_loc_with_skew(ev_b, sigma, line, a)
    delta = a / np.sqrt(1 + a**2)
    book_mean = loc + sigma * delta * np.sqrt(2 / np.pi)
    assert book_mean > ev_b  # right skew: mean exceeds the median


def test_fused_loc_book_skew_zero_matches_current_blend():
    args = dict(w=0.8, ev_a=21.0, ev_b=20.0, cv=0.3, dist="SkewNormal", sigma=5.0, skew_alpha=1.5)
    base = fused_loc(**args)
    with_zero = fused_loc(**args, line=20.0, book_skew=0.0)
    for got, exp in zip(with_zero, base, strict=True):
        assert got == pytest.approx(exp)


def test_fused_loc_book_skew_enters_the_blended_skew():
    args = dict(w=0.8, ev_a=21.0, ev_b=20.0, cv=0.3, dist="SkewNormal", sigma=5.0, skew_alpha=1.5)
    _, _, skew_zero, _ = fused_loc(**args, line=20.0, book_skew=0.0)
    _, _, skew_pos, _ = fused_loc(**args, line=20.0, book_skew=2.0)
    # blended skew = w*model_skew + (1-w)*book_skew, so a positive book skew lifts it.
    assert skew_pos > skew_zero
    assert skew_pos == pytest.approx(0.8 * 1.5 + 0.2 * 2.0)
