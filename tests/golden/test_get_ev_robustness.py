"""``get_ev`` must never invert a book price into a blown or crashing mean.

Two ill-conditioned bands corrupt the archive when left unguarded:

1. **ZINB/ZAGamma gate underflow** — when the de-vigged under-prob sits at or
   below the zero-inflation ``gate`` (common for high-zero count cells like NBA
   ``BLK`` where ``gate`` ~0.63), the gate-stripped base CDF goes non-positive
   and the NegBin inversion runs away to thousands. Only ``add_dfs`` passes a
   ``gate``, so this is the DFS-path blowup.
2. **SkewNormal mean→∞ asymptote** — the scale grows with the mean, so
   ``cdf(line, mean)`` asymptotes to ``Phi(-1/cv)`` instead of 0. An under-prob a
   hair above that floor inverts to a runaway mean (rushing-yards evs in the
   millions) and can hand ``brentq`` a same-sign bracket that raises.

Both degenerate bands must collapse to the neutral line, while genuinely
well-posed prices keep inverting to a real mean.
"""

import numpy as np
import pytest

from sportstradamus.helpers.archive import Archive
from sportstradamus.helpers.distributions import get_ev

# NBA BLK calibration (high zero-inflation count cell).
BLK_CV = 0.5435
BLK_GATE = 0.6258
# NFL rushing-yards calibration (continuous SkewNormal cell).
RY_CV = 0.905


def test_zinb_under_at_or_below_gate_returns_line():
    # A realistic two-sided BLK o/u 1.5 (-130/+110) de-vigs to under ~0.46, which
    # is below the 0.63 gate: the unguarded inversion blew this to ~5900.
    for under in (0.06, 0.30, 0.457, 0.60):
        ev = get_ev(1.5, under, BLK_CV, dist="ZINB", gate=BLK_GATE)
        assert ev == pytest.approx(1.5), f"under={under} should fall back to the line, got {ev}"


def test_zinb_under_above_gate_still_inverts():
    # Above the gate the inversion is well-posed and must produce a real mean,
    # not the neutral-line fallback.
    ev = get_ev(1.5, 0.80, BLK_CV, dist="ZINB", gate=BLK_GATE)
    assert 0 < ev < 10
    assert ev != pytest.approx(1.5)


def test_skewnormal_asymptote_band_returns_line():
    from scipy.stats import norm

    asymptote = norm.cdf(-1.0 / RY_CV)
    # under just above the asymptote inverted to ~1.9e6 before the guard.
    for under in (asymptote - 1e-3, asymptote + 1e-4, asymptote + 5e-3):
        ev = get_ev(40.5, float(np.clip(under, 1e-6, 1 - 1e-6)), RY_CV, dist="SkewNormal")
        assert ev == pytest.approx(40.5), f"under={under} near asymptote should return the line, got {ev}"


def test_skewnormal_no_crash_or_blowup_across_under_grid():
    # Sweep the whole under range incl. the asymptote band: never raise, never
    # exceed a sane multiple of the line.
    for under in np.linspace(0.001, 0.999, 500):
        ev = get_ev(40.5, float(under), RY_CV, dist="SkewNormal")
        assert np.isfinite(ev) and ev <= 5 * 40.5, f"under={under} produced {ev}"


def test_skewnormal_well_posed_inverts_near_line():
    # A balanced book (under=0.5, symmetric) implies a mean at the line.
    ev = get_ev(40.5, 0.5, RY_CV, dist="SkewNormal")
    assert 30 < ev < 55


class _CaptureArchive:
    """Duck-typed ``self`` exposing only what ``add_dfs`` touches (no DuckDB)."""

    def __init__(self):
        self.evs = []

    def _stage_book_ev(self, league, market, date, entity, book, ev, observed_at=None):
        self.evs.append(ev)

    def _stage_line(self, *args, **kwargs):
        pass


def test_add_dfs_missing_under_side_is_symmetric_not_blown():
    # A Rivals/H2H-style DFS pick carries a single Boost and no Boost_Under; the
    # one-sided no_vig fabrication used to drive the count-cell ev into the
    # thousands. It must price as a symmetric ~even pick instead.
    offer = {
        "Player": "Nikola Jokic",
        "League": "NBA",
        "Market": "BLK",
        "Line": 1.5,
        "Date": "2025-12-23",
        "Boost": 1,
    }
    cap = _CaptureArchive()
    Archive.add_dfs(cap, [offer], "Underdog", {})
    assert cap.evs and cap.evs[0] < 5, f"missing under side blew the ev: {cap.evs}"
