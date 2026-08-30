"""``get_ev`` must never invert a book price into a blown or crashing mean.

Two ill-conditioned bands corrupt the archive when left unguarded:

1. **ZINB/ZAGamma gate underflow** — when the de-vigged under-prob sits at or
   *just above* the zero-inflation ``gate`` (common for high-zero count cells
   like NBA ``BLK`` ~0.63 or ``FG3M`` ~0.34), the gate-stripped base CDF is
   non-positive or a hair above zero and the NegBin inversion runs away to
   hundreds-to-thousands. Only ``add_dfs`` passes a ``gate``, so this is the
   DFS-path blowup that broke the 2026-06-05 FG3M slate (a Sleeper under-prob
   just above the gate inverted to ~7800 and was written to the archive).
2. **SkewNormal mean→∞ asymptote** — the scale grows with the mean, so
   ``cdf(line, mean)`` asymptotes to ``Phi(-1/cv)`` instead of 0. An under-prob a
   hair above that floor inverts to a runaway mean (rushing-yards evs in the
   millions) and can hand ``brentq`` a same-sign bracket that raises.

Both degenerate bands must clamp to the plausibility cap (``SN_MAX_MEAN_FACTOR ×
line``) — bounded, never a runaway, and decoding back through ``get_odds`` to the
nearest achievable probability — while genuinely well-posed prices keep inverting
to a real mean.
"""

import numpy as np
import pytest

from sportstradamus.helpers.archive import Archive
from sportstradamus.helpers.distributions import (
    SN_MAX_MEAN_FACTOR,
    UNDERDOG_BOOST_BASELINE,
    get_ev,
)

# NBA BLK calibration (high zero-inflation count cell).
BLK_CV = 0.5435
BLK_GATE = 0.6258
# NBA FG3M calibration (moderate-zero count cell, gate ~0.34). The 2026-06-05
# blowup: a Sleeper DFS under-prob a hair *above* the gate stripped to a
# near-zero base CDF and inverted to ~7800, which add_dfs wrote to the archive.
FG3M_CV = 0.5660
FG3M_GATE = 0.3370
# NFL rushing-yards calibration (continuous SkewNormal cell).
RY_CV = 0.905


def test_zinb_under_at_or_below_gate_clamps_to_cap():
    # A realistic two-sided BLK o/u 1.5 (-130/+110) de-vigs to under ~0.46, below
    # the 0.63 gate: the under-prob sits beneath the gated count's floor, so no
    # finite mean reproduces it (the unguarded inversion blew this to ~5900). It
    # must clamp to the cap, which decodes back to the gate floor — not the line,
    # whose decode re-adds the full gate to an overconfident ~0.8.
    for under in (0.06, 0.30, 0.457, 0.60):
        ev = get_ev(1.5, under, BLK_CV, dist="ZINB", gate=BLK_GATE)
        assert ev == pytest.approx(SN_MAX_MEAN_FACTOR * 1.5), f"under={under} got {ev}"


def test_zinb_under_above_gate_still_inverts():
    # Above the gate the inversion is well-posed and must produce a real mean,
    # not the neutral-line fallback.
    ev = get_ev(1.5, 0.80, BLK_CV, dist="ZINB", gate=BLK_GATE)
    assert 0 < ev < 10
    assert ev != pytest.approx(1.5)


def test_zinb_under_just_above_gate_no_runaway():
    # The 2026-06-05 FG3M bug: an under-prob just *above* the gate strips to a
    # tiny-positive base CDF that the unguarded inversion blew to 90-7800 for a
    # 2.5 line. The cap must bound it — unders far enough above the gate still
    # invert to a real mean, the rest clamp to the ceiling.
    for under in (FG3M_GATE + 0.002, FG3M_GATE + 0.03, FG3M_GATE + 0.10):
        ev = get_ev(2.5, under, FG3M_CV, dist="ZINB", gate=FG3M_GATE)
        assert np.isfinite(ev) and ev <= SN_MAX_MEAN_FACTOR * 2.5, f"under={under} got {ev}"


def test_zinb_well_above_gate_fg3m_still_inverts():
    # Far enough above the gate the base CDF is large enough to invert sanely; the
    # fallback must not over-trigger and erase every FG3M book signal.
    ev = get_ev(2.5, 0.70, FG3M_CV, dist="ZINB", gate=FG3M_GATE)
    assert 0 < ev < 5
    assert ev != pytest.approx(2.5)


def test_zinb_no_runaway_across_under_grid():
    # Sweep the whole under range for a moderate-gate count cell: never exceed a
    # sane multiple of the line (the unguarded inversion hit ~7800 for a 2.5 line).
    for under in np.linspace(0.001, 0.999, 500):
        ev = get_ev(2.5, float(under), FG3M_CV, dist="ZINB", gate=FG3M_GATE)
        assert np.isfinite(ev) and ev <= 5 * 2.5, f"under={under} produced {ev}"


def test_skewnormal_asymptote_band_clamps_to_cap():
    from scipy.stats import norm

    asymptote = norm.cdf(-1.0 / RY_CV)
    # under near the mean→∞ asymptote inverted to ~1.9e6 before the guard; it must
    # clamp to the plausibility cap instead.
    for under in (asymptote - 1e-3, asymptote + 1e-4, asymptote + 5e-3):
        ev = get_ev(40.5, float(np.clip(under, 1e-6, 1 - 1e-6)), RY_CV, dist="SkewNormal")
        assert ev == pytest.approx(SN_MAX_MEAN_FACTOR * 40.5), f"under={under} got {ev}"


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
        self.under_probs = []
        self.rungs = []

    def _stage_book_ev(
        self, league, market, date, entity, book, ev, observed_at=None, under_prob=None, line=None
    ):
        self.evs.append(ev)
        self.under_probs.append(under_prob)

    def _stage_line(self, *args, **kwargs):
        pass

    def add_ladder(self, league, market, date, entity, book, rungs, observed_at=None):
        self.rungs.extend(rungs)


def test_add_dfs_boost_only_offer_prices_symmetric_never_blown_or_clamped():
    """A ``Boost``-only DFS pick prices as a symmetric standard pick, never a blown mean.

    A Rivals/H2H-style pick carries a single both-ways Boost and no per-side
    multipliers, so its payout-implied quote is the even 50/50 (the boost value
    cancels in a symmetric devig). BLK's high zero rate floors that price below
    the gate, so the inversion has no solution and lands on ``get_ev``'s ceiling
    — which must be stored as NULL, never a bound dressed as a mean.
    """
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
    assert len(cap.evs) == 1
    assert cap.under_probs == [pytest.approx(0.5)]
    assert cap.evs[0] is None or 0 < cap.evs[0] < SN_MAX_MEAN_FACTOR * 1.5, (
        f"stored a clamped or blown ev as a mean: {cap.evs}"
    )


def test_add_dfs_one_sided_underdog_offer_converts_through_baseline():
    """An Underdog one-sided boost is a payout *modifier*, not decimal odds.

    The raw 1.67 modifier scales the standard 1.78 slot payout to decimal
    2.97, so the stored under-probability is its payout-implied complement
    ``1 - 1/2.97`` — not a fabricated symmetric 0.5, and not the raw
    modifier's breakeven.
    """
    offer = {
        "Player": "Nikola Jokic",
        "League": "NBA",
        "Market": "PTS",
        "Line": 25.5,
        "Date": "2025-12-23",
        "Boost_Over": 1.67,
        "Boost_Under": 0.0,
    }
    cap = _CaptureArchive()
    Archive.add_dfs(cap, [offer], "Underdog", {})
    expected_under = 1 - 1 / (1.67 * UNDERDOG_BOOST_BASELINE)
    assert cap.under_probs == [pytest.approx(expected_under)]
    assert cap.rungs == [(25.5, pytest.approx(1 - expected_under))]


# --- Gate-refuted quote persistence (the 2026-06+ NFL tds poisoning) ---------
#
# A sharp quote whose under-prob sits at or below the calibrated gate has no
# implied mean (population zi vs quoted-star selection). Both writers must store a
# NULL ev with the shape-free quote — never the clamp ceiling as a real mean.


def test_player_prop_gate_refuted_quote_stores_null_ev(monkeypatch):
    from sportstradamus import moneylines

    monkeypatch.setitem(moneylines.stat_dist.setdefault("NFL", {}), "tds", "ZINB")
    monkeypatch.setitem(moneylines.stat_cv.setdefault("NFL", {}), "tds", 0.35)
    monkeypatch.setattr(moneylines, "book_gate", lambda lg, mkt, dist: 0.777)

    market = {
        "key": "player_tds",
        "outcomes": [
            {"description": "Star Scorer", "name": "Over", "price": 1.52, "point": 0.5},
            {"description": "Star Scorer", "name": "Under", "price": 2.45, "point": 0.5},
        ],
    }
    odds = {}
    moneylines._event_player_props(
        {"key": "fanduel"}, market, "NFL", {"NFL": {"player_tds": "tds"}}, odds
    )
    entry = odds["tds"]["Star Scorer"]
    assert entry["EV"]["fanduel"] is None
    line, under = entry["Quotes"]["fanduel"]
    assert line == 0.5
    assert 0 < under < 0.777


def _tds_prop_ev(monkeypatch, dist, gate, over, under):
    """Archive one NFL tds quote through the parser and return the ev it stored."""
    from sportstradamus import moneylines

    monkeypatch.setitem(moneylines.stat_dist.setdefault("NFL", {}), "tds", dist)
    monkeypatch.setitem(moneylines.stat_cv.setdefault("NFL", {}), "tds", 0.35)
    monkeypatch.setattr(moneylines, "book_gate", lambda lg, mkt, d: gate)

    market = {
        "key": "player_tds",
        "outcomes": [
            {"description": "Star Scorer", "name": "Over", "price": over, "point": 0.5},
            {"description": "Star Scorer", "name": "Under", "price": under, "point": 0.5},
        ],
    }
    odds = {}
    moneylines._event_player_props(
        {"key": "fanduel"}, market, "NFL", {"NFL": {"player_tds": "tds"}}, odds
    )
    return odds["tds"]["Star Scorer"]["EV"]["fanduel"]


def test_player_prop_that_inverts_stores_the_solved_mean(monkeypatch):
    # De-vigged under ~0.49 — well above the 0.30 gate and inside the band ZINB can
    # reproduce at a mean below the cap, so brentq lands on a real solution.
    ev = _tds_prop_ev(monkeypatch, "ZINB", 0.30, over=1.90, under=2.00)
    assert ev is not None and np.isfinite(ev)
    assert 0 < ev < SN_MAX_MEAN_FACTOR * 0.5, "a solved mean must sit strictly under the cap"


def test_player_prop_clamped_by_an_ungated_family_stores_null_ev(monkeypatch):
    # A heavy over favourite (-1000) leaves a de-vigged under of ~0.12, below what an
    # ungated NegBin reaches at any mean up to the cap, so get_ev returns its ceiling.
    # The ceiling is a bound, not a mean, and must not be persisted just because
    # zero-inflation was not what made the inversion unsolvable.
    ev = _tds_prop_ev(monkeypatch, "NegBin", None, over=1.10, under=8.00)
    assert ev is None


class _StageCapture:
    def __init__(self):
        self.staged = []

    def _stage_book_ev(
        self, league, market, date, entity, book, ev, observed_at=None, under_prob=None, line=None
    ):
        self.staged.append((book, ev, under_prob))

    def _stage_line(self, *args, **kwargs):
        pass


def test_merge_player_books_stages_null_ev_with_quote():
    cap = _StageCapture()
    Archive.merge_player_books(
        cap,
        "NFL",
        "tds",
        "2026-09-13",
        "Star Scorer",
        {"fanduel": None, "draftkings": 0.9, "espn": None},
        book_quotes={"fanduel": (0.5, 0.383), "draftkings": (0.5, 0.82)},
    )
    books = {b: (ev, up) for b, ev, up in cap.staged}
    assert books["fanduel"] == (None, 0.383)
    assert books["draftkings"] == (0.9, 0.82)
    assert "espn" not in books


def test_stage_book_ev_none_stores_null():
    import datetime as _dt

    class _Buf:
        pass

    buf = _Buf()
    buf._pending_odds = []
    Archive._stage_book_ev(
        buf,
        "NFL",
        "tds",
        _dt.date(2026, 9, 13),
        "Star Scorer",
        "fanduel",
        None,
        under_prob=0.383,
        line=0.5,
    )
    (row,) = buf._pending_odds
    assert row[5] is None
    assert row[7] == pytest.approx(0.383)
    assert row[8] == pytest.approx(0.5)
