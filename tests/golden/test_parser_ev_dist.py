"""The props parser must invert each book price under the market's own
distribution and zero-inflation gate, not the ``get_ev`` SkewNormal default.

``stat_dist`` routes a count market (blocks/steals/tds) to ZINB/NegBin and
``book_gate`` supplies its zero-inflation gate; the parser threads both into
``get_ev`` so the archived book mean is the exact inverse of the gated
``get_odds`` call the book-fallback decode (``_book_cell_params``) reads it back
with. Whenever ``get_ev`` returns its ``SN_MAX_MEAN_FACTOR × line`` ceiling, no mean
reproduced the price, so the parser stores a NULL ev beside the shape-free quote
(the 2026-06 NFL tds poisoning fix) rather than persisting the bound. A gate
refuted by a sharp under-prob is one way to land there; an ungated family too
tight for a long price is another. A quote the family *can* reproduce still
inverts under it, landing on a different mean than the SkewNormal default would —
the routing this guards. Regression guard for the moneylines.py parser dist+gate
wiring.
"""

import datetime

import pytest

from sportstradamus.helpers import book_gate, stat_cv, stat_dist
from sportstradamus.helpers.distributions import SN_MAX_MEAN_FACTOR, get_ev, no_vig_odds
from sportstradamus.moneylines import _archive_event_props


class _CaptureArchive:
    def __init__(self):
        self.book_evs = {}

    def merge_player_books(
        self, league, market, date, player, book_evs, lines, observed_at=None, book_quotes=None
    ):
        self.book_evs[(market, player)] = dict(book_evs)

    def set_team_books(self, *args, **kwargs):
        pass

    def add_ladder(self, *args, **kwargs):
        pass


def test_low_count_prop_ev_inverts_under_market_dist_and_gate():
    # A center priced ~89% to clear 0.5 blocks: over short (1.05), under long
    # (9.0) → de-vigged under ~0.10, below NBA BLK's ~0.63 zero-inflation gate.
    # The parser must invert under the market's ZINB *and* book_gate; that exact
    # call is the inverse of the gated get_odds the fallback decode reads it with,
    # and it lands on the cap (2.5) here, distinct from the SkewNormal default's
    # ~2.02. A sub-band quote clamps to the cap rather than running away.
    game = {
        "commence_time": "2025-12-23T23:00:00Z",
        "home_team": "Denver Nuggets",
        "away_team": "Phoenix Suns",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_blocks",
                        "outcomes": [
                            {
                                "description": "Nikola Jokic",
                                "name": "Over",
                                "price": 1.05,
                                "point": 0.5,
                            },
                            {
                                "description": "Nikola Jokic",
                                "name": "Under",
                                "price": 9.0,
                                "point": 0.5,
                            },
                        ],
                    }
                ],
            }
        ],
    }
    archive = _CaptureArchive()
    _archive_event_props(
        archive, game, "NBA", {"NBA": {"player_blocks": "BLK"}}, datetime.date(2025, 12, 23)
    )

    ev = archive.book_evs[("BLK", "Nikola Jokic")]["draftkings"]
    dist = stat_dist["NBA"]["BLK"]
    under = no_vig_odds(1.05, 9.0)[1]
    gate = book_gate("NBA", "BLK", dist)
    # The de-vigged under (~0.10) sits below BLK's zero-inflation gate: the quote
    # refutes the population zero rate, so the parser must store a NULL ev (the
    # shape-free quote carries the price) — proof the gate was consulted. The
    # ungated SkewNormal default would have inverted to a real mean here.
    if not gate:
        pytest.skip("NBA BLK carries no zero-inflation gate in this calibration (CI stub)")
    assert under <= gate, "fixture must sit below the live gate"
    assert ev is None, f"gate-refuted quote must store NULL ev, got {ev}"


def test_count_prop_above_gate_inverts_under_market_dist():
    # Same market, priced the other way: over long (2.8), under short (1.45) →
    # de-vigged under ~0.66, above BLK's gate. The parser must invert under the
    # market's ZINB + gate — the exact inverse of the gated get_odds decode —
    # and land on a different mean than the SkewNormal default would.
    game = {
        "commence_time": "2025-12-23T23:00:00Z",
        "home_team": "Denver Nuggets",
        "away_team": "Phoenix Suns",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_blocks",
                        "outcomes": [
                            {
                                "description": "Nikola Jokic",
                                "name": "Over",
                                "price": 2.8,
                                "point": 0.5,
                            },
                            {
                                "description": "Nikola Jokic",
                                "name": "Under",
                                "price": 1.45,
                                "point": 0.5,
                            },
                        ],
                    }
                ],
            }
        ],
    }
    archive = _CaptureArchive()
    _archive_event_props(
        archive, game, "NBA", {"NBA": {"player_blocks": "BLK"}}, datetime.date(2025, 12, 23)
    )

    ev = archive.book_evs[("BLK", "Nikola Jokic")]["draftkings"]
    cv = stat_cv["NBA"]["BLK"]
    dist = stat_dist["NBA"]["BLK"]
    under = no_vig_odds(2.8, 1.45)[1]
    gate = book_gate("NBA", "BLK", dist)
    assert gate is None or under > gate, "fixture must sit above the live gate"
    # Expectation is derived from live calibration rather than pinned, because stat_cv/
    # stat_zi are recomputed by every meditate: what must hold is that the parser routes
    # through the market's own family and drops the ceiling, whichever side of the cap
    # this cell's current calibration puts the quote on.
    expected_ev = get_ev(0.5, under, cv, dist=dist, gate=gate)
    if expected_ev >= SN_MAX_MEAN_FACTOR * 0.5:
        assert ev is None, f"a clamped inversion is a bound, not a mean: {ev}"
    else:
        assert ev == pytest.approx(expected_ev), (
            f"parser must invert under {dist}+book_gate, got ev={ev:.4f} vs {expected_ev:.4f} "
            "(the SkewNormal default diverges here)"
        )
