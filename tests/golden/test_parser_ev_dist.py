"""The props parser must invert each book price under the market's own
distribution and zero-inflation gate, not the ``get_ev`` SkewNormal default.

``stat_dist`` routes a count market (blocks/steals/tds) to ZINB/NegBin and
``book_gate`` supplies its zero-inflation gate; the parser threads both into
``get_ev`` so the archived book mean is the exact inverse of the gated
``get_odds`` call the book-fallback decode (``_book_cell_params``) reads it back
with. A center priced to clear a low line implies an under far below the count's
representable band, so the inversion clamps to the plausibility cap
(``SN_MAX_MEAN_FACTOR × line``) — bounded, never the old SkewNormal-asymptote
runaway — and lands on a different mean than the SkewNormal default would, which
is the routing this guards. Regression guard for the moneylines.py parser
dist+gate wiring.
"""

import datetime

import pytest

from sportstradamus.helpers import book_gate, stat_cv, stat_dist
from sportstradamus.helpers.distributions import SN_MAX_MEAN_FACTOR, get_ev, no_vig_odds
from sportstradamus.moneylines import _archive_event_props


class _CaptureArchive:
    def __init__(self):
        self.book_evs = {}

    def merge_player_books(self, league, market, date, player, book_evs, lines, observed_at=None):
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
    cv = stat_cv["NBA"]["BLK"]
    dist = stat_dist["NBA"]["BLK"]
    under = no_vig_odds(1.05, 9.0)[1]
    # The parser must thread the market dist + book_gate into get_ev; the
    # SkewNormal default (or a missing gate that changed the inversion) would
    # land on a different mean.
    expected_ev = get_ev(0.5, under, cv, dist=dist, gate=book_gate("NBA", "BLK", dist))
    assert ev == pytest.approx(expected_ev), (
        f"parser must invert under {dist}+book_gate, got ev={ev:.4f} vs {expected_ev:.4f} "
        "(the SkewNormal default diverges here)"
    )
    assert 0 < ev <= SN_MAX_MEAN_FACTOR * 0.5, f"sub-band quote must clamp, not run away: {ev}"
