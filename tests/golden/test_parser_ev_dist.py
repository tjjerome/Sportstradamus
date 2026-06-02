"""The props parser must invert each book price under the market's own
distribution, not the ``get_ev`` SkewNormal default.

For low-count markets (blocks/steals/tds, lines 0.5-1.5) the SkewNormal
inversion is ill-conditioned: its CDF asymptotes to ``Phi(-1/cv)``, so a
de-vigged under-prob just above that asymptote drives the solved mean toward
infinity and the archived ``ev`` explodes (up to ``line * 1e6``). Passing the
real distribution (ZINB/NegBin for counts) keeps the inversion bounded and
makes it the true inverse of the ``get_odds`` call training reads it back with.
Regression guard for the moneylines.py parser dist fix.
"""

import datetime

from sportstradamus.helpers import stat_cv, stat_dist
from sportstradamus.helpers.distributions import get_odds, no_vig_odds
from sportstradamus.moneylines import _archive_event_props


class _CaptureArchive:
    def __init__(self):
        self.book_evs = {}

    def merge_player_books(self, league, market, date, player, book_evs, lines, observed_at=None):
        self.book_evs[(market, player)] = dict(book_evs)

    def set_team_books(self, *args, **kwargs):
        pass


def test_low_count_prop_ev_round_trips_under_market_dist():
    # A center priced ~89% to clear 0.5 blocks: over short (1.05 decimal),
    # under long (9.0). Inverting this under the market's real ZINB keeps the
    # archived ev a true inverse of the get_odds call training reads it back
    # with; the SkewNormal default does not (and blows up in the asymptote band).
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
                            {"description": "Nikola Jokic", "name": "Over", "price": 1.05, "point": 0.5},
                            {"description": "Nikola Jokic", "name": "Under", "price": 9.0, "point": 0.5},
                        ],
                    }
                ],
            }
        ],
    }
    archive = _CaptureArchive()
    _archive_event_props(archive, game, "NBA", {"NBA": {"player_blocks": "BLK"}}, datetime.date(2025, 12, 23))

    ev = archive.book_evs[("BLK", "Nikola Jokic")]["draftkings"]
    cv = stat_cv["NBA"]["BLK"]
    dist = stat_dist["NBA"]["BLK"]
    expected_under = no_vig_odds(1.05, 9.0)[1]
    # Training recovers p_book via get_odds under the market's real dist; the
    # parser must have used the same dist for that to recover the book's quote.
    recovered_under = get_odds(0.5, ev, dist, cv=cv)
    assert abs(recovered_under - expected_under) < 0.02, (
        f"parser ev must round-trip to the book under-prob {expected_under:.3f}, got {recovered_under:.3f} "
        f"(ev={ev:.3f}) — the SkewNormal default breaks this"
    )
    assert 0 < ev < 10, f"low-count ev should stay bounded, got {ev}"
