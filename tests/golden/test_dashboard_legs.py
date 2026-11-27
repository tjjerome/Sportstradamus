"""Pins for the leg helpers every surface shares: the picks' record line."""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.legs import record_line


def _record(play: str, league: str, kind: str, n: int, roi: float) -> dict:
    return {
        "play": play,
        "league": league,
        "kind": kind,
        "n": n,
        "dates": 30,
        "first_date": "2026-09-01",
        "last_date": "2026-11-25",
        "hit_rate": 0.561,
        "breakeven_rate": 0.546,
        "roi": roi,
        "roi_lo": roi - 0.068,
        "roi_hi": roi + 0.069,
    }


# Every other play's row for the leg's league and kind sits before the ev_pick row, and the
# ev_pick rows of other leagues and kinds around it, so a lookup that skips any of the three
# keys reads a wrong row.
_RECORD = pd.DataFrame(
    [
        _record("model_edge", "NHL", "underdog_even", 900, -0.081),
        _record("ev_rank", "NHL", "underdog_even", 310, 0.012),
        _record("ev_pick", "NBA", "underdog_even", 40, 0.05),
        _record("ev_pick", "NHL", "all", 324, 0.017),
        _record("ev_pick", "NHL", "sleeper", 55, -0.03),
        _record("ev_pick", "NHL", "underdog_even", 269, 0.027),
        _record("model_edge", "NFL", "underdog_even", 120, -0.04),
    ]
)


def test_the_record_line_reads_the_picks_row_of_the_legs_league_and_kind():
    line, hover = record_line(_RECORD, "NHL", "underdog_even")
    assert line == "NHL even picks: 269 legs, +2.7%"
    assert "-4.1% to +9.6%" in hover
    assert record_line(_RECORD, "NHL", "sleeper")[0] == "NHL Sleeper picks: 55 legs, -3.0%"


def test_a_kind_with_no_picks_settled_shows_no_other_plays_record():
    assert record_line(_RECORD, "NFL", "underdog_even") == ("NFL even picks: 0 legs yet", None)
    assert record_line(_RECORD, "NHL", "underdog_cut") == (
        "NHL reduced-payout picks: 0 legs yet",
        None,
    )
