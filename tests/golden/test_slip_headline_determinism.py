"""Owner-requested determinism pin for the live slip thesis headline.

``slip_headline`` must be a pure function of the *canonical leg-set*: editing a
slip away and back reproduces the original headline byte-for-byte, and leg order
never changes the string. The exact prose is the phrase bank's job (pinned in
``test_stories.py``); here we only prove path- and order-independence.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.slip_engine import slip_headline
from sportstradamus.prediction.stories.context import GameCtx

_OFFERS = pd.DataFrame(
    [
        {"Player": "Jayson Tatum", "Bet": "Over", "Line": 28.5, "Market": "PTS",
         "Game": "BOS/PHI", "Team": "BOS", "Opponent": "PHI", "Position": "F1"},
        {"Player": "Jayson Tatum", "Bet": "Over", "Line": 8.5, "Market": "REB",
         "Game": "BOS/PHI", "Team": "BOS", "Opponent": "PHI", "Position": "F1"},
        {"Player": "Joel Embiid", "Bet": "Under", "Line": 30.5, "Market": "PTS",
         "Game": "BOS/PHI", "Team": "PHI", "Opponent": "BOS", "Position": "C1"},
        {"Player": "Nikola Jokic", "Bet": "Over", "Line": 9.5, "Market": "AST",
         "Game": "DEN/MIA", "Team": "DEN", "Opponent": "MIA", "Position": "C1"},
        {"Player": "Nikola Jokic", "Bet": "Over", "Line": 12.5, "Market": "REB",
         "Game": "DEN/MIA", "Team": "DEN", "Opponent": "MIA", "Position": "C1"},
    ]
)

_CTXS = {
    "BOS/PHI": GameCtx(league="NBA", game="BOS/PHI", date="2026-06-13", shape="blowout"),
    "DEN/MIA": GameCtx(league="NBA", game="DEN/MIA", date="2026-06-13", shape="even"),
}


def _leg(desc):
    return {"Desc": desc}


# A: a Tatum-driven BOS/PHI family → "player" archetype (subject = Jayson Tatum).
_SLIP_A = [
    _leg("Jayson Tatum Over 28.5 PTS - 59.0%, 1.00x"),
    _leg("Jayson Tatum Over 8.5 REB - 56.0%, 1.00x"),
    _leg("Joel Embiid Under 30.5 PTS - 57.0%, 1.00x"),
]
# B: a different leg-set entirely (Jokic in DEN/MIA) → a different subject/string.
_SLIP_B = [
    _leg("Nikola Jokic Over 9.5 AST - 60.0%, 1.00x"),
    _leg("Nikola Jokic Over 12.5 REB - 58.0%, 1.00x"),
]


def test_headline_path_independent():
    h_a = slip_headline(_SLIP_A, _OFFERS, _CTXS)
    assert h_a  # the Tatum family headlines — a real, non-empty string
    h_b = slip_headline(_SLIP_B, _OFFERS, _CTXS)
    assert h_b != h_a  # editing to a different leg-set changes the story
    h_a_again = slip_headline(_SLIP_A, _OFFERS, _CTXS)
    assert h_a_again == h_a  # ...and editing back reproduces it byte-for-byte


def test_headline_order_independent():
    h_a = slip_headline(_SLIP_A, _OFFERS, _CTXS)
    assert slip_headline(list(reversed(_SLIP_A)), _OFFERS, _CTXS) == h_a
