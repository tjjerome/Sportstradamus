"""Prophecy prose: per-offer "the case" strings and per-family thesis headlines.

Pipeline-side replacement for the dashboard's render-time phrase bank. Both
``attach_*`` functions are pure (no I/O, no archive, no network), deterministic,
and produce the prose the dashboard reads verbatim from the parquet snapshots
(``Why`` on ``current_offers``, ``Thesis`` on ``current_parlays``). Diversity
comes from bank size + context keying + a per-snapshot ``Date`` seed — never
``random`` — so a given snapshot always renders the same copy, yet the same
matchup-shape rotates headlines day to day.

Game shape (shootout / grind / blowout / coinflip ...) is read from the offer
columns ``O/U`` (raw game total) and ``Moneyline``. In this snapshot
``Moneyline`` is the team's *implied win probability* in ``(0, 1)`` (not American
odds): the favorite is the side nearer 1.0 and ``max|p - 0.5|`` over a game's
offers measures how lopsided the matchup is.
"""

from sportstradamus.prediction.stories.context import build_game_context
from sportstradamus.prediction.stories.legs import parse_leg
from sportstradamus.prediction.stories.menu import build_game_stories
from sportstradamus.prediction.stories.thesis import attach_parlay_theses
from sportstradamus.prediction.stories.why import attach_offer_why

STORIES_VERSION: str = "p3a"

__all__ = [
    "STORIES_VERSION",
    "attach_offer_why",
    "attach_parlay_theses",
    "build_game_context",
    "build_game_stories",
    "parse_leg",
]
