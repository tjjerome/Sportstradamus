"""Live DFS scrapers: Underdog Fantasy (:mod:`.underdog`) and Sleeper (:mod:`.sleeper`).

These are the two DFS platforms still scraped directly. Sportsbook odds
(DraftKings, FanDuel, Pinnacle, Caesars, etc.) are ingested via the Odds
API in :mod:`sportstradamus.moneylines`; the deprecated per-book scrapers
live in ``src/deprecated/books_deprecated.py``.
"""

from sportstradamus.books.sleeper import get_sleeper
from sportstradamus.books.underdog import get_ud

__all__ = ["get_sleeper", "get_ud"]
