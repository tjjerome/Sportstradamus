"""Render-only translation from a market slug to its human display name.

The slug (``PTS``, ``qb-yards``, ...) stays the logic key everywhere — training,
the archive, offer matching. This module only maps that key to prose for
display; nothing here is parsed back.
"""

import importlib.resources as pkg_resources
import json
from functools import cache

from sportstradamus import data
from sportstradamus.helpers.archive import archive_market


@cache
def _market_display() -> dict:
    with (pkg_resources.files(data) / "config/market_display.json").open() as f:
        return json.load(f)


def market_display_name(league: str, slug: str) -> str:
    """Display label for a market slug; the slug stays the logic key everywhere.

    The label is read under the league's own key, so a board name from a platform's
    map (NHL ``AST``, NBA ``fantasy points underdog``) reads like the league's other rows.
    """
    key = archive_market(league, slug)
    return _market_display().get(league, {}).get(key, key)
