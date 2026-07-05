"""Render-only translation from a market slug to its human display name.

The slug (``PTS``, ``qb-yards``, ...) stays the logic key everywhere — training,
the archive, offer matching. This module only maps that key to prose for
display; nothing here is parsed back.
"""

import importlib.resources as pkg_resources
import json
from functools import cache

from sportstradamus import data


@cache
def _market_display() -> dict:
    with (pkg_resources.files(data) / "config/market_display.json").open() as f:
        return json.load(f)


def market_display_name(league: str, slug: str) -> str:
    """Display label for a market slug; the slug stays the logic key everywhere."""
    return _market_display().get(league, {}).get(slug, slug)
