"""Valence and category totality pins for ``prediction.stories.legs``.

Every live market key must resolve through ``_stat_category``, and every
negative-market slug must land in ``"mistakes"`` ahead of the needle loop —
pinning the fix for the needle collisions that read mistake stats as thriving
ones (interceptions/sacks taken as stops, batter strikeouts as k's,
goalsAgainst / runs allowed / hits allowed as scoring).
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

from sportstradamus import data
from sportstradamus.prediction.stories.legs import _NEGATIVE_MARKETS, _stat_category


def _stat_meta_markets() -> set[str]:
    resource = pkg_resources.files(data) / "config" / "stat_meta.json"
    meta = json.loads(resource.read_text(encoding="utf-8"))
    return {market for markets in meta.values() for market in markets}


def test_every_market_resolves_to_a_category():
    for market in sorted(_stat_meta_markets() | {"TOV"}):
        category = _stat_category(market)
        assert isinstance(category, str) and category, market


def test_negative_markets_land_in_mistakes():
    for slug in sorted(_NEGATIVE_MARKETS):
        assert _stat_category(slug) == "mistakes", slug


def test_negative_needle_collisions_resolved():
    assert _stat_category("interceptions") == "mistakes"  # not "stops"
    assert _stat_category("sacks taken") == "mistakes"  # not "stops"
    assert _stat_category("batter strikeouts") == "mistakes"  # not "k's"
    assert _stat_category("goalsAgainst") == "mistakes"  # not "scoring"
    assert _stat_category("runs allowed") == "mistakes"  # not "scoring"
    assert _stat_category("hits allowed") == "mistakes"  # not "scoring"
    assert _stat_category("1st inning hits allowed") == "mistakes"  # not "scoring"
    assert _stat_category("TOV") == "mistakes"


def test_positive_cousins_stay_positive():
    assert _stat_category("walks") != "mistakes"
    assert _stat_category("pitcher strikeouts") == "k's"
    assert _stat_category("hits") == "scoring"
    assert _stat_category("goals") == "scoring"
