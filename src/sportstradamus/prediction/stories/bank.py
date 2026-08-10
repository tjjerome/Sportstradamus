"""Phrase-bank lookups for prophecy prose: JSON-backed cells plus the fallback chain.

Template strings live in ``data/config/`` JSON — a phrase bank is data, not
code. ``voice_bank.json`` holds thesis cells nested ``voice → archetype →
shape → direction → category → [variants]``; voices are ``basketball`` (NBA
and WNBA), ``football``, ``hockey``, ``baseball``, and the league-neutral
``shared`` safety net every chain ends on. Direction keys are *narrative*
(thrive/fade), not bet-literal: "Over" cells describe a thriving subject even
when a negative-market leg's actual bet is Under, so templates outside the
``mistakes`` category must carry mood, never literal bet-side words.
``why_bank.json`` holds the per-offer case clauses and story-dek clauses
(``{clause: {branch: [variants]}}`` plus the pronoun map). This module is the
only reader and stays pure stdlib so the dashboard can import it live.
"""

import functools
import importlib.resources as pkg_resources
import json

from sportstradamus import data


@functools.cache
def _bank(filename: str = "voice_bank.json") -> dict:
    """Load and memoize a packaged phrase-bank JSON by filename."""
    resource = pkg_resources.files(data) / "config" / filename
    return json.loads(resource.read_text(encoding="utf-8"))


def why_bank() -> dict:
    """The why/dek phrase bank: pronoun map plus clause → branch → variants."""
    return _bank("why_bank.json")


def team_assets() -> dict:
    """Team display assets: ``{league: {abbrev: {primary, secondary, name}}}``."""
    return _bank("team_assets.json")


def bank_cell(voice: str, archetype: str, shape: str, direction: str, category: str) -> list[str]:
    """Return the template variants for a cell via the voice → shared fallback chain.

    Tries ``(voice, archetype, shape, direction, category)``, then the same key
    under ``shared``, then ``shared`` with shape ``"even"``, and finally the
    guaranteed ``shared (archetype, "even", direction, "production")`` endpoint
    — never a KeyError, never an empty list, for any valid archetype/direction.
    A voice missing from the bank reads straight from ``shared``.
    """
    bank = _bank()
    shared = bank["shared"]
    for phrases, shape_key, category_key in (
        (bank.get(voice, shared), shape, category),
        (shared, shape, category),
        (shared, "even", category),
    ):
        cell = phrases.get(archetype, {}).get(shape_key, {}).get(direction, {}).get(category_key)
        if cell:
            return cell
    return shared[archetype]["even"][direction]["production"]
