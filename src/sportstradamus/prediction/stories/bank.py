"""Voice-bank lookup for prophecy theses: JSON-backed cells plus the fallback chain.

The template strings live in ``data/config/voice_bank.json`` — a phrase bank
is data, not code — nested ``voice → archetype → shape → direction →
category → [variants]``. Voices are ``basketball`` (NBA and WNBA),
``football``, ``hockey``, ``baseball``, and the league-neutral ``shared``
safety net every chain ends on. This module is the only reader and stays
pure stdlib so the dashboard can import it live.
"""

import functools
import importlib.resources as pkg_resources
import json

from sportstradamus import data


@functools.cache
def _bank() -> dict:
    """Load and memoize the nested voice-bank dict from the packaged JSON."""
    resource = pkg_resources.files(data) / "config" / "voice_bank.json"
    return json.loads(resource.read_text(encoding="utf-8"))


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
