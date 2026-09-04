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
    """Return the template variants for a cell via the shape-first fallback chain.

    Shape outranks both voice and category: a shootout keeps shootout copy even
    when the voice authors no such cell and the category has to degrade to
    ``production``, because shape-blind prose contradicts the game it describes
    ("low ceiling — fade the scoring" on a Coors slate). Under a shape node the
    chain probes ``category`` then ``production`` and nothing else, so the
    authoring invariant the design rests on is that every authored shaped node
    carries a ``production`` cell. Only when neither the voice nor ``shared``
    authors *any* cell for that shape and direction does the chain fall to
    ``"even"``, and finally to the guaranteed ``shared (archetype, "even",
    direction, "production")`` endpoint. A voice missing from the bank reads
    straight from ``shared``.
    """
    bank = _bank()
    shared = bank["shared"]
    voiced = bank.get(voice, shared)
    for phrases, shape_key, category_key in (
        (voiced, shape, category),
        (voiced, shape, "production"),
        (shared, shape, category),
        (shared, shape, "production"),
        # Defense in depth: while shared authors every archetype/shape/direction
        # at production, no mapped league reaches these last two or the endpoint.
        (voiced, "even", category),
        (shared, "even", category),
    ):
        cell = phrases.get(archetype, {}).get(shape_key, {}).get(direction, {}).get(category_key)
        if cell:
            return cell
    return shared[archetype]["even"][direction]["production"]
