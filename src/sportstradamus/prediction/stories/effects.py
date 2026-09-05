"""The ``up`` / ``down`` effect clauses a Mixed headline names.

A Mixed leg-set thrives on both sides at once, so its headline has to say *what*
rises and *what* falls rather than announce that the slip plays both ways. The
engine builds those two clauses here and hands them to the template as slots.
Every word of them comes from ``stat_words.json``, valence included — a negative
market's thriving clause reads "the turnovers stay down" because the bank says
so, never because the engine reasoned about a literal bet word.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sportstradamus.helpers import market_display_name
from sportstradamus.prediction.stories.bank import stat_words
from sportstradamus.prediction.stories.context import Leg
from sportstradamus.prediction.stories.legs import narrative_side


def split_effects(legs: Sequence[Leg], voice: str, league: str) -> dict[str, str]:
    """The rising and falling clauses for a Mixed leg-set, as ``{"up", "down"}``.

    Both narrative sides hold legs by construction — Mixed *is* the split. Each
    side normally speaks through its most common stat family, but two sides of
    one family would name the same noun twice, so there the clauses name the
    highest-conviction player on each side and the board that leg sits on.
    """
    thrive = [leg for leg in legs if narrative_side(leg) == "Over"]
    fade = [leg for leg in legs if narrative_side(leg) == "Under"]
    table = stat_words()[voice]
    up_family, down_family = _modal_family(thrive), _modal_family(fade)
    if up_family != down_family:
        up, down = table[up_family], table[down_family]
        return {
            "up": f"{up['noun']} {up['thrive']}",
            "down": f"{down['noun']} {down['fade']}",
        }
    shared_family = table[up_family]
    return {
        "up": _owned_clause(shared_family["owned_thrive"], thrive, league),
        "down": _owned_clause(shared_family["owned_fade"], fade, league),
    }


def _modal_family(side: Sequence[Leg]) -> str:
    counts = Counter(leg.category for leg in side)
    return min(counts, key=lambda family: (-counts[family], family))


def _owned_clause(template: str, side: Sequence[Leg], league: str) -> str:
    leg = max(side, key=lambda candidate: (candidate.win_prob or 0.0, candidate.player))
    board = market_display_name(league, leg.market)
    return template.format(who=leg.player, what=_prose_case(board))


def _prose_case(board: str) -> str:
    """A board name lowercased for mid-clause prose, acronyms left alone.

    "Total Bases" reads as a title in a column header and as shouting inside a
    sentence, but "PRA" and "RBIs" are how those boards are spelled anywhere.
    """
    return " ".join(
        word if word.isupper() or (word.endswith("s") and word[:-1].isupper()) else word.lower()
        for word in board.split()
    )
