"""Pipeline adapter: attach the per-leg-set ``Thesis`` headline to parlays.

``attach_parlay_theses`` parses each parlay row's legs, enriches them against the
scored offers, builds per-game context, and routes each *distinct leg-set*
through the v2 engine. A slate-uniqueness pass guarantees no two leg-sets in the
same ``(League, Date)`` render the same headline. Parlay-family clustering plays
no part here — the ``Family`` column survives only as a Slips grouping device.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from sportstradamus.prediction.stories.context import build_game_context, ctxs_from_frame
from sportstradamus.prediction.stories.engine import thesis_variants
from sportstradamus.prediction.stories.legs import enrich_legs, parse_leg


@dataclass
class _Pick:
    """A leg-set's rendered variant list + the chosen index the slate pass bumps."""

    variants: list[str]
    index: int
    text: str


def attach_parlay_theses(
    parlays: pd.DataFrame,
    offers: pd.DataFrame,
    *,
    corr=None,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a ``Thesis`` column: one headline per distinct leg-set, written on every row.

    ``context`` is the precomputed ``current_game_context`` frame and ``corr`` the
    ``current_game_corr`` slices; when ``context`` is omitted it is built from
    ``offers`` directly (the dashboard rail's live-edit path). The frame is
    mutated in place and returned.
    """
    if parlays.empty:
        parlays["Thesis"] = pd.Series(dtype=str)
        return parlays

    if context is None:
        context = build_game_context(offers, {})
    ctxs = ctxs_from_frame(context, corr)
    leg_cols = [c for c in parlays.columns if c.startswith("Leg ")]

    picks: dict[tuple, _Pick] = {}
    row_keys: list[tuple] = []
    for _, row in parlays.iterrows():
        legs = enrich_legs(
            [leg for leg in (parse_leg(row.get(c)) for c in leg_cols) if leg], offers
        )
        key = (
            row["League"],
            str(row.get("Date", "")),
            tuple(sorted((leg.player, leg.market, leg.bet) for leg in legs)),
        )
        row_keys.append(key)
        if key not in picks:
            variants, idx, _ = thesis_variants(legs, ctxs)
            picks[key] = _Pick(variants, idx, variants[idx] if variants else "")
    _dedupe_slate(picks)

    parlays["Thesis"] = [picks[k].text for k in row_keys]
    return parlays


def _dedupe_slate(picks: dict[tuple, _Pick]) -> None:
    """Guarantee distinct theses within each ``(League, Date)`` slate.

    Walks each slate's leg-sets in key order; on a collision advances to the next
    rendered variant in the cell, falling back to a numeric suffix once the cell
    is exhausted.
    """
    slates: dict[tuple, list[tuple]] = defaultdict(list)
    for key in picks:
        slates[(key[0], key[1])].append(key)
    for keys in slates.values():
        seen: set[str] = set()
        for key in sorted(keys):
            pick = picks[key]
            if not pick.text:
                continue
            _bump_until_unique(pick, seen)
            seen.add(pick.text)


def _bump_until_unique(pick: _Pick, seen: set[str]) -> None:
    for step in range(len(pick.variants)):
        idx = (pick.index + step) % len(pick.variants)
        if pick.variants[idx] not in seen:
            pick.index, pick.text = idx, pick.variants[idx]
            return
    suffix = 2
    while f"{pick.text} ({suffix})" in seen:
        suffix += 1
    pick.text = f"{pick.text} ({suffix})"
