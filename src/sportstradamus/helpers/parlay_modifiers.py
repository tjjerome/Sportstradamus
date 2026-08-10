"""Expected-vs-actual quote reconciliation for pick'em correlation modifiers.

Platform slip quotes are computed server-side and automated authed probing is
off the table (dfs-products locked decision, 2026-07-10), so extraction
inverts the human path: price a slip the way ``correlation.py`` would — leg
multipliers x parlay rake x known ``banned_combos.json`` modifiers — and let
the operator report the app's actual quote through the dashboard Modifiers
page. The residual factors out the one unknown pair modifier
(``[same-direction, opposite-direction]`` slots, the ``_leg_pair_corr_boost``
contract); a slip with no same-game pairs calibrates the per-leg-count parlay
rake instead (``parlay_rake.json``).

Captures land in ``data/runtime/modifier_overrides.json`` — never the
committed configs directly. ``helpers.config`` merges the overlay at import
so scoring picks corrections up immediately; :func:`fold_overlay` applies the
overlay onto the committed configs (nightly on the server, post-pull and on
``sync_from_prod.sh`` with ``prune=True``). The overlay persists until a
pulled checkout already carries its values, so cron git pulls can reset the
two config files without losing corrections.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from importlib import resources
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

from sportstradamus import data
from sportstradamus.helpers.config import apply_modifier_overrides
from sportstradamus.helpers.io import MODIFIER_OVERRIDES_PATH

# A learned modifier is a ratio of two operator-read quotes; the app displays
# them at 2 decimals, so a third digit is already at the noise floor.
MODIFIER_DECIMALS = 3
# Quotes within half a percent of expected carry no signal worth writing.
MATCH_TOLERANCE = 0.005


class Leg(NamedTuple):
    game: str
    team: str
    league: str
    key: str
    higher: bool
    mult: float


class Pair(NamedTuple):
    league: str
    relation: str
    display_key: str
    disk_key: str | None
    slot: int
    value: float | None
    legs: tuple[int, int]

    def group_key(self) -> tuple[str, str, str, int]:
        """Identity used to attribute a residual to this pair's modifier slot."""
        return self.league, self.relation, self.disk_key or self.display_key, self.slot


def slip_pairs(legs: list[Leg], platform_mods: dict) -> list[Pair]:
    """Classify same-game leg pairs and look up their known modifiers.

    ``platform_mods`` is one platform's section of ``banned_combos.json``
    (league -> team/opponent tables). Cross-game pairs are skipped (modifier 1
    by construction); same-game legs share a league. Slot 0 is
    same-direction, slot 1 opposite — the ``_leg_pair_corr_boost`` contract.
    """
    pairs = []
    for i, j in combinations(range(len(legs)), 2):
        a, b = legs[i], legs[j]
        if a.game != b.game:
            continue
        relation = "team" if a.team == b.team else "opponent"
        slot = 0 if a.higher == b.higher else 1
        table = platform_mods.get(a.league, {"team": {}, "opponent": {}})[relation]
        disk_key = next(
            (k for k in (f"{a.key} & {b.key}", f"{b.key} & {a.key}") if k in table),
            None,
        )
        value = table[disk_key][slot] if disk_key else None
        pairs.append(Pair(a.league, relation, f"{a.key} & {b.key}", disk_key, slot, value, (i, j)))
    return pairs


def reconcile(legs, pairs, rake, actual, expected):
    """Return (kind, target, value) for what the actual quote teaches.

    kind: "match" | "rake" | "solve" | "drift" | "ambiguous" | "uncalibrated"
    """
    if abs(actual / expected - 1) < MATCH_TOLERANCE:
        return "match", None, None
    if not pairs:
        return "rake", len(legs), actual / math.prod(leg.mult for leg in legs)
    if rake is None:
        return "uncalibrated", len(legs), None
    unknown = Counter(p.group_key() for p in pairs if p.value is None)
    if len(unknown) > 1:
        return "ambiguous", sorted(unknown), None
    ratio = actual / expected
    if len(unknown) == 1:
        ((target, count),) = unknown.items()
        return "solve", target, ratio ** (1 / count)
    known = Counter(p.group_key() for p in pairs)
    if len(known) != 1:
        return "ambiguous", sorted(known), None
    ((target, count),) = known.items()
    old = next(p.value for p in pairs)
    return "drift", target, old * ratio ** (1 / count)


def _prune_synced_modifiers(modifiers: dict, combos: dict) -> int:
    """Drop overlay pair entries the committed combos already carry; return the count."""
    pruned = 0
    for platform, leagues in list(modifiers.items()):
        for league, relations in list(leagues.items()):
            for relation, entries in list(relations.items()):
                committed = combos.get(platform, {}).get(league, {}).get(relation, {})
                for key in [k for k, v in entries.items() if committed.get(k) == v]:
                    del entries[key]
                    pruned += 1
                if not entries:
                    del relations[relation]
            if not relations:
                del leagues[league]
        if not leagues:
            del modifiers[platform]
    return pruned


def _prune_synced_rakes(rake_overrides: dict, rake_config: dict) -> int:
    """Drop overlay rake entries the committed rake config already carries."""
    pruned = 0
    for platform, table in list(rake_overrides.items()):
        committed = rake_config.get(platform, {})
        for k in [k for k, v in table.items() if committed.get(k) == v]:
            del table[k]
            pruned += 1
        if not table:
            del rake_overrides[platform]
    return pruned


def fold_overlay(
    overlay_path: Path | None = None,
    combos_path: Path | None = None,
    rake_path: Path | None = None,
    *,
    prune: bool = False,
) -> dict[str, int]:
    """Apply the dashboard-captured overlay onto the committed modifier configs.

    The overlay is never cleared outright — it persists as the source of truth
    for corrections devel hasn't absorbed yet, so a git pull that resets the
    two config files loses nothing. With ``prune=True`` (post-pull, dev sync)
    entries whose value already equals the committed one are dropped first;
    without it (nightly) everything is re-applied idempotently.
    """
    config_dir = resources.files(data) / "config"
    overlay_path = Path(str(overlay_path or MODIFIER_OVERRIDES_PATH))
    combos_path = Path(str(combos_path or config_dir / "banned_combos.json"))
    rake_path = Path(str(rake_path or config_dir / "parlay_rake.json"))
    if not overlay_path.is_file():
        return {"pairs": 0, "rakes": 0, "pruned": 0}

    overlay = json.loads(overlay_path.read_text())
    modifiers = overlay.get("modifiers", {})
    rake_overrides = overlay.get("rake", {})
    combos = json.loads(combos_path.read_text())
    rake_config = json.loads(rake_path.read_text()) if rake_path.exists() else {}

    pruned = 0
    if prune:
        pruned = _prune_synced_modifiers(modifiers, combos)
        pruned += _prune_synced_rakes(rake_overrides, rake_config)
        overlay_path.write_text(
            json.dumps({"modifiers": modifiers, "rake": rake_overrides}, indent=2) + "\n"
        )

    apply_modifier_overrides(combos, modifiers)
    combos_path.write_text(json.dumps(combos, indent=4) + "\n")
    for platform_name, table in rake_overrides.items():
        rake_config.setdefault(platform_name, {}).update(table)
    rake_path.write_text(json.dumps(rake_config, indent=4) + "\n")

    n_pairs = sum(
        len(entries)
        for leagues in modifiers.values()
        for relations in leagues.values()
        for entries in relations.values()
    )
    n_rakes = sum(len(t) for t in rake_overrides.values())
    return {"pairs": n_pairs, "rakes": n_rakes, "pruned": pruned}
