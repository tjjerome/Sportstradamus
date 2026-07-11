#!/usr/bin/env python3
"""Interactive reconciler for pick'em correlation-modifier extraction.

Platform slip quotes are computed server-side and automated authed probing is
off the table (dfs-products locked decision, 2026-07-10), so this tool
inverts the human path instead: it prices a slip the way ``correlation.py``
would — leg multipliers x parlay rake x known ``banned_combos.json``
modifiers — shows the expected quote, and the operator types in the app's
actual quote. The residual factors out the one unknown pair modifier, which
is written back to ``banned_combos.json`` (``[same-direction,
opposite-direction]`` slots, matching ``_leg_pair_corr_boost``). A slip with
no same-game pairs calibrates the per-leg-count parlay rake instead
(``parlay_rake.json``).

Leg spec — whitespace-separated legs, quote a leg whose market name contains
spaces::

    <game>:<team>:<POS.market>:<h|l>:<multiplier>

    1:SD:TEAM.moneyline:h:1.79 "1:SD:P.pitcher strikeouts:h:1.62" 2:LAD:B.hits+runs+rbi:h:2.33

Team/prediction legs use the ``TEAM.<market>`` key convention. Enter the
app-displayed multiplier for prediction legs — the platform's own quote math
uses the fee-excluded display figure, so it reconciles; fee adjustment
belongs to EV computation, not extraction.

Usage
-----
    poetry run python -m sportstradamus.scripts.calibrate_parlay_modifiers --league MLB
"""

from __future__ import annotations

import json
import math
import shlex
from collections import Counter
from importlib import resources
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import click

from sportstradamus import data
from sportstradamus.helpers.config import apply_modifier_overrides
from sportstradamus.helpers.io import MODIFIER_OVERRIDES_PATH

# A learned modifier is a ratio of two operator-read quotes; the app displays
# them at 2 decimals, so a third digit is already at the noise floor. Public
# (no leading underscore) — dashboard/surfaces/lab_modifiers.py imports it to
# keep its rounding policy in sync with this CLI's.
MODIFIER_DECIMALS = 3
# Quotes within half a percent of expected carry no signal worth writing.
_MATCH_TOLERANCE = 0.005

_HIGHER_WORDS = {"h", "higher", "o", "over"}
_LOWER_WORDS = {"l", "lower", "u", "under"}


class Leg(NamedTuple):
    game: str
    team: str
    key: str
    higher: bool
    mult: float


class Pair(NamedTuple):
    relation: str
    display_key: str
    disk_key: str | None
    slot: int
    value: float | None

    def group_key(self) -> tuple[str, str, int]:
        """Identity used to attribute a residual to this pair's modifier slot."""
        return self.relation, self.disk_key or self.display_key, self.slot


def parse_slip(text: str, *, min_legs: int = 2) -> list[Leg]:
    """Parse a whitespace-separated slip spec into legs (see module doc)."""
    legs = []
    for tok in shlex.split(text):
        parts = tok.split(":")
        if len(parts) != 5:
            raise ValueError(f"leg '{tok}' needs game:team:POS.market:h|l:multiplier")
        game, team, key, direction, mult = parts
        if "." not in key:
            raise ValueError(f"leg '{tok}': market must be POS.market, e.g. B.hits")
        d = direction.lower()
        if d not in _HIGHER_WORDS | _LOWER_WORDS:
            raise ValueError(f"leg '{tok}': direction must be one of h/l/higher/lower")
        try:
            multiplier = float(mult)
        except ValueError as exc:
            raise ValueError(f"leg '{tok}': bad multiplier '{mult}'") from exc
        legs.append(Leg(game, team, key, d in _HIGHER_WORDS, multiplier))
    if len(legs) < min_legs:
        raise ValueError(f"a slip needs at least {min_legs} legs")
    return legs


def slip_pairs(legs: list[Leg], league_mods: dict) -> list[Pair]:
    """Classify same-game leg pairs and look up their known modifiers.

    Cross-game pairs are skipped (modifier 1 by construction). Slot 0 is
    same-direction, slot 1 opposite — the ``_leg_pair_corr_boost`` contract.
    """
    pairs = []
    for i, j in combinations(range(len(legs)), 2):
        a, b = legs[i], legs[j]
        if a.game != b.game:
            continue
        relation = "team" if a.team == b.team else "opponent"
        slot = 0 if a.higher == b.higher else 1
        table = league_mods[relation]
        disk_key = next(
            (k for k in (f"{a.key} & {b.key}", f"{b.key} & {a.key}") if k in table),
            None,
        )
        value = table[disk_key][slot] if disk_key else None
        pairs.append(Pair(relation, f"{a.key} & {b.key}", disk_key, slot, value))
    return pairs


def _write_modifier(path: Path, platform: str, league: str, pair: Pair, value: float) -> None:
    config = json.loads(path.read_text())
    table = config[platform][league][pair.relation]
    entry = table.get(pair.disk_key or pair.display_key, [1.0, 1.0])
    entry[pair.slot] = value
    table[pair.disk_key or pair.display_key] = entry
    path.write_text(json.dumps(config, indent=4) + "\n")


def _write_rake(path: Path, platform: str, n_legs: int, value: float) -> None:
    config = json.loads(path.read_text()) if path.exists() else {}
    config.setdefault(platform, {})[str(n_legs)] = value
    path.write_text(json.dumps(config, indent=4) + "\n")


def _fold_overlay(overlay_path: Path, combos_path: Path, rake_path: Path) -> None:
    """Migrate dashboard-captured overrides into the committed configs."""
    if not overlay_path.is_file():
        click.echo(f"no overlay at {overlay_path}")
        return
    overlay = json.loads(overlay_path.read_text())
    modifiers = overlay.get("modifiers", {})
    rake_overrides = overlay.get("rake", {})

    combos = json.loads(combos_path.read_text())
    apply_modifier_overrides(combos, modifiers)
    combos_path.write_text(json.dumps(combos, indent=4) + "\n")

    rake_config = json.loads(rake_path.read_text()) if rake_path.exists() else {}
    for platform_name, table in rake_overrides.items():
        rake_config.setdefault(platform_name, {}).update(table)
    rake_path.write_text(json.dumps(rake_config, indent=4) + "\n")

    overlay_path.write_text(json.dumps({"modifiers": {}, "rake": {}}, indent=4) + "\n")
    n_pairs = sum(
        len(entries)
        for leagues in modifiers.values()
        for relations in leagues.values()
        for entries in relations.values()
    )
    n_rakes = sum(len(t) for t in rake_overrides.values())
    click.echo(f"folded {n_pairs} pair modifiers + {n_rakes} rakes; overlay cleared")


def reconcile(legs, pairs, rake, actual, expected):
    """Return (kind, target, value) for what the actual quote teaches.

    kind: "match" | "rake" | "solve" | "drift" | "ambiguous" | "uncalibrated"
    """
    if abs(actual / expected - 1) < _MATCH_TOLERANCE:
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


@click.command()
@click.option("--platform", default="Underdog", show_default=True)
@click.option("--league", default=None, help="League section of banned_combos.json, e.g. MLB")
@click.option(
    "--combos-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override banned_combos.json location (tests/maintenance).",
)
@click.option(
    "--rake-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override parlay_rake.json location (tests/maintenance).",
)
@click.option(
    "--fold-overlay",
    is_flag=True,
    help="Fold data/runtime/modifier_overrides.json (dashboard captures) into the committed configs, then clear it.",
)
@click.option(
    "--overlay-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override modifier_overrides.json location (tests/maintenance).",
)
def calibrate_parlay_modifiers(
    platform, league, combos_path, rake_path, fold_overlay, overlay_path
):
    """Solve unknown pair modifiers from expected-vs-actual slip quotes."""
    config_dir = resources.files(data) / "config"
    combos_path = combos_path or Path(str(config_dir / "banned_combos.json"))
    rake_path = rake_path or Path(str(config_dir / "parlay_rake.json"))

    if fold_overlay:
        overlay_path = overlay_path or Path(str(MODIFIER_OVERRIDES_PATH))
        _fold_overlay(overlay_path, combos_path, rake_path)
        return

    if league is None:
        raise click.UsageError("--league is required in interactive mode")
    league = league.upper()
    combos = json.loads(combos_path.read_text())
    if platform not in combos or league not in combos[platform]:
        raise click.UsageError(f"{platform}/{league} not present in {combos_path}")

    rakes = json.loads(rake_path.read_text()).get(platform, {}) if rake_path.exists() else {}
    click.echo("enter slips (blank or q to quit); leg spec: game:team:POS.market:h|l:multiplier")

    while True:
        text = click.prompt("slip", default="", show_default=False)
        if not text or text.lower() in {"q", "quit"}:
            break
        try:
            legs = parse_slip(text)
        except ValueError as exc:
            click.echo(f"  {exc}")
            continue

        league_mods = combos[platform][league]
        pairs = slip_pairs(legs, league_mods)
        rake = rakes.get(str(len(legs)))
        product = math.prod(leg.mult for leg in legs)
        known = math.prod(p.value for p in pairs if p.value is not None)
        expected = (rake if rake is not None else 1.0) * product * known

        for p in pairs:
            slot_name = "same-dir" if p.slot == 0 else "opp-dir"
            shown = f"{p.value}" if p.value is not None else "UNKNOWN"
            click.echo(f"  pair [{p.relation}/{slot_name}] {p.display_key}: {shown}")
        rake_note = (
            f"rake[{len(legs)}]={rake}" if rake is not None else f"rake[{len(legs)}] UNCALIBRATED"
        )
        click.echo(
            f"  expected: {expected:.3f}x  (product {product:.3f} x {rake_note} x known {known:.3f})"
        )

        actual = click.prompt("actual quote (blank if it matches)", default=0.0, show_default=False)
        if not actual:
            continue

        kind, target, value = reconcile(legs, pairs, rake, actual, expected)
        if kind == "match":
            click.echo("  matches expected — nothing to learn")
        elif kind == "uncalibrated":
            click.echo(
                f"  rake[{target}] unknown — quote an all-cross-game {target}-leg slip first"
            )
        elif kind == "ambiguous":
            click.echo(f"  cannot attribute residual across multiple pair groups: {target}")
        elif kind == "rake":
            value = round(value, MODIFIER_DECIMALS)
            if click.confirm(f"  write rake[{target}] = {value} for {platform}?"):
                _write_rake(rake_path, platform, target, value)
                rakes[str(target)] = value
        else:
            relation, key, slot = target
            value = round(value, MODIFIER_DECIMALS)
            verb = "solve" if kind == "solve" else "drift-update"
            slot_name = "same-dir" if slot == 0 else "opp-dir"
            pair = next(p for p in pairs if p.group_key() == target)
            if click.confirm(f"  {verb} [{relation}/{slot_name}] {key} = {value}, write?"):
                _write_modifier(combos_path, platform, league, pair, value)
                combos = json.loads(combos_path.read_text())


if __name__ == "__main__":
    calibrate_parlay_modifiers()
