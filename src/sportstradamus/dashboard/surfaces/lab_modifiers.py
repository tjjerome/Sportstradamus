"""Lab Modifiers — expected-vs-actual quote reconciler over dashboard slips.

Prices the slip already built in the session (the rail) or a locked slip the
way ``correlation.py`` would; the operator enters the app's actual quoted
payout and the solved pair modifier or parlay rake lands in
``data/runtime/modifier_overrides.json`` — never the committed configs
directly. ``helpers.config`` merges the overlay at load so scoring picks the
correction up immediately; the nightly ``reflect`` run folds it into
``banned_combos.json`` / ``parlay_rake.json`` (see
``helpers.parlay_modifiers``). The Games tab links here via its
"Payout incorrect?" chip.
"""

import json
import math
from importlib import resources
from itertools import combinations

import streamlit as st

from sportstradamus import data
from sportstradamus.dashboard.components.hero import page_hero
from sportstradamus.dashboard.data import load_current_offers
from sportstradamus.dashboard.legs import find_offer_idx
from sportstradamus.helpers.config import apply_modifier_overrides
from sportstradamus.helpers.io import (
    read_modifier_overrides,
    read_user_slips,
    write_modifier_overrides,
)
from sportstradamus.helpers.parlay_modifiers import (
    MATCH_TOLERANCE,
    MODIFIER_DECIMALS,
    Leg,
    reconcile,
    slip_pairs,
)
from sportstradamus.leg_schema import leg_label

_CONFIG_DIR = resources.files(data) / "config"


def _effective_tables(platform: str) -> tuple[dict, dict]:
    """Committed configs with the runtime overlay merged, string-keyed.

    ``helpers.config.banned`` is frozenset-keyed for parlay scoring; the
    reconciler needs the on-disk string keys, so it re-reads the json.
    """
    combos = json.loads((_CONFIG_DIR / "banned_combos.json").read_text())
    overlay = read_modifier_overrides()
    apply_modifier_overrides(combos, overlay.get("modifiers", {}))
    rake_file = _CONFIG_DIR / "parlay_rake.json"
    rakes = json.loads(rake_file.read_text()).get(platform, {}) if rake_file.is_file() else {}
    rakes.update(overlay.get("rake", {}).get(platform, {}))
    return combos.get(platform, {}), rakes


def _pos_prefix(league: str, position) -> str:
    """Banned-combos key prefix from a snapshot Position value."""
    text = "" if position is None else str(position)
    if text.lower() == "nan":
        text = ""
    if league == "MLB":
        return "P" if text.upper() in {"P", "SP", "RP"} else "B"
    return text


def _default_key(leg: dict, offers, platform: str) -> str:
    idx = find_offer_idx(leg, offers, platform)
    position = offers.at[idx, "Position"] if idx is not None else None
    prefix = _pos_prefix(leg["league"], position)
    return f"{prefix}.{leg['market']}" if prefix else str(leg["market"])


def _removal_hints(legs: list[Leg], labels: list[str], platform_mods: dict) -> list[str]:
    """Legs whose removal leaves exactly one unknown pair group to solve for."""
    hints = []
    for i in range(len(legs)):
        rest = legs[:i] + legs[i + 1 :]
        unknown = {p.group_key() for p in slip_pairs(rest, platform_mods) if p.value is None}
        if len(unknown) == 1:
            hints.append(labels[i])
    return hints


def _ambiguity_warning(n_unknown: int, hints: list[str]) -> str:
    msg = f"{n_unknown} unknown pair modifiers on this slip — one quote can only solve one."
    if hints:
        return f"{msg} Remove {' or '.join(hints)} to isolate a single unknown, then report again."
    return f"{msg} Add or remove a leg so exactly one unknown pair remains, then report again."


def _slot_name(slot: int) -> str:
    """Human label for a pair modifier's same-direction (0) / opposite-direction (1) slot."""
    return "same-dir" if slot == 0 else "opp-dir"


def _write_pair_override(
    platform: str, combos_platform: dict, group: tuple[str, str, str, int], value: float
) -> None:
    """Upsert one solved pair-modifier slot into the runtime overlay."""
    league, relation, key_str, slot = group
    league_mods = combos_platform.get(league, {"team": {}, "opponent": {}})
    entry = list(league_mods.get(relation, {}).get(key_str, [1.0, 1.0]))
    entry[slot] = value
    overlay = read_modifier_overrides()
    overlay.setdefault("modifiers", {}).setdefault(platform, {}).setdefault(league, {}).setdefault(
        relation, {}
    )[key_str] = entry
    write_modifier_overrides(overlay)


def _write_rake_override(platform: str, n_legs: int, value: float) -> None:
    """Upsert one solved parlay-rake entry into the runtime overlay."""
    overlay = read_modifier_overrides()
    overlay.setdefault("rake", {}).setdefault(platform, {})[str(n_legs)] = value
    write_modifier_overrides(overlay)


page_hero("MODEL LAB · MODIFIERS", "Modifier Reconciler")

slips = read_user_slips()
source_options = ["Current slip"] + ([] if slips.empty else ["Locked slip"])
source = st.radio("Slip source", source_options, horizontal=True)

if source == "Current slip":
    raw_legs = list(st.session_state.get("slip_legs", []))
    platform = st.session_state.get("slip_platform", "Underdog")
    source_id = "rail"
else:
    slips = slips.sort_values("saved_at", ascending=False)
    labels = {
        f"{row.headline} ({str(row.saved_at)[:10]})": row.slip_id for row in slips.itertuples()
    }
    picked = st.selectbox("Locked slip", list(labels))
    slip_row = slips.loc[slips["slip_id"] == labels[picked]].iloc[0]
    raw_legs = [dict(leg) for leg in slip_row["legs"]]
    platform = str(slip_row["platform"])
    source_id = str(slip_row["slip_id"])[:8]

if len(raw_legs) < 2:
    st.info("Build a slip on the Games tab first (two legs minimum), or pick a locked slip.")
    st.stop()

offers = load_current_offers()
combos_platform, rakes = _effective_tables(platform)

legs_input: list[Leg] = []
leg_labels: list[str] = []
st.caption("Leg | pair key (edit to match banned_combos convention) | bet | app multiplier")
for i, leg in enumerate(raw_legs):
    c_label, c_key, c_bet, c_mult = st.columns([3, 3, 2, 2])
    label = leg_label(leg)
    c_label.markdown(label)
    key_val = c_key.text_input(
        "pair key",
        value=_default_key(leg, offers, platform),
        key=f"mod_key_{source_id}_{i}",
        label_visibility="collapsed",
    )
    bet_val = c_bet.selectbox(
        "bet",
        ["Over", "Under"],
        index=0 if str(leg["bet"]) in {"Over", "Higher"} else 1,
        key=f"mod_bet_{source_id}_{i}",
        label_visibility="collapsed",
    )
    mult_val = c_mult.number_input(
        "multiplier",
        min_value=1.0,
        value=float(leg.get("boost") or 1.0),
        step=0.01,
        key=f"mod_mult_{source_id}_{i}",
        label_visibility="collapsed",
    )
    legs_input.append(
        Leg(
            str(leg["game"]),
            str(leg["team"]),
            str(leg["league"]),
            key_val,
            bet_val == "Over",
            float(mult_val),
        )
    )
    leg_labels.append(label)

pairs = slip_pairs(legs_input, combos_platform)
rake = rakes.get(str(len(legs_input)))
product = math.prod(leg.mult for leg in legs_input)
known = math.prod(p.value for p in pairs if p.value is not None)
expected = (rake if rake is not None else 1.0) * product * known

for p in pairs:
    shown = "UNKNOWN" if p.value is None else f"{p.value}"
    st.caption(f"pair [{p.league}/{p.relation}/{_slot_name(p.slot)}] {p.display_key}: {shown}")
rake_note = (
    f"rake[{len(legs_input)}]={rake}"
    if rake is not None
    else f"rake[{len(legs_input)}] uncalibrated"
)
st.metric(
    "Expected quote",
    f"{expected:.3f}x",
    help=f"product {product:.3f} x {rake_note} x known {known:.3f}",
)

unknown_groups = {p.group_key() for p in pairs if p.value is None}
if len(unknown_groups) > 1:
    st.warning(
        _ambiguity_warning(
            len(unknown_groups), _removal_hints(legs_input, leg_labels, combos_platform)
        )
    )

actual = st.number_input(
    "Actual quoted payout (x)",
    min_value=0.0,
    value=0.0,
    step=0.01,
    key=f"mod_actual_{source_id}",
)
pending_key = f"mod_pending_{source_id}"
pairwise_key = f"mod_pairwise_{source_id}"
if st.button("Save corrected modifiers", type="primary", disabled=actual <= 0):
    st.session_state[pending_key] = True

if st.session_state.get(pending_key):
    # A stale leg multiplier is the likeliest cause of a residual — odds move
    # constantly, modifiers rarely. Rake staleness only reaches the quote when
    # a prediction (TEAM.) leg puts the slip on the parlay-rake structure;
    # player-only slips ride the fixed pick'em payout tables.
    check_msg = (
        "Quick check: do the leg multipliers above match what the app shows right now? "
        "Odds move — edit any leg that changed and the expected quote updates."
    )
    if any(leg.key.startswith("TEAM.") for leg in legs_input):
        check_msg += (
            f" Team-line slips also ride rake[{len(legs_input)}] — if the legs check out "
            f"and the quote is still off, report a {len(legs_input)}-leg slip with no "
            "same-game pairs first to re-verify it."
        )
    st.info(check_msg)
    cancel_col, confirm_col = st.columns(2)
    if cancel_col.button("Cancel"):
        st.session_state.pop(pending_key, None)
        st.rerun()
    if confirm_col.button("Confirm save", type="primary", disabled=actual <= 0):
        st.session_state.pop(pending_key, None)
        kind, target, value = reconcile(legs_input, pairs, rake, actual, expected)
        if kind == "match":
            st.info("Matches expected — nothing to learn.")
        elif kind == "uncalibrated":
            st.warning(
                f"rake[{target}] unknown — report a {target}-leg slip "
                "with no same-game pairs first."
            )
        elif kind == "ambiguous" and unknown_groups:
            st.warning(
                _ambiguity_warning(
                    len(target), _removal_hints(legs_input, leg_labels, combos_platform)
                )
            )
        elif kind == "ambiguous":
            st.session_state[pairwise_key] = True
        elif kind == "rake":
            value = round(value, MODIFIER_DECIMALS)
            _write_rake_override(platform, target, value)
            st.success(f"Saved rake[{target}] = {value} for {platform} to the overlay.")
        else:
            league, relation, key_str, slot = target
            value = round(value, MODIFIER_DECIMALS)
            _write_pair_override(platform, combos_platform, target, value)
            st.success(
                f"Saved [{league}/{relation}/{_slot_name(slot)}] {key_str} = {value} to the overlay."
            )

# Pairwise isolation: every pair modifier on the slip is already on record and
# legs + rake were just confirmed, so the residual has no unique attribution.
# Each pair's own 2-leg quote solves its modifier directly.
if st.session_state.get(pairwise_key):
    st.info(
        "Every pair modifier here is already on record, so one quote can't say which went "
        "stale. Quote each pair alone in the app and enter the payouts below — any modifier "
        "that moved gets updated. If the app refuses a pair as a bare 2-leg slip (e.g. team "
        "line + same-team player), rebuild it as pair + one cross-game filler and report "
        "that slip separately — a single known pair solves exactly."
    )
    rake2 = rakes.get("2")
    cross = next(
        (
            (i, j)
            for i, j in combinations(range(len(legs_input)), 2)
            if legs_input[i].game != legs_input[j].game
        ),
        None,
    )
    rake2_quote = 0.0
    if rake2 is None and cross is not None:
        rake2_quote = st.number_input(
            f"2-leg cross-game payout — {leg_labels[cross[0]]} + {leg_labels[cross[1]]} "
            "(calibrates rake[2])",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key=f"mod_rake2_{source_id}",
        )
    pair_quotes = [
        st.number_input(
            f"Pair payout — {leg_labels[p.legs[0]]} + {leg_labels[p.legs[1]]}",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key=f"mod_pairq_{source_id}_{n}",
        )
        for n, p in enumerate(pairs)
    ]
    update_col, done_col = st.columns(2)
    if done_col.button("Done"):
        st.session_state.pop(pairwise_key, None)
        st.rerun()
    if update_col.button("Update stale modifiers", type="primary"):
        if rake2 is None and rake2_quote > 0:
            rake2 = round(
                rake2_quote / (legs_input[cross[0]].mult * legs_input[cross[1]].mult),
                MODIFIER_DECIMALS,
            )
            _write_rake_override(platform, 2, rake2)
            st.success(f"Calibrated rake[2] = {rake2} from the cross-game pair.")
        if rake2 is None:
            st.warning(
                "rake[2] unknown — enter the cross-game 2-leg payout above (or report any "
                "2-leg slip with no same-game pair) first."
            )
        else:
            for p, quote in zip(pairs, pair_quotes, strict=True):
                if quote <= 0:
                    continue
                mult = legs_input[p.legs[0]].mult * legs_input[p.legs[1]].mult
                new = round(quote / (rake2 * mult), MODIFIER_DECIMALS)
                if p.value and abs(new / p.value - 1) < MATCH_TOLERANCE:
                    st.caption(f"{p.display_key}: unchanged ({p.value}).")
                    continue
                _write_pair_override(platform, combos_platform, p.group_key(), new)
                st.success(
                    f"Updated [{p.league}/{p.relation}/{_slot_name(p.slot)}] {p.display_key} = "
                    f"{new} in the overlay."
                )

overlay_now = read_modifier_overrides()
n_pending = sum(
    len(entries)
    for leagues_d in overlay_now.get("modifiers", {}).values()
    for relations in leagues_d.values()
    for entries in relations.values()
) + sum(len(t) for t in overlay_now.get("rake", {}).values())
if n_pending:
    st.caption(
        f"{n_pending} correction(s) live in the overlay — scoring uses them now; "
        "the nightly reflect run folds them into the committed configs."
    )
